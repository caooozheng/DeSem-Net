from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from clipuie.config import ExperimentConfig
from clipuie.engine.checkpoint import save_checkpoint
from clipuie.engine.evaluator import Evaluator
from clipuie.losses import CompositeReconstructionLoss, select_best_outputs
from clipuie.utils import compute_psnr_batch


class ClipUIETrainer:
    def __init__(self, model: torch.nn.Module, config: ExperimentConfig, device: torch.device, run_dirs: dict[str, Path]) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.run_dirs = run_dirs
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_checkpoint_dir = run_dirs["checkpoint_runs"] / self.run_id
        self.run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.loss_fn = CompositeReconstructionLoss().to(device)
        self._build_optimizers()
        self.evaluator = Evaluator(model=self.model, device=device, config=config.evaluation, prediction_dir=run_dirs["predictions"])
        if config.training.pretrained_checkpoint:
            checkpoint = torch.load(config.training.pretrained_checkpoint, map_location=device)
            self.model.load_state_dict(checkpoint, strict=config.training.strict_load)

    def _region_loss(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        foreground_weight = float(self.config.training.lambda_foreground)
        background_weight = float(self.config.training.lambda_background)
        if foreground_weight <= 0 and background_weight <= 0:
            return prediction.new_tensor(0.0)

        mask = mask.to(device=prediction.device, dtype=prediction.dtype).clamp(0.0, 1.0)
        background_mask = 1.0 - mask
        region_loss = prediction.new_tensor(0.0)
        if foreground_weight > 0:
            region_loss = region_loss + foreground_weight * F.l1_loss(prediction * mask, target * mask)
        if background_weight > 0:
            region_loss = region_loss + background_weight * F.l1_loss(prediction * background_mask, target * background_mask)
        return region_loss

    def _build_optimizers(self) -> None:
        generator_params = []
        router_params = []
        for name, parameter in self.model.named_parameters():
            if "adaptive_route" in name:
                router_params.append(parameter)
            else:
                generator_params.append(parameter)
        betas = tuple(self.config.optimizer.betas)
        self.optimizer_g = AdamW(generator_params, lr=self.config.optimizer.generator_lr, betas=betas, weight_decay=self.config.optimizer.weight_decay)
        self.optimizer_router = AdamW(router_params, lr=self.config.optimizer.router_lr, betas=betas, weight_decay=self.config.optimizer.weight_decay)
        self.scheduler_g = MultiStepLR(self.optimizer_g, milestones=self.config.scheduler.generator_milestones, gamma=self.config.scheduler.generator_gamma)
        self.scheduler_router = MultiStepLR(self.optimizer_router, milestones=self.config.scheduler.router_milestones, gamma=self.config.scheduler.router_gamma)

    def fit(self, train_loader: torch.utils.data.DataLoader, eval_loader: torch.utils.data.DataLoader) -> dict[str, float]:
        best_psnr = float("-inf")
        best_metrics: dict[str, float] = {}
        for epoch in range(self.config.training.epochs):
            train_metrics = self._train_one_epoch(train_loader, epoch)
            eval_metrics = self.evaluator.evaluate(eval_loader)
            self.scheduler_g.step()
            self.scheduler_router.step()
            if eval_metrics["psnr_256"] > best_psnr:
                best_psnr = eval_metrics["psnr_256"]
                best_metrics = eval_metrics
                save_checkpoint(self.model.state_dict(), self.run_dirs["checkpoints"] / "best.pth")
                save_checkpoint(self.model.state_dict(), self.run_checkpoint_dir / "best.pth")
            if self.config.training.checkpoint_interval > 0 and epoch % self.config.training.checkpoint_interval == 0:
                save_checkpoint(self.model.state_dict(), self.run_checkpoint_dir / f"epoch-{epoch:05d}.pth")
            print({"epoch": epoch, "train_loss": train_metrics["loss"], **eval_metrics})
        return best_metrics

    def _train_one_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            masks = batch["mask"].to(self.device)
            prompts = list(batch["prompt"])
            self.optimizer_g.zero_grad()
            recon_input = self.model.forward_recon(inputs, masks, prompts)
            loss = self.loss_fn(recon_input, inputs)
            loss = loss + self._region_loss(recon_input, inputs, masks)
            recon_target = self.model.forward_recon(targets, masks, prompts)
            loss = loss + self.loss_fn(recon_target, targets)
            loss = loss + self._region_loss(recon_target, targets, masks)
            loss.backward(retain_graph=True)
            output, style_loss = self.model.forward_style_loss(inputs, targets, masks, prompts)
            routed_loss = self.loss_fn(output, targets) + style_loss * self.config.training.lambda_style
            routed_loss = routed_loss + self._region_loss(output, targets, masks)
            routed_loss.backward(retain_graph=True)
            total_loss = routed_loss
            if epoch >= self.config.training.route_start_epoch:
                with torch.no_grad():
                    output_r = inputs
                    psnrs = [compute_psnr_batch(output_r, targets).unsqueeze(-1)]
                    for _ in range(self.config.model.num_branch - 1):
                        output_r = self.model.forward(output_r, masks, prompts)
                        psnrs.append(compute_psnr_batch(output_r, targets).unsqueeze(-1))
                    max_idx = torch.argmax(torch.cat(psnrs, dim=1), dim=1)
                self.optimizer_router.zero_grad()
                output, logits, style_loss, output_list = self.model.forward_route_style_loss(
                    inputs,
                    targets,
                    masks,
                    prompts,
                    return_logits=True,
                    return_proc_outs=True,
                )
                route_loss = F.cross_entropy(logits, max_idx)
                total_loss = self.loss_fn(output, targets) + route_loss * self.config.training.lambda_route + style_loss * self.config.training.lambda_style
                total_loss = total_loss + self._region_loss(output, targets, masks)
                best_outputs, valid_targets = select_best_outputs(output_list, targets, max_idx)
                if best_outputs is not None:
                    total_loss = total_loss + self.loss_fn(best_outputs, valid_targets)
                total_loss.backward(retain_graph=True)
                self.optimizer_router.step()
            self.optimizer_g.step()
            running_loss += float(total_loss.detach().cpu().item())
        return {"loss": running_loss / max(1, len(train_loader))}
