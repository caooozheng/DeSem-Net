from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from clipuie.config import ExperimentConfig
from clipuie.engine.checkpoint import load_model_state, save_checkpoint
from clipuie.engine.evaluator import Evaluator
from clipuie.losses import CompositeReconstructionLoss, LabColorConsistencyLoss, SoftHistogramLoss, select_best_outputs
from clipuie.models.ops import GetGradientNopadding
from clipuie.utils import compute_psnr_batch


def compute_ssim_batch_torch(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    padding = window_size // 2
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mu_x = F.avg_pool2d(prediction, window_size, stride=1, padding=padding, count_include_pad=False)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding, count_include_pad=False)
    sigma_x = F.avg_pool2d(prediction * prediction, window_size, stride=1, padding=padding, count_include_pad=False) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(target * target, window_size, stride=1, padding=padding, count_include_pad=False) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(prediction * target, window_size, stride=1, padding=padding, count_include_pad=False) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)


def compute_color_consistency_batch(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    pred_mean = prediction.mean(dim=(2, 3))
    target_mean = target.mean(dim=(2, 3))
    pred_std = prediction.std(dim=(2, 3), unbiased=False)
    target_std = target.std(dim=(2, 3), unbiased=False)
    mean_error = torch.abs(pred_mean - target_mean).mean(dim=1)
    contrast_error = torch.abs(pred_std - target_std).mean(dim=1)
    return (1.0 - mean_error - 0.5 * contrast_error).clamp(0.0, 1.0)


class ClipUIETrainer:
    def __init__(self, model: torch.nn.Module, config: ExperimentConfig, device: torch.device, run_dirs: dict[str, Path]) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.run_dirs = run_dirs
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_checkpoint_dir = run_dirs["checkpoint_runs"] / self.run_id
        self.run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.loss_fn = CompositeReconstructionLoss(
            l1_weight=config.training.loss_l1_weight,
            l2_weight=config.training.loss_l2_weight,
            perceptual_weight=config.training.loss_perceptual_weight,
            gradient_weight=config.training.loss_gradient_weight,
        ).to(device)
        self.lab_color_loss = LabColorConsistencyLoss().to(device)
        self.histogram_loss = SoftHistogramLoss().to(device)
        self.get_gradient = GetGradientNopadding().to(device)
        self._apply_trainable_filter()
        self._build_optimizers()
        eval_config = replace(config.evaluation, save_images=False)
        self.evaluator = Evaluator(model=self.model, device=device, config=eval_config, prediction_dir=run_dirs["predictions"])
        self.start_epoch = 0
        self.best_psnr = float("-inf")
        self.best_metrics: dict[str, float] = {}
        self.use_ema = bool(config.training.use_ema)
        self.ema_decay = float(config.training.ema_decay)
        self.ema_state: dict[str, torch.Tensor] | None = None
        self._active_phase_index: int | None = None
        self._load_training_state()
        if self.use_ema and self.ema_state is None:
            self._init_ema()

    def _apply_trainable_filter(self) -> None:
        keywords = list(self.config.training.trainable_keywords)
        if not keywords:
            return
        trainable_count = 0
        frozen_count = 0
        for name, parameter in self.model.named_parameters():
            is_trainable = any(keyword in name for keyword in keywords)
            parameter.requires_grad_(is_trainable)
            if is_trainable:
                trainable_count += parameter.numel()
            else:
                frozen_count += parameter.numel()
        print(
            "Applied trainable parameter filter: "
            f"trainable={trainable_count:,}, frozen={frozen_count:,}, keywords={keywords}"
        )

    def _load_training_state(self) -> None:
        if self.config.training.resume_checkpoint:
            checkpoint = torch.load(self.config.training.resume_checkpoint, map_location=self.device)
            if "model_state_dict" not in checkpoint:
                raise ValueError("resume_checkpoint must be a full checkpoint saved by the trainer, not a plain model state_dict.")
            checkpoint_start_epoch = int(checkpoint.get("epoch", -1)) + 1
            checkpoint_best_psnr = float(checkpoint.get("best_psnr", float("-inf")))
            checkpoint_best_metrics = dict(checkpoint.get("best_metrics", {}))
            load_result, skipped_keys = load_model_state(self.model, checkpoint, strict=self.config.training.strict_load)
            if self.use_ema and "ema_state_dict" in checkpoint:
                self.ema_state = {
                    key: value.detach().clone().cpu()
                    for key, value in checkpoint["ema_state_dict"].items()
                }
            if load_result.missing_keys or load_result.unexpected_keys:
                print(
                    "Checkpoint model key mismatch: "
                    f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}"
                )
                if load_result.missing_keys:
                    print(f"Missing keys sample: {load_result.missing_keys[:10]}")
                if load_result.unexpected_keys:
                    print(f"Unexpected keys sample: {load_result.unexpected_keys[:10]}")
                if skipped_keys:
                    print(f"Skipped shape-mismatched keys sample: {skipped_keys[:10]}")
            model_state_changed = bool(load_result.missing_keys or load_result.unexpected_keys or skipped_keys)
            try:
                if model_state_changed:
                    raise ValueError("model parameters changed relative to checkpoint")
                self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
                if self.optimizer_router is not None and "optimizer_router_state_dict" in checkpoint:
                    self.optimizer_router.load_state_dict(checkpoint["optimizer_router_state_dict"])
                self.scheduler_g.load_state_dict(checkpoint["scheduler_g_state_dict"])
                if self.scheduler_router is not None and "scheduler_router_state_dict" in checkpoint:
                    self.scheduler_router.load_state_dict(checkpoint["scheduler_router_state_dict"])
                loaded_optimizer_state = True
            except (KeyError, ValueError) as exc:
                loaded_optimizer_state = False
                print(
                    "Optimizer or scheduler state is incompatible with the current model; "
                    "restarting from epoch 0 with loaded model weights and freshly initialized optimizer state."
                )
                print(f"Optimizer resume detail: {exc}")
            if loaded_optimizer_state:
                self.start_epoch = checkpoint_start_epoch
                self.best_psnr = checkpoint_best_psnr
                self.best_metrics = checkpoint_best_metrics
                run_id = checkpoint.get("run_id")
                if run_id:
                    self.run_id = str(run_id)
                    self.run_checkpoint_dir = self.run_dirs["checkpoint_runs"] / self.run_id
                    self.run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                resume_mode = "full training state"
            else:
                self.start_epoch = 0
                self.best_psnr = float("-inf")
                self.best_metrics = {}
                resume_mode = "model weights only"
            print(f"Resumed training from {self.config.training.resume_checkpoint} at epoch {self.start_epoch} ({resume_mode}).")
            return

        if self.config.training.pretrained_checkpoint:
            checkpoint = torch.load(self.config.training.pretrained_checkpoint, map_location=self.device)
            load_result, skipped_keys = load_model_state(self.model, checkpoint, strict=self.config.training.strict_load)
            if skipped_keys:
                print(f"Skipped shape-mismatched pretrained keys sample: {skipped_keys[:10]}")

    def _init_ema(self) -> None:
        self.ema_state = {
            key: value.detach().clone().cpu()
            for key, value in self.model.state_dict().items()
        }

    @torch.no_grad()
    def _update_ema(self) -> None:
        if not self.use_ema or self.ema_state is None:
            return
        model_state = self.model.state_dict()
        for key, value in model_state.items():
            value_cpu = value.detach().cpu()
            if not value.is_floating_point():
                self.ema_state[key] = value_cpu.clone()
                continue
            self.ema_state[key].mul_(self.ema_decay).add_(value_cpu, alpha=1.0 - self.ema_decay)

    @contextmanager
    def _ema_scope(self):
        if not self.use_ema or self.ema_state is None:
            yield
            return
        raw_state = {
            key: value.detach().clone().cpu()
            for key, value in self.model.state_dict().items()
        }
        self.model.load_state_dict(self.ema_state, strict=False)
        try:
            yield
        finally:
            self.model.load_state_dict(raw_state, strict=False)

    def _checkpoint_state(self, epoch: int, use_ema_model: bool = False) -> dict:
        current_state = self.model.state_dict()
        model_state = self.ema_state if use_ema_model and self.ema_state is not None else current_state
        state = {
            "epoch": epoch,
            "run_id": self.run_id,
            "best_psnr": self.best_psnr,
            "best_metrics": self.best_metrics,
            "model_state_dict": {key: value.detach().cpu().clone() for key, value in model_state.items()},
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_router_state_dict": self.optimizer_router.state_dict() if self.optimizer_router is not None else None,
            "scheduler_g_state_dict": self.scheduler_g.state_dict(),
            "scheduler_router_state_dict": self.scheduler_router.state_dict() if self.scheduler_router is not None else None,
        }
        if self.ema_state is not None:
            state["ema_state_dict"] = {key: value.detach().cpu().clone() for key, value in self.ema_state.items()}
        if use_ema_model and self.ema_state is not None:
            state["raw_model_state_dict"] = {key: value.detach().cpu().clone() for key, value in current_state.items()}
        return state

    def _region_loss(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        foreground_weight = float(self.config.training.lambda_foreground)
        background_weight = float(self.config.training.lambda_background)
        if foreground_weight <= 0 and background_weight <= 0:
            return prediction.new_tensor(0.0)

        mask = mask.to(device=prediction.device, dtype=prediction.dtype).clamp(0.0, 1.0)
        background_mask = 1.0 - mask
        region_loss = prediction.new_tensor(0.0)
        if foreground_weight > 0:
            region_loss = region_loss + foreground_weight * self._masked_l1_loss(prediction, target, mask)
        if background_weight > 0:
            region_loss = region_loss + background_weight * self._masked_l1_loss(prediction, target, background_mask)
        return region_loss

    def _final_regularization(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        aux: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        total = prediction.new_tensor(0.0)
        if self.config.training.lambda_physical > 0:
            total = total + self.config.training.lambda_physical * self._physical_consistency_loss(prediction, inputs, aux)
        if self.config.training.lambda_lab > 0:
            total = total + self.config.training.lambda_lab * self.lab_color_loss(prediction, target)
        if self.config.training.lambda_histogram > 0:
            total = total + self.config.training.lambda_histogram * self.histogram_loss(prediction, target)
        if self.config.training.lambda_boundary > 0:
            total = total + self.config.training.lambda_boundary * self._boundary_gradient_loss(prediction, target, mask)
        if self.config.training.lambda_foreground_texture > 0:
            total = total + self.config.training.lambda_foreground_texture * self._masked_gradient_loss(prediction, target, mask)
        return total

    @staticmethod
    def _physical_consistency_loss(
        prediction: torch.Tensor,
        inputs: torch.Tensor,
        aux: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if aux is None:
            return prediction.new_tensor(0.0)
        transmission = aux["transmission"].to(device=prediction.device, dtype=prediction.dtype)
        ambient = aux["ambient"].to(device=prediction.device, dtype=prediction.dtype)
        reconstructed_input = prediction.clamp(0.0, 1.0) * transmission + ambient * (1.0 - transmission)
        reconstruction_loss = F.l1_loss(reconstructed_input, inputs.clamp(0.0, 1.0))
        smooth_h = torch.abs(transmission[:, :, :, 1:] - transmission[:, :, :, :-1]).mean()
        smooth_v = torch.abs(transmission[:, :, 1:, :] - transmission[:, :, :-1, :]).mean()
        return reconstruction_loss + 0.05 * (smooth_h + smooth_v)

    def _boundary_gradient_loss(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=prediction.device, dtype=prediction.dtype).clamp(0.0, 1.0)
        mask_min = -F.max_pool2d(-mask, kernel_size=5, stride=1, padding=2)
        mask_max = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)
        boundary = (mask_max - mask_min).clamp(0.0, 1.0)
        return self._masked_gradient_loss(prediction, target, boundary)

    def _masked_gradient_loss(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        prediction_gradient = self.get_gradient(prediction.clamp(0.0, 1.0), gray=False)
        target_gradient = self.get_gradient(target.clamp(0.0, 1.0), gray=False)
        return self._masked_l1_loss(prediction_gradient, target_gradient, mask)

    @staticmethod
    def _masked_l1_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        abs_error = torch.abs(prediction - target) * mask
        per_sample_error = abs_error.sum(dim=(1, 2, 3))
        per_sample_area = mask.sum(dim=(1, 2, 3)).clamp_min(1.0) * prediction.size(1)
        return (per_sample_error / per_sample_area).mean()

    def _build_optimizers(self) -> None:
        generator_params = []
        router_params = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "adaptive_route" in name:
                router_params.append(parameter)
            else:
                generator_params.append(parameter)
        betas = tuple(self.config.optimizer.betas)
        self.optimizer_g = AdamW(generator_params, lr=self.config.optimizer.generator_lr, betas=betas, weight_decay=self.config.optimizer.weight_decay)
        self.optimizer_router = (
            AdamW(router_params, lr=self.config.optimizer.router_lr, betas=betas, weight_decay=self.config.optimizer.weight_decay)
            if router_params
            else None
        )
        self.scheduler_g = MultiStepLR(self.optimizer_g, milestones=self.config.scheduler.generator_milestones, gamma=self.config.scheduler.generator_gamma)
        self.scheduler_router = (
            MultiStepLR(self.optimizer_router, milestones=self.config.scheduler.router_milestones, gamma=self.config.scheduler.router_gamma)
            if self.optimizer_router is not None
            else None
        )

    def _apply_phase_schedule(self, epoch: int) -> None:
        phases = sorted(
            list(self.config.training.phase_schedule),
            key=lambda item: int(item.get("start_epoch", 0)),
        )
        active_index = None
        active_phase = None
        for index, phase in enumerate(phases):
            if epoch >= int(phase.get("start_epoch", 0)):
                active_index = index
                active_phase = phase
        if active_phase is None:
            return

        for key, value in active_phase.items():
            if key in {"name", "start_epoch", "generator_lr", "router_lr"}:
                continue
            if hasattr(self.config.training, key):
                setattr(self.config.training, key, value)

        if "generator_lr" in active_phase:
            for param_group in self.optimizer_g.param_groups:
                param_group["lr"] = float(active_phase["generator_lr"])
        if "router_lr" in active_phase and self.optimizer_router is not None:
            for param_group in self.optimizer_router.param_groups:
                param_group["lr"] = float(active_phase["router_lr"])

        if active_index != self._active_phase_index:
            self._active_phase_index = active_index
            if any(
                key in active_phase
                for key in ("loss_l1_weight", "loss_l2_weight", "loss_perceptual_weight", "loss_gradient_weight")
            ):
                self.loss_fn = CompositeReconstructionLoss(
                    l1_weight=self.config.training.loss_l1_weight,
                    l2_weight=self.config.training.loss_l2_weight,
                    perceptual_weight=self.config.training.loss_perceptual_weight,
                    gradient_weight=self.config.training.loss_gradient_weight,
                ).to(self.device)
            phase_name = active_phase.get("name", f"phase_{active_index}")
            generator_lr = self.optimizer_g.param_groups[0]["lr"]
            router_lr = self.optimizer_router.param_groups[0]["lr"] if self.optimizer_router is not None else None
            print(
                "Activated training phase: "
                f"{phase_name} at epoch {epoch}, generator_lr={generator_lr}, router_lr={router_lr}"
            )

    def _select_route_targets(self, output_candidates: list[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        psnr_weight = float(self.config.training.route_score_psnr_weight)
        ssim_weight = float(self.config.training.route_score_ssim_weight)
        color_weight = float(self.config.training.route_score_color_weight)
        scores = []
        for output_candidate in output_candidates:
            psnr_score = compute_psnr_batch(output_candidate, targets).clamp(0.0, 40.0) / 40.0
            ssim_score = compute_ssim_batch_torch(output_candidate, targets)
            color_score = compute_color_consistency_batch(output_candidate, targets)
            scores.append(psnr_weight * psnr_score + ssim_weight * ssim_score + color_weight * color_score)
        return torch.argmax(torch.stack(scores, dim=1), dim=1)

    def _route_targets_from_outputs(self, output_list: list[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._select_route_targets([output.detach() for output in output_list], targets)

    def _branch_supervision_loss(self, output_list: list[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        branch_weight = float(self.config.training.lambda_branch_supervision)
        if branch_weight <= 0 or not output_list:
            return targets.new_tensor(0.0)
        weights = targets.new_tensor([0.25, 1.0, 0.5])
        if len(output_list) != int(weights.numel()):
            weights = torch.ones(len(output_list), device=targets.device, dtype=targets.dtype)
        weights = weights / weights.sum().clamp_min(1e-6)
        total = targets.new_tensor(0.0)
        for branch_output, weight in zip(output_list, weights):
            total = total + weight * self.loss_fn(branch_output, targets)
        return branch_weight * total

    def _psnr_mse_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_weight = float(self.config.training.lambda_psnr_mse)
        if mse_weight <= 0:
            return prediction.new_tensor(0.0)
        return mse_weight * F.mse_loss(prediction, target)

    def _ssim_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_weight = float(self.config.training.lambda_ssim_loss)
        if ssim_weight <= 0:
            return prediction.new_tensor(0.0)
        return ssim_weight * (1.0 - compute_ssim_batch_torch(prediction, target).mean())

    def _color_mse_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        color_weight = float(self.config.training.lambda_color_mse)
        if color_weight <= 0:
            return prediction.new_tensor(0.0)
        prediction = prediction.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)
        pred_mean = prediction.mean(dim=(2, 3))
        target_mean = target.mean(dim=(2, 3))
        pred_std = prediction.std(dim=(2, 3), unbiased=False)
        target_std = target.std(dim=(2, 3), unbiased=False)
        return color_weight * (F.mse_loss(pred_mean, target_mean) + 0.5 * F.mse_loss(pred_std, target_std))

    def _select_training_output(self, route_output: torch.Tensor, output_list: list[torch.Tensor]) -> torch.Tensor:
        branch_index = self.config.evaluation.output_branch_index
        if branch_index is None or not output_list:
            return route_output
        branch_index = int(branch_index)
        if branch_index < 0 or branch_index >= len(output_list):
            return route_output
        return output_list[branch_index]

    def fit(self, train_loader: torch.utils.data.DataLoader, eval_loader: torch.utils.data.DataLoader) -> dict[str, float]:
        for epoch in range(self.start_epoch, self.config.training.epochs):
            self._apply_phase_schedule(epoch)
            train_metrics = self._train_one_epoch(train_loader, epoch)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            with self._ema_scope():
                eval_metrics = self.evaluator.evaluate(eval_loader)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            self.scheduler_g.step()
            if self.scheduler_router is not None:
                self.scheduler_router.step()
            latest_state = self._checkpoint_state(epoch)
            save_checkpoint(latest_state, self.run_dirs["checkpoints"] / "latest.pth")
            save_checkpoint(latest_state, self.run_checkpoint_dir / "latest.pth")
            if eval_metrics["psnr_256"] > self.best_psnr:
                self.best_psnr = eval_metrics["psnr_256"]
                self.best_metrics = eval_metrics
                best_state = self._checkpoint_state(epoch, use_ema_model=self.use_ema)
                save_checkpoint(best_state, self.run_dirs["checkpoints"] / "best.pth")
                save_checkpoint(best_state, self.run_checkpoint_dir / "best.pth")
            if self.config.training.checkpoint_interval > 0 and epoch % self.config.training.checkpoint_interval == 0:
                save_checkpoint(self._checkpoint_state(epoch), self.run_checkpoint_dir / f"epoch-{epoch:05d}.pth")
            print({"epoch": epoch, "train_loss": train_metrics["loss"], **eval_metrics})
        return self.best_metrics

    def _train_one_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            masks = batch["mask"].to(self.device)
            prompts = list(batch["prompt"])
            self.optimizer_g.zero_grad(set_to_none=True)
            if self.optimizer_router is not None:
                self.optimizer_router.zero_grad(set_to_none=True)

            recon_weight = float(self.config.training.lambda_recon)
            if recon_weight > 0:
                recon_input = self.model.forward_recon(inputs, masks, prompts)
                recon_input_loss = recon_weight * (self.loss_fn(recon_input, inputs) + self._region_loss(recon_input, inputs, masks))
                recon_input_loss.backward()

                recon_target = self.model.forward_recon(targets, masks, prompts)
                recon_target_loss = recon_weight * (self.loss_fn(recon_target, targets) + self._region_loss(recon_target, targets, masks))
                recon_target_loss.backward()

            route_active = epoch >= self.config.training.route_start_epoch
            if not route_active:
                output, style_loss, aux = self.model.forward_style_loss(inputs, targets, masks, prompts, return_aux=True)
                route_report_loss = self.loss_fn(output, targets) + style_loss * self.config.training.lambda_style
                route_report_loss = route_report_loss + self._region_loss(output, targets, masks)
                route_report_loss = route_report_loss + self._final_regularization(output, targets, inputs, masks, aux)
                route_report_loss = route_report_loss + self._psnr_mse_loss(output, targets)
                route_report_loss = route_report_loss + self._ssim_loss(output, targets)
                route_report_loss = route_report_loss + self._color_mse_loss(output, targets)
                route_report_loss.backward()
            else:
                output, logits, style_loss, output_list, aux = self.model.forward_route_style_loss(
                    inputs,
                    targets,
                    masks,
                    prompts,
                    return_logits=True,
                    return_proc_outs=True,
                    return_aux=True,
                )
                max_idx = self._route_targets_from_outputs(output_list, targets)
                training_output = self._select_training_output(output, output_list)
                route_loss = F.cross_entropy(logits, max_idx)
                route_report_loss = self.loss_fn(training_output, targets) + route_loss * self.config.training.lambda_route
                route_report_loss = route_report_loss + self._region_loss(training_output, targets, masks)
                route_report_loss = route_report_loss + self._final_regularization(training_output, targets, inputs, masks, aux)
                route_report_loss = route_report_loss + self._psnr_mse_loss(training_output, targets)
                route_report_loss = route_report_loss + self._ssim_loss(training_output, targets)
                route_report_loss = route_report_loss + self._color_mse_loss(training_output, targets)
                route_report_loss = route_report_loss + self._branch_supervision_loss(output_list, targets)
                best_outputs, valid_targets = select_best_outputs(output_list, targets, max_idx)
                if best_outputs is not None:
                    route_report_loss = route_report_loss + 0.25 * self.loss_fn(best_outputs, valid_targets)
                    route_report_loss = route_report_loss + 0.25 * self._psnr_mse_loss(best_outputs, valid_targets)
                    route_report_loss = route_report_loss + 0.25 * self._ssim_loss(best_outputs, valid_targets)
                    route_report_loss = route_report_loss + 0.25 * self._color_mse_loss(best_outputs, valid_targets)
                route_report_loss.backward()

            if route_active:
                if self.optimizer_router is not None:
                    self.optimizer_router.step()
            self.optimizer_g.step()
            self._update_ema()
            running_loss += float(route_report_loss.detach().cpu().item())
        return {"loss": running_loss / max(1, len(train_loader))}
