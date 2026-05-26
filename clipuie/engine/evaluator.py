from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from clipuie.config import EvaluationSection
from clipuie.utils import ImageMetricRunner, tensor_to_uint8


class Evaluator:
    def __init__(self, model: torch.nn.Module, device: torch.device, config: EvaluationSection, prediction_dir: Path | None = None) -> None:
        self.model = model
        self.device = device
        self.config = config
        self.prediction_dir = prediction_dir
        self.metric_runner = ImageMetricRunner(config.compute_uiqm, config.compute_uciqe)
        self.calibration = self._load_calibration(config.calibration_path)
        if config.save_images and prediction_dir is not None:
            (prediction_dir / "output").mkdir(parents=True, exist_ok=True)
            (prediction_dir / "comparison").mkdir(parents=True, exist_ok=True)
            (prediction_dir / "input").mkdir(parents=True, exist_ok=True)
            (prediction_dir / "target").mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> dict[str, float]:
        self.model.eval()
        aggregated: dict[str, list[float]] = defaultdict(list)
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            masks = batch["mask"].to(self.device)
            names = batch["name"]
            prompts = list(batch["prompt"])
            if self.config.self_ensemble:
                outputs = self._predict_self_ensemble(inputs, masks, prompts)
                _, output_list = self._predict_once(
                    inputs,
                    masks,
                    prompts,
                    need_branch_outputs=self.config.compute_branch_metrics or self.config.output_branch_index is not None,
                )
            else:
                outputs, output_list = self._predict_once(
                    inputs,
                    masks,
                    prompts,
                    need_branch_outputs=self.config.compute_branch_metrics or self.config.output_branch_index is not None,
                )
            outputs = self._apply_calibration(outputs)
            for index, name in enumerate(names):
                input_sample = inputs[index : index + 1]
                target_sample = targets[index : index + 1]
                output_sample = outputs[index : index + 1]
                metrics = self.metric_runner.evaluate_pair(output_sample, target_sample)
                for key, value in metrics.items():
                    aggregated[key].append(value)
                if self.config.compute_branch_metrics:
                    for branch_index, branch_output in enumerate(output_list):
                        branch_metrics = self.metric_runner.evaluate_pair(branch_output[index : index + 1], target_sample)
                        aggregated[f"psnr_branch_{branch_index}"].append(branch_metrics["psnr_256"])
                if self.config.save_images and self.prediction_dir is not None:
                    input_uint8 = tensor_to_uint8(input_sample)
                    output_uint8 = tensor_to_uint8(output_sample)
                    target_uint8 = tensor_to_uint8(target_sample)
                    canvas = np.concatenate(
                        [input_uint8, output_uint8, target_uint8],
                        axis=1,
                    )
                    Image.fromarray(output_uint8.astype(np.uint8)).save(self.prediction_dir / "output" / name)
                    Image.fromarray(input_uint8.astype(np.uint8)).save(self.prediction_dir / "input" / name)
                    Image.fromarray(target_uint8.astype(np.uint8)).save(self.prediction_dir / "target" / name)
                    Image.fromarray(canvas.astype(np.uint8)).save(self.prediction_dir / "comparison" / name)
        return {key: float(np.mean(values)) for key, values in aggregated.items()}

    def _load_calibration(self, path: str | None) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not path:
            return None
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        matrix = torch.tensor(raw["matrix"], dtype=torch.float32, device=self.device)
        bias = torch.tensor(raw["bias"], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        return matrix, bias

    def _apply_calibration(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.calibration is None:
            return outputs
        matrix, bias = self.calibration
        calibrated = torch.einsum("ij,bjhw->bihw", matrix, outputs) + bias
        return calibrated

    def _predict_once(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        prompts: list[str],
        need_branch_outputs: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        route_result = self.model.forward_route(
            inputs,
            masks,
            prompts,
            return_logits=True,
            return_proc_outs=need_branch_outputs,
            hard_route=self.config.hard_route,
        )
        if need_branch_outputs:
            outputs, _, output_list = route_result
        else:
            outputs, _ = route_result
            output_list = []
        if self.config.output_branch_index is not None:
            branch_index = int(self.config.output_branch_index)
            if branch_index < 0 or branch_index >= len(output_list):
                raise ValueError(
                    f"evaluation.output_branch_index={branch_index} is out of range for "
                    f"{len(output_list)} available branches."
                )
            outputs = output_list[branch_index]
        return outputs, output_list

    @staticmethod
    def _augment_tensor(tensor: torch.Tensor, mode: int) -> torch.Tensor:
        if mode & 1:
            tensor = torch.flip(tensor, dims=(-1,))
        if mode & 2:
            tensor = torch.flip(tensor, dims=(-2,))
        if mode & 4:
            tensor = tensor.transpose(-1, -2)
        return tensor.contiguous()

    @staticmethod
    def _deaugment_tensor(tensor: torch.Tensor, mode: int) -> torch.Tensor:
        if mode & 4:
            tensor = tensor.transpose(-1, -2)
        if mode & 2:
            tensor = torch.flip(tensor, dims=(-2,))
        if mode & 1:
            tensor = torch.flip(tensor, dims=(-1,))
        return tensor.contiguous()

    def _predict_self_ensemble(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        prompts: list[str],
    ) -> torch.Tensor:
        outputs = []
        for mode in range(8):
            aug_inputs = self._augment_tensor(inputs, mode)
            aug_masks = self._augment_tensor(masks, mode)
            aug_outputs, _ = self._predict_once(
                aug_inputs,
                aug_masks,
                prompts,
                need_branch_outputs=self.config.output_branch_index is not None,
            )
            outputs.append(self._deaugment_tensor(aug_outputs, mode))
        return torch.stack(outputs, dim=0).mean(dim=0)
