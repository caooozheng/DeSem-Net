from __future__ import annotations

from collections import defaultdict
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
            outputs, _, output_list = self.model.forward_route(
                inputs,
                masks,
                prompts,
                return_logits=True,
                return_proc_outs=True,
            )
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
                    canvas = np.concatenate(
                        [tensor_to_uint8(input_sample), tensor_to_uint8(output_sample), tensor_to_uint8(target_sample)],
                        axis=1,
                    )
                    Image.fromarray(canvas.astype(np.uint8)).save(self.prediction_dir / name)
        return {key: float(np.mean(values)) for key, values in aggregated.items()}
