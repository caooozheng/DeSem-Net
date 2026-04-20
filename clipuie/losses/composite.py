from __future__ import annotations

import torch
import torch.nn as nn

from clipuie.losses.perceptual import VGG19PerceptualLoss
from clipuie.losses.pixel import WeightedPixelLoss
from clipuie.models.ops import GetGradientNopadding


class CompositeReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pixel = WeightedPixelLoss()
        self.perceptual = VGG19PerceptualLoss()
        self.gradient = nn.L1Loss()
        self.get_gradient = GetGradientNopadding()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pixel_loss = self.pixel(prediction, target)
        perceptual_loss = self.perceptual(prediction, target)
        gradient_loss = self.gradient(
            self.get_gradient(prediction, gray=False),
            self.get_gradient(target, gray=False),
        )
        return pixel_loss + 0.3 * perceptual_loss + 0.1 * gradient_loss


def select_best_outputs(
    output_list: list[torch.Tensor],
    ground_truth: torch.Tensor,
    max_indices: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    valid_mask = max_indices != 0
    if valid_mask.sum().item() == 0:
        return None, None
    outputs_cat = torch.stack(output_list, dim=1)
    valid_outputs = outputs_cat[valid_mask]
    valid_indices = max_indices[valid_mask]
    gather_index = valid_indices.view(-1, 1, 1, 1, 1).expand(-1, 1, *ground_truth.shape[1:])
    best_output = torch.gather(valid_outputs, dim=1, index=gather_index).squeeze(1)
    return best_output, ground_truth[valid_mask]
