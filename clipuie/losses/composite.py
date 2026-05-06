from __future__ import annotations

import torch
import torch.nn as nn

from clipuie.losses.perceptual import VGG19PerceptualLoss
from clipuie.losses.pixel import WeightedPixelLoss
from clipuie.models.ops import GetGradientNopadding


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    image = image.clamp(0.0, 1.0)
    linear = torch.where(image > 0.04045, ((image + 0.055) / 1.055).pow(2.4), image / 12.92)
    r, g, b = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    xyz = torch.cat([x / 0.95047, y, z / 1.08883], dim=1)
    epsilon = 0.008856
    kappa = 903.3
    f_xyz = torch.where(xyz > epsilon, xyz.clamp_min(1e-6).pow(1.0 / 3.0), (kappa * xyz + 16.0) / 116.0)
    fx, fy, fz = f_xyz[:, 0:1], f_xyz[:, 1:2], f_xyz[:, 2:3]
    lab_l = (116.0 * fy - 16.0) / 100.0
    lab_a = (500.0 * (fx - fy) + 128.0) / 255.0
    lab_b = (200.0 * (fy - fz) + 128.0) / 255.0
    return torch.cat([lab_l, lab_a, lab_b], dim=1)


class LabColorConsistencyLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_lab = rgb_to_lab(prediction)
        target_lab = rgb_to_lab(target)
        mean_loss = torch.mean(torch.abs(pred_lab.mean(dim=(2, 3)) - target_lab.mean(dim=(2, 3))))
        std_loss = torch.mean(torch.abs(pred_lab.std(dim=(2, 3), unbiased=False) - target_lab.std(dim=(2, 3), unbiased=False)))
        return mean_loss + 0.5 * std_loss


class SoftHistogramLoss(nn.Module):
    def __init__(self, bins: int = 16, sigma: float = 0.03) -> None:
        super().__init__()
        self.bins = bins
        self.sigma = sigma
        self.register_buffer("centers", torch.linspace(0.0, 1.0, bins).view(1, 1, bins, 1))

    def _histogram(self, image: torch.Tensor) -> torch.Tensor:
        values = image.clamp(0.0, 1.0).flatten(2).unsqueeze(2)
        weights = torch.exp(-0.5 * ((values - self.centers) / self.sigma) ** 2)
        hist = weights.mean(dim=-1)
        return hist / hist.sum(dim=2, keepdim=True).clamp_min(1e-6)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(self._histogram(prediction) - self._histogram(target)))


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
