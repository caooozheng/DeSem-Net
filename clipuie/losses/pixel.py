from __future__ import annotations

import torch
import torch.nn as nn


class WeightedPixelLoss(nn.Module):
    def __init__(self, l1_weight: float = 0.8, l2_weight: float = 0.2) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1_weight * self.l1(prediction, target) + self.l2_weight * self.l2(prediction, target)
