from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class VGG19PerceptualLoss(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.vgg = models.vgg19(pretrained=pretrained).features
        for parameter in self.vgg.parameters():
            parameter.requires_grad_(False)

    def get_features(self, image: torch.Tensor, layers: dict[str, str] | None = None) -> dict[str, torch.Tensor]:
        if layers is None:
            layers = {"30": "conv5_2"}
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, layer: str = "conv5_2") -> torch.Tensor:
        pred_features = self.get_features(prediction)
        target_features = self.get_features(target)
        return torch.mean((target_features[layer] - pred_features[layer]) ** 2)
