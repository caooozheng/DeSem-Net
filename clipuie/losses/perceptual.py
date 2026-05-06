from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

try:
    from torchvision.models import VGG19_Weights
except ImportError:  # pragma: no cover - compatibility with older torchvision
    VGG19_Weights = None


class VGG19PerceptualLoss(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        if pretrained and VGG19_Weights is not None:
            self.vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        else:
            self.vgg = models.vgg19(pretrained=pretrained).features
        self.vgg.eval()
        for parameter in self.vgg.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image.clamp(0.0, 1.0) - self.mean) / self.std

    def get_features(self, image: torch.Tensor, layers: dict[str, str] | None = None) -> dict[str, torch.Tensor]:
        if layers is None:
            layers = {"30": "conv5_2"}
        features = {}
        x = self._normalize(image)
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, layer: str = "conv5_2") -> torch.Tensor:
        pred_features = self.get_features(prediction)
        target_features = self.get_features(target)
        return torch.mean((target_features[layer] - pred_features[layer]) ** 2)
