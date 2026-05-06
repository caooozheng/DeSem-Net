from __future__ import annotations

import torch


def _infer_color_cast(channel_means: torch.Tensor) -> str:
    red, green, blue = [float(value) for value in channel_means]
    dominant = max(red, green, blue)
    if dominant == blue and blue - red > 0.03:
        return "a noticeable blue color cast"
    if dominant == green and green - red > 0.03:
        return "a noticeable green color cast"
    if red < 0.25 and blue < 0.35:
        return "a dim and muted tone"
    return "a relatively balanced color distribution"


def _infer_degradation_tags(image: torch.Tensor, mask: torch.Tensor) -> list[str]:
    channel_means = image.mean(dim=(1, 2))
    red, green, blue = [float(value) for value in channel_means]
    brightness = float(image.mean().item())
    contrast = float(image.std().item())
    foreground_ratio = float(mask.mean().item())
    tags = []
    if blue - red > 0.03:
        tags.append("blue_cast")
    if green - red > 0.03:
        tags.append("green_cast")
    if brightness < 0.35:
        tags.append("low_light")
    if contrast < 0.16:
        tags.append("haze_scattering")
    if foreground_ratio < 0.2:
        tags.append("small_foreground")
    if not tags:
        tags.append("clear_balanced")
    return tags


def _infer_contrast(std_value: float) -> str:
    if std_value < 0.12:
        return "very low contrast"
    if std_value < 0.18:
        return "low contrast"
    if std_value < 0.25:
        return "moderate contrast"
    return "relatively strong contrast"


def _infer_brightness(mean_value: float) -> str:
    if mean_value < 0.25:
        return "dark illumination"
    if mean_value < 0.45:
        return "dim illumination"
    if mean_value < 0.65:
        return "medium illumination"
    return "bright illumination"


def _infer_foreground(mask: torch.Tensor) -> str:
    foreground_ratio = float(mask.mean().item())
    if foreground_ratio < 0.15:
        return "a small foreground subject"
    if foreground_ratio < 0.4:
        return "a moderate foreground subject"
    return "a dominant foreground subject"


def build_underwater_prompt(image: torch.Tensor, mask: torch.Tensor) -> str:
    channel_means = image.mean(dim=(1, 2))
    brightness = float(image.mean().item())
    contrast = float(image.std().item())
    degradation_tags = ", ".join(_infer_degradation_tags(image, mask))
    prompt = (
        "Enhance this underwater image with natural color restoration and clear structure. "
        f"Degradation tags: {degradation_tags}. "
        f"The scene shows {_infer_foreground(mask)}, {_infer_brightness(brightness)}, "
        f"{_infer_contrast(contrast)}, and {_infer_color_cast(channel_means)}. "
        "Recover foreground detail, improve visibility, suppress haze and scattering, "
        "and keep the background smooth and realistic."
    )
    return prompt
