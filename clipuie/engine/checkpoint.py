from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(state_dict: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)


def extract_model_state(checkpoint: dict) -> dict:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_model_state(model: nn.Module, checkpoint: dict, strict: bool = False):
    state_dict = extract_model_state(checkpoint)
    if strict:
        return model.load_state_dict(state_dict, strict=True), []

    model_state = model.state_dict()
    compatible_state = {}
    skipped_keys = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible_state[key] = value
        elif key in model_state:
            skipped_keys.append(key)
    load_result = model.load_state_dict(compatible_state, strict=False)
    return load_result, skipped_keys
