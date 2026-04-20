from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def create_run_directories(output_dir: str, experiment_name: str) -> dict[str, Path]:
    root = Path(output_dir) / experiment_name
    directories = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
        "predictions": root / "predictions",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories
