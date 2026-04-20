from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(state_dict: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
