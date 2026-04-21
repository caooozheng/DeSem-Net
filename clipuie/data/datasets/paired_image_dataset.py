from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from clipuie.config import DatasetSection
from clipuie.data.prompts import build_underwater_prompt


class PairedImageDataset(Dataset):
    def __init__(self, root: str, config: DatasetSection) -> None:
        self.root = Path(root)
        self.image_size = config.image_size
        self.input_dir = self.root / "input"
        self.target_dir = self.root / "target"
        self.mask_dir = self.root / config.mask_dir_name if config.mask_dir_name else None
        self.mask_suffix = config.mask_suffix
        self.mask_fallback_value = float(config.mask_fallback_value)
        self.names = sorted(path.name for path in self.input_dir.iterdir() if path.is_file())
        if not self.names:
            raise RuntimeError(f"No paired images found under {self.root}")

    def __len__(self) -> int:
        return len(self.names)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    def _load_mask(self, name: str) -> torch.Tensor:
        if self.mask_dir is None:
            return torch.full((1, self.image_size, self.image_size), self.mask_fallback_value, dtype=torch.float32)

        mask_path = self.mask_dir / f"{Path(name).stem}{self.mask_suffix}"
        if not mask_path.exists():
            return torch.full((1, self.image_size, self.image_size), self.mask_fallback_value, dtype=torch.float32)

        mask = np.load(mask_path)
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, but got shape {mask.shape} for {mask_path}")
        mask = cv2.resize(mask.astype(np.float32), (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = np.clip(mask, 0.0, 1.0)
        return torch.from_numpy(mask).unsqueeze(0).float()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        name = self.names[index]
        input_tensor = self._load_rgb(self.input_dir / name)
        target_tensor = self._load_rgb(self.target_dir / name)
        mask_tensor = self._load_mask(name)
        return {
            "name": name,
            "input": input_tensor,
            "target": target_tensor,
            "mask": mask_tensor,
            "prompt": build_underwater_prompt(input_tensor, mask_tensor),
        }
