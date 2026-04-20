from __future__ import annotations

from torch.utils.data import DataLoader

from clipuie.config import DatasetSection
from clipuie.data.datasets import PairedImageDataset


def _build_loader(dataset: PairedImageDataset, config: DatasetSection, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )


def build_dataloaders(config: DatasetSection) -> dict[str, DataLoader]:
    loaders = {
        "train": _build_loader(PairedImageDataset(config.train_root, config), config, shuffle=True),
        "test": _build_loader(PairedImageDataset(config.test_root, config), config, shuffle=False),
    }
    if config.val_root:
        loaders["val"] = _build_loader(PairedImageDataset(config.val_root, config), config, shuffle=False)
    return loaders
