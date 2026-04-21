from __future__ import annotations

import argparse
from pathlib import Path

import torch

from clipuie.config import load_config
from clipuie.data import build_dataloaders
from clipuie.engine import ClipUIETrainer
from clipuie.models import build_model
from clipuie.utils import create_run_directories, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ClipUIe experiments.")
    default_config = Path(__file__).resolve().parent / "configs" / "clipuie_uieb_baseline.yaml"
    parser.add_argument(
        "--config",
        default=str(default_config),
        help=f"Path to a YAML config file. Defaults to {default_config}.",
    )
    parser.add_argument("--device", default=None, help="Device override, for example `cpu`, `cuda`, or `cuda:1`.")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index shortcut. For example `--gpu 1` means `cuda:1`.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.experiment.seed)
    requested_device = args.device or (f"cuda:{args.gpu}" if args.gpu is not None else config.runtime.device)
    device = resolve_device(requested_device)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = config.runtime.cudnn_benchmark

    run_dirs = create_run_directories(config.experiment.output_dir, config.experiment.name)
    loaders = build_dataloaders(config.dataset)
    eval_key = "val" if config.training.validate_on == "val" and "val" in loaders else "test"
    trainer = ClipUIETrainer(build_model(config.model, config.multimodal), config, device, run_dirs)
    print(trainer.fit(loaders["train"], loaders[eval_key]))


if __name__ == "__main__":
    main()
