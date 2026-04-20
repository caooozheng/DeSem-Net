from __future__ import annotations

import argparse
from pathlib import Path

import torch

from clipuie.config import load_config
from clipuie.data import build_dataloaders
from clipuie.engine import Evaluator
from clipuie.models import build_model
from clipuie.utils import create_run_directories, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ClipUIe experiments.")
    default_config = Path(__file__).resolve().parent / "configs" / "clipuie_uieb_baseline.yaml"
    parser.add_argument(
        "--config",
        default=str(default_config),
        help=f"Path to a YAML config file. Defaults to {default_config}.",
    )
    parser.add_argument("--device", default=None, help="Device override, for example `cpu`, `cuda`, or `cuda:1`.")
    parser.add_argument("--gpu", type=int, default=1, help="GPU index shortcut. For example `--gpu 1` means `cuda:1`.")
    parser.add_argument("--checkpoint", default=None, help="Path to the checkpoint file. Overrides the YAML setting.")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Evaluation split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.experiment.seed)
    requested_device = args.device or (f"cuda:{args.gpu}" if args.gpu is not None else config.runtime.device)
    device = resolve_device(requested_device)
    run_dirs = create_run_directories(config.experiment.output_dir, config.experiment.name)
    loaders = build_dataloaders(config.dataset)
    split = args.split if args.split in loaders else "test"
    default_best_checkpoint = run_dirs["checkpoints"] / "best.pth"
    checkpoint_path = args.checkpoint or config.evaluation.checkpoint or config.training.pretrained_checkpoint
    if not checkpoint_path and default_best_checkpoint.exists():
        checkpoint_path = str(default_best_checkpoint)
    if not checkpoint_path:
        raise ValueError("No checkpoint configured. Set evaluation.checkpoint or training.pretrained_checkpoint in YAML, or pass --checkpoint.")

    model = build_model(config.model).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    evaluator = Evaluator(model, device, config.evaluation, run_dirs["predictions"])
    print(evaluator.evaluate(loaders[split]))


if __name__ == "__main__":
    main()
