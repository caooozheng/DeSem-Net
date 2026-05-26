from __future__ import annotations

import argparse
from pathlib import Path

import torch

from clipuie.config import load_config
from clipuie.data import build_dataloaders
from clipuie.engine import Evaluator
from clipuie.engine.checkpoint import load_model_state
from clipuie.models import build_model
from clipuie.utils import create_run_directories, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ClipUIe experiments.")
    default_config = Path(__file__).resolve().parent / "sam_integration/configs" / "clipuie_euvp_scene_sam_clip_llm.yaml"
    parser.add_argument(
        "--config",
        default=str(default_config),
        help=f"Path to a YAML config file. Defaults to {default_config}.",
    )
    parser.add_argument("--device", default=None, help="Device override, for example `cpu`, `cuda`, or `cuda:1`.")
    parser.add_argument("--gpu", type=int, default=2, help="GPU index shortcut. For example `--gpu 1` means `cuda:1`.")
    parser.add_argument("--checkpoint", default=None, help="Path to the checkpoint file. Overrides the YAML setting.")
    parser.add_argument("--calibration", default=None, help="Path to an RGB affine calibration JSON.")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Evaluation split.")
    parser.add_argument("--save-images", action="store_true", help="Save output and comparison images for visual inspection.")
    parser.add_argument("--soft-route", action="store_true", help="Use soft routing during evaluation instead of config hard_route.")
    parser.add_argument("--output-branch", type=int, default=None, help="Use a fixed branch as the final output, for example 1 for branch_1.")
    parser.add_argument("--route-output", action="store_true", help="Use the model route output instead of a fixed output branch from the config.")
    parser.add_argument("--branch-metrics", action="store_true", help="Temporarily report per-branch PSNR metrics.")
    parser.add_argument("--self-ensemble", action="store_true", help="Use flip/transpose test-time self-ensemble for the final output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.save_images:
        config.evaluation.save_images = True
    if args.calibration:
        config.evaluation.calibration_path = args.calibration
    if args.soft_route:
        config.evaluation.hard_route = False
    if args.route_output:
        config.evaluation.output_branch_index = None
    if args.output_branch is not None:
        config.evaluation.output_branch_index = args.output_branch
    if args.branch_metrics:
        config.evaluation.compute_branch_metrics = True
    if args.self_ensemble:
        config.evaluation.self_ensemble = True
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

    model = build_model(config.model, config.multimodal).to(device)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        checkpoint_meta = {
            key: checkpoint[key]
            for key in ("epoch", "run_id", "best_psnr")
            if key in checkpoint
        }
        if checkpoint_meta:
            print(f"Checkpoint metadata: {checkpoint_meta}")
    load_result, skipped_keys = load_model_state(model, checkpoint, strict=config.training.strict_load)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            "Checkpoint key mismatch: "
            f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}"
        )
        if load_result.missing_keys:
            print(f"Missing keys sample: {load_result.missing_keys[:10]}")
        if load_result.unexpected_keys:
            print(f"Unexpected keys sample: {load_result.unexpected_keys[:10]}")
        if skipped_keys:
            print(f"Skipped shape-mismatched keys sample: {skipped_keys[:10]}")
    evaluator = Evaluator(model, device, config.evaluation, run_dirs["predictions"])
    metrics = evaluator.evaluate(loaders[split])
    print(metrics)
    if config.evaluation.save_images:
        print(f"Saved predictions to: {run_dirs['predictions']}")


if __name__ == "__main__":
    main()
