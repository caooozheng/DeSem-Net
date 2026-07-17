from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from clipuie.config import load_config
from clipuie.data.prompts import build_underwater_prompt
from clipuie.engine.checkpoint import load_model_state
from clipuie.models import build_model
from clipuie.utils import resolve_device, seed_everything, tensor_to_uint8
from clipuie.utils.metrics import calculate_uciqe, calculate_uiqm


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class OptionalIqaMetrics:
    def __init__(
        self,
        metric_names: list[str],
        device: torch.device,
        allow_missing: bool = True,
        local_uranker_config: str | None = None,
        local_uranker_checkpoint: str | None = None,
        metric_options: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.metrics = {}
        self.local_uranker = None
        self.allow_missing = allow_missing
        metric_options = metric_options or {}
        if "uranker" in metric_names and local_uranker_config:
            self.local_uranker = LocalURankerMetric(
                config_path=local_uranker_config,
                checkpoint_path=local_uranker_checkpoint,
                device=device,
                allow_missing=allow_missing,
            )
            metric_names = [metric_name for metric_name in metric_names if metric_name != "uranker"]
        if not metric_names:
            return
        try:
            import pyiqa
        except ModuleNotFoundError as exc:
            if allow_missing:
                print("Optional IQA metrics skipped: `pyiqa` is not installed.")
                return
            raise RuntimeError("NIQE/URanker evaluation requires `pyiqa`. Install it with `pip install pyiqa`.") from exc

        for metric_name in metric_names:
            try:
                metric = pyiqa.create_metric(metric_name, device=device, as_loss=False, **metric_options.get(metric_name, {}))
            except Exception as exc:
                if allow_missing:
                    print(f"Optional IQA metric skipped: failed to create `{metric_name}` ({exc}).")
                    continue
                raise RuntimeError(f"Failed to create IQA metric `{metric_name}` with pyiqa.") from exc
            if hasattr(metric, "eval"):
                metric.eval()
            self.metrics[metric_name] = metric

    @torch.no_grad()
    def evaluate(self, image: torch.Tensor) -> dict[str, float]:
        values = {}
        if self.local_uranker is not None:
            values["uranker_256"] = self.local_uranker.evaluate(image)
        if not self.metrics:
            return values
        image = image.detach().clamp(0.0, 1.0)
        for metric_name, metric in self.metrics.items():
            score = metric(image)
            if isinstance(score, (tuple, list)):
                score = score[0]
            values[f"{metric_name}_256"] = float(torch.as_tensor(score).detach().cpu().mean().item())
        return values


class LocalURankerMetric:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str | None,
        device: torch.device,
        allow_missing: bool = True,
    ) -> None:
        self.device = device
        try:
            from sam_integration.uranker.uranker_utils import build_model, get_option
        except ModuleNotFoundError as exc:
            if allow_missing:
                print(f"Local URanker skipped: missing dependency ({exc}).")
                self.model = None
                return
            raise RuntimeError("Local URanker requires its package dependencies, especially `timm` and `einops`.") from exc
        opt = get_option(config_path)["model"]
        if checkpoint_path:
            opt["resume_ckpt_path"] = checkpoint_path
        if not opt.get("resume_ckpt_path"):
            message = "Local URanker skipped: no checkpoint configured."
            if allow_missing:
                print(message)
                self.model = None
                return
            raise RuntimeError(message)
        opt["cuda"] = device.type == "cuda"
        self.model = build_model(opt).to(device)
        self.model.eval()

    @staticmethod
    def _pad_to_multiple(image: torch.Tensor, multiple: int = 32) -> torch.Tensor:
        _, _, height, width = image.shape
        padded_height = ((height + multiple - 1) // multiple) * multiple
        padded_width = ((width + multiple - 1) // multiple) * multiple
        left = (padded_width - width) // 2
        right = padded_width - width - left
        top = (padded_height - height) // 2
        bottom = padded_height - height - top
        if left == 0 and right == 0 and top == 0 and bottom == 0:
            return image
        return torch.nn.functional.pad(image, (left, right, top, bottom), mode="constant", value=0.0)

    @staticmethod
    def _histogram_token(image: torch.Tensor, bins: int = 64) -> torch.Tensor:
        tokens = []
        for sample in image.detach().clamp(0.0, 1.0):
            channels = []
            for channel in range(sample.shape[0]):
                hist = torch.histc(sample[channel], bins=bins, min=0.0, max=1.0)
                channels.append(hist)
            tokens.append(torch.cat(channels, dim=0))
        return torch.stack(tokens, dim=0).unsqueeze(1)

    @torch.no_grad()
    def evaluate(self, image: torch.Tensor) -> float:
        if self.model is None:
            return float("nan")
        image = image.detach().to(self.device).clamp(0.0, 1.0)
        image = self._pad_to_multiple(image)
        histogram = self._histogram_token(image).to(self.device)
        output = self.model(image, histogram)
        score = output["final_result"] if isinstance(output, dict) else output
        return float(torch.as_tensor(score).detach().cpu().mean().item())


def load_no_ref_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to load no-reference config files.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return raw or {}


def pick_value(args: argparse.Namespace, config: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    value = getattr(args, key)
    if value is not None:
        return value
    section_values = config.get(section, {})
    if isinstance(section_values, dict) and key in section_values:
        return section_values[key]
    return config.get(key, default)


class NoReferenceImageDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int,
        input_subdir: str | None = None,
        preserve_size: bool = False,
        max_size: int | None = None,
        mask_dir_name: str | None = None,
        mask_suffix: str = ".npy",
        mask_fallback_value: float = 1.0,
    ) -> None:
        self.root = Path(root)
        self.image_size = int(image_size)
        self.preserve_size = preserve_size
        self.max_size = max_size
        if input_subdir:
            self.input_dir = self.root / input_subdir
        elif (self.root / "input").is_dir():
            self.input_dir = self.root / "input"
        elif (self.root / "test").is_dir():
            self.input_dir = self.root / "test"
        else:
            self.input_dir = self.root
        self.mask_dir = self.root / mask_dir_name if mask_dir_name else None
        self.mask_suffix = mask_suffix
        self.mask_fallback_value = float(mask_fallback_value)
        self.names = sorted(
            path.name
            for path in self.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.names:
            raise RuntimeError(f"No images found under {self.input_dir}")

    def __len__(self) -> int:
        return len(self.names)

    @staticmethod
    def _round_to_multiple(value: int, multiple: int = 32) -> int:
        return max(multiple, int(round(value / multiple)) * multiple)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.preserve_size and self.max_size:
            height, width = image.shape[:2]
            long_side = max(height, width)
            if long_side > self.max_size:
                scale = self.max_size / float(long_side)
                new_width = self._round_to_multiple(int(round(width * scale)))
                new_height = self._round_to_multiple(int(round(height * scale)))
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        elif not self.preserve_size:
            image = cv2.resize(image, (self.image_size, self.image_size))
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    @staticmethod
    def load_original_rgb(path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_mask(self, name: str) -> torch.Tensor:
        if self.mask_dir is None:
            if self.preserve_size:
                return torch.empty(0, dtype=torch.float32)
            return torch.full((1, self.image_size, self.image_size), self.mask_fallback_value, dtype=torch.float32)
        mask_path = self.mask_dir / f"{Path(name).stem}{self.mask_suffix}"
        if not mask_path.exists():
            if self.preserve_size:
                return torch.empty(0, dtype=torch.float32)
            return torch.full((1, self.image_size, self.image_size), self.mask_fallback_value, dtype=torch.float32)
        mask = np.load(mask_path)
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, but got shape {mask.shape} for {mask_path}")
        if not self.preserve_size:
            mask = cv2.resize(mask.astype(np.float32), (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = np.clip(mask, 0.0, 1.0)
        return torch.from_numpy(mask).unsqueeze(0).float()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        name = self.names[index]
        input_tensor = self._load_rgb(self.input_dir / name)
        mask_tensor = self._load_mask(name)
        if mask_tensor.numel() == 0:
            mask_tensor = torch.full((1, input_tensor.shape[1], input_tensor.shape[2]), self.mask_fallback_value, dtype=torch.float32)
        return {
            "name": name,
            "input": input_tensor,
            "input_path": str(self.input_dir / name),
            "mask": mask_tensor,
            "prompt": build_underwater_prompt(input_tensor, mask_tensor),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-reference evaluation with a trained Clip-UIE checkpoint.")
    parser.add_argument("--config", default=None, help="Path to a no-reference YAML config file.")
    parser.add_argument("--base-config", default=None, help="Path to the trained model YAML. Overrides model.config in --config.")
    parser.add_argument("--data-root", default=None, help="No-reference dataset root, or a directory containing input images.")
    parser.add_argument("--input-subdir", default=None, help="Image subdirectory under --data-root, for example `test`.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to evaluation.checkpoint in the YAML.")
    parser.add_argument("--output-dir", default=None, help="Directory for enhanced images and metric files.")
    parser.add_argument("--device", default=None, help="Device override, for example `cpu`, `cuda`, or `cuda:1`.")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index shortcut. For example `--gpu 1` means `cuda:1`.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count override.")
    parser.add_argument("--output-branch", type=int, default=None, help="Use a fixed output branch, for example 1.")
    parser.add_argument("--route-output", action="store_true", help="Use the model route output instead of a fixed branch.")
    parser.add_argument("--soft-route", action="store_true", help="Use soft routing instead of config hard_route.")
    parser.add_argument("--self-ensemble", action="store_true", default=None, help="Use flip/transpose test-time self-ensemble.")
    parser.add_argument("--niqe", action="store_true", default=None, help="Compute NIQE with pyiqa.")
    parser.add_argument("--uranker", action="store_true", default=None, help="Compute URanker with pyiqa.")
    parser.add_argument("--strict-iqa", action="store_true", help="Fail if optional NIQE/URanker metrics cannot be loaded.")
    parser.add_argument("--metric-size", default=None, choices=["model", "original"], help="Compute no-reference metrics on model-size output or resized-back original-size output.")
    parser.add_argument("--preserve-size", action="store_true", default=None, help="Run inference at each image's original size. Forces batch size 1.")
    parser.add_argument("--max-size", type=int, default=None, help="When preserving aspect ratio, resize the long side to at most this value.")
    parser.add_argument("--input-metrics", action="store_true", default=None, help="Also compute optional IQA metrics on input images.")
    return parser.parse_args()


def augment_tensor(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    if mode & 1:
        tensor = torch.flip(tensor, dims=(-1,))
    if mode & 2:
        tensor = torch.flip(tensor, dims=(-2,))
    if mode & 4:
        tensor = tensor.transpose(-1, -2)
    return tensor.contiguous()


def deaugment_tensor(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    if mode & 4:
        tensor = tensor.transpose(-1, -2)
    if mode & 2:
        tensor = torch.flip(tensor, dims=(-2,))
    if mode & 1:
        tensor = torch.flip(tensor, dims=(-1,))
    return tensor.contiguous()


def apply_postprocess(outputs: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    if not config.get("enabled", False):
        return outputs
    method = str(config.get("method", "bilateral")).lower()
    blend = float(config.get("blend", 0.25))
    blend = max(0.0, min(1.0, blend))
    if blend <= 0:
        return outputs

    processed = []
    for sample in outputs.detach().clamp(0.0, 1.0):
        image = (
            sample.mul(255.0)
            .add(0.5)
            .clamp(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        if method == "gaussian":
            kernel_size = int(config.get("kernel_size", 3))
            if kernel_size % 2 == 0:
                kernel_size += 1
            filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), float(config.get("sigma", 0.8)))
        elif method == "median":
            kernel_size = int(config.get("kernel_size", 3))
            if kernel_size % 2 == 0:
                kernel_size += 1
            filtered = cv2.medianBlur(image, kernel_size)
        else:
            filtered = cv2.bilateralFilter(
                image,
                int(config.get("diameter", 5)),
                float(config.get("sigma_color", 20)),
                float(config.get("sigma_space", 7)),
            )
        mixed = cv2.addWeighted(image, 1.0 - blend, filtered, blend, 0.0)
        processed.append(torch.from_numpy(mixed).permute(2, 0, 1).float() / 255.0)
    return torch.stack(processed, dim=0).to(device=outputs.device, dtype=outputs.dtype)


@torch.no_grad()
def predict_once(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    masks: torch.Tensor,
    prompts: list[str],
    hard_route: bool,
    output_branch_index: int | None,
) -> torch.Tensor:
    need_branch_outputs = output_branch_index is not None
    route_result = model.forward_route(
        inputs,
        masks,
        prompts,
        return_logits=True,
        return_proc_outs=need_branch_outputs,
        hard_route=hard_route,
    )
    if need_branch_outputs:
        outputs, _, output_list = route_result
        branch_index = int(output_branch_index)
        if branch_index < 0 or branch_index >= len(output_list):
            raise ValueError(f"output_branch_index={branch_index} is out of range for {len(output_list)} branches.")
        return output_list[branch_index]
    outputs, _ = route_result
    return outputs


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    masks: torch.Tensor,
    prompts: list[str],
    hard_route: bool,
    output_branch_index: int | None,
    self_ensemble: bool,
) -> torch.Tensor:
    if not self_ensemble:
        return predict_once(model, inputs, masks, prompts, hard_route, output_branch_index)
    outputs = []
    for mode in range(8):
        aug_inputs = augment_tensor(inputs, mode)
        aug_masks = augment_tensor(masks, mode)
        aug_outputs = predict_once(model, aug_inputs, aug_masks, prompts, hard_route, output_branch_index)
        outputs.append(deaugment_tensor(aug_outputs, mode))
    return torch.stack(outputs, dim=0).mean(dim=0)


def evaluate_no_reference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    hard_route: bool,
    output_branch_index: int | None,
    self_ensemble: bool,
    iqa_metrics: OptionalIqaMetrics,
    metric_size: str,
    compute_input_metrics: bool,
    postprocess_config: dict[str, Any],
) -> dict[str, float]:
    output_image_dir = output_dir / "output"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    model.eval()
    for batch in tqdm(dataloader, desc="No-reference evaluating", leave=False):
        inputs = batch["input"].to(device)
        masks = batch["mask"].to(device)
        prompts = list(batch["prompt"])
        names = list(batch["name"])
        input_paths = list(batch["input_path"])
        outputs = predict(model, inputs, masks, prompts, hard_route, output_branch_index, self_ensemble)
        outputs = apply_postprocess(outputs, postprocess_config)
        for index, (name, input_path) in enumerate(zip(names, input_paths)):
            input_sample = inputs[index : index + 1]
            output_sample = outputs[index : index + 1]
            output_uint8 = tensor_to_uint8(output_sample)
            metric_output_uint8 = output_uint8
            metric_input_uint8 = tensor_to_uint8(input_sample)
            metric_output_sample = output_sample
            metric_input_sample = input_sample
            if metric_size == "original":
                original_input_uint8 = NoReferenceImageDataset.load_original_rgb(Path(input_path))
                original_height, original_width = original_input_uint8.shape[:2]
                metric_output_uint8 = cv2.resize(output_uint8, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
                metric_input_uint8 = original_input_uint8
                metric_output_sample = (
                    torch.from_numpy(metric_output_uint8)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device=device, dtype=output_sample.dtype)
                    / 255.0
                )
                metric_input_sample = (
                    torch.from_numpy(metric_input_uint8)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device=device, dtype=input_sample.dtype)
                    / 255.0
                )
            uiqm = float(calculate_uiqm(metric_output_uint8))
            uciqe = float(calculate_uciqe(metric_output_uint8))
            optional_metrics = iqa_metrics.evaluate(metric_output_sample)
            input_optional_metrics = {}
            if compute_input_metrics:
                input_optional_metrics = {
                    f"input_{key}": value
                    for key, value in iqa_metrics.evaluate(metric_input_sample).items()
                }
            Image.fromarray(output_uint8.astype(np.uint8)).save(output_image_dir / name)
            rows.append({"name": name, "uiqm_256": uiqm, "uciqe_256": uciqe, **optional_metrics, **input_optional_metrics})

    metric_keys = [key for key in rows[0].keys() if key != "name"] if rows else []
    metrics = {key: float(np.mean([row[key] for row in rows if key in row])) for key in metric_keys}
    metrics["num_images"] = float(len(rows))
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", *metric_keys])
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def main() -> None:
    args = parse_args()
    no_ref_config = load_no_ref_config(args.config)
    base_config_path = args.base_config or no_ref_config.get("base_config") or no_ref_config.get("model", {}).get("config") or "configs/uieb.yaml"
    data_root = pick_value(args, no_ref_config, "dataset", "data_root")
    if not data_root:
        raise ValueError("No no-reference dataset configured. Set dataset.data_root in YAML or pass --data-root.")

    config = load_config(base_config_path)
    seed_everything(config.experiment.seed)
    gpu = pick_value(args, no_ref_config, "runtime", "gpu")
    device_override = pick_value(args, no_ref_config, "runtime", "device")
    requested_device = device_override or (f"cuda:{gpu}" if gpu is not None else config.runtime.device)
    device = resolve_device(requested_device)
    checkpoint_path = pick_value(args, no_ref_config, "model", "checkpoint", config.evaluation.checkpoint or config.training.pretrained_checkpoint)
    if not checkpoint_path:
        raise ValueError("No checkpoint configured. Pass --checkpoint or set evaluation.checkpoint in YAML.")

    output_branch_index = config.evaluation.output_branch_index
    if args.route_output:
        output_branch_index = None
    output_branch_index = pick_value(args, no_ref_config, "inference", "output_branch", output_branch_index)
    hard_route = bool(no_ref_config.get("inference", {}).get("hard_route", config.evaluation.hard_route))
    if args.soft_route:
        hard_route = False
    self_ensemble_config = no_ref_config.get("inference", {}).get("self_ensemble", config.evaluation.self_ensemble)
    self_ensemble = bool(args.self_ensemble if args.self_ensemble is not None else self_ensemble_config)
    metric_config = no_ref_config.get("metrics", {})
    compute_niqe = bool(args.niqe if args.niqe is not None else metric_config.get("niqe", False))
    compute_uranker = bool(args.uranker if args.uranker is not None else metric_config.get("uranker", False))
    allow_missing_iqa = not bool(args.strict_iqa or metric_config.get("strict_iqa", False))
    iqa_metric_names = []
    if compute_niqe:
        iqa_metric_names.append(str(metric_config.get("niqe_name", "niqe")))
    if compute_uranker:
        iqa_metric_names.append(str(metric_config.get("uranker_name", "uranker")))
    uranker_local_config = metric_config.get("uranker_local_config")
    uranker_checkpoint = metric_config.get("uranker_checkpoint")
    metric_size = pick_value(args, no_ref_config, "metrics", "metric_size", "model")
    compute_input_metrics = bool(pick_value(args, no_ref_config, "metrics", "input_metrics", False))
    preserve_size = bool(pick_value(args, no_ref_config, "inference", "preserve_size", False))
    max_size = pick_value(args, no_ref_config, "inference", "max_size")
    postprocess_config = no_ref_config.get("postprocess", {})
    metric_options = {}
    if compute_niqe:
        niqe_options = {
            "color_space": metric_config.get("niqe_color_space", "ycbcr"),
            "crop_border": int(metric_config.get("niqe_crop_border", 0)),
            "test_y_channel": bool(metric_config.get("niqe_test_y_channel", True)),
            "version": metric_config.get("niqe_version", "original"),
        }
        if metric_config.get("niqe_pretrained_model_path"):
            niqe_options["pretrained_model_path"] = metric_config["niqe_pretrained_model_path"]
        metric_options[str(metric_config.get("niqe_name", "niqe"))] = niqe_options
    image_size = int(no_ref_config.get("dataset", {}).get("image_size", config.dataset.image_size))
    input_subdir = pick_value(args, no_ref_config, "dataset", "input_subdir")
    mask_dir_name = no_ref_config.get("dataset", {}).get("mask_dir_name", config.dataset.mask_dir_name)
    mask_suffix = no_ref_config.get("dataset", {}).get("mask_suffix", config.dataset.mask_suffix)
    mask_fallback_value = float(no_ref_config.get("dataset", {}).get("mask_fallback_value", config.dataset.mask_fallback_value))

    dataset = NoReferenceImageDataset(
        data_root,
        image_size=image_size,
        input_subdir=input_subdir,
        preserve_size=preserve_size,
        max_size=int(max_size) if max_size else None,
        mask_dir_name=mask_dir_name,
        mask_suffix=mask_suffix,
        mask_fallback_value=mask_fallback_value,
    )
    batch_size = 1 if preserve_size else pick_value(args, no_ref_config, "runtime", "batch_size", config.dataset.batch_size)
    num_workers = pick_value(args, no_ref_config, "runtime", "num_workers", config.dataset.num_workers)
    dataloader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(no_ref_config.get("runtime", {}).get("pin_memory", config.dataset.pin_memory)),
    )

    model = build_model(config.model, config.multimodal).to(device)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_result, skipped_keys = load_model_state(model, checkpoint, strict=config.training.strict_load)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            "Checkpoint key mismatch: "
            f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}"
        )
        if skipped_keys:
            print(f"Skipped shape-mismatched keys sample: {skipped_keys[:10]}")

    iqa_metrics = OptionalIqaMetrics(
        iqa_metric_names,
        device=device,
        allow_missing=allow_missing_iqa,
        local_uranker_config=uranker_local_config,
        local_uranker_checkpoint=uranker_checkpoint,
        metric_options=metric_options,
    )
    output_dir_value = pick_value(args, no_ref_config, "output", "output_dir")
    output_dir = Path(output_dir_value) if output_dir_value else Path(config.experiment.output_dir) / f"{config.experiment.name}_no_ref"
    metrics = evaluate_no_reference(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        hard_route=hard_route,
        output_branch_index=output_branch_index,
        self_ensemble=self_ensemble,
        iqa_metrics=iqa_metrics,
        metric_size=str(metric_size),
        compute_input_metrics=compute_input_metrics,
        postprocess_config=postprocess_config,
    )
    print(metrics)
    print(f"Saved no-reference outputs to: {output_dir}")


if __name__ == "__main__":
    main()
