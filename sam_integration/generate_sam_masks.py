from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SAM masks for EUVP-Dark datasets.")
    parser.add_argument("--dataset-root",  default="../../Clip-UIE/UnderwaterDatasets/EUVP-Dark", help="Dataset root that contains train/val/test.")
    parser.add_argument("--sam-checkpoint", default="../sam_integration/weight/sam_vit_h_4b8939.pth", help="Path to the SAM checkpoint.")
    parser.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--gpu", type=int, default=2, help="CUDA GPU index. Falls back to CPU when CUDA is unavailable.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Dataset splits to process.")
    parser.add_argument("--output-dir-name", default="mask_sam", help="Mask directory name created under each split.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top masks to merge.")
    parser.add_argument("--min-area", type=int, default=500, help="Minimum region area to keep.")
    parser.add_argument("--max-area-ratio", type=float, default=0.85, help="Drop masks larger than this image-area ratio.")
    parser.add_argument("--no-blur", action="store_true", help="Disable Gaussian blur on the merged mask.")
    return parser.parse_args()


def resolve_device(gpu_index: int) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    gpu_count = torch.cuda.device_count()
    if gpu_index < 0 or gpu_index >= gpu_count:
        raise ValueError(f"Invalid --gpu={gpu_index}. Available CUDA device indices: 0 to {gpu_count - 1}.")
    return f"cuda:{gpu_index}"


def save_default_mask(mask_dir: Path, image_stem: str, shape: tuple[int, int], value: float = 0.5) -> None:
    fallback = np.full(shape, value, dtype=np.float32)
    np.save(mask_dir / f"{image_stem}.npy", fallback)


def generate_mask_for_split(
    image_dir: Path,
    mask_dir: Path,
    mask_generator: SamAutomaticMaskGenerator,
    top_k: int,
    min_area: int,
    max_area_ratio: float,
    blur: bool,
) -> None:
    mask_dir.mkdir(parents=True, exist_ok=True)
    image_list = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})

    for image_path in tqdm(image_list, desc=f"Processing {image_dir.parent.name}"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        image_area = height * width
        masks = mask_generator.generate(image)

        if not masks:
            save_default_mask(mask_dir, image_path.stem, (height, width))
            continue

        def mask_rank(item: dict[str, object]) -> float:
            return float(item.get("predicted_iou", 1.0)) * float(item["area"])

        final_mask = np.zeros((height, width), dtype=np.float32)
        ranked_masks = sorted(masks, key=mask_rank, reverse=True)

        kept_count = 0
        for mask_item in ranked_masks:
            area = int(mask_item["area"])
            if area < min_area:
                continue
            if area > image_area * max_area_ratio:
                continue
            segmentation = mask_item["segmentation"]
            score = float(mask_item.get("predicted_iou", 1.0))
            final_mask[segmentation] = np.maximum(final_mask[segmentation], score)
            kept_count += 1
            if kept_count >= top_k:
                break

        if final_mask.max() <= 0:
            save_default_mask(mask_dir, image_path.stem, (height, width))
            continue

        if blur:
            final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)

        final_mask = np.clip(final_mask, 0.0, 1.0).astype(np.float32)
        np.save(mask_dir / f"{image_path.stem}.npy", final_mask)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    sam_checkpoint = Path(args.sam_checkpoint).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not sam_checkpoint.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")

    device = resolve_device(args.gpu)
    print(f"Loading SAM model on {device} ...")
    sam = sam_model_registry[args.model_type](checkpoint=str(sam_checkpoint))
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=200,
    )

    for split in args.splits:
        image_dir = dataset_root / split / "input"
        if not image_dir.exists():
            print(f"Skip split `{split}` because {image_dir} does not exist.")
            continue
        mask_dir = dataset_root / split / args.output_dir_name
        print(f"Generating masks for {split}: {image_dir}")
        generate_mask_for_split(
            image_dir=image_dir,
            mask_dir=mask_dir,
            mask_generator=mask_generator,
            top_k=args.top_k,
            min_area=args.min_area,
            max_area_ratio=args.max_area_ratio,
            blur=not args.no_blur,
        )

    print("All masks generated successfully.")


if __name__ == "__main__":
    main()
