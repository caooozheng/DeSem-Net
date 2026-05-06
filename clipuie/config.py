from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional


@dataclass
class ExperimentSection:
    name: str
    seed: int = 42
    output_dir: str = "artifacts"


@dataclass
class DatasetSection:
    train_root: str
    test_root: str
    val_root: Optional[str] = None
    image_size: int = 256
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    mask_dir_name: Optional[str] = None
    mask_suffix: str = ".npy"
    mask_fallback_value: float = 1.0


@dataclass
class ModelSection:
    name: str = "clipuie_net"
    num_branch: int = 3
    n_feat: int = 32
    n_rcb: int = 2
    chan_factor: int = 2
    bias: bool = True
    use_sam_mask: bool = False
    use_dual_region_branch: bool = False
    region_branch_rcb: int = 1
    region_fusion_strength: float = 0.2
    multimodal_aux_strength: float = 0.03
    use_multimodal_initial_condition: bool = False
    use_fg_bg_decoder: bool = False
    fg_bg_decoder_blocks: int = 1
    fg_bg_decoder_strength: float = 0.1
    use_frequency_refinement: bool = False
    frequency_refinement_strength: float = 0.05
    use_physical_head: bool = False


@dataclass
class MultimodalSection:
    enabled: bool = False
    clip_model_name: str = "openai/clip-vit-base-patch32"
    llm_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    freeze_clip: bool = True
    freeze_llm: bool = True
    prompt_max_length: int = 96
    adapter_hidden_dim: int = 256


@dataclass
class OptimizerSection:
    name: str = "adamw"
    generator_lr: float = 1e-4
    router_lr: float = 2e-5
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class SchedulerSection:
    generator_milestones: list[int] = field(default_factory=lambda: [25, 50, 75, 100])
    generator_gamma: float = 0.5
    router_milestones: list[int] = field(default_factory=lambda: [25, 50, 75, 100])
    router_gamma: float = 0.3


@dataclass
class TrainingSection:
    epochs: int = 200
    route_start_epoch: int = 0
    checkpoint_interval: int = 20
    validate_on: str = "val"
    pretrained_checkpoint: Optional[str] = None
    resume_checkpoint: Optional[str] = None
    strict_load: bool = False
    lambda_style: float = 10.0
    lambda_route: float = 1.0
    lambda_foreground: float = 0.0
    lambda_background: float = 0.0
    lambda_physical: float = 0.0
    lambda_lab: float = 0.0
    lambda_histogram: float = 0.0
    lambda_boundary: float = 0.0
    lambda_foreground_texture: float = 0.0
    route_score_psnr_weight: float = 0.7
    route_score_ssim_weight: float = 0.2
    route_score_color_weight: float = 0.1


@dataclass
class EvaluationSection:
    checkpoint: Optional[str] = None
    save_images: bool = False
    hard_route: bool = False
    output_branch_index: Optional[int] = None
    compute_uiqm: bool = True
    compute_uciqe: bool = True
    compute_branch_metrics: bool = True


@dataclass
class RuntimeSection:
    device: str = "cuda"
    cudnn_benchmark: bool = True


@dataclass
class ExperimentConfig:
    experiment: ExperimentSection
    dataset: DatasetSection
    model: ModelSection = field(default_factory=ModelSection)
    multimodal: MultimodalSection = field(default_factory=MultimodalSection)
    optimizer: OptimizerSection = field(default_factory=OptimizerSection)
    scheduler: SchedulerSection = field(default_factory=SchedulerSection)
    training: TrainingSection = field(default_factory=TrainingSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)
    runtime: RuntimeSection = field(default_factory=RuntimeSection)


def _load_yaml_module() -> Any:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required. Install it with `pip install PyYAML`.") from exc
    return yaml


def _build_section(section_type: type[Any], values: dict[str, Any] | None) -> Any:
    if not values:
        return section_type()
    allowed_keys = {item.name for item in fields(section_type)}
    return section_type(**{key: value for key, value in values.items() if key in allowed_keys})


def load_config(path: str | Path) -> ExperimentConfig:
    yaml = _load_yaml_module()
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ExperimentConfig(
        experiment=_build_section(ExperimentSection, raw["experiment"]),
        dataset=_build_section(DatasetSection, raw["dataset"]),
        model=_build_section(ModelSection, raw.get("model", {})),
        multimodal=_build_section(MultimodalSection, raw.get("multimodal", {})),
        optimizer=_build_section(OptimizerSection, raw.get("optimizer", {})),
        scheduler=_build_section(SchedulerSection, raw.get("scheduler", {})),
        training=_build_section(TrainingSection, raw.get("training", {})),
        evaluation=_build_section(EvaluationSection, raw.get("evaluation", {})),
        runtime=_build_section(RuntimeSection, raw.get("runtime", {})),
    )
