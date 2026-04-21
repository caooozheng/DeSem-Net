from __future__ import annotations

from dataclasses import dataclass, field
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
    strict_load: bool = False
    lambda_style: float = 10.0
    lambda_route: float = 1.0
    lambda_foreground: float = 0.0
    lambda_background: float = 0.0


@dataclass
class EvaluationSection:
    checkpoint: Optional[str] = None
    save_images: bool = False
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


def load_config(path: str | Path) -> ExperimentConfig:
    yaml = _load_yaml_module()
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ExperimentConfig(
        experiment=ExperimentSection(**raw["experiment"]),
        dataset=DatasetSection(**raw["dataset"]),
        model=ModelSection(**raw.get("model", {})),
        multimodal=MultimodalSection(**raw.get("multimodal", {})),
        optimizer=OptimizerSection(**raw.get("optimizer", {})),
        scheduler=SchedulerSection(**raw.get("scheduler", {})),
        training=TrainingSection(**raw.get("training", {})),
        evaluation=EvaluationSection(**raw.get("evaluation", {})),
        runtime=RuntimeSection(**raw.get("runtime", {})),
    )
