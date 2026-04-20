from __future__ import annotations

from clipuie.config import ModelSection
from clipuie.models.architectures.clipuie_net import ClipUIENet


def build_model(config: ModelSection) -> ClipUIENet:
    if config.name != "clipuie_net":
        raise ValueError(f"Unsupported model: {config.name}")
    return ClipUIENet(
        num_branch=config.num_branch,
        n_feat=config.n_feat,
        n_rcb=config.n_rcb,
        chan_factor=config.chan_factor,
        bias=config.bias,
        use_sam_mask=config.use_sam_mask,
    )
