from __future__ import annotations

from clipuie.config import ModelSection, MultimodalSection
from clipuie.models.architectures.clipuie_net import ClipUIENet


def build_model(config: ModelSection, multimodal_config: MultimodalSection | None = None) -> ClipUIENet:
    if config.name != "clipuie_net":
        raise ValueError(f"Unsupported model: {config.name}")
    return ClipUIENet(
        num_branch=config.num_branch,
        n_feat=config.n_feat,
        n_rcb=config.n_rcb,
        chan_factor=config.chan_factor,
        bias=config.bias,
        use_sam_mask=config.use_sam_mask,
        use_dual_region_branch=config.use_dual_region_branch,
        region_branch_rcb=config.region_branch_rcb,
        region_fusion_strength=config.region_fusion_strength,
        multimodal_aux_strength=config.multimodal_aux_strength,
        use_multimodal_initial_condition=config.use_multimodal_initial_condition,
        use_fg_bg_decoder=config.use_fg_bg_decoder,
        fg_bg_decoder_blocks=config.fg_bg_decoder_blocks,
        fg_bg_decoder_strength=config.fg_bg_decoder_strength,
        use_frequency_refinement=config.use_frequency_refinement,
        frequency_refinement_strength=config.frequency_refinement_strength,
        use_physical_head=config.use_physical_head,
        multimodal_config=multimodal_config,
    )
