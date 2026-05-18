from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from clipuie.config import MultimodalSection
from clipuie.models.multimodal import MultimodalConditionAdapter
from clipuie.models.ops import AFF, AtrousBlock, DownSample, GetGradientNopadding1CGray, NonLocalSparseAttention, RCB, UpSample


class StyleLossModule(nn.Module):
    def forward(self, x_bot: torch.Tensor, gt_bot: torch.Tensor) -> torch.Tensor:
        batch = x_bot.size(0)
        gram_x = torch.bmm(x_bot.view(batch, x_bot.size(1), -1), x_bot.view(batch, x_bot.size(1), -1).transpose(1, 2))
        gram_gt = torch.bmm(gt_bot.view(batch, gt_bot.size(1), -1), gt_bot.view(batch, gt_bot.size(1), -1).transpose(1, 2))
        return torch.mean((gram_gt - gram_x) ** 2) / (x_bot.size(1) * x_bot.size(2) * x_bot.size(3))


class GramGlobalWeightNet(nn.Module):
    def __init__(self, channels: int, num_weights: int = 3) -> None:
        super().__init__()
        proj_dim = max(1, channels // 4)
        self.proj_a = nn.Linear(channels, proj_dim)
        self.proj_b = nn.Linear(channels, proj_dim)
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim * proj_dim + channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        logits = []
        for x in xs:
            batch, channels, _, _ = x.shape
            gram = torch.bmm(x.view(batch, channels, -1), x.view(batch, channels, -1).transpose(1, 2))
            a = self.proj_a(gram)
            b = self.proj_b(gram.transpose(1, 2))
            gb = torch.bmm(a.transpose(-1, -2), b).view(batch, -1)
            pooled = self.pool(x).view(batch, channels)
            logits.append(self.mlp(torch.cat([pooled, gb], dim=1)))
        return torch.cat(logits, dim=1)

# MGRA
class MaskGuidedResidualAdapter(nn.Module):
    def __init__(self, channels: int, activation: nn.Module, num_rcb: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.trunk = nn.Sequential(*[RCB(channels, activation, bias=bias) for _ in range(num_rcb)])
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=bias),
            activation,
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
        )
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias)
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        boundary = 4.0 * mask * (1.0 - mask)
        modulation = torch.sigmoid(self.mask_encoder(torch.cat([mask, 1.0 - mask, boundary], dim=1)))
        return self.out_proj(self.trunk(x) * modulation)


class RegionSemanticDegradationControl(nn.Module):
    def __init__(
        self,
        channels: int,
        activation: nn.Module,
        condition_dim: int = 0,
        strength: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.strength = strength
        self.condition_dim = condition_dim
        hidden = max(16, channels // 2)
        descriptor_dim = channels * 3 + condition_dim
        self.alpha_mlp = nn.Sequential(
            nn.Linear(descriptor_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        self.fg_delta = nn.Sequential(
            RCB(channels, activation, bias=bias),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
        )
        self.bg_delta = nn.Sequential(
            RCB(channels, activation, bias=bias),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
        )
        self.bd_delta = nn.Sequential(
            RCB(channels, activation, bias=bias),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
        )
        nn.init.zeros_(self.alpha_mlp[-1].weight)
        nn.init.constant_(self.alpha_mlp[-1].bias, -2.0)
        for branch in (self.fg_delta, self.bg_delta, self.bd_delta):
            nn.init.zeros_(branch[-1].weight)
            if branch[-1].bias is not None:
                nn.init.zeros_(branch[-1].bias)

    @staticmethod
    def _masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weighted = features * mask
        area = mask.sum(dim=(2, 3)).clamp_min(1.0)
        return weighted.sum(dim=(2, 3)) / area

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        foreground = mask
        background = 1.0 - mask
        boundary = (4.0 * mask * background).clamp(0.0, 1.0)
        descriptors = [
            self._masked_mean(features, foreground),
            self._masked_mean(features, background),
            self._masked_mean(features, boundary),
        ]
        if self.condition_dim > 0:
            if condition is None:
                condition = features.new_zeros(features.size(0), self.condition_dim)
            descriptors.append(condition.to(device=features.device, dtype=features.dtype))
        alphas = torch.sigmoid(self.alpha_mlp(torch.cat(descriptors, dim=1))).view(features.size(0), 3, 1, 1, 1)
        fg_residual = self.fg_delta(features * foreground) * foreground
        bg_residual = self.bg_delta(features * background) * background
        bd_residual = self.bd_delta(features * boundary) * boundary
        residuals = torch.stack([fg_residual, bg_residual, bd_residual], dim=1)
        return features + self.strength * torch.tanh((alphas * residuals).sum(dim=1))


class ForegroundBackgroundDecoder(nn.Module):
    def __init__(self, channels: int, activation: nn.Module, num_blocks: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=bias),
            activation,
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
        )
        self.fg_decoder = nn.Sequential(
            *[RCB(channels, activation, bias=bias) for _ in range(num_blocks)],
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
        )
        self.bg_decoder = nn.Sequential(
            *[RCB(channels, activation, bias=bias) for _ in range(num_blocks)],
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(3, max(1, channels // 2), kernel_size=3, padding=1, bias=bias),
            activation,
            nn.Conv2d(max(1, channels // 2), 1, kernel_size=3, padding=1, bias=bias),
        )
        nn.init.zeros_(self.fg_decoder[-1].weight)
        nn.init.zeros_(self.bg_decoder[-1].weight)
        nn.init.zeros_(self.fusion_gate[-1].weight)
        if self.fg_decoder[-1].bias is not None:
            nn.init.zeros_(self.fg_decoder[-1].bias)
        if self.bg_decoder[-1].bias is not None:
            nn.init.zeros_(self.bg_decoder[-1].bias)
        if self.fusion_gate[-1].bias is not None:
            nn.init.zeros_(self.fusion_gate[-1].bias)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        boundary = 4.0 * mask * (1.0 - mask)
        mask_context = self.mask_encoder(torch.cat([mask, 1.0 - mask, boundary], dim=1))
        fg_delta = self.fg_decoder(features * mask + mask_context)
        bg_delta = self.bg_decoder(features * (1.0 - mask) + mask_context)
        gate_input = torch.cat([mask, 1.0 - mask, boundary], dim=1)
        fusion_mask = torch.clamp(mask + 0.1 * torch.tanh(self.fusion_gate(gate_input)), 0.0, 1.0)
        return fg_delta * fusion_mask + bg_delta * (1.0 - fusion_mask)


class FrequencyRefinementBlock(nn.Module):
    def __init__(self, channels: int, activation: nn.Module, strength: float = 0.05, bias: bool = True) -> None:
        super().__init__()
        self.strength = strength
        self.low_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.high_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
            activation,
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
        )
        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=bias)
        nn.init.zeros_(self.fusion.weight)
        if self.fusion.bias is not None:
            nn.init.zeros_(self.fusion.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        low = F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)
        high = features - low
        refined = self.fusion(torch.cat([self.low_proj(low), self.high_proj(high)], dim=1))
        return features + self.strength * torch.tanh(refined)


class WaveletDetailRefinementBlock(nn.Module):
    def __init__(self, channels: int, activation: nn.Module, strength: float = 0.05, bias: bool = True) -> None:
        super().__init__()
        self.strength = strength
        self.detail_encoder = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=bias),
            activation,
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
        )
        self.detail_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=bias),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.detail_encoder[-1].weight)
        if self.detail_encoder[-1].bias is not None:
            nn.init.zeros_(self.detail_encoder[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        even_h = features[:, :, 0::2, :]
        odd_h = features[:, :, 1::2, :]
        if odd_h.shape[-2:] != even_h.shape[-2:]:
            odd_h = F.pad(odd_h, (0, 0, 0, even_h.size(2) - odd_h.size(2)))
        low_h = 0.5 * (even_h + odd_h)
        high_h = 0.5 * (even_h - odd_h)
        even_w = low_h[:, :, :, 0::2]
        odd_w = low_h[:, :, :, 1::2]
        if odd_w.shape[-2:] != even_w.shape[-2:]:
            odd_w = F.pad(odd_w, (0, even_w.size(3) - odd_w.size(3), 0, 0))
        ll = 0.5 * (even_w + odd_w)
        hl = 0.5 * (even_w - odd_w)
        lh = F.avg_pool2d(high_h, kernel_size=(1, 2), stride=(1, 2), ceil_mode=True, count_include_pad=False)
        ll_up = F.interpolate(ll, size=features.shape[-2:], mode="bilinear", align_corners=False)
        hl_up = F.interpolate(hl, size=features.shape[-2:], mode="bilinear", align_corners=False)
        lh_up = F.interpolate(lh, size=features.shape[-2:], mode="bilinear", align_corners=False)
        high = features - ll_up
        detail = self.detail_encoder(torch.cat([high, hl_up, lh_up], dim=1))
        return features + self.strength * torch.tanh(detail) * self.detail_gate(high.abs())


class LightweightTransformerContextBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        strength: float = 0.05,
        num_heads: int = 4,
        pool_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.strength = strength
        self.pool_size = pool_size
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True, bias=bias)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2, bias=bias),
            nn.GELU(),
            nn.Linear(channels * 2, channels, bias=bias),
        )
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        height, width = features.shape[-2:]
        pooled = F.adaptive_avg_pool2d(features, (self.pool_size, self.pool_size))
        tokens = pooled.flatten(2).transpose(1, 2)
        attn_in = self.norm1(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.norm2(tokens))
        context = tokens.transpose(1, 2).view(features.size(0), features.size(1), self.pool_size, self.pool_size)
        context = F.interpolate(context, size=(height, width), mode="bilinear", align_corners=False)
        return features + self.strength * torch.tanh(self.out_proj(context))


class LearnableWhiteBalanceCorrection(nn.Module):
    def __init__(self, strength: float = 0.08, hidden_dim: int = 16, bias: bool = True) -> None:
        super().__init__()
        self.strength = strength
        self.mlp = nn.Sequential(
            nn.Linear(9, hidden_dim, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 6, bias=bias),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        stats_image = image.clamp(0.0, 1.0)
        channel_mean = stats_image.mean(dim=(2, 3))
        channel_std = stats_image.std(dim=(2, 3), unbiased=False)
        gray_mean = channel_mean.mean(dim=1, keepdim=True)
        color_cast = gray_mean - channel_mean
        gain, bias = self.mlp(torch.cat([channel_mean, channel_std, color_cast], dim=1)).chunk(2, dim=1)
        gain = 1.0 + self.strength * torch.tanh(gain).view(-1, 3, 1, 1)
        bias = self.strength * torch.tanh(bias).view(-1, 3, 1, 1)
        return image * gain + bias


class ImageRefinementBlock(nn.Module):
    def __init__(self, channels: int, bias: bool = True) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ImageSpaceRefinementHead(nn.Module):
    def __init__(self, channels: int = 32, strength: float = 0.1, bias: bool = True) -> None:
        super().__init__()
        self.strength = strength
        self.entry = nn.Sequential(
            nn.Conv2d(12, channels, kernel_size=3, padding=1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.refine_high = ImageRefinementBlock(channels, bias=bias)
        self.down = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.refine_low = ImageRefinementBlock(channels * 2, bias=bias)
        self.up = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1, bias=bias),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        if self.fuse[-1].bias is not None:
            nn.init.zeros_(self.fuse[-1].bias)

    def forward(
        self,
        output: torch.Tensor,
        base: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if base is None:
            return output
        base = base.to(device=output.device, dtype=output.dtype)
        if mask is None:
            mask = output.new_ones((output.size(0), 1, output.size(2), output.size(3)))
        if mask.shape[-2:] != output.shape[-2:]:
            mask = F.interpolate(mask, size=output.shape[-2:], mode="bilinear", align_corners=False)
        mask = mask.to(device=output.device, dtype=output.dtype).clamp(0.0, 1.0)
        background = 1.0 - mask
        boundary = (4.0 * mask * background).clamp(0.0, 1.0)
        high = self.refine_high(self.entry(torch.cat([output, base, output - base, mask, background, boundary], dim=1)))
        low = self.refine_low(self.down(high))
        low = F.interpolate(self.up(low), size=high.shape[-2:], mode="bilinear", align_corners=False)
        correction = self.fuse(torch.cat([high, low], dim=1))
        return output + self.strength * torch.tanh(correction)


class PhysicalParameterHead(nn.Module):
    def __init__(self, channels: int, activation: nn.Module, bias: bool = True) -> None:
        super().__init__()
        hidden = max(8, channels // 2)
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=bias),
            activation,
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=bias),
            activation,
        )
        self.transmission = nn.Conv2d(hidden, 1, kernel_size=3, padding=1, bias=bias)
        self.ambient = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, 3, kernel_size=1, bias=bias),
        )

    def forward(self, features: torch.Tensor, output_size: tuple[int, int]) -> dict[str, torch.Tensor]:
        hidden = self.trunk(features)
        transmission = torch.sigmoid(self.transmission(hidden))
        if transmission.shape[-2:] != output_size:
            transmission = F.interpolate(transmission, size=output_size, mode="bilinear", align_corners=False)
        transmission = 0.05 + 0.9 * transmission
        ambient = torch.sigmoid(self.ambient(hidden))
        return {"transmission": transmission, "ambient": ambient}


class ClipUIENet(nn.Module):
    def __init__(
        self,
        num_branch: int = 3,
        n_feat: int = 32,
        n_rcb: int = 2,
        chan_factor: int = 2,
        bias: bool = True,
        use_sam_mask: bool = False,
        use_dual_region_branch: bool = False,
        region_branch_rcb: int = 1,
        region_fusion_strength: float = 0.2,
        use_rsdc: bool = False,
        rsdc_strength: float = 0.1,
        multimodal_aux_strength: float = 0.03,
        use_multimodal_initial_condition: bool = False,
        use_fg_bg_decoder: bool = False,
        fg_bg_decoder_blocks: int = 1,
        fg_bg_decoder_strength: float = 0.1,
        use_frequency_refinement: bool = False,
        frequency_refinement_strength: float = 0.05,
        use_wavelet_refinement: bool = False,
        wavelet_refinement_strength: float = 0.05,
        use_transformer_context: bool = False,
        transformer_context_strength: float = 0.05,
        use_white_balance_correction: bool = False,
        white_balance_strength: float = 0.08,
        use_residual_output: bool = False,
        residual_output_strength: float = 0.8,
        use_output_refinement: bool = False,
        output_refinement_strength: float = 0.1,
        use_physical_head: bool = False,
        multimodal_config: MultimodalSection | None = None,
    ) -> None:
        super().__init__()
        self.num_branch = num_branch
        self.use_sam_mask = use_sam_mask
        self.use_dual_region_branch = use_dual_region_branch
        self.region_fusion_strength = region_fusion_strength
        self.use_rsdc = use_rsdc
        self.multimodal_aux_strength = multimodal_aux_strength
        self.use_multimodal_initial_condition = use_multimodal_initial_condition
        self.use_fg_bg_decoder = use_fg_bg_decoder
        self.fg_bg_decoder_strength = fg_bg_decoder_strength
        self.use_frequency_refinement = use_frequency_refinement
        self.use_wavelet_refinement = use_wavelet_refinement
        self.use_transformer_context = use_transformer_context
        self.use_white_balance_correction = use_white_balance_correction
        self.use_residual_output = use_residual_output
        self.residual_output_strength = residual_output_strength
        self.use_output_refinement = use_output_refinement
        self.use_physical_head = use_physical_head
        self.multimodal_enabled = multimodal_config.enabled if multimodal_config is not None else False
        self.act = nn.LeakyReLU(0.1, True)
        atrous = [1, 2, 3, 4]
        self.dau_top = nn.Sequential(*[RCB(int(n_feat * chan_factor ** 0), self.act, bias=bias) for _ in range(n_rcb)])
        self.dau_mid = nn.Sequential(*[RCB(int(n_feat * chan_factor ** 1), self.act, bias=bias) for _ in range(n_rcb)])
        self.dau_bot = nn.Sequential(*[RCB(int(n_feat * chan_factor ** 2), self.act, bias=bias) for _ in range(n_rcb)])
        self.nl_mid = NonLocalSparseAttention(channels=int(n_feat * chan_factor ** 1))
        self.nl_bot = NonLocalSparseAttention(channels=int(n_feat * chan_factor ** 2))
        self.atb_top = AtrousBlock(int(n_feat * chan_factor ** 0), 3, 1, self.act, atrous)
        self.atb_mid = AtrousBlock(int(n_feat * chan_factor ** 1), 3, 1, self.act, atrous)
        self.atb_bot = AtrousBlock(int(n_feat * chan_factor ** 2), 3, 1, self.act, atrous)
        self.down2 = nn.Sequential(DownSample(int((chan_factor ** 0) * n_feat), 2, chan_factor))
        self.down4 = nn.Sequential(
            DownSample(int((chan_factor ** 0) * n_feat), 2, chan_factor),
            DownSample(int((chan_factor ** 1) * n_feat), 2, chan_factor),
        )
        self.up21_1 = UpSample(int((chan_factor ** 1) * n_feat), 2, chan_factor)
        self.up21_2 = UpSample(int((chan_factor ** 1) * n_feat), 2, chan_factor)
        self.up32_1 = UpSample(int((chan_factor ** 2) * n_feat), 2, chan_factor)
        self.up32_2 = UpSample(int((chan_factor ** 2) * n_feat), 2, chan_factor)
        input_channels = 5 if use_sam_mask else 4
        self.conv_in = nn.Conv2d(input_channels, n_feat, kernel_size=3, padding=1, bias=bias)
        self.conv_mid = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=bias)
        self.conv_out = nn.Conv2d(n_feat, 3, kernel_size=3, padding=1, bias=bias)
        self.aff_top = AFF(int(n_feat * chan_factor ** 0), self.act)
        self.aff_mid = AFF(int(n_feat * chan_factor ** 1), self.act)
        self.aff_final = AFF(n_feat, self.act)
        self.get_gradient = GetGradientNopadding1CGray()
        self.b_concat_1 = nn.Conv2d(2 * n_feat, n_feat, kernel_size=3, padding=1, bias=bias)
        self.b_block_1 = RCB(2 * n_feat, self.act, bias=bias)
        self.b_concat_2 = nn.Conv2d(2 * n_feat, n_feat, kernel_size=3, padding=1, bias=bias)
        self.b_block_2 = RCB(2 * n_feat, self.act, bias=bias)
        self.style_loss = StyleLossModule()
        self.adaptive_route = GramGlobalWeightNet(channels=n_feat * chan_factor ** 2, num_weights=num_branch)
        if self.use_frequency_refinement:
            self.frequency_refinement = FrequencyRefinementBlock(
                channels=n_feat,
                activation=self.act,
                strength=frequency_refinement_strength,
                bias=bias,
            )
        else:
            self.frequency_refinement = None
        if self.use_wavelet_refinement:
            self.wavelet_refinement = WaveletDetailRefinementBlock(
                channels=n_feat,
                activation=self.act,
                strength=wavelet_refinement_strength,
                bias=bias,
            )
        else:
            self.wavelet_refinement = None
        if self.use_transformer_context:
            self.transformer_context = LightweightTransformerContextBlock(
                channels=int(n_feat * chan_factor ** 2),
                strength=transformer_context_strength,
                bias=bias,
            )
        else:
            self.transformer_context = None
        if self.use_white_balance_correction:
            self.white_balance_correction = LearnableWhiteBalanceCorrection(
                strength=white_balance_strength,
                bias=bias,
            )
        else:
            self.white_balance_correction = None
        if self.use_output_refinement:
            self.output_refinement = ImageSpaceRefinementHead(
                strength=output_refinement_strength,
                bias=bias,
            )
        else:
            self.output_refinement = None
        if self.use_physical_head:
            self.physical_head = PhysicalParameterHead(channels=n_feat, activation=self.act, bias=bias)
        else:
            self.physical_head = None
        if self.use_fg_bg_decoder:
            self.fg_bg_decoder = ForegroundBackgroundDecoder(
                channels=n_feat,
                activation=self.act,
                num_blocks=fg_bg_decoder_blocks,
                bias=bias,
            )
        else:
            self.fg_bg_decoder = None
        if self.use_dual_region_branch:
            self.region_top = MaskGuidedResidualAdapter(
                int(n_feat * chan_factor ** 0),
                self.act,
                num_rcb=region_branch_rcb,
                bias=bias,
            )
            self.region_bot = MaskGuidedResidualAdapter(
                int(n_feat * chan_factor ** 2),
                self.act,
                num_rcb=region_branch_rcb,
                bias=bias,
            )
        else:
            self.region_top = None
            self.region_bot = None
        rsdc_condition_dim = multimodal_config.adapter_hidden_dim if self.multimodal_enabled and multimodal_config is not None else 0
        if self.use_rsdc:
            self.rsdc = RegionSemanticDegradationControl(
                channels=int(n_feat * chan_factor ** 0),
                activation=self.act,
                condition_dim=rsdc_condition_dim,
                strength=rsdc_strength,
                bias=bias,
            )
        else:
            self.rsdc = None
        if self.multimodal_enabled:
            self.multimodal_adapter = MultimodalConditionAdapter(multimodal_config)
            self.cond_top = nn.Linear(multimodal_config.adapter_hidden_dim, 2 * int(n_feat * chan_factor ** 0))
            self.cond_mid = nn.Linear(multimodal_config.adapter_hidden_dim, 2 * int(n_feat * chan_factor ** 1))
            self.cond_bot = nn.Linear(multimodal_config.adapter_hidden_dim, 2 * int(n_feat * chan_factor ** 2))
            nn.init.zeros_(self.cond_top.weight)
            nn.init.zeros_(self.cond_top.bias)
            nn.init.zeros_(self.cond_mid.weight)
            nn.init.zeros_(self.cond_mid.bias)
            nn.init.zeros_(self.cond_bot.weight)
            nn.init.zeros_(self.cond_bot.bias)
        else:
            self.multimodal_adapter = None

    def _apply_condition(self, features: torch.Tensor, condition: torch.Tensor, projector: nn.Linear) -> torch.Tensor:
        gamma, beta = projector(condition).chunk(2, dim=1)
        gamma = (self.multimodal_aux_strength * torch.tanh(gamma)).unsqueeze(-1).unsqueeze(-1)
        beta = (self.multimodal_aux_strength * torch.tanh(beta)).unsqueeze(-1).unsqueeze(-1)
        return features * (1.0 + gamma) + beta

    def _prepare_mask(self, mask: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return mask.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x_grad = self.get_gradient(x)
        features = [x.clone(), x_grad]
        if self.use_sam_mask:
            features.append(self._prepare_mask(mask, x))
        x_str = self.conv_in(torch.cat(features, dim=1))
        x_style = self.down4(x_str)
        return x_str, x_style

# MGRA
    def _apply_region_branch(
        self,
        x_str: torch.Tensor,
        x_style: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_dual_region_branch:
            return x_str, x_style
        top_mask = self._prepare_mask(mask, x_str)
        bot_mask = self._prepare_mask(mask, x_style)
        top_residual = self.region_top(x_str, top_mask)
        bot_residual = self.region_bot(x_style, bot_mask)
        return x_str + self.region_fusion_strength * top_residual, x_style + self.region_fusion_strength * bot_residual

    def transfer(
        self,
        x_str: torch.Tensor,
        x_style: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_mid = self.down2(x_str)
        x_str1 = self.dau_top(self.atb_top(x_str))
        x_mid1 = self.dau_mid(self.atb_mid(x_mid))
        x_style1 = self.dau_bot(self.atb_bot(x_style))
        if self.transformer_context is not None:
            x_style1 = self.transformer_context(x_style1)
        if condition is not None:
            x_str1 = self._apply_condition(x_str1, condition, self.cond_top)
            x_mid1 = self._apply_condition(x_mid1, condition, self.cond_mid)
            x_style1 = self._apply_condition(x_style1, condition, self.cond_bot)
        x_mid1 = self.aff_mid(x_mid1, self.up32_1(x_style1))
        x_str1 = self.aff_top(x_str1, self.up21_1(x_mid1))
        x_str2 = self.dau_top(self.atb_top(x_str1))
        x_mid2 = self.dau_mid(self.nl_mid(x_mid1))
        x_style2 = self.dau_bot(self.nl_bot(x_style1))
        if self.transformer_context is not None:
            x_style2 = self.transformer_context(x_style2)
        if condition is not None:
            x_str2 = self._apply_condition(x_str2, condition, self.cond_top)
            x_mid2 = self._apply_condition(x_mid2, condition, self.cond_mid)
            x_style2 = self._apply_condition(x_style2, condition, self.cond_bot)
        x_mid2 = self.aff_mid(x_mid2, self.up32_2(x_style2))
        x_str2 = self.aff_top(x_str2, self.up21_2(x_mid2))
        mid_out = self.conv_mid(x_str2) + x_str2
        x_cat_1 = self.b_concat_1(self.b_block_1(torch.cat([x_str, x_str1], dim=1)))
        x_cat_2 = self.b_concat_2(self.b_block_2(torch.cat([x_cat_1, x_str2], dim=1)))
        return self.aff_final(mid_out, x_cat_2), x_style2

    def _compute_multimodal_condition(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
        prompts: list[str] | None = None,
    ) -> torch.Tensor | None:
        if not self.multimodal_enabled:
            return None
        if prompts is None:
            raise ValueError("Prompts are required when multimodal support is enabled.")
        prepared_mask = self._prepare_mask(mask, inputs)
        return self.multimodal_adapter(inputs, prepared_mask, prompts)

    def _apply_initial_condition(
        self,
        x_str: torch.Tensor,
        x_style: torch.Tensor,
        condition: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if condition is None or not self.use_multimodal_initial_condition:
            return x_str, x_style
        return self._apply_condition(x_str, condition, self.cond_top), self._apply_condition(x_style, condition, self.cond_bot)

    def _apply_rsdc(
        self,
        x_str: torch.Tensor,
        mask: torch.Tensor | None,
        condition: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.rsdc is None:
            return x_str
        top_mask = self._prepare_mask(mask, x_str)
        return self.rsdc(x_str, top_mask, condition)

    def _compute_branch_features(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
        prompts: list[str] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        x_str, x_style = self.encode(inputs, mask)
        x_str, x_style = self._apply_region_branch(x_str, x_style, mask)
        condition = self._compute_multimodal_condition(inputs, mask, prompts)
        x_str, x_style = self._apply_initial_condition(x_str, x_style, condition)
        x_str = self._apply_rsdc(x_str, mask, condition)
        x_style_branch_outs = [x_style]
        x_str_branch_outs = [x_str]
        for _ in range(self.num_branch - 1):
            x_str, x_style = self.transfer(x_str, x_style, condition)
            x_style_branch_outs.append(x_style)
            x_str_branch_outs.append(x_str)
        return x_str_branch_outs, x_style_branch_outs

    def forward_candidates(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
        prompts: list[str] | None = None,
    ) -> list[torch.Tensor]:
        x_str_branch_outs, _ = self._compute_branch_features(inputs, mask, prompts)
        return [self.decode(branch_x_str, mask, inputs) for branch_x_str in x_str_branch_outs]

    def decode(
        self,
        out_f: torch.Tensor,
        mask: torch.Tensor | None = None,
        base: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.frequency_refinement is not None:
            out_f = self.frequency_refinement(out_f)
        if self.wavelet_refinement is not None:
            out_f = self.wavelet_refinement(out_f)
        if self.use_fg_bg_decoder:
            decode_mask = self._prepare_mask(mask, out_f)
            feature_delta = self.fg_bg_decoder(out_f, decode_mask)
            out_f = out_f + self.fg_bg_decoder_strength * torch.tanh(feature_delta)
        output = self.conv_out(out_f)
        if self.use_residual_output and base is not None:
            output = base + self.residual_output_strength * torch.tanh(output)
        if self.white_balance_correction is not None:
            output = self.white_balance_correction(output)
        if self.output_refinement is not None:
            output = self.output_refinement(output, base, mask)
        return output

    def _physical_aux(self, features: torch.Tensor, output: torch.Tensor) -> dict[str, torch.Tensor] | None:
        if self.physical_head is None:
            return None
        return self.physical_head(features, output.shape[-2:])

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None, prompts: list[str] | None = None) -> torch.Tensor:
        x_str, x_style = self.encode(inputs, mask)
        x_str, x_style = self._apply_region_branch(x_str, x_style, mask)
        condition = self._compute_multimodal_condition(inputs, mask, prompts)
        x_str, x_style = self._apply_initial_condition(x_str, x_style, condition)
        x_str = self._apply_rsdc(x_str, mask, condition)
        x_str, _ = self.transfer(x_str, x_style, condition)
        return self.decode(x_str, mask, inputs)

    def forward_recon(self, inputs: torch.Tensor, mask: torch.Tensor | None = None, prompts: list[str] | None = None) -> torch.Tensor:
        x_str, x_style = self.encode(inputs, mask)
        x_str, _ = self._apply_region_branch(x_str, x_style, mask)
        return self.decode(x_str, mask, inputs)

    def forward_style_loss(
        self,
        inputs: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor | None = None,
        prompts: list[str] | None = None,
        return_aux: bool = False,
    ):
        x_str, x_style = self.encode(inputs, mask)
        x_str, x_style = self._apply_region_branch(x_str, x_style, mask)
        condition = self._compute_multimodal_condition(inputs, mask, prompts)
        x_str, x_style = self._apply_initial_condition(x_str, x_style, condition)
        x_str = self._apply_rsdc(x_str, mask, condition)
        x_str, x_style = self.transfer(x_str, x_style, condition)
        _, gt_style = self.encode(gt, mask)
        output = self.decode(x_str, mask, inputs)
        if return_aux:
            return output, self.style_loss(x_style, gt_style), self._physical_aux(x_str, output)
        return output, self.style_loss(x_style, gt_style)

    def forward_route(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
        prompts: list[str] | None = None,
        return_logits: bool = False,
        return_proc_outs: bool = False,
        hard_route: bool = False,
    ):
        x_str_branch_outs, x_style_branch_outs = self._compute_branch_features(inputs, mask, prompts)
        x_str = x_str_branch_outs[0]
        x_style = x_style_branch_outs[0]
        x_style_w = torch.zeros_like(x_style)
        x_str_w = torch.zeros_like(x_str)
        x_outs = [self.decode(branch_x_str, mask, inputs) for branch_x_str in x_str_branch_outs] if return_proc_outs else []
        weight_logits = self.adaptive_route(x_style_branch_outs)
        if hard_route:
            route_indices = torch.argmax(weight_logits, dim=1)
            weights = F.one_hot(route_indices, num_classes=self.num_branch).to(dtype=x_str.dtype, device=x_str.device)
        else:
            weights = F.softmax(weight_logits, dim=1)
        for branch_index in range(self.num_branch):
            weight = weights[:, branch_index].view(-1, 1, 1, 1)
            x_style_w += weight * x_style_branch_outs[branch_index]
            x_str_w += weight * x_str_branch_outs[branch_index]
        output = self.decode(x_str_w, mask, inputs)
        if return_logits and return_proc_outs:
            return output, weight_logits, x_outs
        if return_logits:
            return output, weight_logits
        return output

    def forward_route_style_loss(
        self,
        inputs: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor | None = None,
        prompts: list[str] | None = None,
        return_logits: bool = False,
        return_proc_outs: bool = False,
        return_aux: bool = False,
    ):
        x_str_branch_outs, x_style_branch_outs = self._compute_branch_features(inputs, mask, prompts)
        x_str = x_str_branch_outs[0]
        x_style = x_style_branch_outs[0]
        x_style_w = torch.zeros_like(x_style)
        x_str_w = torch.zeros_like(x_str)
        x_outs = []
        if return_proc_outs:
            x_outs.extend(self.decode(branch_x_str, mask, inputs) for branch_x_str in x_str_branch_outs)
        weight_logits = self.adaptive_route(x_style_branch_outs)
        weights = F.softmax(weight_logits, dim=1)
        for branch_index in range(self.num_branch):
            weight = weights[:, branch_index].view(-1, 1, 1, 1)
            x_style_w += weight * x_style_branch_outs[branch_index]
            x_str_w += weight * x_str_branch_outs[branch_index]
        output = self.decode(x_str_w, mask, inputs)
        _, gt_style = self.encode(gt, mask)
        style_loss = self.style_loss(x_style_w, gt_style)
        aux = self._physical_aux(x_str_w, output)
        if return_logits and return_proc_outs and return_aux:
            return output, weight_logits, style_loss, x_outs, aux
        if return_logits and return_proc_outs:
            return output, weight_logits, style_loss, x_outs
        if return_logits and return_aux:
            return output, weight_logits, style_loss, aux
        if return_logits:
            return output, weight_logits, style_loss
        if return_aux:
            return output, style_loss, aux
        return output, style_loss
