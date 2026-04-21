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


class ClipUIENet(nn.Module):
    def __init__(
        self,
        num_branch: int = 3,
        n_feat: int = 32,
        n_rcb: int = 2,
        chan_factor: int = 2,
        bias: bool = True,
        use_sam_mask: bool = False,
        multimodal_config: MultimodalSection | None = None,
    ) -> None:
        super().__init__()
        self.num_branch = num_branch
        self.use_sam_mask = use_sam_mask
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

    @staticmethod
    def _apply_condition(features: torch.Tensor, condition: torch.Tensor, projector: nn.Linear) -> torch.Tensor:
        gamma, beta = projector(condition).chunk(2, dim=1)
        gamma = (0.1 * torch.tanh(gamma)).unsqueeze(-1).unsqueeze(-1)
        beta = (0.1 * torch.tanh(beta)).unsqueeze(-1).unsqueeze(-1)
        return features * (1.0 + gamma) + beta

    def _prepare_mask(self, mask: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return mask.to(device=x.device, dtype=x.dtype).clamp_(0.0, 1.0)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x_grad = self.get_gradient(x)
        features = [x.clone(), x_grad]
        if self.use_sam_mask:
            features.append(self._prepare_mask(mask, x))
        x_str = self.conv_in(torch.cat(features, dim=1))
        x_style = self.down4(x_str)
        return x_str, x_style

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
        if condition is not None:
            x_str1 = self._apply_condition(x_str1, condition, self.cond_top)
            x_mid1 = self._apply_condition(x_mid1, condition, self.cond_mid)
            x_style1 = self._apply_condition(x_style1, condition, self.cond_bot)
        x_mid1 = self.aff_mid(x_mid1, self.up32_1(x_style1))
        x_str1 = self.aff_top(x_str1, self.up21_1(x_mid1))
        x_str2 = self.dau_top(self.atb_top(x_str1))
        x_mid2 = self.dau_mid(self.nl_mid(x_mid1))
        x_style2 = self.dau_bot(self.nl_bot(x_style1))
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
        if condition is None:
            return x_str, x_style
        return self._apply_condition(x_str, condition, self.cond_top), self._apply_condition(x_style, condition, self.cond_bot)

    def decode(self, out_f: torch.Tensor) -> torch.Tensor:
        return self.conv_out(out_f)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None, prompts: list[str] | None = None) -> torch.Tensor:
        x_str, x_style = self.encode(inputs, mask)
        condition = self._compute_multimodal_condition(inputs, mask, prompts)
        x_str, x_style = self._apply_initial_condition(x_str, x_style, condition)
        x_str, _ = self.transfer(x_str, x_style, condition)
        return self.decode(x_str)

    def forward_recon(self, inputs: torch.Tensor, mask: torch.Tensor | None = None, prompts: list[str] | None = None) -> torch.Tensor:
        x_str, _ = self.encode(inputs, mask)
        condition = self._compute_multimodal_condition(inputs, mask, prompts)
        if condition is not None:
            x_str = self._apply_condition(x_str, condition, self.cond_top)
        return self.decode(x_str)

    def forward_style_loss(
        self,
        inputs: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor | None = None,
        prompts: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_str, x_style = self.encode(inputs, mask)
        condition = self._compute_multimodal_condition(inputs, mask, prompts)
        x_str, x_style = self._apply_initial_condition(x_str, x_style, condition)
        x_str, x_style = self.transfer(x_str, x_style, condition)
        _, gt_style = self.encode(gt, mask)
        return self.decode(x_str), self.style_loss(x_style, gt_style)

    def forward_route(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
        prompts: list[str] | None = None,
        return_logits: bool = False,
        return_proc_outs: bool = False,
    ):
        x_str, x_style = self.encode(inputs, mask)
        condition = self._compute_multimodal_condition(inputs, mask, prompts)
        x_str, x_style = self._apply_initial_condition(x_str, x_style, condition)
        x_style_w = torch.zeros_like(x_style)
        x_str_w = torch.zeros_like(x_str)
        x_style_branch_outs = [x_style]
        x_str_branch_outs = [x_str]
        x_outs = [self.decode(x_str)]
        for _ in range(self.num_branch - 1):
            x_str, x_style = self.transfer(x_str, x_style, condition)
            x_style_branch_outs.append(x_style)
            x_str_branch_outs.append(x_str)
            if return_proc_outs:
                x_outs.append(self.decode(x_str))
        weight_logits = self.adaptive_route(x_style_branch_outs)
        weights = F.softmax(weight_logits, dim=1)
        for branch_index in range(self.num_branch):
            weight = weights[:, branch_index].view(-1, 1, 1, 1)
            x_style_w += weight * x_style_branch_outs[branch_index]
            x_str_w += weight * x_str_branch_outs[branch_index]
        output = self.decode(x_str_w)
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
    ):
        x_str, x_style = self.encode(inputs, mask)
        condition = self._compute_multimodal_condition(inputs, mask, prompts)
        x_str, x_style = self._apply_initial_condition(x_str, x_style, condition)
        x_style_w = torch.zeros_like(x_style)
        x_str_w = torch.zeros_like(x_str)
        x_style_branch_outs = [x_style]
        x_str_branch_outs = [x_str]
        x_outs = [inputs]
        for _ in range(self.num_branch - 1):
            x_str, x_style = self.transfer(x_str, x_style, condition)
            x_style_branch_outs.append(x_style)
            x_str_branch_outs.append(x_str)
            if return_proc_outs:
                x_outs.append(self.decode(x_str))
        weight_logits = self.adaptive_route(x_style_branch_outs)
        weights = F.softmax(weight_logits, dim=1)
        for branch_index in range(self.num_branch):
            weight = weights[:, branch_index].view(-1, 1, 1, 1)
            x_style_w += weight * x_style_branch_outs[branch_index]
            x_str_w += weight * x_str_branch_outs[branch_index]
        output = self.decode(x_str_w)
        _, gt_style = self.encode(gt, mask)
        style_loss = self.style_loss(x_style_w, gt_style)
        if return_logits and return_proc_outs:
            return output, weight_logits, style_loss, x_outs
        if return_logits:
            return output, weight_logits, style_loss
        return output, style_loss
