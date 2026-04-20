from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GetGradientNopadding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel_v = torch.FloatTensor([[-0, -1, -0], [0, 0, 0], [0, 1, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_h = torch.FloatTensor([[-0, 0, 0], [-1, 0, 1], [-0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(kernel_v, requires_grad=False)
        self.weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)

    def rgb_to_grayscale(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 3:
            return (x * self.weights.to(x.device)).sum(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        return x

    def forward(self, inp_feat: torch.Tensor, gray: bool = False) -> torch.Tensor:
        if gray:
            inp_feat = self.rgb_to_grayscale(inp_feat)
        outputs = []
        for channel in range(inp_feat.shape[1]):
            feat = inp_feat[:, channel]
            feat_v = F.conv2d(feat.unsqueeze(1), self.weight_v, padding=1)
            feat_h = F.conv2d(feat.unsqueeze(1), self.weight_h, padding=1)
            outputs.append(torch.sqrt(feat_v.pow(2) + feat_h.pow(2) + 1e-6))
        return torch.cat(outputs, dim=1)


class GetGradientNopadding1CGray(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel_v = torch.FloatTensor([[-0, -1, -0], [0, 0, 0], [0, 1, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_h = torch.FloatTensor([[-0, 0, 0], [-1, 0, 1], [-0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(kernel_v, requires_grad=False)
        self.weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)

    def rgb_to_grayscale(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 3:
            return (x * self.weights.to(x.device)).sum(dim=1, keepdim=False).unsqueeze(1)
        return x

    def forward(self, inp_feat: torch.Tensor) -> torch.Tensor:
        inp_feat = self.rgb_to_grayscale(inp_feat)
        outputs = []
        for channel in range(inp_feat.shape[1]):
            feat = inp_feat[:, channel]
            feat_v = F.conv2d(feat.unsqueeze(1), self.weight_v, padding=1)
            feat_h = F.conv2d(feat.unsqueeze(1), self.weight_h, padding=1)
            outputs.append(torch.sqrt(feat_v.pow(2) + feat_h.pow(2) + 1e-6))
        return torch.cat(outputs, dim=1)


class Down(nn.Module):
    def __init__(self, in_channels: int, chan_factor: int, bias: bool = False) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels * chan_factor), 1, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DownSample(nn.Module):
    def __init__(self, in_channels: int, scale_factor: int, chan_factor: int = 2) -> None:
        super().__init__()
        modules = []
        current = in_channels
        for _ in range(int(math.log2(scale_factor))):
            modules.append(Down(current, chan_factor))
            current = int(current * chan_factor)
        self.body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, chan_factor: int, bias: bool = False) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels // chan_factor), 1, bias=bias),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, in_channels: int, scale_factor: int, chan_factor: int = 2) -> None:
        super().__init__()
        modules = []
        current = in_channels
        for _ in range(int(math.log2(scale_factor))):
            modules.append(Up(current, chan_factor))
            current = int(current // chan_factor)
        self.body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ContextBlock(nn.Module):
    def __init__(self, n_feat: int, activation: nn.Module, bias: bool = True) -> None:
        super().__init__()
        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            activation,
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = x.size()
        input_x = x.view(batch, channel, height * width).unsqueeze(1)
        context_mask = self.softmax(self.conv_mask(x).view(batch, 1, height * width)).unsqueeze(3)
        context = torch.matmul(input_x, context_mask).view(batch, channel, 1, 1)
        return x + self.channel_add_conv(context)


class RCB(nn.Module):
    def __init__(self, n_feat: int, act: nn.Module, bias: bool = True, kernel_size: int = 3) -> None:
        super().__init__()
        self.act = act
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
        )
        self.gcnet = ContextBlock(n_feat, act, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        res = self.act(self.gcnet(res))
        return x + res


class AFF(nn.Module):
    def __init__(self, channels: int, activation: nn.Module, reduction: int = 4) -> None:
        super().__init__()
        inter_channels = int(channels // reduction)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1),
            activation,
            nn.Conv2d(inter_channels, channels, kernel_size=1),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1),
            activation,
            nn.Conv2d(inter_channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        xa = x + residual
        wei = torch.sigmoid(self.local_att(xa) + self.global_att(xa))
        return 2 * x * wei + 2 * residual * (1 - wei)


class AtrousBlock(nn.Module):
    def __init__(self, mid_channels: int, kernel_size: int, stride: int, activation: nn.Module, atrous: list[int]) -> None:
        super().__init__()
        self.atrous_layers = nn.Sequential(
            *[
                nn.Conv2d(mid_channels, mid_channels // 2, kernel_size, stride, dilation=d, padding=d)
                for d in atrous
            ]
        )
        self.conv = nn.Conv2d(mid_channels * 2, mid_channels, 1)
        self.act = activation
        self.att = AFF(mid_channels, activation)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.atrous_layers[0](data))
        x2 = self.act(self.atrous_layers[1](data))
        x3 = self.act(self.atrous_layers[2](data))
        x4 = self.act(self.atrous_layers[3](data))
        x_total = self.act(self.conv(torch.cat((x1, x2, x3, x4), 1)))
        return self.att(data, x_total)
