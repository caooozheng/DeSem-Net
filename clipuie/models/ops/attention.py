from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def batched_index_select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None]).expand(-1, -1, last_dim)


class NonLocalSparseAttention(nn.Module):
    def __init__(
        self,
        n_hashes: int = 4,
        channels: int = 64,
        k_size: int = 3,
        reduction: int = 4,
        chunk_size: int = 144,
        res_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.n_hashes = n_hashes
        self.reduction = reduction
        self.res_scale = res_scale
        self.conv_match = nn.Conv2d(channels, channels // reduction, k_size, padding=k_size // 2, bias=True)
        self.conv_assembly = nn.Conv2d(channels, channels, 1, bias=True)

    def lsh(self, hash_buckets: int, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets // 2)
        random_rotations = torch.randn(rotations_shape, dtype=x.dtype, device=x.device).expand(batch, -1, -1, -1)
        rotated_vecs = torch.einsum("btf,bfhi->bhti", x, random_rotations)
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
        hash_codes = torch.argmax(rotated_vecs, dim=-1)
        offsets = torch.arange(self.n_hashes, device=x.device).view(1, -1, 1) * hash_buckets
        return (hash_codes + offsets).reshape(batch, -1)

    @staticmethod
    def add_adjacent_buckets(x: torch.Tensor) -> torch.Tensor:
        x_extra_back = torch.cat([x[:, :, -1:, ...], x[:, :, :-1, ...]], dim=2)
        x_extra_forward = torch.cat([x[:, :, 1:, ...], x[:, :, :1, ...]], dim=2)
        return torch.cat([x, x_extra_back, x_extra_forward], dim=3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = inputs.shape
        x_embed = self.conv_match(inputs).view(batch, -1, height * width).permute(0, 2, 1).contiguous()
        y_embed = self.conv_assembly(inputs).view(batch, -1, height * width).permute(0, 2, 1).contiguous()
        sequence_length, channels = x_embed.shape[-2:]
        hash_buckets = min(sequence_length // self.chunk_size + (sequence_length // self.chunk_size) % 2, 128)
        hash_codes = self.lsh(hash_buckets, x_embed).detach()
        _, indices = hash_codes.sort(dim=-1)
        _, undo_sort = indices.sort(dim=-1)
        mod_indices = indices % sequence_length
        x_embed_sorted = batched_index_select(x_embed, mod_indices)
        y_embed_sorted = batched_index_select(y_embed, mod_indices)
        padding = self.chunk_size - sequence_length % self.chunk_size if sequence_length % self.chunk_size != 0 else 0
        x_buckets = x_embed_sorted.reshape(batch, self.n_hashes, -1, channels)
        y_buckets = y_embed_sorted.reshape(batch, self.n_hashes, -1, channels * self.reduction)
        if padding:
            x_buckets = torch.cat([x_buckets, x_buckets[:, :, -padding:, :].clone()], dim=2)
            y_buckets = torch.cat([y_buckets, y_buckets[:, :, -padding:, :].clone()], dim=2)
        x_buckets = x_buckets.reshape(batch, self.n_hashes, -1, self.chunk_size, channels)
        y_buckets = y_buckets.reshape(batch, self.n_hashes, -1, self.chunk_size, channels * self.reduction)
        x_match = self.add_adjacent_buckets(F.normalize(x_buckets, p=2, dim=-1, eps=5e-5))
        y_buckets = self.add_adjacent_buckets(y_buckets)
        raw_score = torch.einsum("bhkie,bhkje->bhkij", x_buckets, x_match)
        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
        score = torch.exp(raw_score - bucket_score)
        bucket_score = bucket_score.reshape(batch, self.n_hashes, -1)
        ret = torch.einsum("bukij,bukje->bukie", score, y_buckets).reshape(batch, self.n_hashes, -1, channels * self.reduction)
        if padding:
            ret = ret[:, :, :-padding, :].clone()
            bucket_score = bucket_score[:, :, :-padding].clone()
        ret = ret.reshape(batch, -1, channels * self.reduction)
        bucket_score = bucket_score.reshape(batch, -1)
        ret = batched_index_select(ret, undo_sort)
        bucket_score = bucket_score.gather(1, undo_sort)
        ret = ret.reshape(batch, self.n_hashes, sequence_length, channels * self.reduction)
        bucket_score = bucket_score.reshape(batch, self.n_hashes, sequence_length, 1)
        probs = F.softmax(bucket_score, dim=1)
        ret = torch.sum(ret * probs, dim=1)
        ret = ret.permute(0, 2, 1).view(batch, -1, height, width).contiguous()
        return ret * self.res_scale + inputs
