"""Linear attention with rotary position encoding and grouped channel normalization."""

import math

import torch
from torch import nn


class RoPE(nn.Module):
    """Dynamic 2D rotary positional encoding for channel-last feature maps."""

    def __init__(self, base=10000):
        super().__init__()
        self.base = base

    def generate_rotations(self, x):
        spatial_dims = x.shape[1:-1]
        feature_dim = x.shape[-1]
        if len(spatial_dims) != 2 or feature_dim % (2 * len(spatial_dims)):
            raise ValueError(f"RoPE expects BHWC with channels divisible by 4, got {tuple(x.shape)}")
        frequencies_per_axis = feature_dim // (2 * len(spatial_dims))
        theta = 1 / (
            self.base ** (torch.arange(frequencies_per_axis, dtype=x.dtype, device=x.device) / frequencies_per_axis)
        )
        grids = torch.meshgrid(
            *(torch.arange(size, dtype=x.dtype, device=x.device) for size in spatial_dims), indexing="ij"
        )
        angles = torch.cat(tuple(grid.unsqueeze(-1) * theta for grid in grids), dim=-1)
        return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def forward(self, x):
        rotations = torch.view_as_complex(self.generate_rotations(x).contiguous())
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
        return torch.view_as_real(rotations * x_complex).flatten(-2)


class GroupedAttentionNormalization(nn.Module):
    """Normalize channel groups independently at every spatial position."""

    def __init__(self, channels, num_groups=8, eps=1e-6):
        super().__init__()
        self.num_groups = math.gcd(channels, num_groups)
        self.eps = eps

    def forward(self, x):
        b, c, h, w = x.shape
        channel_last = x.permute(0, 2, 3, 1).reshape(b, h, w, self.num_groups, c // self.num_groups)
        mean = channel_last.mean(dim=-1, keepdim=True)
        variance = channel_last.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (channel_last - mean) * torch.rsqrt(variance + self.eps)
        return normalized.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class MLLAttentionWithGroupedNorm(nn.Module):
    """Multi-head linear attention using grouped normalization instead of global normalization."""

    def __init__(self, dim, num_heads=4, qkv_bias=True, num_groups=8, eps=1e-6):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if dim % 4:
            raise ValueError(f"dim={dim} must be divisible by 4 for 2D RoPE")
        self.dim = dim
        self.num_heads = num_heads
        self.eps = eps
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE()
        self.group_norm = GroupedAttentionNormalization(dim, num_groups, eps)

    def forward(self, x):
        x = self.group_norm(x)
        b, c, h, w = x.shape
        n = h * w
        head_dim = c // self.num_heads
        tokens = x.flatten(2).transpose(1, 2)
        qk = self.qk(tokens).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k = self.elu(qk[0]) + 1.0, self.elu(qk[1]) + 1.0

        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, self.num_heads, head_dim)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, self.num_heads, head_dim)
        q_rope = q_rope.permute(0, 2, 1, 3)
        k_rope = k_rope.permute(0, 2, 1, 3)
        q, k, value = (tensor.reshape(b, n, self.num_heads, head_dim).permute(0, 2, 1, 3) for tensor in (q, k, tokens))

        normalizer = torch.reciprocal(q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + self.eps)
        scale = n**-0.5
        key_value = (k_rope.transpose(-2, -1) * scale) @ (value * scale)
        attended = q_rope @ key_value * normalizer
        attended = attended.transpose(1, 2).reshape(b, n, c).transpose(1, 2).reshape(b, c, h, w)
        value_map = value.permute(0, 2, 1, 3).reshape(b, n, c).transpose(1, 2).reshape(b, c, h, w)
        return attended + self.lepe(value_map)
