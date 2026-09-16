"""Dual-style DySample upsampling with learnable feature fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def normal_init(module, mean=0.0, std=1.0, bias=0.0):
    """Initialize a module with a normal weight distribution."""
    if getattr(module, "weight", None) is not None:
        nn.init.normal_(module.weight, mean, std)
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, value, bias=0.0):
    """Initialize a module with constant weights and bias."""
    if getattr(module, "weight", None) is not None:
        nn.init.constant_(module.weight, value)
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, bias)


class DySample_UP(nn.Module):
    """Dynamic upsampler supporting the ``lp`` and ``pl`` sampling styles."""

    def __init__(self, in_channels, scale=2, style="lp", groups=4, dyscope=False):
        super().__init__()
        if style not in {"lp", "pl"}:
            raise ValueError(f"Unsupported DySample style: {style}")
        if in_channels < groups or in_channels % groups:
            raise ValueError(f"in_channels={in_channels} must be divisible by groups={groups}")
        if style == "pl" and (in_channels < scale**2 or in_channels % scale**2):
            raise ValueError(f"pl style requires in_channels={in_channels} to be divisible by scale^2={scale**2}")

        self.scale = scale
        self.style = style
        self.groups = groups
        offset_in_channels = in_channels // scale**2 if style == "pl" else in_channels
        offset_out_channels = 2 * groups if style == "pl" else 2 * groups * scale**2

        self.offset = nn.Conv2d(offset_in_channels, offset_out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(offset_in_channels, offset_out_channels, 1)
            constant_init(self.scope, 0.0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid(h, h, indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def sample(self, x, offset):
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)
        coords_h = torch.arange(h, dtype=x.dtype, device=x.device) + 0.5
        coords_w = torch.arange(w, dtype=x.dtype, device=x.device) + 0.5
        coords = torch.stack(torch.meshgrid(coords_w, coords_h, indexing="ij")).transpose(1, 2)
        coords = coords.unsqueeze(1).unsqueeze(0)
        normalizer = x.new_tensor([w, h]).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(b, -1, h, w), self.scale)
        coords = coords.reshape(b, 2, -1, self.scale * h, self.scale * w)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(
            x.reshape(b * self.groups, -1, h, w),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(b, -1, self.scale * h, self.scale * w)

    def forward_lp(self, x):
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        shuffled = F.pixel_shuffle(x, self.scale)
        if hasattr(self, "scope"):
            offset = F.pixel_unshuffle(self.offset(shuffled) * self.scope(shuffled).sigmoid(), self.scale) * 0.5
        else:
            offset = F.pixel_unshuffle(self.offset(shuffled), self.scale) * 0.25
        return self.sample(x, offset + self.init_pos)

    def forward(self, x):
        return self.forward_pl(x) if self.style == "pl" else self.forward_lp(x)


class DySampleFusion(nn.Module):
    """Fuse local-preserving (lp) and pixel-shuffled (pl) DySample outputs."""

    def __init__(self, in_channels, scale=2, groups=4, dyscope=False):
        super().__init__()
        self.lp_upsampler = DySample_UP(in_channels, scale, "lp", groups, dyscope)
        self.pl_upsampler = DySample_UP(in_channels, scale, "pl", groups, dyscope)
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, 1)

    def forward(self, x):
        return self.fusion(torch.cat((self.lp_upsampler(x), self.pl_upsampler(x)), dim=1))
