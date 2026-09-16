"""AvgPool2d w/ Same Padding.

Hacked together by / Copyright 2020 Ross Wightman
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._fx import register_notrace_module
from .helpers import to_2tuple
from .padding import get_padding_value, pad_same


def avg_pool2d_same(
    x: torch.Tensor,
    kernel_size: list[int],
    stride: list[int],
    padding: list[int] = (0, 0),
    ceil_mode: bool = False,
    count_include_pad: bool = True,
):
    # FIXME how to deal with count_include_pad vs not for external padding?
    x = pad_same(x, kernel_size, stride)
    return F.avg_pool2d(x, kernel_size, stride, (0, 0), ceil_mode, count_include_pad)


@register_notrace_module
class AvgPool2dSame(nn.AvgPool2d):
    """Tensorflow like 'SAME' wrapper for 2D average pooling."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] | str = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        super().__init__(kernel_size, stride, (0, 0), ceil_mode, count_include_pad)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride)
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)


def max_pool2d_same(
    x: torch.Tensor,
    kernel_size: list[int],
    stride: list[int],
    padding: list[int] = (0, 0),
    dilation: list[int] = (1, 1),
    ceil_mode: bool = False,
):
    x = pad_same(x, kernel_size, stride, value=-float("inf"))
    return F.max_pool2d(x, kernel_size, stride, (0, 0), dilation, ceil_mode)


@register_notrace_module
class MaxPool2dSame(nn.MaxPool2d):
    """Tensorflow like 'SAME' wrapper for 2D max pooling."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] | str = 0,
        dilation: int | tuple[int, int] = 1,
        ceil_mode: bool = False,
    ):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        super().__init__(kernel_size, stride, (0, 0), dilation, ceil_mode)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride, value=-float("inf"))
        return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)


def create_pool2d(pool_type, kernel_size, stride=None, **kwargs):
    stride = stride or kernel_size
    padding = kwargs.pop("padding", "")
    padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, **kwargs)
    if is_dynamic:
        if pool_type == "avg":
            return AvgPool2dSame(kernel_size, stride=stride, **kwargs)
        elif pool_type == "max":
            return MaxPool2dSame(kernel_size, stride=stride, **kwargs)
        else:
            assert False, f"Unsupported pool type {pool_type}"
    else:
        if pool_type == "avg":
            return nn.AvgPool2d(kernel_size, stride=stride, padding=padding, **kwargs)
        elif pool_type == "max":
            return nn.MaxPool2d(kernel_size, stride=stride, padding=padding, **kwargs)
        else:
            assert False, f"Unsupported pool type {pool_type}"
