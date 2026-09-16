"""PyTorch FX Based Feature Extraction Helpers
Using https://pytorch.org/vision/stable/feature_extraction.html.
"""

from __future__ import annotations

import torch
from timm.layers import (
    Format,
    create_feature_extractor,
    get_graph_node_names,
    get_notrace_functions,
    get_notrace_modules,
    is_notrace_function,
    is_notrace_module,
    register_notrace_function,
    register_notrace_module,
)
from torch import nn

from ._features import _get_feature_info, _get_return_layers

__all__ = [
    "FeatureGraphNet",
    "GraphExtractNet",
    "create_feature_extractor",
    "get_graph_node_names",
    "get_notrace_functions",
    "get_notrace_modules",
    "is_notrace_function",
    "is_notrace_module",
    "register_notrace_function",
    "register_notrace_module",
]


class FeatureGraphNet(nn.Module):
    """A FX Graph based feature extractor that works with the model feature_info metadata."""

    return_dict: torch.jit.Final[bool]

    def __init__(
        self,
        model: nn.Module,
        out_indices: tuple[int, ...],
        out_map: dict | None = None,
        output_fmt: str = "NCHW",
        return_dict: bool = False,
    ):
        super().__init__()
        self.feature_info = _get_feature_info(model, out_indices)
        if out_map is not None:
            assert len(out_map) == len(out_indices)
        self.output_fmt = Format(output_fmt)
        return_nodes = _get_return_layers(self.feature_info, out_map)
        self.graph_module = create_feature_extractor(model, return_nodes)
        self.return_dict = return_dict

    def forward(self, x):
        out = self.graph_module(x)
        if self.return_dict:
            return out
        return list(out.values())


class GraphExtractNet(nn.Module):
    """A standalone feature extraction wrapper that maps dict -> list or single tensor NOTE: * one can use
    feature_extractor directly if dictionary output is desired * unlike FeatureGraphNet, this is intended to be used
    standalone and not with model feature_info metadata for builtin feature extraction mode *
    create_feature_extractor can be used directly if dictionary output is acceptable.

    Args:
        model: model to extract features from
        return_nodes: node names to return features from (dict or list)
        squeeze_out: if only one output, and output in list format, flatten to single tensor
        return_dict: return as dictionary from extractor with node names as keys, ignores squeeze_out arg
    """

    return_dict: torch.jit.Final[bool]

    def __init__(
        self,
        model: nn.Module,
        return_nodes: dict[str, str] | list[str],
        squeeze_out: bool = True,
        return_dict: bool = False,
    ):
        super().__init__()
        self.squeeze_out = squeeze_out
        self.graph_module = create_feature_extractor(model, return_nodes)
        self.return_dict = return_dict

    def forward(self, x) -> list[torch.Tensor] | torch.Tensor:
        out = self.graph_module(x)
        if self.return_dict:
            return out
        out = list(out.values())
        return out[0] if self.squeeze_out and len(out) == 1 else out
