from typing import Any, Dict, Iterable, Protocol, Type, Union

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias
try:
    from typing import TypeVar
except ImportError:
    pass

import torch
import torch.optim

try:
    from torch.optim.optimizer import ParamsT
except (ImportError, TypeError):
    ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


OptimType = Type[torch.optim.Optimizer]


class OptimizerCallable(Protocol):
    """Protocol for optimizer constructor signatures."""

    def __call__(self, params: ParamsT, **kwargs) -> torch.optim.Optimizer: ...


__all__ = ["OptimType", "OptimizerCallable", "ParamsT"]
