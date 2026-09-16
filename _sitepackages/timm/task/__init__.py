"""Training task abstractions for timm.

This module provides task-based abstractions for training loops where each task
encapsulates both the forward pass and loss computation, returning a dictionary
with loss components and outputs for logging.
"""

from .classification import ClassificationTask
from .distillation import DistillationTeacher, FeatureDistillationTask, LogitDistillationTask
from .task import TrainingTask
from .token_distillation import TokenDistillationTask, TokenDistillationTeacher

__all__ = [
    "ClassificationTask",
    "DistillationTeacher",
    "FeatureDistillationTask",
    "LogitDistillationTask",
    "TokenDistillationTask",
    "TokenDistillationTeacher",
    "TrainingTask",
]
