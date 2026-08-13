"""Training phase contracts and implementations."""

from src.train.phases.base import TrainingPhase
from src.train.phases.label_only import LabelOnlyPhase
from src.train.phases.registry import get_phase, registered_phases

__all__ = [
    "LabelOnlyPhase",
    "TrainingPhase",
    "get_phase",
    "registered_phases",
]
