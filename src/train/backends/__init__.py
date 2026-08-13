"""Fine-tuning backend contracts and implementations."""

from src.train.backends.contracts import (
    BackendFitResult,
    BackendPrediction,
    CheckpointEvent,
    CheckpointObserver,
    FineTuneRequest,
    FineTuningBackend,
    GenerationSpec,
    LoadedCheckpoint,
    LoraSpec,
    MetricEvent,
    MetricSink,
    ModelLoadSpec,
    PredictionSample,
    RuntimeInfo,
    TrainerSpec,
)
from src.train.backends.unsloth import UnslothBackend

__all__ = [
    "BackendFitResult",
    "BackendPrediction",
    "CheckpointEvent",
    "CheckpointObserver",
    "FineTuneRequest",
    "FineTuningBackend",
    "GenerationSpec",
    "LoadedCheckpoint",
    "LoraSpec",
    "MetricEvent",
    "MetricSink",
    "ModelLoadSpec",
    "PredictionSample",
    "RuntimeInfo",
    "TrainerSpec",
    "UnslothBackend",
]
