"""Typed source rows for reproducible thesis figures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingHistoryPoint:
    """Logged optimization values at one global step."""

    step: int
    epoch: float
    train_loss: float | None
    eval_loss: float | None
    learning_rate: float | None


@dataclass(frozen=True, slots=True)
class CheckpointMetricPoint:
    """Quality and validation loss at one saved checkpoint."""

    checkpoint_id: str
    epoch: float
    top1_accuracy: float
    macro_f1: float
    balanced_accuracy: float
    eval_loss: float


@dataclass(frozen=True, slots=True)
class DistributionPoint:
    """Count for a category in a named split or source grouping."""

    split: str
    category: str
    count: int


@dataclass(frozen=True, slots=True)
class TrainableParameterPoint:
    """Trainable parameter count for one model component."""

    component: str
    parameter_count: int


@dataclass(frozen=True, slots=True)
class ResourcePoint:
    """Resource monitor sample collected during training."""

    elapsed_seconds: float
    step: int | None
    throughput_samples_per_second: float | None
    allocated_vram_gib: float | None
    gpu_utilization_percent: float | None
    power_watts: float | None
    temperature_celsius: float | None


@dataclass(frozen=True, slots=True)
class QualityCostPoint:
    """Run-level quality and cost coordinates."""

    run_id: str
    experiment_id: str
    top1_accuracy: float
    macro_f1: float
    gpu_hours: float
    peak_vram_gib: float
    trainable_parameters: int
