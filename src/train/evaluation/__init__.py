"""Public evaluation API for the E1 label-only training phase."""

from .comparison import aggregate_seed_runs, compare_paired_runs
from .labels import LabelAlias, LabelVocabulary, canonicalize_predictions
from .metrics import evaluate_predictions
from .models import (
    CheckpointScore,
    ClassificationMetrics,
    ComparableRun,
    ConfidenceInterval,
    MetricAggregate,
    PairedComparison,
    PerClassMetrics,
    PredictionInput,
    PredictionRecord,
    RunContract,
    SeedAggregate,
)
from .selection import select_best_checkpoint

__all__ = [
    "CheckpointScore",
    "ClassificationMetrics",
    "ComparableRun",
    "ConfidenceInterval",
    "LabelAlias",
    "LabelVocabulary",
    "MetricAggregate",
    "PairedComparison",
    "PerClassMetrics",
    "PredictionInput",
    "PredictionRecord",
    "RunContract",
    "SeedAggregate",
    "aggregate_seed_runs",
    "canonicalize_predictions",
    "compare_paired_runs",
    "evaluate_predictions",
    "select_best_checkpoint",
]
