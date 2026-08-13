"""Immutable run-level results shared by orchestration modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.train.evaluation import ClassificationMetrics


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    """Durable outcome of one complete E1 execution."""

    run_directory: Path
    best_checkpoint: Path
    best_metrics: ClassificationMetrics
