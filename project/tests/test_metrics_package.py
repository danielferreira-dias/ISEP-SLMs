"""Contracts for the canonical E3 training-metrics namespace."""

from __future__ import annotations

import math

import pytest

from project.metrics.contracts import MetricEvent as CanonicalMetricEvent
from project.metrics.resource_metrics import resource_summary
from project.metrics.resources import LocalResourceMonitor
from project.metrics.trainer_events import (
    CheckpointEvent as CanonicalCheckpointEvent,
)
from project.metrics.trainer_events import TrainerEventBridge
from src.train.backends.contracts import CheckpointEvent as LegacyCheckpointEvent
from src.train.backends.contracts import MetricEvent as LegacyMetricEvent
from src.train.execution.callbacks import TrainerEventBridge as LegacyTrainerEventBridge
from src.train.execution.resources import LocalResourceMonitor as LegacyResourceMonitor
from src.train.resource_metrics import resource_summary as legacy_resource_summary


def test_legacy_metric_imports_are_identity_preserving_facades() -> None:
    """Historical E1/E2 imports resolve to the canonical project objects."""

    assert LegacyMetricEvent is CanonicalMetricEvent
    assert LegacyCheckpointEvent is CanonicalCheckpointEvent
    assert LegacyTrainerEventBridge is TrainerEventBridge
    assert LegacyResourceMonitor is LocalResourceMonitor
    assert legacy_resource_summary is resource_summary


def test_metric_event_rejects_non_finite_values() -> None:
    """A corrupt scalar cannot enter any canonical metric sink."""

    with pytest.raises(ValueError, match="finite"):
        CanonicalMetricEvent(name="loss", value=math.nan, step=1)
