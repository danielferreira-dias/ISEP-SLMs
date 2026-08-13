"""Framework-independent contract implemented by training phases."""

from __future__ import annotations

from typing import Protocol

from src.train.domain import (
    FormattedExample,
    LabeledImageSample,
    Taxonomy,
    TrainingPhaseName,
)


class TrainingPhase(Protocol):
    """Render source samples into one phase-specific supervision format."""

    @property
    def name(self) -> TrainingPhaseName:
        """Return the stable phase identifier."""

    @property
    def taxonomy(self) -> Taxonomy:
        """Return the closed taxonomy rendered by the phase."""

    def format_example(self, sample: LabeledImageSample) -> FormattedExample:
        """Render a typed source sample for a multimodal chat collator."""
