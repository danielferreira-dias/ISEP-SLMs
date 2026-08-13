"""Explicit registry of implemented training phases."""

from __future__ import annotations

from collections.abc import Callable

from src.train.domain import Taxonomy, TrainingPhaseName
from src.train.phases.base import TrainingPhase
from src.train.phases.label_only import LabelOnlyPhase

PhaseFactory = Callable[[Taxonomy], TrainingPhase]


_PHASE_FACTORIES: dict[TrainingPhaseName, PhaseFactory] = {
    TrainingPhaseName.E1_LABEL: lambda taxonomy: LabelOnlyPhase(taxonomy=taxonomy),
}


def get_phase(name: TrainingPhaseName, taxonomy: Taxonomy) -> TrainingPhase:
    """Construct one implemented phase from the explicit registry."""

    try:
        factory = _PHASE_FACTORIES[name]
    except KeyError as exc:
        raise ValueError(f"Training phase {name!s} is not implemented") from exc
    return factory(taxonomy)


def registered_phases() -> tuple[TrainingPhaseName, ...]:
    """Return implemented phase names in stable order."""

    return tuple(_PHASE_FACTORIES)
