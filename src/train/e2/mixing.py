"""Deterministic, no-replacement task interleaving for E2."""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload


class DeterministicTaskInterleave(Sequence[dict[str, object]]):
    """Spread two or three task datasets without replacement."""

    def __init__(
        self,
        diagnosis: Sequence[dict[str, object]],
        morphology: Sequence[dict[str, object]],
        caption: Sequence[dict[str, object]] | None = None,
    ) -> None:
        """Create an exact all-row schedule with stable proportional spacing."""

        if not diagnosis or not morphology:
            raise ValueError("Both E2 task datasets must be non-empty")
        if caption is not None and not caption:
            raise ValueError("The optional E2 caption dataset must be non-empty")
        self._datasets = (
            (diagnosis, morphology)
            if caption is None
            else (diagnosis, morphology, caption)
        )
        counts = tuple(len(dataset) for dataset in self._datasets)
        self._schedule = (
            _proportional_schedule(counts[0], counts[1])
            if len(counts) == 2
            else _proportional_schedule_many(counts)
        )

    def __len__(self) -> int:
        """Return the exact sum of both source task counts."""

        return len(self._schedule)

    @overload
    def __getitem__(self, index: int) -> dict[str, object]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, object]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> dict[str, object] | list[dict[str, object]]:
        """Return one scheduled row or a materialized slice."""

        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        normalized = index if index >= 0 else len(self) + index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("E2 interleave index out of range")
        task_index, source_index = self._schedule[normalized]
        return self._datasets[task_index][source_index]

    @property
    def task_counts(self) -> tuple[int, ...]:
        """Return task cardinalities in diagnosis/morphology/caption order."""

        return tuple(len(dataset) for dataset in self._datasets)


def _proportional_schedule(
    diagnosis_count: int,
    morphology_count: int,
) -> tuple[tuple[int, int], ...]:
    total = diagnosis_count + morphology_count
    used = [0, 0]
    schedule: list[tuple[int, int]] = []
    for position in range(total):
        desired_diagnosis = round((position + 1) * diagnosis_count / total)
        task = 0 if used[0] < desired_diagnosis else 1
        schedule.append((task, used[task]))
        used[task] += 1
    if used != [diagnosis_count, morphology_count]:
        raise RuntimeError("E2 task interleave did not consume every row exactly once")
    return tuple(schedule)


def _proportional_schedule_many(
    counts: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    """Create a stable weighted-fair schedule for three or more tasks."""

    if len(counts) < 3 or any(count <= 0 for count in counts):
        raise ValueError("Multi-task scheduling requires at least three counts")
    total = sum(counts)
    used = [0 for _ in counts]
    schedule: list[tuple[int, int]] = []
    for position in range(total):
        candidates = [
            index for index, count in enumerate(counts) if used[index] < count
        ]
        task = max(
            candidates,
            key=lambda index: (
                ((position + 1) * counts[index] / total) - used[index],
                -index,
            ),
        )
        schedule.append((task, used[task]))
        used[task] += 1
    if tuple(used) != counts:
        raise RuntimeError("E2 task interleave did not consume every row exactly once")
    return tuple(schedule)
