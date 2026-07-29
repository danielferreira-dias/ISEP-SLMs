"""Deterministic, model-independent benchmark subset selection."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from src.benchmark.results import canonical_hash


def select_units(
    frame: pd.DataFrame,
    *,
    unit_column: str,
    task_column: str,
    limit: int | None,
    seed: int,
    benchmark_release_hash: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select stable units and preserve every task belonging to those units.

    A confusion-set unit is a pair and therefore expands to two task rows.
    For other benchmarks the unit and task identifiers can be the same column.
    """

    missing = {unit_column, task_column} - set(frame.columns)
    if missing:
        raise ValueError(
            "Selection frame is missing columns: "
            + ", ".join(sorted(missing))
        )
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be a positive integer")
    if frame[task_column].astype(str).duplicated().any():
        raise ValueError(f"{task_column} values must be unique")

    units = sorted(set(frame[unit_column].astype(str)))
    scores = {
        unit: _unit_score(
            unit=unit,
            seed=seed,
            benchmark_release_hash=benchmark_release_hash,
        )
        for unit in units
    }
    ordered_units = sorted(units, key=lambda unit: (scores[unit], unit))
    if limit is not None:
        ordered_units = ordered_units[:limit]
    selected_set = set(ordered_units)
    selected = frame[
        frame[unit_column].astype(str).isin(selected_set)
    ].copy()
    selected["_selection_unit_order"] = selected[unit_column].astype(
        str
    ).map({unit: index for index, unit in enumerate(ordered_units)})
    selected = selected.sort_values(
        ["_selection_unit_order", task_column],
        kind="mergesort",
    ).drop(columns=["_selection_unit_order"])
    selected = selected.reset_index(drop=True)

    task_ids = selected[task_column].astype(str).tolist()
    selection = {
        "algorithm": "sha256_lowest_v1",
        "seed": int(seed),
        "benchmark_release_hash": benchmark_release_hash,
        "unit_column": unit_column,
        "task_column": task_column,
        "requested_limit": limit,
        "selected_unit_count": len(ordered_units),
        "selected_task_count": len(task_ids),
        "unit_ids": ordered_units,
        "task_ids": task_ids,
    }
    selection["selection_hash"] = canonical_hash(selection)
    return selected, selection


def task_seed(run_seed: int, task_id: str) -> int:
    """Derive a stable non-negative 31-bit seed for one task."""

    digest = hashlib.sha256(
        f"{int(run_seed)}\0{task_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _unit_score(
    *,
    unit: str,
    seed: int,
    benchmark_release_hash: str,
) -> str:
    return hashlib.sha256(
        (
            f"{benchmark_release_hash}\0{int(seed)}\0{unit}"
        ).encode("utf-8")
    ).hexdigest()
