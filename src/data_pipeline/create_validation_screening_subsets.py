"""Build fixed, nested Validation cohorts for teacher screening.

The cohorts are class-balanced development views of the frozen
ISEPDermaBench Validation splits. They contain task IDs only; images and
isolated scoring references remain in the benchmark release. The 100-unit
cohort is always a prefix of the largest available expansion cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


SEED = 42
TASK_SPECS: dict[str, dict[str, Any]] = {
    "visual_top_k": {
        "unit_column": "task_id",
        "stratum_columns": ("reference_disease_id",),
        "cohort_sizes": (100, 200),
    },
    "visual_confusion_sets": {
        "unit_column": "pair_id",
        "stratum_columns": (
            "confusion_set_id",
            "reference_disease_id",
        ),
        "cohort_sizes": (100, 200),
    },
    "evidence_grounded_diagnosis": {
        "unit_column": "task_id",
        "stratum_columns": ("reference_disease_id",),
        "cohort_sizes": (100, 137),
    },
    "open_ended_diagnosis": {
        "unit_column": "task_id",
        "stratum_columns": ("reference_disease_id",),
        "cohort_sizes": (100,),
    },
}


def build_validation_screening_subsets(
    *,
    release_root: Path,
    output_root: Path,
    seed: int = SEED,
) -> dict[str, Any]:
    """Write task-ID cohorts and return their reproducibility manifest."""

    release_path = release_root / "release.json"
    release_sha256 = _file_sha256(release_path)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "id": "validation_teacher_screening_v1",
        "benchmark_release": "ISEPDermaBench/1.5.0",
        "benchmark_release_sha256": release_sha256,
        "evaluation_set": "validation",
        "seed": int(seed),
        "selection_algorithm": (
            "class_balanced_nested_round_robin_v1"
        ),
        "purpose": (
            "Development-only paired teacher screening; not final reporting"
        ),
        "tasks": {},
    }

    for task_name, spec in TASK_SPECS.items():
        task_frame = _read_shards(
            release_root / "tasks" / task_name,
            split="validation",
        )
        reference_frame = _read_shards(
            release_root / "references" / task_name,
            split="validation",
        )
        unit_column = str(spec["unit_column"])
        units = _build_unit_frame(
            task_frame=task_frame,
            reference_frame=reference_frame,
            unit_column=unit_column,
            stratum_columns=tuple(spec["stratum_columns"]),
        )
        ordered_units = _balanced_nested_order(
            units,
            unit_column=unit_column,
            seed=seed,
            release_sha256=release_sha256,
        )
        task_entry: dict[str, Any] = {
            "selection_unit": (
                "image_pair" if unit_column == "pair_id" else "case"
            ),
            "available_units": len(ordered_units),
            "available_tasks": int(len(task_frame)),
            "cohorts": [],
        }
        for requested_size in spec["cohort_sizes"]:
            size = min(int(requested_size), len(ordered_units))
            selected_units = ordered_units[:size]
            selected_task_ids = _expand_task_ids(
                task_frame,
                unit_column=unit_column,
                selected_units=selected_units,
            )
            suffix = "pairs" if unit_column == "pair_id" else "cases"
            filename = f"{task_name}_{size}_{suffix}.task_ids.txt"
            output_path = output_root / filename
            _write_task_ids(
                output_path,
                task_name=task_name,
                unit_column=unit_column,
                unit_count=size,
                task_ids=selected_task_ids,
                seed=seed,
                release_sha256=release_sha256,
            )
            task_entry["cohorts"].append(
                {
                    "unit_count": size,
                    "task_count": len(selected_task_ids),
                    "task_ids_file": filename,
                    "sha256": _file_sha256(output_path),
                    "is_initial_screen": size == 100,
                }
            )
        manifest["tasks"][task_name] = task_entry

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _read_shards(directory: Path, *, split: str) -> pd.DataFrame:
    paths = sorted(directory.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No {split} Parquet shards in {directory}")
    return pd.concat(
        [pq.read_table(path).to_pandas() for path in paths],
        ignore_index=True,
    )


def _build_unit_frame(
    *,
    task_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    unit_column: str,
    stratum_columns: tuple[str, ...],
) -> pd.DataFrame:
    reference_columns = [
        "task_id",
        "reference_disease_id",
        "source",
        "confusion_set_id",
    ]
    merged = task_frame.merge(
        reference_frame[
            [column for column in reference_columns if column in reference_frame]
        ],
        on="task_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_reference"),
    )
    if len(merged) != len(task_frame):
        raise ValueError("Task/reference join is incomplete")
    if unit_column == "pair_id":
        pair_sizes = merged.groupby("pair_id", dropna=False).size()
        if not pair_sizes.eq(2).all():
            raise ValueError("Every confusion-set pair must contain two tasks")
        condition_sets = merged.groupby("pair_id")["condition"].agg(set)
        expected = {"low_confusability", "high_confusability"}
        if not condition_sets.map(lambda value: value == expected).all():
            raise ValueError("Every pair must contain low and high conditions")
    columns = [unit_column, "source", *stratum_columns]
    units = merged[columns].drop_duplicates(unit_column).copy()
    for column in columns:
        units[column] = units[column].fillna("unknown").astype(str)
    units["_stratum"] = units[list(stratum_columns)].agg("|".join, axis=1)
    return units


def _balanced_nested_order(
    units: pd.DataFrame,
    *,
    unit_column: str,
    seed: int,
    release_sha256: str,
) -> list[str]:
    """Round-robin classes and sources while preserving a stable prefix."""

    per_stratum: dict[str, list[str]] = {}
    for stratum, stratum_frame in units.groupby("_stratum", sort=True):
        source_queues: dict[str, list[str]] = {}
        for source, source_frame in stratum_frame.groupby("source", sort=True):
            source_queues[str(source)] = sorted(
                source_frame[unit_column].astype(str).tolist(),
                key=lambda unit: _stable_score(
                    release_sha256, seed, stratum, source, unit
                ),
            )
        source_order = sorted(
            source_queues,
            key=lambda source: _stable_score(
                release_sha256, seed, stratum, source
            ),
        )
        ordered: list[str] = []
        while any(source_queues.values()):
            for source in source_order:
                if source_queues[source]:
                    ordered.append(source_queues[source].pop(0))
        per_stratum[str(stratum)] = ordered

    stratum_order = sorted(
        per_stratum,
        key=lambda stratum: _stable_score(release_sha256, seed, stratum),
    )
    ordered_units: list[str] = []
    while any(per_stratum.values()):
        for stratum in stratum_order:
            if per_stratum[stratum]:
                ordered_units.append(per_stratum[stratum].pop(0))
    if len(ordered_units) != len(set(ordered_units)):
        raise ValueError("Selection algorithm produced duplicate units")
    return ordered_units


def _expand_task_ids(
    task_frame: pd.DataFrame,
    *,
    unit_column: str,
    selected_units: list[str],
) -> list[str]:
    order = {unit: index for index, unit in enumerate(selected_units)}
    selected = task_frame[
        task_frame[unit_column].astype(str).isin(order)
    ].copy()
    selected["_unit_order"] = selected[unit_column].astype(str).map(order)
    condition_order = {
        "low_confusability": 0,
        "high_confusability": 1,
    }
    selected["_condition_order"] = (
        selected["condition"].map(condition_order).fillna(0)
        if "condition" in selected
        else 0
    )
    selected = selected.sort_values(
        ["_unit_order", "_condition_order", "task_id"],
        kind="stable",
    )
    return selected["task_id"].astype(str).tolist()


def _write_task_ids(
    path: Path,
    *,
    task_name: str,
    unit_column: str,
    unit_count: int,
    task_ids: list[str],
    seed: int,
    release_sha256: str,
) -> None:
    header = [
        "# ISEPDermaBench fixed Validation teacher-screening cohort",
        f"# task: {task_name}",
        f"# selection_unit: {unit_column}",
        f"# selected_units: {unit_count}",
        f"# selected_tasks: {len(task_ids)}",
        f"# seed: {seed}",
        f"# benchmark_release_sha256: {release_sha256}",
    ]
    path.write_text("\n".join([*header, *task_ids, ""]), encoding="utf-8")


def _stable_score(*parts: object) -> str:
    return hashlib.sha256(
        "\0".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("data/benchmarks/ISEPDermaBench"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/benchmarks/ISEPDermaBench/metadata/"
            "validation_screening_v1"
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    manifest = build_validation_screening_subsets(
        release_root=args.release_root.resolve(),
        output_root=args.output_root.resolve(),
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
