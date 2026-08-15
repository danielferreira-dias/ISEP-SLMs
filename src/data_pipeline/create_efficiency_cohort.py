"""Build the fixed ISEPDermaBench cohort for model-efficiency comparisons.

The cohort contains 400 measured requests: 100 Top-K cases, 50 complete
confusion pairs (100 requests), 100 evidence cases, and 100 open-ended cases.
Ten additional Top-K cases are reserved exclusively for the unmeasured 1→10
multimodal gate required before each model enters the measured cohort.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data_pipeline.create_validation_screening_subsets import (
    _balanced_nested_order,
    _build_unit_frame,
    _expand_task_ids,
    _file_sha256,
    _read_shards,
    _write_task_ids,
)

SEED = 42


@dataclass(frozen=True, slots=True)
class CohortSpec:
    """Deterministic selection rule for one measured benchmark task."""

    benchmark_id: str
    unit_column: str
    stratum_columns: tuple[str, ...]
    unit_count: int


COHORT_SPECS: dict[str, CohortSpec] = {
    "visual_top_k": CohortSpec(
        "visual_top_k_closed_set",
        "task_id",
        ("reference_disease_id",),
        100,
    ),
    "visual_confusion_sets": CohortSpec(
        "visual_disease_confusion_sets",
        "pair_id",
        ("confusion_set_id", "reference_disease_id"),
        50,
    ),
    "evidence_grounded_diagnosis": CohortSpec(
        "evidence_grounded_diagnosis",
        "task_id",
        ("reference_disease_id",),
        100,
    ),
    "open_ended_diagnosis": CohortSpec(
        "open_ended_diagnosis",
        "task_id",
        ("reference_disease_id",),
        100,
    ),
}


def build_efficiency_cohort(
    *, release_root: Path, output_root: Path, seed: int = SEED
) -> dict[str, Any]:
    """Create task-ID files and a cryptographic cohort manifest.

    Args:
        release_root: Frozen local ISEPDermaBench release directory.
        output_root: Destination for ID-only cohort artifacts.
        seed: Deterministic class/source balancing seed.

    Returns:
        The persisted manifest as a JSON-compatible mapping.
    """

    release_path = release_root / "release.json"
    release = _read_json_object(release_path)
    release_sha256 = _file_sha256(release_path)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks: dict[str, object] = {}
    total_requests = 0
    smoke_task_ids: list[str] | None = None

    for task_name, spec in COHORT_SPECS.items():
        unit_column = spec.unit_column
        unit_count = spec.unit_count
        stratum_columns = spec.stratum_columns
        task_frame = _read_shards(
            release_root / "tasks" / task_name,
            split="internal_benchmark",
        )
        reference_frame = _read_shards(
            release_root / "references" / task_name,
            split="internal_benchmark",
        )
        units = _build_unit_frame(
            task_frame=task_frame,
            reference_frame=reference_frame,
            unit_column=unit_column,
            stratum_columns=stratum_columns,
        )
        ordered_units = _balanced_nested_order(
            units,
            unit_column=unit_column,
            seed=seed,
            release_sha256=release_sha256,
        )
        if len(ordered_units) < unit_count:
            raise ValueError(f"{task_name} has only {len(ordered_units)} units")
        selected_units = ordered_units[:unit_count]
        task_ids = _expand_task_ids(
            task_frame,
            unit_column=unit_column,
            selected_units=selected_units,
        )
        suffix = "pairs" if unit_column == "pair_id" else "cases"
        path = output_root / f"{task_name}_{unit_count}_{suffix}.task_ids.txt"
        _write_task_ids(
            path,
            task_name=task_name,
            unit_column=unit_column,
            unit_count=unit_count,
            task_ids=task_ids,
            seed=seed,
            release_sha256=release_sha256,
        )
        total_requests += len(task_ids)
        tasks[spec.benchmark_id] = {
            "source_task": task_name,
            "selection_unit": "image_pair" if unit_column == "pair_id" else "case",
            "unit_count": unit_count,
            "request_count": len(task_ids),
            "task_ids_file": path.name,
            "sha256": _file_sha256(path),
        }
        if task_name == "visual_top_k":
            if len(ordered_units) < unit_count + 10:
                raise ValueError("visual_top_k has fewer than 10 independent gates")
            smoke_task_ids = _expand_task_ids(
                task_frame,
                unit_column=unit_column,
                selected_units=ordered_units[unit_count : unit_count + 10],
            )
            if len(smoke_task_ids) != 10:
                raise ValueError("smoke selection must contain exactly 10 requests")

    if total_requests != 400 or smoke_task_ids is None:
        raise ValueError(
            f"efficiency cohort must contain 400 requests, found {total_requests}"
        )
    warmup_path = output_root / "warmup_visual_top_k_1_case.task_ids.txt"
    _write_task_ids(
        warmup_path,
        task_name="visual_top_k",
        unit_column="task_id",
        unit_count=1,
        task_ids=smoke_task_ids[:1],
        seed=seed,
        release_sha256=release_sha256,
    )
    smoke_path = output_root / "smoke_visual_top_k_10_cases.task_ids.txt"
    _write_task_ids(
        smoke_path,
        task_name="visual_top_k",
        unit_column="task_id",
        unit_count=10,
        task_ids=smoke_task_ids,
        seed=seed,
        release_sha256=release_sha256,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "id": "isep_efficiency_cohort_v1",
        "purpose": (
            "Same-hardware quality, latency, throughput, memory, energy, "
            "and cost comparison"
        ),
        "benchmark_release": _release_label(release),
        "benchmark_release_sha256": release_sha256,
        "evaluation_set": "internal_benchmark",
        "seed": seed,
        "selection_algorithm": "class_balanced_nested_round_robin_v1",
        "measured_request_count": total_requests,
        "warmup": {
            "measured": False,
            "request_count": 1,
            "task_ids_file": warmup_path.name,
            "sha256": _file_sha256(warmup_path),
        },
        "smoke_gate": {
            "measured": False,
            "request_count": 10,
            "task_ids_file": smoke_path.name,
            "sha256": _file_sha256(smoke_path),
        },
        "tasks": tasks,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _release_label(release: dict[str, Any]) -> str:
    metadata = release.get("release")
    if not isinstance(metadata, dict):
        raise ValueError("release.json is missing the release object")
    identifier = metadata.get("id")
    version = metadata.get("version")
    if not isinstance(identifier, str) or not isinstance(version, str):
        raise ValueError("release id and version must be strings")
    return f"{identifier}/{version}"


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    """Build the repository's canonical efficiency cohort."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("data/benchmarks/ISEPDermaBench"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/benchmarks/ISEPDermaBench/metadata/efficiency_cohort_v1"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    manifest = build_efficiency_cohort(
        release_root=args.release_root.resolve(),
        output_root=args.output_root.resolve(),
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
