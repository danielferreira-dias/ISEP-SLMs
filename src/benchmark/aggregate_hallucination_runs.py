"""Aggregate nested parent and expansion hallucination benchmark runs.

The ISEPDermaBench v1 hallucination cohorts are strict subsets of the v2
cohorts.  This utility joins a completed parent run with a completed run over
only the new task IDs, validates that the two selections are disjoint and
complete, recomputes deterministic metrics over the full cohort, and creates
a normal self-contained HTML report without repeating provider inference.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.benchmark.executor import _record_to_prediction
from src.benchmark.hallucination import (
    compute_dermatology_counterfactual_metrics,
    compute_general_visual_hallucination_metrics,
)
from src.benchmark.report import generate_run_report
from src.benchmark.results import read_jsonl


EXPECTED_COUNTS = {
    "general_visual_hallucination_audit": (100, 200, 300),
    "dermatology_counterfactual_hallucination": (50, 150, 200),
}


def aggregate_runs(
    *,
    root: Path,
    parent_run: Path,
    expansion_run: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Validate and aggregate two completed nested-cohort runs."""

    parent_run = parent_run.resolve()
    expansion_run = expansion_run.resolve()
    output_directory = output_directory.resolve()
    parent_manifest = _yaml(parent_run / "run_manifest.yaml")
    expansion_manifest = _yaml(expansion_run / "run_manifest.yaml")
    if parent_manifest.get("status") != "completed":
        raise ValueError("Parent run is not completed")
    if expansion_manifest.get("status") != "completed":
        raise ValueError("Expansion run is not completed")
    parent_model = str(_mapping(parent_manifest, "model")["id"])
    expansion_model = str(_mapping(expansion_manifest, "model")["id"])
    if parent_model != expansion_model:
        raise ValueError("Parent and expansion model IDs differ")
    benchmark_id = str(_mapping(expansion_manifest, "benchmark")["id"])
    if str(_mapping(parent_manifest, "benchmark")["id"]) != benchmark_id:
        raise ValueError("Parent and expansion benchmark IDs differ")
    if benchmark_id not in EXPECTED_COUNTS:
        raise ValueError(f"Unsupported aggregate benchmark: {benchmark_id}")

    parent_records = read_jsonl(parent_run / "predictions.jsonl")
    expansion_records = read_jsonl(expansion_run / "predictions.jsonl")
    expected_parent, expected_expansion, expected_total = EXPECTED_COUNTS[
        benchmark_id
    ]
    if len(parent_records) != expected_parent:
        raise ValueError(
            f"Expected {expected_parent} parent records, got {len(parent_records)}"
        )
    if len(expansion_records) != expected_expansion:
        raise ValueError(
            "Expected "
            f"{expected_expansion} expansion records, got {len(expansion_records)}"
        )
    parent_ids = {str(row["task_id"]) for row in parent_records}
    expansion_ids = {str(row["task_id"]) for row in expansion_records}
    overlap = parent_ids & expansion_ids
    if overlap:
        raise ValueError(f"Parent/expansion overlap: {len(overlap)} task IDs")
    records = parent_records + expansion_records
    combined_ids = {str(row["task_id"]) for row in records}
    if len(combined_ids) != expected_total:
        raise ValueError("Combined task IDs are not complete and unique")
    release_ids = _benchmark_task_ids(root, benchmark_id)
    if combined_ids != release_ids:
        raise ValueError(
            "Combined task IDs do not equal the complete expanded release"
        )

    predictions = [_record_to_prediction(row) for row in records]
    if benchmark_id == "general_visual_hallucination_audit":
        metrics = compute_general_visual_hallucination_metrics(predictions)
    else:
        metrics = compute_dermatology_counterfactual_metrics(predictions)
    counts = Counter(str(row.get("status", "")) for row in records)
    counts["total"] = len(records)

    output_directory.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output_directory / "predictions.jsonl", records)
    prompts = _unique_by_task_id(
        read_jsonl(parent_run / "rendered_prompts.jsonl")
        + read_jsonl(expansion_run / "rendered_prompts.jsonl")
    )
    _write_jsonl(output_directory / "rendered_prompts.jsonl", prompts)
    shutil.copy2(
        expansion_run / "config_snapshot.yaml",
        output_directory / "config_snapshot.yaml",
    )
    _json(output_directory / "metrics.json", metrics)
    _json(
        output_directory / "selection.json",
        {
            "algorithm": "nested_parent_plus_expansion_v1",
            "benchmark_id": benchmark_id,
            "model_id": parent_model,
            "parent_run": str(parent_run),
            "expansion_run": str(expansion_run),
            "selected_task_count": len(records),
            "task_ids": [str(row["task_id"]) for row in records],
        },
    )
    _json(
        output_directory / "environment.json",
        {
            "aggregation_only": True,
            "provider_inference_repeated": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    aggregate_manifest = dict(expansion_manifest)
    aggregate_manifest["status"] = "completed"
    aggregate_manifest["aggregation"] = {
        "method": "nested_parent_plus_expansion_v1",
        "parent_run": str(parent_run),
        "expansion_run": str(expansion_run),
        "provider_inference_repeated": False,
    }
    aggregate_manifest["evaluation"] = dict(
        _mapping(expansion_manifest, "evaluation")
    )
    aggregate_manifest["evaluation"]["selected_units"] = expected_total
    aggregate_manifest["evaluation"]["selected_tasks"] = expected_total
    aggregate_manifest["counts"] = dict(sorted(counts.items()))
    aggregate_manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    (output_directory / "run_manifest.yaml").write_text(
        yaml.safe_dump(aggregate_manifest, sort_keys=False),
        encoding="utf-8",
    )
    image_loader = _image_loader(root, benchmark_id)
    report = generate_run_report(
        output_directory,
        image_loader=image_loader,
    )
    return {
        "benchmark_id": benchmark_id,
        "model_id": parent_model,
        "sample_count": len(records),
        "metrics_path": str(output_directory / "metrics.json"),
        "report_path": str(report),
    }


def _image_loader(root: Path, benchmark_id: str):
    task_dir = (
        root.resolve()
        / "data/benchmarks/ISEPDermaBench/tasks"
        / benchmark_id
    )
    paths = sorted(task_dir.glob("validation-*.parquet"))
    table = pa.concat_tables([pq.read_table(path) for path in paths])
    rows = table.select(["task_id", "image"]).to_pylist()
    images = {
        str(row["task_id"]): bytes(row["image"]["bytes"])
        for row in rows
    }

    def load(image_uri: str) -> bytes:
        task_id = image_uri.removeprefix("embedded://").rsplit("/", 1)[-1]
        return images[task_id]

    return load


def _benchmark_task_ids(root: Path, benchmark_id: str) -> set[str]:
    task_dir = (
        root.resolve()
        / "data/benchmarks/ISEPDermaBench/tasks"
        / benchmark_id
    )
    paths = sorted(task_dir.glob("validation-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No task Parquets in {task_dir}")
    table = pa.concat_tables(
        [pq.read_table(path, columns=["task_id"]) for path in paths]
    )
    return {str(value) for value in table.column("task_id").to_pylist()}


def _unique_by_task_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        result[str(row["task_id"])] = row
    return list(result.values())


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise ValueError(f"Missing manifest mapping: {key}")
    return nested


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return value


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--expansion-run", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate_runs(
        root=args.project_root,
        parent_run=args.parent_run,
        expansion_run=args.expansion_run,
        output_directory=args.output_directory,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
