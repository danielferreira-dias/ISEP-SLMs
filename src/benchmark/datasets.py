"""Benchmark manifest loading and conversion to executable samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.benchmark.results import file_sha256
from src.benchmark.runner import BenchmarkSample
from src.benchmark.selection import select_units


@dataclass(frozen=True, slots=True)
class LoadedBenchmarkDataset:
    """Selected manifest rows and their stable execution samples."""

    manifest_path: Path
    manifest_sha256: str
    release_sha256: str
    evaluation_set: str
    frame: pd.DataFrame
    samples: tuple[BenchmarkSample, ...]
    selection: dict[str, Any]


def load_benchmark_dataset(
    *,
    root: Path,
    config: dict[str, Any],
    evaluation_set: str | None,
    limit: int | None,
    seed: int,
) -> LoadedBenchmarkDataset:
    """Load, validate, and deterministically subset one benchmark manifest."""

    benchmark = _mapping(config, "benchmark")
    dataset = _mapping(config, "dataset")
    task = str(benchmark.get("task", ""))
    selected_evaluation_set, relative_manifest = _resolve_manifest(
        dataset=dataset,
        evaluation_set=evaluation_set,
    )
    manifest_path = _inside_root(root, relative_manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Benchmark manifest does not exist: {manifest_path}"
        )
    frame = pq.read_table(manifest_path).to_pandas()
    unit_column, task_column = _selection_columns(task, dataset)
    _validate_columns(
        frame=frame,
        dataset=dataset,
        task=task,
        unit_column=unit_column,
        task_column=task_column,
    )
    release_path_value = dataset.get("release_manifest")
    release_path = (
        _inside_root(root, str(release_path_value))
        if isinstance(release_path_value, str)
        else None
    )
    release_sha256 = (
        file_sha256(release_path)
        if release_path is not None and release_path.is_file()
        else file_sha256(manifest_path)
    )
    selected, selection = select_units(
        frame,
        unit_column=unit_column,
        task_column=task_column,
        limit=limit,
        seed=seed,
        benchmark_release_hash=release_sha256,
    )
    samples = tuple(
        _row_to_sample(row, task=task, dataset=dataset)
        for row in selected.to_dict(orient="records")
    )
    return LoadedBenchmarkDataset(
        manifest_path=manifest_path,
        manifest_sha256=file_sha256(manifest_path),
        release_sha256=release_sha256,
        evaluation_set=selected_evaluation_set,
        frame=selected,
        samples=samples,
        selection=selection,
    )


def _resolve_manifest(
    *,
    dataset: dict[str, Any],
    evaluation_set: str | None,
) -> tuple[str, str]:
    evaluation_sets = dataset.get("evaluation_sets")
    default = dataset.get("default_evaluation_set")
    if isinstance(evaluation_sets, dict) and evaluation_sets:
        selected = evaluation_set or (
            str(default) if default is not None else next(iter(evaluation_sets))
        )
        if selected not in evaluation_sets:
            raise ValueError(
                f"Unknown evaluation set {selected!r}; expected one of "
                + ", ".join(sorted(str(key) for key in evaluation_sets))
            )
        entry = evaluation_sets[selected]
        if isinstance(entry, str):
            return selected, entry
        if isinstance(entry, dict) and isinstance(entry.get("manifest"), str):
            return selected, str(entry["manifest"])
        raise ValueError(
            f"Evaluation set {selected!r} must define a manifest"
        )

    if evaluation_set is not None:
        raise ValueError(
            "This benchmark does not declare named evaluation sets"
        )
    if isinstance(dataset.get("task_manifest"), str):
        return "default", str(dataset["task_manifest"])
    if isinstance(dataset.get("manifest"), str):
        return "default", str(dataset["manifest"])
    raise ValueError("Benchmark dataset does not define a manifest")


def _selection_columns(
    task: str,
    dataset: dict[str, Any],
) -> tuple[str, str]:
    if task == "visual_disease_contrast_ranking":
        return (
            str(dataset.get("pair_id_column", "pair_id")),
            str(dataset.get("task_id_column", "task_id")),
        )
    sample_id = str(dataset.get("sample_id_column", "sample_id"))
    return sample_id, sample_id


def _validate_columns(
    *,
    frame: pd.DataFrame,
    dataset: dict[str, Any],
    task: str,
    unit_column: str,
    task_column: str,
) -> None:
    required = {
        unit_column,
        task_column,
        str(dataset.get("image_column", "image_uri")),
    }
    if task == "visual_disease_contrast_ranking":
        required.update(
            {
                str(dataset.get("label_column", "disease_id")),
                str(
                    dataset.get(
                        "candidate_ids_column",
                        "candidate_disease_ids",
                    )
                ),
            }
        )
    elif task == "visual_disease_ranking":
        required.add(str(dataset.get("label_column", "disease_id")))
    elif task == "evidence_grounded_visual_diagnosis":
        required.update(
            {
                "morphology_concept_ids",
                "score_morphology",
                "score_description",
                "score_diagnosis",
            }
        )
    else:
        raise ValueError(f"Unsupported benchmark task: {task!r}")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Benchmark manifest is missing columns: "
            + ", ".join(missing)
        )


def _row_to_sample(
    row: dict[str, Any],
    *,
    task: str,
    dataset: dict[str, Any],
) -> BenchmarkSample:
    sample_column = str(dataset.get("sample_id_column", "sample_id"))
    image_column = str(dataset.get("image_column", "image_uri"))
    label_column = str(
        dataset.get(
            "label_column",
            dataset.get("disease_label_column", "disease_id"),
        )
    )
    sample_id = str(row[sample_column])
    task_id = (
        str(row[str(dataset.get("task_id_column", "task_id"))])
        if task == "visual_disease_contrast_ranking"
        else sample_id
    )
    candidate_ids: tuple[str, ...] | None = None
    if task == "visual_disease_contrast_ranking":
        raw_candidates = row[
            str(
                dataset.get(
                    "candidate_ids_column",
                    "candidate_disease_ids",
                )
            )
        ]
        candidate_ids = tuple(str(value) for value in raw_candidates)
    disease_value = row.get(label_column)
    disease_id = "" if disease_value is None else str(disease_value)
    return BenchmarkSample(
        sample_id=sample_id,
        image_uri=str(row[image_column]),
        disease_id=disease_id,
        task_id=task_id,
        candidate_disease_ids=candidate_ids,
        metadata={
            str(key): _python_scalar(value)
            for key, value in row.items()
        },
    )


def _inside_root(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Path escapes repository root: {relative!r}")
    return path


def _mapping(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected {key!r} to be a mapping")
    return value


def _python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple, set)):
        return [_python_scalar(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
        except (TypeError, ValueError):
            pass
        else:
            return _python_scalar(converted)
    return value
