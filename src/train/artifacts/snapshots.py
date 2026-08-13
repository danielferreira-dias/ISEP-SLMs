"""Canonical persistence boundary for independently comparable runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from src.train.evaluation.models import ComparableRun

from .predictions import read_prediction_parquet, write_prediction_files
from .serialization import (
    classification_metrics_from_json,
    classification_metrics_to_json,
    run_contract_from_json,
    run_contract_to_json,
)
from .store import ArtifactStore
from .types import JsonValue, RunSnapshotArtifacts


def write_comparable_run_snapshot(
    store: ArtifactStore,
    run: ComparableRun,
) -> RunSnapshotArtifacts:
    """Persist all typed inputs required for later paired comparison."""

    metadata: dict[str, JsonValue] = {
        "experiment_id": run.experiment_id,
        "run_id": run.run_id,
        "seed": run.seed,
        "duration_seconds": run.duration_seconds,
        "gpu_hours": run.gpu_hours,
        "peak_vram_gib": run.peak_vram_gib,
        "trainable_parameters": run.trainable_parameters,
    }
    metadata_path = store.write_json("manifests", "comparison_run.json", metadata)
    contract_path = store.write_json(
        "manifests",
        "run_contract.json",
        run_contract_to_json(run.contract),
    )
    metrics_path = store.write_json(
        "metrics",
        "classification.json",
        classification_metrics_to_json(run.metrics),
    )
    predictions = write_prediction_files(
        run.predictions, store.layout.predictions, stem="final"
    )
    return RunSnapshotArtifacts(
        metadata_path=metadata_path,
        contract_path=contract_path,
        metrics_path=metrics_path,
        predictions=predictions,
    )


def _read_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"Required run artefact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping(value: object, context: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _required_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _required_integer(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    return value


def _optional_float(value: object, context: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{context} must be numeric or null")
    return float(value)


def _optional_integer(value: object, context: str) -> int | None:
    if value is None:
        return None
    return _required_integer(value, context)


def load_comparable_run(run_directory: Path) -> ComparableRun:
    """Restore and validate a comparable run from its canonical files."""

    store = ArtifactStore.at(run_directory)
    metadata = _mapping(
        _read_json(store.path("manifests", "comparison_run.json")),
        "comparison_run",
    )
    contract = run_contract_from_json(
        _read_json(store.path("manifests", "run_contract.json"))
    )
    metrics = classification_metrics_from_json(
        _read_json(store.path("metrics", "classification.json"))
    )
    predictions = read_prediction_parquet(store.path("predictions", "final.parquet"))
    return ComparableRun(
        experiment_id=_required_string(metadata.get("experiment_id"), "experiment_id"),
        run_id=_required_string(metadata.get("run_id"), "run_id"),
        seed=_required_integer(metadata.get("seed"), "seed"),
        contract=contract,
        predictions=predictions,
        metrics=metrics,
        duration_seconds=_optional_float(
            metadata.get("duration_seconds"), "duration_seconds"
        ),
        gpu_hours=_optional_float(metadata.get("gpu_hours"), "gpu_hours"),
        peak_vram_gib=_optional_float(metadata.get("peak_vram_gib"), "peak_vram_gib"),
        trainable_parameters=_optional_integer(
            metadata.get("trainable_parameters"), "trainable_parameters"
        ),
    )
