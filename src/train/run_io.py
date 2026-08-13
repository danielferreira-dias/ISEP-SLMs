"""Run-directory persistence and frozen-release reopening helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from src.train.artifacts import ArtifactStore, write_prediction_files
from src.train.artifacts.serialization import classification_metrics_to_json
from src.train.config import TrainingConfig
from src.train.data import inspect_data_release, validate_source_shards
from src.train.data.source import source_release_sha256
from src.train.domain import JsonValue, PreparedRelease
from src.train.environment import collect_environment
from src.train.evaluate import EvaluationResult
from src.train.scientific import config_hash, resolved_config_document


@dataclass(frozen=True, slots=True)
class ResourceSummary:
    """Measured training efficiency values used in thesis comparisons."""

    duration_seconds: float | None
    gpu_hours: float | None
    peak_vram_gib: float | None
    train_samples_per_second: float | None
    maximum_power_watts: float | None
    maximum_temperature_celsius: float | None


def open_frozen_release(config: TrainingConfig) -> PreparedRelease:
    """Open an existing release and reject config or source drift.

    Unlike ``prepare-data``, this function never computes or writes a split.
    """

    root = config.resolve_path(config.dataset.release_directory)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Frozen release is missing: {root}. Run `isep-train prepare-data`."
        )
    validate_source_shards(config)
    audit = inspect_data_release(root)
    manifest = _mapping(_read_json(root / "release.json"), "release manifest")
    release = _child_mapping(manifest, "release")
    identity = _child_mapping(release, "identity")
    expected_data_hash = hashlib.sha256(
        config.dataset.model_dump_json().encode()
    ).hexdigest()
    expectations = {
        "source_release_sha256": source_release_sha256(config),
        "assignment_sha256": audit.assignment_sha256,
        "data_config_sha256": expected_data_hash,
    }
    for key, expected in expectations.items():
        if identity.get(key) != expected:
            raise ValueError(f"Frozen release identity mismatch for {key}")
    if release.get("id") != config.dataset.release_id:
        raise ValueError("Frozen release ID differs from the training config")
    return PreparedRelease(
        root=root,
        release_manifest_path=root / "release.json",
        assignments_path=root / "assignments.parquet",
        train_manifest_path=root / "sft_train.parquet",
        dev_manifest_path=root / "sft_dev.parquet",
        dev_panel_manifest_path=root / "dev_panel.parquet",
        audit=audit,
    )


def write_run_manifests(
    *,
    store: ArtifactStore,
    config: TrainingConfig,
    release: PreparedRelease,
    prompt: str,
    execution_profile: str,
) -> None:
    """Persist configuration, environment, data, model, and prompt identity."""

    if config.source_config_path is not None:
        source = config.source_config_path.read_text(encoding="utf-8")
        store.write_text("manifests", "config.original.yaml", source)
    store.write_json(
        "manifests",
        "config.resolved.json",
        resolved_config_document(config),
    )
    store.write_json(
        "manifests",
        "execution_context.json",
        {
            "project_root": str(config.project_root),
            "source_config_path": (
                str(config.source_config_path)
                if config.source_config_path is not None
                else None
            ),
            "config_sha256": config_hash(config),
            "execution_profile": execution_profile,
        },
    )
    store.write_json(
        "manifests",
        "dataset_release.json",
        _json_object(_read_json(release.release_manifest_path), "dataset release"),
    )
    store.write_json(
        "manifests",
        "dataset_audit.json",
        _json_object(asdict(release.audit), "dataset audit"),
    )
    store.write_json(
        "manifests",
        "environment.json",
        collect_environment(config.project_root),
    )
    store.write_text("manifests", "prompt.txt", prompt + "\n")
    store.write_json(
        "manifests",
        "model.json",
        _json_object(
            json.loads(config.model.model_dump_json()),
            "model config",
        ),
    )


def load_run_config(run_directory: Path) -> TrainingConfig:
    """Restore a strictly validated config from a run manifest."""

    store = ArtifactStore.at(run_directory)
    document = _json_object(
        _read_json(store.path("manifests", "config.resolved.json")),
        "resolved config",
    )
    context = _mapping(
        _read_json(store.path("manifests", "execution_context.json")),
        "execution context",
    )
    project_root = context.get("project_root")
    if not isinstance(project_root, str) or not project_root:
        raise ValueError("Run execution context has no project_root")
    document["project_root"] = project_root
    source_path = context.get("source_config_path")
    if source_path is not None:
        if not isinstance(source_path, str):
            raise ValueError("source_config_path must be a string or null")
        document["source_config_path"] = source_path
    return TrainingConfig.model_validate(document, strict=True)


def load_execution_profile(run_directory: Path) -> str:
    """Return the immutable ``full`` or ``smoke`` execution profile."""

    context = _mapping(
        _read_json(run_directory / "manifests" / "execution_context.json"),
        "execution context",
    )
    value = context.get("execution_profile")
    if value not in {"full", "smoke"}:
        raise ValueError("Run execution profile must be 'full' or 'smoke'")
    return str(value)


def persist_evaluation(
    store: ArtifactStore,
    result: EvaluationResult,
) -> None:
    """Write checkpoint predictions and metrics without clinical images."""

    stem = f"{result.subset.value}__{_safe_id(result.checkpoint_id)}"
    write_prediction_files(result.predictions, store.layout.predictions, stem=stem)
    payload = classification_metrics_to_json(result.metrics)
    payload["checkpoint_id"] = result.checkpoint_id
    payload["subset"] = result.subset.value
    payload["epoch"] = result.epoch
    payload["eval_loss"] = result.eval_loss
    store.write_json("metrics", f"{stem}.json", payload)


def checkpoint_directories(run_directory: Path) -> tuple[Path, ...]:
    """Return Trainer checkpoints ordered by global step."""

    root = run_directory / "checkpoints"
    return tuple(
        sorted(
            (path for path in root.glob("checkpoint-*") if path.is_dir()),
            key=lambda path: _checkpoint_step(path.name),
        )
    )


def resource_summary(run_directory: Path) -> ResourceSummary:
    """Aggregate backend and NVML logs into run-level efficiency metrics."""

    duration, throughput = _backend_efficiency_metrics(run_directory)
    peak_memory = 0.0
    maximum_power = 0.0
    maximum_temperature = 0.0
    observed_memory = observed_power = observed_temperature = False
    resources_path = run_directory / "logs" / "resources.jsonl"
    if resources_path.is_file():
        for line in resources_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = _mapping(json.loads(line), "resource sample")
            memory = _optional_float(sample.get("gpu_memory_used_bytes"))
            power = _optional_float(sample.get("gpu_power_watts"))
            temperature = _optional_float(sample.get("gpu_temperature_celsius"))
            if memory is not None:
                observed_memory = True
                peak_memory = max(peak_memory, memory)
            if power is not None:
                observed_power = True
                maximum_power = max(maximum_power, power)
            if temperature is not None:
                observed_temperature = True
                maximum_temperature = max(maximum_temperature, temperature)
    return ResourceSummary(
        duration_seconds=duration,
        gpu_hours=duration / 3600.0 if duration is not None else None,
        peak_vram_gib=(peak_memory / (1024.0**3) if observed_memory else None),
        train_samples_per_second=throughput,
        maximum_power_watts=maximum_power if observed_power else None,
        maximum_temperature_celsius=(
            maximum_temperature if observed_temperature else None
        ),
    )


def _backend_efficiency_metrics(
    run_directory: Path,
) -> tuple[float | None, float | None]:
    """Aggregate duration and throughput across successful backend sessions.

    A resumed run produces one TRL runtime and throughput measurement per
    backend invocation. Durations are additive, while throughput is weighted
    by each session's duration. Runs created before ``backend_sessions.json``
    was introduced retain their legacy latest-result behaviour.
    """

    sessions_path = run_directory / "manifests" / "backend_sessions.json"
    if sessions_path.is_file():
        value = _read_json(sessions_path)
        if not isinstance(value, list):
            raise ValueError(f"backend sessions must be an array: {sessions_path}")
        session_metrics: list[tuple[float, float | None]] = []
        for index, item in enumerate(value):
            session = _mapping(item, f"backend session {index}")
            metrics_value = session.get("metrics")
            if not isinstance(metrics_value, Mapping):
                continue
            runtime = _optional_float(metrics_value.get("train_runtime"))
            if runtime is None:
                continue
            throughput = _optional_float(metrics_value.get("train_samples_per_second"))
            session_metrics.append((runtime, throughput))
        if session_metrics:
            total_duration = sum(runtime for runtime, _ in session_metrics)
            weighted = tuple(
                (runtime, throughput)
                for runtime, throughput in session_metrics
                if runtime > 0.0 and throughput is not None
            )
            total_weight = sum(runtime for runtime, _ in weighted)
            aggregate_throughput = (
                sum(runtime * throughput for runtime, throughput in weighted)
                / total_weight
                if total_weight > 0.0
                else None
            )
            return total_duration, aggregate_throughput

    backend_path = run_directory / "manifests" / "backend_result.json"
    if not backend_path.is_file():
        return None, None
    backend = _mapping(_read_json(backend_path), "backend result")
    metrics_value = backend.get("metrics")
    if not isinstance(metrics_value, Mapping):
        return None, None
    return (
        _optional_float(metrics_value.get("train_runtime")),
        _optional_float(metrics_value.get("train_samples_per_second")),
    )


def _checkpoint_step(name: str) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", name)
    if match is None:
        raise ValueError(f"Invalid checkpoint directory: {name}")
    return int(match.group(1))


def _safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"Unsafe checkpoint identifier: {value!r}")
    return value


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON {path}: {exc}") from exc


def _mapping(value: object, context: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _child_mapping(value: Mapping[object, object], key: str) -> Mapping[object, object]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise ValueError(f"{key} must be an object")
    return child


def _json_object(value: object, context: str) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    result: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{context} has a non-string key")
        result[key] = _json_value(item, context)
    return result


def _json_value(value: object, context: str) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value(item, context) for item in value]
    if isinstance(value, Mapping):
        return _json_object(value, context)
    raise ValueError(f"{context} contains {type(value).__name__}")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
