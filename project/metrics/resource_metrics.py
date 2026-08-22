"""Run-level training efficiency aggregation for thesis artefacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ResourceSummary:
    """Measured training efficiency values used in thesis comparisons."""

    duration_seconds: float | None
    gpu_hours: float | None
    peak_vram_gib: float | None
    average_vram_gib: float | None
    peak_ram_gib: float | None
    average_ram_gib: float | None
    train_samples_per_second: float | None
    train_tokens_per_second: float | None
    train_steps_per_second: float | None
    average_step_seconds: float | None
    average_gpu_utilization_percent: float | None
    maximum_power_watts: float | None
    average_power_watts: float | None
    energy_wh: float | None
    maximum_temperature_celsius: float | None
    average_temperature_celsius: float | None
    checkpoint_size_gib: float | None
    trainable_parameters: int | None


@dataclass(frozen=True, slots=True)
class _BackendEfficiency:
    duration_seconds: float | None
    samples_per_second: float | None
    tokens_per_second: float | None
    steps_per_second: float | None


def resource_summary(run_directory: Path) -> ResourceSummary:
    """Aggregate backend and NVML logs into run-level efficiency metrics."""

    backend = _backend_efficiency_metrics(run_directory)
    gpu_memory: list[float] = []
    ram: list[float] = []
    utilization: list[float] = []
    power: list[float] = []
    temperature: list[float] = []
    energy_watt_seconds = 0.0
    previous_elapsed: float | None = None
    previous_power: float | None = None
    resources_path = run_directory / "logs" / "resources.jsonl"
    if resources_path.is_file():
        for line in resources_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = _mapping(json.loads(line), "resource sample")
            memory = _optional_float(sample.get("gpu_memory_used_bytes"))
            resident = _optional_float(sample.get("process_rss_bytes"))
            gpu_utilization = _optional_float(sample.get("gpu_utilization_percent"))
            watts = _optional_float(sample.get("gpu_power_watts"))
            celsius = _optional_float(sample.get("gpu_temperature_celsius"))
            elapsed = _optional_float(sample.get("elapsed_seconds"))
            _append_if_present(gpu_memory, memory)
            _append_if_present(ram, resident)
            _append_if_present(utilization, gpu_utilization)
            _append_if_present(power, watts)
            _append_if_present(temperature, celsius)
            if (
                elapsed is not None
                and watts is not None
                and previous_elapsed is not None
                and previous_power is not None
                and elapsed >= previous_elapsed
            ):
                energy_watt_seconds += ((previous_power + watts) / 2.0) * (
                    elapsed - previous_elapsed
                )
            previous_elapsed = elapsed
            previous_power = watts
    steps_per_second = backend.steps_per_second
    return ResourceSummary(
        duration_seconds=backend.duration_seconds,
        gpu_hours=_divide(backend.duration_seconds, 3600.0),
        peak_vram_gib=_gib(max(gpu_memory)) if gpu_memory else None,
        average_vram_gib=_gib(_mean(gpu_memory)) if gpu_memory else None,
        peak_ram_gib=_gib(max(ram)) if ram else None,
        average_ram_gib=_gib(_mean(ram)) if ram else None,
        train_samples_per_second=backend.samples_per_second,
        train_tokens_per_second=backend.tokens_per_second,
        train_steps_per_second=steps_per_second,
        average_step_seconds=(
            1.0 / steps_per_second
            if steps_per_second is not None and steps_per_second > 0.0
            else None
        ),
        average_gpu_utilization_percent=_mean_or_none(utilization),
        maximum_power_watts=max(power) if power else None,
        average_power_watts=_mean_or_none(power),
        energy_wh=(energy_watt_seconds / 3600.0 if power else None),
        maximum_temperature_celsius=max(temperature) if temperature else None,
        average_temperature_celsius=_mean_or_none(temperature),
        checkpoint_size_gib=_best_checkpoint_size_gib(run_directory),
        trainable_parameters=_trainable_parameters(run_directory),
    )


def _backend_efficiency_metrics(run_directory: Path) -> _BackendEfficiency:
    sessions_path = run_directory / "manifests" / "backend_sessions.json"
    if sessions_path.is_file():
        value = _read_json(sessions_path)
        if not isinstance(value, list):
            raise ValueError(f"backend sessions must be an array: {sessions_path}")
        session_metrics = _session_metrics(value)
        if session_metrics:
            return _BackendEfficiency(
                duration_seconds=sum(item[0] for item in session_metrics),
                samples_per_second=_weighted_metric(session_metrics, 1),
                tokens_per_second=_weighted_metric(session_metrics, 2),
                steps_per_second=_weighted_metric(session_metrics, 3),
            )
    backend_path = run_directory / "manifests" / "backend_result.json"
    if not backend_path.is_file():
        return _BackendEfficiency(None, None, None, None)
    backend = _mapping(_read_json(backend_path), "backend result")
    metrics = backend.get("metrics")
    if not isinstance(metrics, Mapping):
        return _BackendEfficiency(None, None, None, None)
    return _BackendEfficiency(
        _optional_float(metrics.get("train_runtime")),
        _optional_float(metrics.get("train_samples_per_second")),
        _optional_float(metrics.get("train_tokens_per_second")),
        _optional_float(metrics.get("train_steps_per_second")),
    )


def _session_metrics(
    sessions: list[object],
) -> list[tuple[float, float | None, float | None, float | None]]:
    result: list[tuple[float, float | None, float | None, float | None]] = []
    for index, item in enumerate(sessions):
        session = _mapping(item, f"backend session {index}")
        metrics = session.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        runtime = _optional_float(metrics.get("train_runtime"))
        if runtime is not None:
            result.append(
                (
                    runtime,
                    _optional_float(metrics.get("train_samples_per_second")),
                    _optional_float(metrics.get("train_tokens_per_second")),
                    _optional_float(metrics.get("train_steps_per_second")),
                )
            )
    return result


def _weighted_metric(
    sessions: list[tuple[float, float | None, float | None, float | None]],
    index: int,
) -> float | None:
    values: list[tuple[float, float]] = []
    for item in sessions:
        value = item[index]
        if item[0] > 0.0 and value is not None:
            values.append((item[0], value))
    total = sum(duration for duration, _ in values)
    return (
        sum(duration * value for duration, value in values) / total if total else None
    )


def _best_checkpoint_size_gib(run_directory: Path) -> float | None:
    path = run_directory / "manifests" / "best_checkpoint.json"
    if not path.is_file():
        return None
    document = _mapping(_read_json(path), "best checkpoint")
    raw_path = document.get("path")
    checkpoint = Path(raw_path) if isinstance(raw_path, str) else None
    if checkpoint is None or not checkpoint.is_dir():
        checkpoint_id = document.get("checkpoint_id")
        if not isinstance(checkpoint_id, str):
            return None
        checkpoint = run_directory / "checkpoints" / checkpoint_id
    if not checkpoint.is_dir():
        return None
    size = sum(item.stat().st_size for item in checkpoint.rglob("*") if item.is_file())
    return _gib(float(size))


def _trainable_parameters(run_directory: Path) -> int | None:
    path = run_directory / "manifests" / "backend_result.json"
    if not path.is_file():
        return None
    trainable = _mapping(_read_json(path), "backend result").get("trainable_parameters")
    if not isinstance(trainable, Mapping):
        return None
    total = trainable.get("total")
    return total if isinstance(total, int) and not isinstance(total, bool) else None


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping(value: object, context: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _optional_float(value: object) -> float | None:
    return (
        float(value)
        if isinstance(value, int | float) and not isinstance(value, bool)
        else None
    )


def _append_if_present(target: list[float], value: float | None) -> None:
    if value is not None:
        target.append(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _mean_or_none(values: list[float]) -> float | None:
    return _mean(values) if values else None


def _gib(value: float) -> float:
    return value / (1024.0**3)


def _divide(value: float | None, denominator: float) -> float | None:
    return value / denominator if value is not None else None
