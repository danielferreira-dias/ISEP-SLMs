"""Resource and energy telemetry for controlled local benchmark campaigns."""

from __future__ import annotations

import importlib
import json
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from types import ModuleType


@dataclass(frozen=True, slots=True)
class BenchmarkResourceSample:
    """One host and NVIDIA GPU sample without clinical content."""

    timestamp_utc: str
    monotonic_seconds: float
    phase: str
    gpu_utilization_percent: float
    gpu_memory_used_bytes: int
    gpu_memory_total_bytes: int
    gpu_power_watts: float
    gpu_temperature_celsius: float
    system_memory_used_bytes: int
    system_memory_available_bytes: int
    server_process_rss_bytes: int | None
    server_process_cpu_percent: float | None
    session_id: str = "legacy"


class BenchmarkResourceMonitor:
    """Sample process, RAM, VRAM, utilization, power, and temperature."""

    def __init__(
        self,
        *,
        output_root: Path,
        server_pid: int | None,
        interval_seconds: float = 0.5,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._output_root = output_root
        self._interval = interval_seconds
        self._server_pid = server_pid
        self._phase = "initializing"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._samples: list[BenchmarkResourceSample] = []
        self._nvml: ModuleType | None = None
        self._handle: object | None = None
        self._psutil: ModuleType | None = None
        self._process: object | None = None
        self._session_id = uuid.uuid4().hex

    def start(self) -> None:
        """Initialize required probes and start sampling."""

        if self._thread is not None:
            raise RuntimeError("resource monitor already started")
        self._initialize()
        (self._output_root / "logs").mkdir(parents=True, exist_ok=True)
        (self._output_root / "metrics").mkdir(parents=True, exist_ok=True)
        self._samples = _load_existing_samples(
            self._output_root / "logs" / "nvml_samples.jsonl"
        )
        self._thread = threading.Thread(
            target=self._run,
            name="benchmark-resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        """Associate subsequent resource samples with a benchmark task."""

        if not phase:
            raise ValueError("phase cannot be empty")
        with self._lock:
            self._phase = phase

    def stop(self) -> dict[str, object]:
        """Stop sampling, integrate energy, and persist the summary."""

        thread = self._thread
        if thread is None:
            raise RuntimeError("resource monitor was not started")
        self._stop.set()
        thread.join(timeout=self._interval + 5.0)
        if thread.is_alive():
            raise RuntimeError("resource monitor did not stop")
        if self._nvml is not None:
            _call(self._nvml, "nvmlShutdown")
        if self._error is not None:
            raise RuntimeError(f"resource monitor failed: {self._error}")
        summary = _resource_summary(self._samples, self._interval)
        _atomic_json(
            self._output_root / "metrics" / "resource_efficiency_summary.json",
            summary,
        )
        return summary

    def _initialize(self) -> None:
        try:
            nvml = importlib.import_module("pynvml")
            _call(nvml, "nvmlInit")
            handle = _call(nvml, "nvmlDeviceGetHandleByIndex", 0)
            psutil = importlib.import_module("psutil")
        except Exception as exc:
            raise RuntimeError(
                "pynvml and psutil are required for thesis efficiency metrics: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._nvml = nvml
        self._handle = handle
        self._psutil = psutil
        if self._server_pid is not None:
            self._process = _call(psutil, "Process", self._server_pid)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                sample = self._sample()
                self._samples.append(sample)
                self._append(sample)
                self._stop.wait(self._interval)
        except BaseException as exc:
            self._error = exc

    def _sample(self) -> BenchmarkResourceSample:
        nvml = self._required_nvml()
        handle = self._handle
        psutil = self._required_psutil()
        rates = _call(nvml, "nvmlDeviceGetUtilizationRates", handle)
        memory = _call(nvml, "nvmlDeviceGetMemoryInfo", handle)
        power_mw = _call(nvml, "nvmlDeviceGetPowerUsage", handle)
        temperature = _call(
            nvml,
            "nvmlDeviceGetTemperature",
            handle,
            getattr(nvml, "NVML_TEMPERATURE_GPU", 0),
        )
        virtual_memory = _call(psutil, "virtual_memory")
        rss, cpu = self._server_process_metrics()
        with self._lock:
            phase = self._phase
        return BenchmarkResourceSample(
            timestamp_utc=datetime.now(UTC).isoformat(),
            monotonic_seconds=time.monotonic(),
            phase=phase,
            gpu_utilization_percent=_number_attribute(rates, "gpu"),
            gpu_memory_used_bytes=int(_number_attribute(memory, "used")),
            gpu_memory_total_bytes=int(_number_attribute(memory, "total")),
            gpu_power_watts=_number(power_mw) / 1000.0,
            gpu_temperature_celsius=_number(temperature),
            system_memory_used_bytes=int(_number_attribute(virtual_memory, "used")),
            system_memory_available_bytes=int(
                _number_attribute(virtual_memory, "available")
            ),
            server_process_rss_bytes=rss,
            server_process_cpu_percent=cpu,
            session_id=self._session_id,
        )

    def _server_process_metrics(self) -> tuple[int | None, float | None]:
        process = self._process
        if process is None:
            return None, None
        try:
            processes = [process, *_children(process)]
            rss = sum(
                int(_number_attribute(_method(item, "memory_info"), "rss"))
                for item in processes
            )
            cpu = sum(_number(_method(item, "cpu_percent")) for item in processes)
            return rss, cpu
        except Exception:
            return None, None

    def _append(self, sample: BenchmarkResourceSample) -> None:
        path = self._output_root / "logs" / "nvml_samples.jsonl"
        payload = (
            json.dumps(
                asdict(sample), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        )
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, payload.encode("utf-8"))
        finally:
            os.close(descriptor)

    def _required_nvml(self) -> ModuleType:
        if self._nvml is None or self._handle is None:
            raise RuntimeError("NVML is not initialized")
        return self._nvml

    def _required_psutil(self) -> ModuleType:
        if self._psutil is None:
            raise RuntimeError("psutil is not initialized")
        return self._psutil


def _resource_summary(
    samples: list[BenchmarkResourceSample], interval_seconds: float
) -> dict[str, object]:
    if not samples:
        raise RuntimeError("resource monitor produced no samples")
    phases: dict[str, list[BenchmarkResourceSample]] = {}
    for sample in samples:
        phases.setdefault(sample.phase, []).append(sample)
    return {
        "schema_version": 1,
        "energy_method": "trapezoidal integration of raw NVML board power",
        "sampling_interval_seconds": interval_seconds,
        "sample_count": len(samples),
        "total": _summarize_samples(samples),
        "by_phase": {
            name: _summarize_samples(values) for name, values in sorted(phases.items())
        },
    }


def _summarize_samples(samples: list[BenchmarkResourceSample]) -> dict[str, object]:
    sessions: dict[str, list[BenchmarkResourceSample]] = {}
    for sample in samples:
        sessions.setdefault(sample.session_id, []).append(sample)
    energy_wh = 0.0
    duration = 0.0
    ordered: list[BenchmarkResourceSample] = []
    for session_samples in sessions.values():
        session_ordered = sorted(
            session_samples, key=lambda sample: sample.monotonic_seconds
        )
        ordered.extend(session_ordered)
        duration += max(
            0.0,
            session_ordered[-1].monotonic_seconds
            - session_ordered[0].monotonic_seconds,
        )
        for previous, current in pairwise(session_ordered):
            elapsed = max(0.0, current.monotonic_seconds - previous.monotonic_seconds)
            energy_wh += (
                ((previous.gpu_power_watts + current.gpu_power_watts) / 2.0)
                * elapsed
                / 3600.0
            )
    process_rss = [
        sample.server_process_rss_bytes
        for sample in ordered
        if sample.server_process_rss_bytes is not None
    ]
    return {
        "duration_seconds": duration,
        "energy_wh": energy_wh,
        "mean_gpu_power_watts": _mean([sample.gpu_power_watts for sample in ordered]),
        "peak_gpu_power_watts": max(sample.gpu_power_watts for sample in ordered),
        "mean_gpu_utilization_percent": _mean(
            [sample.gpu_utilization_percent for sample in ordered]
        ),
        "peak_gpu_memory_bytes": max(
            sample.gpu_memory_used_bytes for sample in ordered
        ),
        "gpu_memory_total_bytes": ordered[0].gpu_memory_total_bytes,
        "peak_gpu_temperature_celsius": max(
            sample.gpu_temperature_celsius for sample in ordered
        ),
        "peak_system_memory_used_bytes": max(
            sample.system_memory_used_bytes for sample in ordered
        ),
        "peak_server_process_rss_bytes": max(process_rss) if process_rss else None,
    }


def _mean(values: list[float]) -> float:
    value = sum(values) / len(values)
    if not math.isfinite(value):
        raise ValueError("non-finite resource metric")
    return value


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_existing_samples(path: Path) -> list[BenchmarkResourceSample]:
    """Load prior sessions so resumed campaigns retain complete telemetry."""

    if not path.is_file():
        return []
    samples: list[BenchmarkResourceSample] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise ValueError(f"blank telemetry line at {path}:{line_number}")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(
                f"telemetry record must be an object at {path}:{line_number}"
            )
        try:
            samples.append(BenchmarkResourceSample(**payload))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid telemetry record at {path}:{line_number}: {exc}"
            ) from exc
    return samples


def _call(target: object, name: str, *args: object, **kwargs: object) -> object:
    function = getattr(target, name, None)
    if not callable(function):
        raise AttributeError(f"{name} is not callable")
    return function(*args, **kwargs)


def _method(target: object, name: str, *args: object, **kwargs: object) -> object:
    return _call(target, name, *args, **kwargs)


def _children(process: object) -> list[object]:
    value = _method(process, "children", recursive=True)
    if not isinstance(value, list):
        raise TypeError("psutil children() did not return a list")
    return list(value)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("resource value is not numeric")
    return float(value)


def _number_attribute(value: object, name: str) -> float:
    return _number(getattr(value, name, None))
