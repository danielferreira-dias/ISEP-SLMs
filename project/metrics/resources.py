"""Optional process and NVIDIA GPU resource monitoring."""

from __future__ import annotations

import importlib
import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Protocol

from project.metrics.contracts import MetricEvent, MetricSink


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """Capture process and GPU telemetry at one instant."""

    timestamp_utc: str
    elapsed_seconds: float
    process_cpu_percent: float | None
    process_rss_bytes: int | None
    gpu_utilization_percent: float | None
    gpu_memory_used_bytes: int | None
    gpu_power_watts: float | None
    gpu_temperature_celsius: float | None


class ResourceMonitor(Protocol):
    """Lifecycle contract used by the training executor."""

    def start(self) -> None:
        """Start monitoring without blocking training."""

    def stop(self) -> None:
        """Stop monitoring and flush its local artefacts."""


class NoOpResourceMonitor:
    """Resource monitor used when monitoring is explicitly disabled."""

    def start(self) -> None:
        """Perform no work."""

    def stop(self) -> None:
        """Perform no work."""


class LocalResourceMonitor:
    """Sample optional psutil and NVML telemetry on a background thread."""

    def __init__(
        self,
        *,
        output_dir: Path,
        metric_sink: MetricSink,
        interval_seconds: float = 5.0,
    ) -> None:
        """Configure a monitor without importing optional dependencies."""
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._output_dir = output_dir
        self._metric_sink = metric_sink
        self._interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: object | None = None
        self._nvml: ModuleType | None = None
        self._gpu_handle: object | None = None
        self._start_monotonic = 0.0
        self._thread_error: BaseException | None = None

    def start(self) -> None:
        """Probe optional dependencies and start sampling immediately."""
        if self._thread is not None:
            raise RuntimeError("resource monitor has already been started")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        process_error = self._initialize_process()
        nvml_error = self._initialize_nvml()
        _atomic_write_json(
            self._output_dir / "resource_capabilities.json",
            {
                "psutil_available": self._process is not None,
                "nvml_available": self._gpu_handle is not None,
                "psutil_error": process_error,
                "nvml_error": nvml_error,
                "sampling_interval_seconds": self._interval,
            },
        )
        import time

        self._start_monotonic = time.monotonic()
        self._thread = threading.Thread(
            target=self._run,
            name="isep-resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the sampler, wait for it, and close NVML if initialized."""
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout=self._interval + 2.0)
        if thread.is_alive():
            raise RuntimeError("resource monitor thread did not stop")
        if self._nvml is not None and self._gpu_handle is not None:
            try:
                _module_call(self._nvml, "nvmlShutdown")
            finally:
                self._gpu_handle = None
        if self._thread_error is not None:
            raise RuntimeError(
                f"resource monitor failed: {self._thread_error}"
            ) from self._thread_error

    def _initialize_process(self) -> str | None:
        try:
            module = importlib.import_module("psutil")
            self._process = _module_call(module, "Process", os.getpid())
        except Exception as exc:
            self._process = None
            return f"{type(exc).__name__}: {exc}"
        return None

    def _initialize_nvml(self) -> str | None:
        try:
            module = importlib.import_module("pynvml")
            _module_call(module, "nvmlInit")
            self._gpu_handle = _module_call(
                module,
                "nvmlDeviceGetHandleByIndex",
                0,
            )
            self._nvml = module
        except Exception as exc:
            self._nvml = None
            self._gpu_handle = None
            return f"{type(exc).__name__}: {exc}"
        return None

    def _run(self) -> None:
        try:
            while True:
                sample = self._sample()
                self._append_sample(sample)
                self._emit_metrics(sample)
                if self._stop_event.wait(self._interval):
                    break
        except BaseException as exc:
            self._thread_error = exc

    def _sample(self) -> ResourceSample:
        import time

        cpu: float | None = None
        rss: int | None = None
        if self._process is not None:
            try:
                cpu = _numeric_method(self._process, "cpu_percent")
                memory_info = _object_method(self._process, "memory_info")
                rss = _optional_int_attribute(memory_info, "rss")
            except Exception:
                cpu = None
                rss = None

        utilization: float | None = None
        memory_used: int | None = None
        power: float | None = None
        temperature: float | None = None
        if self._nvml is not None and self._gpu_handle is not None:
            try:
                rates = _module_call(
                    self._nvml,
                    "nvmlDeviceGetUtilizationRates",
                    self._gpu_handle,
                )
                memory = _module_call(
                    self._nvml,
                    "nvmlDeviceGetMemoryInfo",
                    self._gpu_handle,
                )
                utilization = _optional_float_attribute(rates, "gpu")
                memory_used = _optional_int_attribute(memory, "used")
                raw_power = _module_call(
                    self._nvml,
                    "nvmlDeviceGetPowerUsage",
                    self._gpu_handle,
                )
                if isinstance(raw_power, (int, float)):
                    power = float(raw_power) / 1000.0
                temperature_constant = getattr(
                    self._nvml,
                    "NVML_TEMPERATURE_GPU",
                    0,
                )
                raw_temperature = _module_call(
                    self._nvml,
                    "nvmlDeviceGetTemperature",
                    self._gpu_handle,
                    temperature_constant,
                )
                if isinstance(raw_temperature, (int, float)):
                    temperature = float(raw_temperature)
            except Exception:
                utilization = memory_used = power = temperature = None

        return ResourceSample(
            timestamp_utc=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.monotonic() - self._start_monotonic,
            process_cpu_percent=cpu,
            process_rss_bytes=rss,
            gpu_utilization_percent=utilization,
            gpu_memory_used_bytes=memory_used,
            gpu_power_watts=power,
            gpu_temperature_celsius=temperature,
        )

    def _append_sample(self, sample: ResourceSample) -> None:
        path = self._output_dir / "resources.jsonl"
        encoded = (
            json.dumps(
                asdict(sample),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, encoded)
        finally:
            os.close(descriptor)

    def _emit_metrics(self, sample: ResourceSample) -> None:
        step = int(sample.elapsed_seconds)
        for name, value in (
            ("resources/process_cpu_percent", sample.process_cpu_percent),
            ("resources/process_rss_bytes", sample.process_rss_bytes),
            ("resources/gpu_utilization_percent", sample.gpu_utilization_percent),
            ("resources/gpu_memory_used_bytes", sample.gpu_memory_used_bytes),
            ("resources/gpu_power_watts", sample.gpu_power_watts),
            ("resources/gpu_temperature_celsius", sample.gpu_temperature_celsius),
        ):
            if value is not None:
                self._metric_sink.write(
                    MetricEvent(
                        name=name,
                        value=value,
                        step=step,
                        timestamp_utc=sample.timestamp_utc,
                    )
                )


def _atomic_write_json(path: Path, value: object) -> None:
    """Atomically persist the monitor capability document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _module_call(module: ModuleType, name: str, *args: object) -> object:
    function = getattr(module, name, None)
    if not callable(function):
        raise AttributeError(f"{module.__name__}.{name} is not callable")
    return function(*args)


def _object_method(instance: object, name: str) -> object:
    method = getattr(instance, name, None)
    if not callable(method):
        raise AttributeError(f"{type(instance).__name__}.{name} is not callable")
    return method()


def _numeric_method(instance: object, name: str) -> float | None:
    value = _object_method(instance, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _optional_float_attribute(instance: object, name: str) -> float | None:
    value = getattr(instance, name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _optional_int_attribute(instance: object, name: str) -> int | None:
    value = getattr(instance, name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)
