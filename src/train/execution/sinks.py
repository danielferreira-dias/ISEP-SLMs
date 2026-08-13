"""Local metric sinks used by the scientific training pipeline."""

from __future__ import annotations

import importlib
import json
import os
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from src.train.backends.contracts import MetricEvent, MetricSink


class _SummaryWriter(Protocol):
    """Subset of TensorBoard's writer used by this package."""

    def add_scalar(
        self,
        tag: str,
        scalar_value: float | int,
        global_step: int,
        walltime: float | None = None,
    ) -> None:
        """Write one scalar event."""

    def flush(self) -> None:
        """Flush buffered events."""

    def close(self) -> None:
        """Close the writer."""


class NullMetricSink:
    """Discard events while preserving the metric-sink contract."""

    def write(self, event: MetricEvent) -> None:
        """Discard one event."""

    def close(self) -> None:
        """Perform no cleanup."""


class JsonlMetricSink:
    """Append canonical scalar events to an fsync-backed JSONL file."""

    def __init__(self, path: Path) -> None:
        """Create the parent directory without truncating an existing log."""
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: MetricEvent) -> None:
        """Append one complete JSON line with a single operating-system write."""
        payload = asdict(event)
        if payload["timestamp_utc"] is None:
            payload["timestamp_utc"] = datetime.now(UTC).isoformat()
        encoded = (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        with self._lock:
            descriptor = os.open(
                self._path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def close(self) -> None:
        """Perform no cleanup because every write opens and closes the file."""


class TensorBoardMetricSink:
    """Write scalar events through an already-created SummaryWriter."""

    def __init__(self, writer: _SummaryWriter) -> None:
        """Wrap a TensorBoard-compatible writer."""
        self._writer = writer

    @classmethod
    def create_if_available(cls, log_dir: Path) -> TensorBoardMetricSink | None:
        """Create a local TensorBoard sink without making it a hard import.

        Args:
            log_dir: Directory in which TensorBoard event files will be stored.

        Returns:
            A sink when PyTorch's TensorBoard integration is installed,
            otherwise ``None``.
        """
        try:
            module = importlib.import_module("torch.utils.tensorboard")
            writer_factory = module.SummaryWriter
            writer = writer_factory(log_dir=str(log_dir))
        except (AttributeError, ImportError, ModuleNotFoundError):
            return None
        return cls(cast(_SummaryWriter, writer))

    def write(self, event: MetricEvent) -> None:
        """Write one scalar with the metric step as its global step."""
        walltime: float | None = None
        if event.timestamp_utc is not None:
            walltime = datetime.fromisoformat(event.timestamp_utc).timestamp()
        self._writer.add_scalar(
            event.name,
            event.value,
            event.step,
            walltime=walltime,
        )

    def close(self) -> None:
        """Flush and close the TensorBoard event writer."""
        self._writer.flush()
        self._writer.close()


class CompositeMetricSink:
    """Fan out every metric to multiple local sinks."""

    def __init__(self, sinks: tuple[MetricSink, ...]) -> None:
        """Retain sinks in deterministic write order."""
        self._sinks = sinks

    def write(self, event: MetricEvent) -> None:
        """Write one event to every configured sink."""
        for sink in self._sinks:
            sink.write(event)

    def close(self) -> None:
        """Close all sinks, raising the first cleanup failure."""
        first_error: Exception | None = None
        for sink in reversed(self._sinks):
            try:
                sink.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


def create_default_metric_sink(
    run_dir: Path,
    *,
    require_tensorboard: bool = False,
) -> MetricSink:
    """Create canonical JSONL tracking and optional local TensorBoard tracking.

    Args:
        run_dir: Root directory for the current training execution.
        require_tensorboard: Fail if the configured writer is unavailable.

    Returns:
        Composite sink that always contains durable JSONL logging.
    """
    sinks: list[MetricSink] = [JsonlMetricSink(run_dir / "logs" / "metrics.jsonl")]
    tensorboard = TensorBoardMetricSink.create_if_available(run_dir / "tensorboard")
    if tensorboard is not None:
        sinks.append(tensorboard)
    elif require_tensorboard:
        raise RuntimeError("TensorBoard is required by this run but is not installed")
    return CompositeMetricSink(tuple(sinks))
