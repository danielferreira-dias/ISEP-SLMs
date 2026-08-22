"""Framework-neutral event bridge for Hugging Face Trainer callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Protocol

from project.metrics.contracts import MetricEvent, MetricSink


@dataclass(frozen=True, slots=True)
class CheckpointEvent:
    """Describe a checkpoint just after the trainer has persisted it."""

    path: Path
    global_step: int
    epoch: float | None


class CheckpointObserver(Protocol):
    """Receive checkpoint events while a fit is still running."""

    def on_checkpoint(self, event: CheckpointEvent) -> None:
        """Record a newly persisted checkpoint."""


class TrainerEventBridge:
    """Translate dynamic Trainer state into typed local events."""

    def __init__(
        self,
        *,
        metric_sink: MetricSink,
        checkpoint_observer: CheckpointObserver,
        output_dir: Path,
        examples_per_step: int,
    ) -> None:
        """Store output consumers without importing Transformers."""
        self._metric_sink = metric_sink
        self._checkpoint_observer = checkpoint_observer
        self._output_dir = output_dir
        if examples_per_step <= 0:
            raise ValueError("examples_per_step must be positive")
        self._examples_per_step = examples_per_step
        self._last_step: int | None = None
        self._last_monotonic: float | None = None

    def on_log(self, *, state: object, logs: object) -> None:
        """Forward numeric Trainer log values as scalar metric events."""
        if not isinstance(logs, dict):
            return
        step = _integer_attribute(state, "global_step", default=0)
        epoch = _optional_float_attribute(state, "epoch")
        timestamp = datetime.now(UTC).isoformat()
        self._emit_step_timing(step, epoch, timestamp)
        for raw_name, raw_value in sorted(logs.items(), key=lambda item: str(item[0])):
            if (
                not isinstance(raw_name, str)
                or isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
            ):
                continue
            self._metric_sink.write(
                MetricEvent(
                    name=raw_name,
                    value=raw_value,
                    step=step,
                    epoch=epoch,
                    timestamp_utc=timestamp,
                )
            )

    def _emit_step_timing(
        self,
        step: int,
        epoch: float | None,
        timestamp: str,
    ) -> None:
        """Emit interval-averaged optimizer timing from consecutive log steps."""

        now = monotonic()
        previous_step = self._last_step
        previous_time = self._last_monotonic
        if (
            previous_step is not None
            and previous_time is not None
            and step > previous_step
        ):
            elapsed = now - previous_time
            step_count = step - previous_step
            if elapsed > 0.0:
                seconds_per_step = elapsed / step_count
                self._metric_sink.write(
                    MetricEvent(
                        name="performance/seconds_per_step",
                        value=seconds_per_step,
                        step=step,
                        epoch=epoch,
                        timestamp_utc=timestamp,
                    )
                )
                self._metric_sink.write(
                    MetricEvent(
                        name="performance/examples_per_second",
                        value=self._examples_per_step / seconds_per_step,
                        step=step,
                        epoch=epoch,
                        timestamp_utc=timestamp,
                    )
                )
        if previous_step is None or step > previous_step:
            self._last_step = step
            self._last_monotonic = now

    def on_save(self, *, state: object) -> None:
        """Notify the execution layer after Trainer persists a checkpoint."""
        step = _integer_attribute(state, "global_step", default=0)
        self._checkpoint_observer.on_checkpoint(
            CheckpointEvent(
                path=self._output_dir / f"checkpoint-{step}",
                global_step=step,
                epoch=_optional_float_attribute(state, "epoch"),
            )
        )


def _integer_attribute(instance: object, name: str, *, default: int) -> int:
    value = getattr(instance, name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)


def _optional_float_attribute(instance: object, name: str) -> float | None:
    value = getattr(instance, name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
