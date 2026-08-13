"""Framework-neutral event bridge for Hugging Face Trainer callbacks."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.train.backends.contracts import (
    CheckpointEvent,
    CheckpointObserver,
    MetricEvent,
    MetricSink,
)


class TrainerEventBridge:
    """Translate dynamic Trainer state into typed local events."""

    def __init__(
        self,
        *,
        metric_sink: MetricSink,
        checkpoint_observer: CheckpointObserver,
        output_dir: Path,
    ) -> None:
        """Store output consumers without importing Transformers."""
        self._metric_sink = metric_sink
        self._checkpoint_observer = checkpoint_observer
        self._output_dir = output_dir

    def on_log(self, *, state: object, logs: object) -> None:
        """Forward numeric Trainer log values as scalar metric events.

        Args:
            state: Dynamic ``TrainerState`` instance.
            logs: Dynamic mapping produced by Hugging Face Trainer.
        """
        if not isinstance(logs, dict):
            return
        step = _integer_attribute(state, "global_step", default=0)
        epoch = _optional_float_attribute(state, "epoch")
        timestamp = datetime.now(UTC).isoformat()
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
