"""Framework-neutral scalar metric contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

MetricValue = float | int


@dataclass(frozen=True, slots=True)
class MetricEvent:
    """Represent one finite scalar observation emitted during training."""

    name: str
    value: MetricValue
    step: int
    epoch: float | None = None
    timestamp_utc: str | None = None

    def __post_init__(self) -> None:
        """Reject non-finite metrics and invalid event coordinates."""
        if not self.name:
            raise ValueError("MetricEvent name must not be empty")
        if not math.isfinite(float(self.value)):
            raise ValueError("MetricEvent value must be finite")
        if self.step < 0:
            raise ValueError("MetricEvent step must be non-negative")
        if self.epoch is not None and not math.isfinite(self.epoch):
            raise ValueError("MetricEvent epoch must be finite when present")


class MetricSink(Protocol):
    """Receive scalar metrics without depending on a tracking vendor."""

    def write(self, event: MetricEvent) -> None:
        """Persist one scalar metric event."""

    def close(self) -> None:
        """Flush resources held by the sink."""
