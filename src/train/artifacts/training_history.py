"""Validated CSV and Parquet snapshots of the canonical metric JSONL log."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from .tables import write_csv_table

TRAINING_HISTORY_COLUMNS = (
    "name",
    "value",
    "step",
    "epoch",
    "timestamp_utc",
)


@dataclass(frozen=True, slots=True)
class TrainingMetricEvent:
    """One fully validated scalar event from the durable training log."""

    name: str
    value: float
    step: int
    epoch: float | None
    timestamp_utc: str


def materialize_training_history(
    jsonl_path: Path,
    csv_path: Path,
    parquet_path: Path,
) -> tuple[TrainingMetricEvent, ...]:
    """Validate JSONL and write authoritative tabular history snapshots.

    The source JSONL is read-only. Both derived outputs use atomic replacement,
    so a failed validation or Parquet write cannot truncate an existing file.

    Args:
        jsonl_path: Canonical append-only metric event log.
        csv_path: Destination for the RFC-4180 history table.
        parquet_path: Destination for the typed Parquet history table.

    Returns:
        Validated events in their original log order.

    Raises:
        FileNotFoundError: If the canonical JSONL log is missing.
        ValueError: If any line violates the fixed event schema.
    """

    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Training metric log is missing: {jsonl_path}")
    events = _read_events(jsonl_path)
    _write_parquet(events, parquet_path)
    write_csv_table(
        csv_path,
        TRAINING_HISTORY_COLUMNS,
        tuple(
            (
                event.name,
                event.value,
                event.step,
                event.epoch,
                event.timestamp_utc,
            )
            for event in events
        ),
    )
    return events


def _read_events(path: Path) -> tuple[TrainingMetricEvent, ...]:
    events: list[TrainingMetricEvent] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise ValueError(f"Metric JSONL line {line_number} is empty")
        try:
            payload: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Metric JSONL line {line_number} is invalid JSON: {exc.msg}"
            ) from exc
        events.append(_event(payload, line_number))
    return tuple(events)


def _event(payload: object, line_number: int) -> TrainingMetricEvent:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Metric JSONL line {line_number} must be an object")
    keys = set(payload.keys())
    expected = set(TRAINING_HISTORY_COLUMNS)
    if keys != expected:
        missing = sorted(str(key) for key in expected - keys)
        extra = sorted(str(key) for key in keys - expected)
        raise ValueError(
            f"Metric JSONL line {line_number} has schema drift; "
            f"missing={missing}, extra={extra}"
        )
    name = payload.get("name")
    value = payload.get("value")
    step = payload.get("step")
    epoch = payload.get("epoch")
    timestamp = payload.get("timestamp_utc")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Metric JSONL line {line_number} has an invalid name")
    numeric_value = _finite_number(value, line_number, "value")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"Metric JSONL line {line_number} has an invalid step")
    numeric_epoch = (
        None if epoch is None else _finite_number(epoch, line_number, "epoch")
    )
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError(
            f"Metric JSONL line {line_number} has an invalid timestamp_utc"
        )
    _validate_utc_timestamp(timestamp, line_number)
    return TrainingMetricEvent(
        name=name,
        value=numeric_value,
        step=step,
        epoch=numeric_epoch,
        timestamp_utc=timestamp,
    )


def _finite_number(value: object, line_number: int, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Metric JSONL line {line_number} has a non-numeric {field}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Metric JSONL line {line_number} has non-finite {field}")
    return result


def _validate_utc_timestamp(value: str, line_number: int) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"Metric JSONL line {line_number} has an invalid timestamp_utc"
        ) from exc
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"Metric JSONL line {line_number} timestamp_utc is not UTC")


def _write_parquet(
    events: tuple[TrainingMetricEvent, ...],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        (
            pa.field("name", pa.string(), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
            pa.field("step", pa.int64(), nullable=False),
            pa.field("epoch", pa.float64(), nullable=True),
            pa.field("timestamp_utc", pa.string(), nullable=False),
        )
    )
    table = pa.table(
        {
            "name": [event.name for event in events],
            "value": [event.value for event in events],
            "step": [event.step for event in events],
            "epoch": [event.epoch for event in events],
            "timestamp_utc": [event.timestamp_utc for event in events],
        },
        schema=schema,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer = cast(Callable[..., None], pq.write_table)
        writer(table, temporary, compression="zstd")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
