"""Atomic CSV and Parquet persistence for prediction-level audit data."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from src.train.evaluation.models import PredictionRecord

from .tables import write_csv_table
from .types import PredictionArtifacts, TableCell

_HEADERS = (
    "sample_id",
    "leakage_group_id",
    "true_label",
    "raw_output",
    "predicted_label",
    "is_valid",
    "is_correct",
    "checkpoint_id",
    "seed",
)


def _row(record: PredictionRecord) -> tuple[TableCell, ...]:
    return (
        record.sample_id,
        record.leakage_group_id,
        record.true_label,
        record.raw_output,
        record.predicted_label,
        record.is_valid,
        record.is_correct,
        record.checkpoint_id,
        record.seed,
    )


def write_prediction_files(
    records: tuple[PredictionRecord, ...],
    directory: Path,
    *,
    stem: str = "final",
) -> PredictionArtifacts:
    """Write audit-equivalent prediction records to CSV and Parquet."""

    if not stem or Path(stem).name != stem:
        raise ValueError(f"Unsafe prediction stem: {stem!r}")
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv_table(
        directory / f"{stem}.csv",
        _HEADERS,
        tuple(_row(record) for record in records),
    )
    table = pa.table(
        {
            "sample_id": [record.sample_id for record in records],
            "leakage_group_id": [record.leakage_group_id for record in records],
            "true_label": [record.true_label for record in records],
            "raw_output": [record.raw_output for record in records],
            "predicted_label": [record.predicted_label for record in records],
            "is_valid": [record.is_valid for record in records],
            "is_correct": [record.is_correct for record in records],
            "checkpoint_id": [record.checkpoint_id for record in records],
            "seed": [record.seed for record in records],
        }
    )
    parquet_path = directory / f"{stem}.parquet"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{parquet_path.name}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_table = cast(
            Callable[..., None],
            pq.write_table,
        )
        write_table(table, temporary, compression="zstd")
        os.replace(temporary, parquet_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return PredictionArtifacts(csv_path=csv_path, parquet_path=parquet_path)


def _optional_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string or null")
    return value


def read_prediction_parquet(path: Path) -> tuple[PredictionRecord, ...]:
    """Read and validate prediction records from the canonical Parquet file."""

    if not path.is_file():
        raise FileNotFoundError(f"Prediction Parquet is missing: {path}")
    read_table = cast(Callable[[Path], pa.Table], pq.read_table)
    columns = read_table(path).to_pydict()
    missing = set(_HEADERS) - columns.keys()
    if missing:
        raise ValueError(f"Prediction Parquet is missing columns: {missing}")
    row_count = len(columns["sample_id"])
    if any(len(columns[name]) != row_count for name in _HEADERS):
        raise ValueError("Prediction Parquet columns have different lengths")
    records: list[PredictionRecord] = []
    for index in range(row_count):
        sample_id = columns["sample_id"][index]
        group_id = columns["leakage_group_id"][index]
        true_label = columns["true_label"][index]
        raw_output = columns["raw_output"][index]
        is_valid = columns["is_valid"][index]
        stored_is_correct = columns["is_correct"][index]
        checkpoint_id = columns["checkpoint_id"][index]
        seed = columns["seed"][index]
        if not all(
            isinstance(value, str)
            for value in (
                sample_id,
                group_id,
                true_label,
                raw_output,
                checkpoint_id,
            )
        ):
            raise ValueError(f"Prediction row {index} has invalid text fields")
        if not isinstance(is_valid, bool):
            raise ValueError(f"Prediction row {index} has invalid is_valid")
        if not isinstance(stored_is_correct, bool):
            raise ValueError(f"Prediction row {index} has invalid is_correct")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"Prediction row {index} has invalid seed")
        record = PredictionRecord(
            sample_id=sample_id,
            leakage_group_id=group_id,
            true_label=true_label,
            raw_output=raw_output,
            predicted_label=_optional_string(
                columns["predicted_label"][index], "predicted_label"
            ),
            is_valid=is_valid,
            checkpoint_id=checkpoint_id,
            seed=seed,
        )
        if record.is_correct != stored_is_correct:
            raise ValueError(f"Prediction row {index} has inconsistent is_correct")
        records.append(record)
    return tuple(records)
