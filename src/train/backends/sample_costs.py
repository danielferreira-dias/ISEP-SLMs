"""Exact per-example token and image-cost capture at the real collator boundary."""

from __future__ import annotations

import csv
import json
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from src.train.backends.unsloth_compat import invoke


@dataclass(frozen=True, slots=True)
class SampleCostRecord:
    """One sample's reproducibility, image geometry, and token-cost evidence."""

    sample_id: str
    split: str
    leakage_group_id: str
    image_width: int
    image_height: int
    pixel_count: int
    resized_width: int
    resized_height: int
    visual_token_count: int
    prompt_token_count: int
    target_token_count: int
    annotation_availability: str
    phase: str
    task: str


class SampleCostAuditingCollator:
    """Wrap the production collator and durably record each unique task row."""

    def __init__(
        self,
        *,
        collator: object,
        processor: object,
        model: object,
        jsonl_path: Path,
    ) -> None:
        """Bind the collator and resume-safe local JSONL destination."""

        self._collator = collator
        self._image_token_id = _image_token_id(model, processor)
        self._path = jsonl_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._seen = _existing_keys(jsonl_path)
        self._lock = threading.Lock()

    def __call__(self, records: Sequence[Mapping[object, object]]) -> object:
        """Collate normally, then capture exact non-padding token counts."""

        batch = invoke(self._collator, records)
        auditable = tuple(record for record in records if _is_auditable(record))
        if len(auditable) == len(records) and auditable:
            self._record(auditable, batch)
        return batch

    def _record(
        self,
        records: tuple[Mapping[object, object], ...],
        batch: object,
    ) -> None:
        fields = _mapping(batch, "collator batch")
        input_rows = _integer_rows(fields.get("input_ids"), "input_ids")
        label_rows = _integer_rows(fields.get("labels"), "labels")
        mask_rows = _integer_rows(fields.get("attention_mask"), "attention_mask")
        if not (len(records) == len(input_rows) == len(label_rows) == len(mask_rows)):
            raise RuntimeError("Collator cost-audit batch cardinality differs")
        for record, input_ids, labels, attention in zip(
            records,
            input_rows,
            label_rows,
            mask_rows,
            strict=True,
        ):
            active = tuple(
                token
                for token, enabled in zip(input_ids, attention, strict=True)
                if enabled
            )
            target_count = sum(token != -100 for token in labels)
            visual_count = sum(token == self._image_token_id for token in active)
            prompt_count = len(active) - target_count - visual_count
            if target_count <= 0 or visual_count <= 0 or prompt_count <= 0:
                raise RuntimeError("Invalid prompt/visual/target token decomposition")
            cost = _cost_record(
                record,
                visual_token_count=visual_count,
                prompt_token_count=prompt_count,
                target_token_count=target_count,
            )
            self._append_once(cost)

    def _append_once(self, record: SampleCostRecord) -> None:
        key = (record.sample_id, record.split, record.task)
        with self._lock:
            if key in self._seen:
                return
            encoded = (
                json.dumps(
                    asdict(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            descriptor = os.open(
                self._path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(descriptor, encoded)
            finally:
                os.close(descriptor)
            self._seen.add(key)


def materialize_sample_costs(
    jsonl_path: Path,
    output_directory: Path,
    *,
    expected_record_count: int | None = None,
) -> int:
    """Validate the append-only audit and materialize CSV plus typed Parquet."""

    if not jsonl_path.is_file():
        if expected_record_count not in (None, 0):
            raise RuntimeError("No per-sample cost audit was recorded")
        return 0
    records = _read_records(jsonl_path)
    if expected_record_count is not None and len(records) != expected_record_count:
        raise RuntimeError(
            "Per-sample cost audit coverage differs from the train+dev datasets: "
            f"expected {expected_record_count}, observed {len(records)}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    names = tuple(SampleCostRecord.__dataclass_fields__)
    csv_path = output_directory / "sample_costs.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    os.replace(temporary_csv, csv_path)
    table = pa.Table.from_pylist([asdict(record) for record in records])
    parquet_path = output_directory / "sample_costs.parquet"
    temporary_parquet = parquet_path.with_suffix(".parquet.tmp")
    write_table = cast(Callable[..., None], pq.write_table)
    write_table(table, temporary_parquet, compression="zstd")
    os.replace(temporary_parquet, parquet_path)
    manifest_path = output_directory / "sample_costs_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_count": len(records),
                "expected_record_count": expected_record_count,
                "coverage_complete": (
                    expected_record_count is not None
                    and len(records) == expected_record_count
                ),
                "token_count_source": (
                    "production_collator_input_ids_attention_mask_and_loss_labels"
                ),
            },
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return len(records)


def _cost_record(
    record: Mapping[object, object],
    *,
    visual_token_count: int,
    prompt_token_count: int,
    target_token_count: int,
) -> SampleCostRecord:
    availability = record.get("annotation_availability")
    if not isinstance(availability, list) or any(
        not isinstance(item, str) for item in availability
    ):
        raise TypeError("annotation_availability must be a string list")
    return SampleCostRecord(
        sample_id=_string(record, "sample_id"),
        split=_string(record, "split"),
        leakage_group_id=_string(record, "leakage_group_id"),
        image_width=_positive_integer(record, "image_width"),
        image_height=_positive_integer(record, "image_height"),
        pixel_count=_positive_integer(record, "pixel_count"),
        resized_width=_positive_integer(record, "resized_width"),
        resized_height=_positive_integer(record, "resized_height"),
        visual_token_count=visual_token_count,
        prompt_token_count=prompt_token_count,
        target_token_count=target_token_count,
        annotation_availability="|".join(sorted(availability)),
        phase=_string(record, "phase"),
        task=_string(record, "task"),
    )


def _is_auditable(record: Mapping[object, object]) -> bool:
    return all(
        key in record
        for key in (
            "split",
            "image_width",
            "annotation_availability",
            "phase",
            "task",
        )
    )


def _image_token_id(model: object, processor: object) -> int:
    owners = (
        getattr(model, "config", None),
        processor,
        getattr(processor, "tokenizer", None),
    )
    for owner in owners:
        value = getattr(owner, "image_token_id", None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    tokenizer = getattr(processor, "tokenizer", processor)
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(converter):
        value = converter("<|image_pad|>")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    raise RuntimeError("Cannot resolve the model's image token ID")


def _integer_rows(value: object, context: str) -> tuple[tuple[int, ...], ...]:
    tolist = getattr(value, "tolist", None)
    converted: object = tolist() if callable(tolist) else value
    if not isinstance(converted, list | tuple):
        raise TypeError(f"{context} must be a matrix")
    rows: list[tuple[int, ...]] = []
    for raw in converted:
        row = raw.tolist() if callable(getattr(raw, "tolist", None)) else raw
        if not isinstance(row, list | tuple) or any(
            not isinstance(item, int) or isinstance(item, bool) for item in row
        ):
            raise TypeError(f"{context} must contain integer rows")
        rows.append(tuple(row))
    return tuple(rows)


def _existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.is_file():
        return set()
    return {
        (record.sample_id, record.split, record.task) for record in _read_records(path)
    }


def _read_records(path: Path) -> tuple[SampleCostRecord, ...]:
    records: list[SampleCostRecord] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            raise ValueError(f"Blank sample-cost line {line_number}")
        value: object = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"Invalid sample-cost line {line_number}")
        records.append(_record_from_json(value, line_number))
    return tuple(
        sorted(records, key=lambda item: (item.split, item.task, item.sample_id))
    )


def _record_from_json(
    value: Mapping[object, object], line_number: int
) -> SampleCostRecord:
    """Convert one persisted record without trusting dynamic JSON types."""

    def integer(key: str) -> int:
        item = value.get(key)
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ValueError(f"Invalid {key} on sample-cost line {line_number}")
        return item

    def string(key: str) -> str:
        item = value.get(key)
        if not isinstance(item, str) or not item:
            raise ValueError(f"Invalid {key} on sample-cost line {line_number}")
        return item

    return SampleCostRecord(
        sample_id=string("sample_id"),
        split=string("split"),
        leakage_group_id=string("leakage_group_id"),
        image_width=integer("image_width"),
        image_height=integer("image_height"),
        pixel_count=integer("pixel_count"),
        resized_width=integer("resized_width"),
        resized_height=integer("resized_height"),
        visual_token_count=integer("visual_token_count"),
        prompt_token_count=integer("prompt_token_count"),
        target_token_count=integer("target_token_count"),
        annotation_availability=string("annotation_availability"),
        phase=string("phase"),
        task=string("task"),
    )


def _mapping(value: object, context: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _string(value: Mapping[object, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise TypeError(f"{key} must be a non-empty string")
    return item


def _positive_integer(value: Mapping[object, object], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
        raise TypeError(f"{key} must be a positive integer")
    return item
