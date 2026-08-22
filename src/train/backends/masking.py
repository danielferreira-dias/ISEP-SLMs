"""Runtime proof that the multimodal collator supervises only the answer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from src.train.backends.unsloth_compat import invoke, invoke_method
from src.train.execution.io import JsonValue, atomic_write_json


@dataclass(frozen=True, slots=True)
class ResponseMaskAudit:
    """Observed label-mask properties for one real collated example."""

    sample_id: str
    task: str | None
    target_label: str
    decoded_supervision: str
    supervised_token_count: int
    ignored_token_count: int
    forbidden_visual_token_count: int


def audit_response_only_mask(
    *,
    collator: object,
    processor: object,
    train_dataset: object,
    output_path: Path,
) -> ResponseMaskAudit:
    """Collate a real item and fail unless prompt/image/padding are ignored."""

    record = _first_record(train_dataset)
    audit = _audit_record(
        collator=collator,
        processor=processor,
        record=record,
    )
    atomic_write_json(output_path, asdict(audit))
    return audit


def audit_response_only_masks(
    *,
    collator: object,
    processor: object,
    train_dataset: object,
    output_path: Path,
) -> tuple[ResponseMaskAudit, ...]:
    """Audit one representative real row for every available training task."""

    audits = tuple(
        _audit_record(collator=collator, processor=processor, record=record)
        for record in _mask_audit_records(train_dataset)
    )
    tasks = [audit.task for audit in audits if audit.task is not None]
    if len(tasks) != len(set(tasks)):
        raise RuntimeError("Assistant-only mask audit received duplicate tasks")
    atomic_write_json(
        output_path,
        cast(
            JsonValue,
            {
                "audit_count": len(audits),
                "tasks": tasks,
                "audits": [asdict(audit) for audit in audits],
            },
        ),
    )
    return audits


def _audit_record(
    *,
    collator: object,
    processor: object,
    record: Mapping[object, object],
) -> ResponseMaskAudit:
    target = _required_string(record, "label")
    sample_id = _required_string(record, "sample_id")
    batch = invoke(collator, [record])
    fields = _mapping(batch, "collator batch")
    labels = _first_integer_row(fields.get("labels"), "labels")
    supervised = tuple(token for token in labels if token != -100)
    ignored = sum(token == -100 for token in labels)
    if not supervised or ignored == 0:
        raise RuntimeError(
            "Assistant-only audit found no supervised answer or no ignored prompt"
        )
    decoded_value = invoke_method(
        processor,
        "batch_decode",
        [list(supervised)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    decoded = _first_string(decoded_value).strip()
    if decoded != target:
        raise RuntimeError(
            "Assistant-only audit decoded supervision outside the target: "
            f"expected {target!r}, found {decoded!r}"
        )
    forbidden = _forbidden_visual_ids(processor)
    forbidden_count = sum(token in forbidden for token in supervised)
    if forbidden_count:
        raise RuntimeError(
            "Assistant-only audit found supervised visual/padding placeholders"
        )
    audit = ResponseMaskAudit(
        sample_id=sample_id,
        task=_optional_string(record, "task"),
        target_label=target,
        decoded_supervision=decoded,
        supervised_token_count=len(supervised),
        ignored_token_count=ignored,
        forbidden_visual_token_count=forbidden_count,
    )
    return audit


def _mask_audit_records(
    dataset: object,
) -> tuple[Mapping[object, object], ...]:
    provider = getattr(dataset, "mask_audit_records", None)
    if not callable(provider):
        return (_first_record(dataset),)
    raw_records = provider()
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise TypeError("mask_audit_records must return a sequence")
    records = tuple(
        _mapping(record, "mask-audit training record") for record in raw_records
    )
    if not records:
        raise TypeError("mask_audit_records returned no training records")
    return records


def _first_record(dataset: object) -> Mapping[object, object]:
    length = getattr(dataset, "__len__", None)
    getter = getattr(dataset, "__getitem__", None)
    if not callable(length) or not callable(getter) or int(length()) < 1:
        raise TypeError("Training dataset cannot provide a masking-audit item")
    return _mapping(getter(0), "training record")


def _mapping(value: object, context: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _required_string(record: Mapping[object, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"Training record {key} must be a non-empty string")
    return value


def _optional_string(record: Mapping[object, object], key: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"Training record {key} must be a non-empty string")
    return value


def _first_integer_row(value: object, context: str) -> tuple[int, ...]:
    first = value[0]  # type: ignore[index]
    tolist = getattr(first, "tolist", None)
    row: object = tolist() if callable(tolist) else first
    if not isinstance(row, list | tuple) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in row
    ):
        raise TypeError(f"{context} must contain an integer row")
    return tuple(row)


def _first_string(value: object) -> str:
    if not isinstance(value, list | tuple) or not value:
        raise TypeError("batch_decode must return a non-empty sequence")
    first = value[0]
    if not isinstance(first, str):
        raise TypeError("batch_decode returned a non-string item")
    return first


def _forbidden_visual_ids(processor: object) -> frozenset[int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    identifiers: set[int] = set()
    for owner in (processor, tokenizer):
        for name in (
            "pad_token_id",
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ):
            value = getattr(owner, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                identifiers.add(value)
    return frozenset(identifiers)
