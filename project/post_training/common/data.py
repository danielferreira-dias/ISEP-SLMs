"""Lazy, audited dataset adapters for the E3 multimodal SFT stage."""

from __future__ import annotations

import hashlib
import io
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast, overload

from PIL import Image

from project.post_training.common.config import (
    DatasetReference,
    E3SFTStageConfig,
)
from src.train.data.images import preprocess_image_with_metadata


class DatasetLike(Protocol):
    """Small surface shared by Hugging Face datasets and test doubles."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> object: ...


type DatasetLoader = Callable[..., object]


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    """Observed cardinalities checked before the GPU is allocated."""

    identity: str
    expected_rows: int
    observed_rows: int
    expected_task_counts: dict[str, int]
    observed_task_counts: dict[str, int]

    def as_manifest(self) -> dict[str, object]:
        """Return a JSON-serializable audit record."""

        return {
            "identity": self.identity,
            "expected_rows": self.expected_rows,
            "observed_rows": self.observed_rows,
            "expected_task_counts": dict(self.expected_task_counts),
            "observed_task_counts": dict(self.observed_task_counts),
        }


class PreparedSFTDataset(Sequence[dict[str, object]]):
    """Validate and inject one decoded image only when a row is requested."""

    def __init__(
        self,
        backing: DatasetLike,
        reference: DatasetReference,
        *,
        audit: DatasetAudit,
        limit: int | None = None,
    ) -> None:
        if limit is not None and limit <= 0:
            raise ValueError("Dataset limit must be positive when provided")
        self._backing = backing
        self._reference = reference
        self._audit = audit
        self._length = min(len(backing), limit) if limit is not None else len(backing)

    @property
    def reference(self) -> DatasetReference:
        """Return the immutable Hub view used by this adapter."""

        return self._reference

    @property
    def audit(self) -> DatasetAudit:
        """Return the preflight cardinality audit."""

        return self._audit

    def __len__(self) -> int:
        return self._length

    def mask_audit_records(self) -> tuple[dict[str, object], ...]:
        """Return one real adapted row per task for collator-mask auditing."""

        tasks = _column_values(self._backing, "task")
        if tasks is None:
            return (self[0],)
        first_indices: dict[str, int] = {}
        for index, raw_task in enumerate(tasks[: self._length]):
            task = _non_empty_string(raw_task, "task")
            first_indices.setdefault(task, index)
        if not first_indices:
            raise ValueError("Training dataset exposes no task for mask audit")
        return tuple(self[index] for task, index in sorted(first_indices.items()))

    @overload
    def __getitem__(self, index: int) -> dict[str, object]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, object]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> dict[str, object] | list[dict[str, object]]:
        """Return one trainer-visible row or a materialized slice."""

        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        normalized = index if index >= 0 else len(self) + index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("E3 SFT dataset index out of range")
        raw = self._backing[normalized]
        if not isinstance(raw, Mapping):
            raise ValueError("E3 backing dataset returned a non-mapping row")
        return _prepare_row(raw, reference=self._reference)


@dataclass(frozen=True, slots=True)
class PreparedDatasets:
    """Trainer-ready train/dev views plus their preflight audits."""

    train: PreparedSFTDataset
    dev: PreparedSFTDataset

    @property
    def train_audit(self) -> DatasetAudit:
        return self.train.audit

    @property
    def dev_audit(self) -> DatasetAudit:
        return self.dev.audit

    def as_manifest(self) -> dict[str, object]:
        """Return identities, full counts, and effective adapter lengths."""

        return {
            "train": {
                **self.train.audit.as_manifest(),
                "effective_rows": len(self.train),
            },
            "dev": {
                **self.dev.audit.as_manifest(),
                "effective_rows": len(self.dev),
            },
        }


def prepare_dataset_rows(
    rows: DatasetLike,
    reference: DatasetReference,
    *,
    limit: int | None = None,
    validate_counts: bool = True,
) -> PreparedSFTDataset:
    """Wrap one lazy dataset and validate its frozen cardinality contract."""

    audit = _audit_dataset(rows, reference, validate_counts=validate_counts)
    return PreparedSFTDataset(rows, reference, audit=audit, limit=limit)


def load_training_datasets(
    config: E3SFTStageConfig,
    dataset_loader: DatasetLoader | None = None,
    *,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    train_limit: int | None = None,
    dev_limit: int | None = None,
    validate_counts: bool = True,
) -> PreparedDatasets:
    """Load pinned Hub views and adapt them without eager image materialization.

    ``dataset_loader`` is injectable so configuration and row contracts can be
    tested without network access.  It must accept the same keyword arguments
    as :func:`datasets.load_dataset`.
    """

    loader = dataset_loader or _default_dataset_loader
    train_backing = _load_one(
        config.datasets.train,
        loader,
        cache_dir=cache_dir,
        token=token,
    )
    dev_backing = _load_one(
        config.datasets.dev,
        loader,
        cache_dir=cache_dir,
        token=token,
    )
    return PreparedDatasets(
        train=prepare_dataset_rows(
            train_backing,
            config.datasets.train,
            limit=train_limit,
            validate_counts=validate_counts,
        ),
        dev=prepare_dataset_rows(
            dev_backing,
            config.datasets.dev,
            limit=dev_limit,
            validate_counts=validate_counts,
        ),
    )


def _default_dataset_loader(*args: object, **kwargs: object) -> object:
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("The training extra with datasets is required") from exc
    return load_dataset(*args, **kwargs)


def _load_one(
    reference: DatasetReference,
    loader: DatasetLoader,
    *,
    cache_dir: str | Path | None,
    token: str | bool | None,
) -> DatasetLike:
    kwargs: dict[str, object] = {
        "name": reference.config,
        "split": reference.split,
        "revision": reference.revision,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
    # Both frozen releases are private. ``True`` asks Hugging Face to use the
    # active environment/CLI credential without copying it into a manifest.
    kwargs["token"] = token if token is not None else True
    loaded = loader(reference.repo_id, **kwargs)
    if not hasattr(loaded, "__len__") or not hasattr(loaded, "__getitem__"):
        raise TypeError("Dataset loader did not return an indexable dataset")
    return _disable_image_decoding(cast(DatasetLike, loaded))


def _audit_dataset(
    rows: DatasetLike,
    reference: DatasetReference,
    *,
    validate_counts: bool,
) -> DatasetAudit:
    observed_rows = len(rows)
    tasks = _column_values(rows, "task")
    if tasks is None:
        default_task = _default_task(reference)
        observed_task_counts = {default_task: observed_rows}
    else:
        normalized_tasks = [_non_empty_string(item, "task") for item in tasks]
        observed_task_counts = dict(sorted(Counter(normalized_tasks).items()))

    expected_task_counts = dict(reference.expected_task_counts)
    if not expected_task_counts and reference.config == "diagnosis":
        expected_task_counts = {"diagnosis": reference.expected_rows}

    if validate_counts and observed_rows != reference.expected_rows:
        raise ValueError(
            f"Dataset {reference.identity} has {observed_rows} rows; "
            f"expected {reference.expected_rows}"
        )
    if validate_counts and observed_task_counts != expected_task_counts:
        raise ValueError(
            f"Dataset {reference.identity} task counts differ: "
            f"observed={observed_task_counts}, expected={expected_task_counts}"
        )
    return DatasetAudit(
        identity=reference.identity,
        expected_rows=reference.expected_rows,
        observed_rows=observed_rows,
        expected_task_counts=expected_task_counts,
        observed_task_counts=observed_task_counts,
    )


def _column_values(rows: DatasetLike, column: str) -> Sequence[object] | None:
    column_names = getattr(rows, "column_names", None)
    if isinstance(column_names, Sequence) and column not in column_names:
        return None
    if column_names is not None:
        try:
            raw = cast(object, rows)[column]  # type: ignore[index]
        except (IndexError, KeyError, TypeError, ValueError):
            pass
        else:
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                return cast(Sequence[object], raw)

    values: list[object] = []
    found = False
    for index in range(len(rows)):
        raw = rows[index]
        if not isinstance(raw, Mapping):
            raise ValueError("E3 backing dataset returned a non-mapping row")
        if column in raw:
            found = True
            values.append(raw[column])
        elif found:
            raise ValueError(f"Dataset mixes rows with and without {column}")
    return values if found else None


def _prepare_row(
    row: Mapping[object, object],
    *,
    reference: DatasetReference,
) -> dict[str, object]:
    raw_image = row.get("image")
    image_integrity_verified = _verify_encoded_image_sha256(
        raw_image,
        row.get("image_sha256"),
    )
    image, geometry = preprocess_image_with_metadata(
        _decoded_image(raw_image),
        max_edge_pixels=512,
    )
    prompt = _non_empty_string(row.get("prompt"), "prompt")
    target = _non_empty_string(row.get("target_text"), "target_text")
    prompt_sha256 = _verify_text_sha256(
        prompt,
        row.get("prompt_sha256"),
        field="prompt_sha256",
    )
    target_sha256, target_integrity_method = _target_integrity(
        row,
        target=target,
        reference=reference,
    )
    original_sample_id = _non_empty_string(row.get("sample_id"), "sample_id")
    row_id_value = row.get("row_id")
    row_id = (
        _non_empty_string(row_id_value, "row_id")
        if row_id_value is not None
        else original_sample_id
    )
    source_sample_value = row.get("source_sample_id")
    source_sample_id = (
        _non_empty_string(source_sample_value, "source_sample_id")
        if source_sample_value is not None
        else original_sample_id
    )
    task_value = row.get("task")
    task = (
        _non_empty_string(task_value, "task")
        if task_value is not None
        else _default_task(reference)
    )
    messages = _inject_image_and_validate_messages(
        row.get("messages"), image=image, prompt=prompt, target=target
    )

    record: dict[str, object] = {
        # The backend's mask audit requires globally unique sample IDs.  E3 has
        # several task rows per source image, hence row_id is the correct unit.
        "sample_id": row_id,
        "row_id": row_id,
        "canonical_sample_id": original_sample_id,
        "source_sample_id": source_sample_id,
        "task": task,
        "task_id": _optional_string(row.get("task_id"), default=task),
        "split": _optional_string(row.get("split"), default=reference.split),
        "label": target,
        "prompt": prompt,
        "prompt_sha256": prompt_sha256,
        "prompt_integrity_method": "published_sha256",
        "target_text": target,
        "target_sha256": target_sha256,
        "target_integrity_method": target_integrity_method,
        "messages": messages,
        # These fields make the existing production-collator cost audit apply
        # to E3 as well.  They are measured after EXIF normalization and the
        # frozen no-upscale/max-edge-512 preprocessing operation.
        "phase": "e3_multitask_sft",
        "image_width": geometry.image_width,
        "image_height": geometry.image_height,
        "pixel_count": geometry.pixel_count,
        "resized_width": geometry.resized_width,
        "resized_height": geometry.resized_height,
        "annotation_availability": [task],
        "image_integrity_verified": image_integrity_verified,
    }
    for key in (
        "leakage_group_id",
        "disease_id",
        "gold_diagnosis",
        "source_dataset",
        "image_sha256",
        "schema_version",
        "quality_status",
    ):
        value = row.get(key)
        if isinstance(value, str) and value:
            record[key] = value
    return record


def _inject_image_and_validate_messages(
    raw_messages: object,
    *,
    image: Image.Image,
    prompt: str,
    target: str,
) -> list[dict[str, object]]:
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
        raise ValueError("messages must be a user/assistant sequence")
    if len(raw_messages) != 2:
        raise ValueError("E3 SFT rows must contain exactly user and assistant")

    messages: list[dict[str, object]] = []
    image_items = 0
    user_texts: list[str] = []
    assistant_texts: list[str] = []
    for message_index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, Mapping):
            raise ValueError("Each message must be a mapping")
        expected_role = "user" if message_index == 0 else "assistant"
        role = _non_empty_string(raw_message.get("role"), "message.role")
        if role != expected_role:
            raise ValueError("E3 messages must be ordered user then assistant")
        raw_content = raw_message.get("content")
        if not isinstance(raw_content, Sequence) or isinstance(
            raw_content, (str, bytes)
        ):
            raise ValueError("message.content must be a sequence")
        content: list[dict[str, object]] = []
        for raw_item in raw_content:
            if not isinstance(raw_item, Mapping):
                raise ValueError("Message content items must be mappings")
            item_type = _non_empty_string(raw_item.get("type"), "content.type")
            if item_type == "image":
                if role != "user":
                    raise ValueError("Only the user message may contain an image")
                image_items += 1
                content.append({"type": "image", "image": image})
            elif item_type == "text":
                text = _non_empty_string(raw_item.get("text"), "content.text")
                content.append({"type": "text", "text": text})
                if role == "user":
                    user_texts.append(text)
                else:
                    assistant_texts.append(text)
            else:
                raise ValueError(f"Unsupported E3 message content type: {item_type}")
        messages.append({"role": role, "content": content})

    if image_items != 1:
        raise ValueError("E3 SFT rows must contain exactly one user image marker")
    if user_texts != [prompt]:
        raise ValueError("User message differs from prompt")
    if assistant_texts != [target]:
        raise ValueError("Assistant message differs from target_text")
    return messages


def _verify_text_sha256(text: str, expected: object, *, field: str) -> str:
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if actual != expected:
        raise ValueError(
            f"{field} differs from its frozen dataset text: "
            f"expected {expected}, observed {actual}"
        )
    return expected


def _target_integrity(
    row: Mapping[object, object],
    *,
    target: str,
    reference: DatasetReference,
) -> tuple[str, str]:
    published_digest = row.get("target_sha256")
    if published_digest is not None:
        return (
            _verify_text_sha256(
                target,
                published_digest,
                field="target_sha256",
            ),
            "published_sha256",
        )
    if reference.config != "diagnosis" or reference.split != "sft_dev":
        raise ValueError("Frozen E3 train row is missing target_sha256")
    gold_diagnosis = _non_empty_string(
        row.get("gold_diagnosis"),
        "gold_diagnosis",
    )
    if target != gold_diagnosis:
        raise ValueError("Historical sft_dev target_text differs from gold_diagnosis")
    return hashlib.sha256(target.encode("utf-8")).hexdigest(), (
        "gold_diagnosis_equality"
    )


def _decoded_image(value: object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _image_from_bytes(bytes(value))
    if isinstance(value, Mapping):
        encoded = value.get("bytes")
        if isinstance(encoded, (bytes, bytearray, memoryview)):
            return _image_from_bytes(bytes(encoded))
        path = value.get("path")
        if isinstance(path, str) and path:
            return _image_from_path(Path(path))
    if isinstance(value, (str, Path)):
        return _image_from_path(Path(value))
    raise ValueError("image must be a decoded PIL image, bytes, or image path")


def _disable_image_decoding(rows: DatasetLike) -> DatasetLike:
    """Keep encoded image bytes available for row-level SHA-256 validation."""

    cast_column = getattr(rows, "cast_column", None)
    columns = getattr(rows, "column_names", None)
    if not callable(cast_column) or not isinstance(columns, Sequence):
        return rows
    if "image" not in columns:
        raise ValueError("Frozen SFT dataset has no image column")
    try:
        from datasets import Image as DatasetImage
    except ImportError as exc:
        raise RuntimeError("The training extra with datasets is required") from exc
    converted = cast_column("image", DatasetImage(decode=False))
    if not hasattr(converted, "__len__") or not hasattr(converted, "__getitem__"):
        raise TypeError("cast_column returned a non-indexable dataset")
    return cast(DatasetLike, converted)


def _verify_encoded_image_sha256(value: object, expected: object) -> bool:
    """Verify original encoded bytes when the frozen row exposes a digest.

    Hugging Face datasets loaded by :func:`load_training_datasets` are cast to
    ``Image(decode=False)``, so production rows take the strict path.  Direct
    test/programmatic adapters may provide a decoded PIL object; that path is
    still valid but explicitly reports that byte-level verification was not
    possible.
    """

    if not isinstance(expected, str) or not expected:
        return False
    invalid_character = any(
        character not in "0123456789abcdef" for character in expected
    )
    if len(expected) != 64 or invalid_character:
        raise ValueError("image_sha256 must be a lowercase SHA-256 digest")
    encoded = _encoded_image_bytes(value)
    if encoded is None:
        return False
    actual = hashlib.sha256(encoded).hexdigest()
    if actual != expected:
        raise ValueError(
            "Encoded image SHA-256 differs from the frozen dataset row: "
            f"expected {expected}, observed {actual}"
        )
    return True


def _encoded_image_bytes(value: object) -> bytes | None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, Mapping):
        encoded = value.get("bytes")
        if isinstance(encoded, (bytes, bytearray, memoryview)):
            return bytes(encoded)
        path = value.get("path")
        if isinstance(path, str) and path:
            return Path(path).expanduser().read_bytes()
    if isinstance(value, (str, Path)):
        return Path(value).expanduser().read_bytes()
    return None


def _image_from_bytes(encoded: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(encoded)) as opened:
            opened.load()
            return opened.copy()
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot decode dataset image bytes") from exc


def _image_from_path(path: Path) -> Image.Image:
    try:
        with Image.open(path.expanduser()) as opened:
            opened.load()
            return opened.copy()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot decode dataset image path: {path}") from exc


def _default_task(reference: DatasetReference) -> str:
    if reference.config == "diagnosis":
        return "diagnosis"
    raise ValueError(f"Dataset {reference.identity} does not expose a task column")


def _non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: object, *, default: str) -> str:
    return value if isinstance(value, str) and value else default
