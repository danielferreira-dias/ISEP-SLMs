"""Typed domain objects shared by the training pipeline.

This module intentionally contains no framework-specific training objects.  The
immutable records form the boundary between data preparation, phase rendering,
and the Unsloth backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from PIL import Image

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class TrainingPhaseName(StrEnum):
    """Runnable supervised phases in the training pipeline."""

    E1_LABEL = "e1_label"
    E2_SKINCON = "e2_skincon"


class VisionTuningProfile(StrEnum):
    """Controlled E1 visual-adaptation conditions."""

    FROZEN_VISION = "frozen_vision"
    UNSLOTH_ALL = "unsloth_all"


class ReleaseSubset(StrEnum):
    """Materialized views of a frozen data release."""

    SFT_TRAIN = "sft_train"
    SFT_DEV = "sft_dev"
    DEV_PANEL = "dev_panel"


@dataclass(frozen=True, slots=True)
class TaxonomyClass:
    """One canonical diagnostic class."""

    disease_id: str
    label: str


@dataclass(frozen=True, slots=True)
class Taxonomy:
    """Ordered closed taxonomy used for prompts and output validation."""

    taxonomy_id: str
    classes: tuple[TaxonomyClass, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        """Return canonical labels in their frozen display order."""

        return tuple(item.label for item in self.classes)

    @property
    def disease_ids(self) -> tuple[str, ...]:
        """Return canonical disease identifiers."""

        return tuple(item.disease_id for item in self.classes)


@dataclass(frozen=True, slots=True)
class LabeledImageSample:
    """One decoded image and its minimum supervised E1 metadata."""

    sample_id: str
    leakage_group_id: str
    disease_id: str
    label: str
    source: str
    image: Image.Image
    subset: str = ""
    image_width: int = 0
    image_height: int = 0
    pixel_count: int = 0
    resized_width: int = 0
    resized_height: int = 0


@dataclass(frozen=True, slots=True)
class ImageMessageContent:
    """Image item in a Hugging Face multimodal chat message."""

    image: Image.Image
    type: Literal["image"] = "image"

    def as_record(self) -> dict[str, object]:
        """Convert the item to the TRL/Unsloth conversation representation."""

        return {"type": self.type, "image": self.image}


@dataclass(frozen=True, slots=True)
class TextMessageContent:
    """Text item in a Hugging Face multimodal chat message."""

    text: str
    type: Literal["text"] = "text"

    def as_record(self) -> dict[str, object]:
        """Convert the item to the TRL/Unsloth conversation representation."""

        return {"type": self.type, "text": self.text}


type MessageContent = ImageMessageContent | TextMessageContent


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One typed user or assistant message."""

    role: Literal["user", "assistant"]
    content: tuple[MessageContent, ...]

    def as_record(self) -> dict[str, object]:
        """Convert the message to a serializable chat record."""

        return {
            "role": self.role,
            "content": [item.as_record() for item in self.content],
        }


@dataclass(frozen=True, slots=True)
class FormattedExample:
    """A phase-rendered example ready for a vision chat collator."""

    sample_id: str
    leakage_group_id: str
    disease_id: str
    label: str
    messages: tuple[ChatMessage, ...]
    subset: str = ""
    source: str = ""
    image_width: int = 0
    image_height: int = 0
    pixel_count: int = 0
    resized_width: int = 0
    resized_height: int = 0

    def as_record(self) -> dict[str, object]:
        """Convert the example to a Hugging Face Dataset-compatible record."""

        record: dict[str, object] = {
            "sample_id": self.sample_id,
            "leakage_group_id": self.leakage_group_id,
            "disease_id": self.disease_id,
            "label": self.label,
            "messages": [message.as_record() for message in self.messages],
        }
        if self.image_width > 0:
            record.update(
                {
                    "phase": TrainingPhaseName.E1_LABEL.value,
                    "task": "diagnosis",
                    "split": self.subset,
                    "source": self.source,
                    "image_width": self.image_width,
                    "image_height": self.image_height,
                    "pixel_count": self.pixel_count,
                    "resized_width": self.resized_width,
                    "resized_height": self.resized_height,
                    "annotation_availability": ["diagnosis"],
                }
            )
        return record


@dataclass(frozen=True, slots=True)
class ReleaseAudit:
    """Verified cardinalities and integrity state of a data release."""

    release_id: str
    source_image_count: int
    source_group_count: int
    class_count: int
    source_count: int
    train_image_count: int
    train_group_count: int
    dev_image_count: int
    dev_group_count: int
    dev_panel_image_count: int
    dev_panel_group_count: int
    group_overlap_count: int
    assignment_sha256: str
    source_release_sha256: str


@dataclass(frozen=True, slots=True)
class PreparedRelease:
    """Paths and audit results for a frozen, ID-only training release."""

    root: Path
    release_manifest_path: Path
    assignments_path: Path
    train_manifest_path: Path
    dev_manifest_path: Path
    dev_panel_manifest_path: Path
    audit: ReleaseAudit
