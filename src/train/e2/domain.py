"""Typed human-supervision domain for E2 diagnosis, morphology, and captions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from PIL import Image

from src.train.domain import ChatMessage


class E2TaskName(StrEnum):
    """Human-authored task configurations admitted to E2."""

    DIAGNOSIS = "diagnosis"
    MORPHOLOGY = "morphology"
    CAPTION = "caption"


@dataclass(frozen=True, slots=True)
class SkinConOntology:
    """Ordered closed SKINCON concept vocabulary."""

    ontology_id: str
    concepts: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject empty or ambiguous concept vocabularies."""

        if not self.ontology_id:
            raise ValueError("SKINCON ontology ID must not be empty")
        if not self.concepts or any(not item for item in self.concepts):
            raise ValueError("SKINCON concepts must not be empty")
        if len(set(self.concepts)) != len(self.concepts):
            raise ValueError("SKINCON concepts must be unique")


@dataclass(frozen=True, slots=True)
class MorphologyTarget:
    """Complete multilabel target derived from one human SKINCON row."""

    positive_concepts: tuple[str, ...]
    all_concepts_annotated: bool

    def canonical_json(self) -> str:
        """Return the byte-stable target expected by the student."""

        return json.dumps(
            {
                "positive_concepts": list(self.positive_concepts),
                "all_concepts_annotated": self.all_concepts_annotated,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class E2HumanSample:
    """One decoded human-supervised E2 task row."""

    sample_id: str
    leakage_group_id: str
    task: E2TaskName
    source: str
    image: Image.Image
    prompt: str
    target_text: str
    disease_id: str | None = None
    label: str | None = None
    morphology: MorphologyTarget | None = None
    source_caption_sha256: str | None = None
    caption_transform_version: str | None = None
    subset: str = ""
    image_sha256: str = ""
    image_width: int = 0
    image_height: int = 0
    pixel_count: int = 0
    resized_width: int = 0
    resized_height: int = 0
    annotation_availability: tuple[E2TaskName, ...] = ()


@dataclass(frozen=True, slots=True)
class E2FormattedExample:
    """One E2 chat conversation ready for the Unsloth vision collator."""

    sample_id: str
    leakage_group_id: str
    task: E2TaskName
    target_text: str
    subset: str
    source: str
    image_sha256: str
    image_width: int
    image_height: int
    pixel_count: int
    resized_width: int
    resized_height: int
    annotation_availability: tuple[E2TaskName, ...]
    messages: tuple[ChatMessage, ...]

    def as_record(self) -> dict[str, object]:
        """Return the minimal record visible to the trainer."""

        return {
            "sample_id": self.sample_id,
            "leakage_group_id": self.leakage_group_id,
            "task_id": self.task.value,
            "phase": "e2_skincon",
            "task": self.task.value,
            "split": self.subset,
            "source": self.source,
            "label": self.target_text,
            "image_sha256": self.image_sha256,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "pixel_count": self.pixel_count,
            "resized_width": self.resized_width,
            "resized_height": self.resized_height,
            "annotation_availability": [
                item.value for item in self.annotation_availability
            ],
            "messages": [message.as_record() for message in self.messages],
        }


@dataclass(frozen=True, slots=True)
class E2Shard:
    """One immutable Parquet shard declared by the release manifest."""

    path: Path
    task: E2TaskName
    subset: str
    rows: int
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class E2ReleaseAudit:
    """Validated identity and shard inventory for ISEPDistillDataset E2."""

    root: Path
    release_manifest_path: Path
    ontology_path: Path
    release_id: str
    schema_version: str
    manifest_sha256: str
    ontology_sha256: str
    shards: tuple[E2Shard, ...]
    diagnosis_train: int
    diagnosis_dev: int
    morphology_train: int
    morphology_dev: int
    caption_train: int
    caption_dev: int
    config_schema_versions: tuple[tuple[E2TaskName, str], ...]
    annotation_availability: tuple[tuple[str, tuple[E2TaskName, ...]], ...]
    ontology: SkinConOntology

    @property
    def training_contract_sha256(self) -> str:
        """Return the immutable dataset identity used by checkpoints."""

        return self.manifest_sha256

    def schema_version_for(self, task: E2TaskName) -> str:
        """Return the frozen row schema used by one configuration."""

        versions = dict(self.config_schema_versions)
        try:
            return versions[task]
        except KeyError as exc:
            raise ValueError(f"E2 release has no schema for {task.value}") from exc

    def annotation_availability_by_image(
        self,
    ) -> dict[str, tuple[E2TaskName, ...]]:
        """Return task availability keyed by immutable image SHA-256."""

        return dict(self.annotation_availability)
