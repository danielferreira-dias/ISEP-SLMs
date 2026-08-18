"""Versioned, source-traceable dermatology terminology for E3 Stage A."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)


def _tuple_from_json(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


type StringTuple = Annotated[tuple[str, ...], BeforeValidator(_tuple_from_json)]


class ImageModality(StrEnum):
    """Image modalities that Stage A may distinguish without metadata leakage."""

    CLINICAL_PHOTO = "clinical_photo"
    DERMOSCOPY = "dermoscopy"
    UNKNOWN = "unknown"


class ConceptCategory(StrEnum):
    """Non-diagnostic visual descriptor families in the frozen lexicon."""

    PRIMARY_LESION = "primary_lesion"
    SURFACE_CHANGE = "surface_change"
    COLOR = "color"
    BORDER = "border"
    SHAPE = "shape"
    PROFILE = "profile"
    ARRANGEMENT = "arrangement"
    DISTRIBUTION = "distribution"
    DERMOSCOPY_ELEMENT = "dermoscopy_element"


class ImageObservability(StrEnum):
    """Whether a concept can be asserted from an image alone."""

    DIRECT = "direct"
    CONDITIONAL = "conditional"


def _concept_category_from_json(value: object) -> object:
    return ConceptCategory(value) if isinstance(value, str) else value


def _image_observability_from_json(value: object) -> object:
    return ImageObservability(value) if isinstance(value, str) else value


type ConceptCategoryValue = Annotated[
    ConceptCategory,
    BeforeValidator(_concept_category_from_json),
]
type ImageObservabilityValue = Annotated[
    ImageObservability,
    BeforeValidator(_image_observability_from_json),
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class TerminologySource(_FrozenModel):
    """One authoritative source used to curate the lexicon."""

    source_id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str = Field(min_length=1)
    organization: str = Field(min_length=1)
    publication_year: int | None = Field(default=None, ge=1900, le=2100)
    url: str = Field(pattern=r"^https://")
    accessed_on: date
    role: str = Field(min_length=1)


class TerminologyConcept(_FrozenModel):
    """One normalized answer-blind visual concept."""

    concept_id: str = Field(pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
    label: str = Field(min_length=1)
    category: ConceptCategoryValue
    definition: str = Field(min_length=1)
    allowed_modalities: Annotated[
        tuple[Literal["clinical_photo", "dermoscopy"], ...],
        BeforeValidator(_tuple_from_json),
    ] = Field(min_length=1)
    image_observability: ImageObservabilityValue
    source_ids: StringTuple = Field(min_length=1)
    usage_note: str | None = None

    @model_validator(mode="after")
    def _validate_modalities(self) -> TerminologyConcept:
        if len(self.allowed_modalities) != len(set(self.allowed_modalities)):
            raise ValueError("allowed_modalities must be unique")
        return self


class DermatologyTerminology(_FrozenModel):
    """Frozen terminology contract injected into the Stage-A teacher prompt."""

    schema_version: Literal[1]
    lexicon_id: str = Field(pattern=r"^[a-z0-9_]+_v[0-9]+$")
    language: Literal["en"]
    scope: str = Field(min_length=1)
    sources: Annotated[
        tuple[TerminologySource, ...], BeforeValidator(_tuple_from_json)
    ] = Field(min_length=1)
    concepts: Annotated[
        tuple[TerminologyConcept, ...], BeforeValidator(_tuple_from_json)
    ] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_identity_and_references(self) -> DermatologyTerminology:
        source_ids = tuple(item.source_id for item in self.sources)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("terminology source IDs must be unique")
        concept_ids = tuple(item.concept_id for item in self.concepts)
        if len(concept_ids) != len(set(concept_ids)):
            raise ValueError("terminology concept IDs must be unique")
        labels = tuple(item.label.casefold() for item in self.concepts)
        if len(labels) != len(set(labels)):
            raise ValueError("terminology concept labels must be unique")
        known_sources = set(source_ids)
        unknown_sources = sorted(
            {
                source_id
                for concept in self.concepts
                for source_id in concept.source_ids
                if source_id not in known_sources
            }
        )
        if unknown_sources:
            raise ValueError(
                "terminology concepts reference unknown sources: "
                + ", ".join(unknown_sources)
            )
        return self

    @property
    def concept_ids(self) -> tuple[str, ...]:
        return tuple(item.concept_id for item in self.concepts)

    def concept(self, concept_id: str) -> TerminologyConcept | None:
        return next(
            (item for item in self.concepts if item.concept_id == concept_id),
            None,
        )

    def prompt_catalog_json(self) -> str:
        """Return the compact, diagnosis-free catalogue injected into Stage A."""

        payload = [
            {
                "allowed_modalities": list(item.allowed_modalities),
                "category": item.category.value,
                "concept_id": item.concept_id,
                "definition": item.definition,
                "image_observability": item.image_observability.value,
                "label": item.label,
                "usage_note": item.usage_note,
            }
            for item in self.concepts
        ]
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def audit_observation(
        self,
        *,
        concept_id: str,
        concept_label: str,
        image_modality: ImageModality,
    ) -> tuple[str, ...]:
        """Return deterministic normalization violations for one observation."""

        concept = self.concept(concept_id)
        if concept is None:
            return ("stage_a_contains_unknown_terminology_concept",)
        reasons: list[str] = []
        if concept_label != concept.label:
            reasons.append("stage_a_terminology_label_mismatch")
        if (
            image_modality is ImageModality.CLINICAL_PHOTO
            and "clinical_photo" not in concept.allowed_modalities
        ):
            reasons.append("stage_a_concept_incompatible_with_image_modality")
        if image_modality is ImageModality.DERMOSCOPY and (
            "dermoscopy" not in concept.allowed_modalities
        ):
            reasons.append("stage_a_concept_incompatible_with_image_modality")
        if image_modality is ImageModality.UNKNOWN and (
            concept.category is ConceptCategory.DERMOSCOPY_ELEMENT
        ):
            reasons.append("stage_a_dermoscopy_concept_requires_confirmed_modality")
        return tuple(reasons)


def load_dermatology_terminology(path: str | Path) -> DermatologyTerminology:
    """Load and strictly validate one immutable terminology YAML resource."""

    resource_path = Path(path)
    document = yaml.safe_load(resource_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Terminology resource must be an object: {resource_path}")
    return DermatologyTerminology.model_validate(document)


def terminology_resource_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
