"""Strict data contracts for the two-stage E3 teacher generation pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class RecordStatus(StrEnum):
    """Scientific review status written by Stage A and Stage B."""

    OK = "ok"
    ERROR = "error"
    REJECTED = "rejected"


class ImageModality(StrEnum):
    """Acquisition type declared by Stage A."""

    CLINICAL = "clinical"
    DERMOSCOPY = "dermoscopy"
    UNKNOWN = "unknown"


class ObservationStatus(StrEnum):
    """Evidence states that do not invent unobserved absence."""

    PRESENT = "present"
    ABSENT_IN_OBSERVED_SCOPE = "absent_in_observed_scope"
    UNCERTAIN = "uncertain"
    NOT_ASSESSABLE = "not_assessable"
    NOT_SHOWN = "not_shown"


class ConfidenceLevel(StrEnum):
    """Closed confidence vocabulary for observations and diagnoses."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class AnchorEvidenceStatus(StrEnum):
    """How strongly the image and Stage A support the private gold anchor."""

    SUPPORTED = "supported"
    WEAK = "weak"
    UNSUPPORTED = "unsupported"


class ResponsePolicy(StrEnum):
    """Response behavior derived from whether the image is evaluable."""

    ANSWER_DIFFERENTIAL = "ANSWER_DIFFERENTIAL"
    REQUEST_NEW_IMAGE = "REQUEST_NEW_IMAGE"


type ObservationConcept = Literal[
    "image.anatomic_site",
    "lesion.count",
    "lesion.primary",
    "lesion.size",
    "lesion.color",
    "lesion.shape",
    "lesion.symmetry",
    "lesion.border_demarcation",
    "lesion.border_regularity",
    "lesion.profile",
    "lesion.surface_texture",
    "lesion.secondary_change",
    "lesion.configuration",
    "lesion.distribution",
    "lesion.additional_feature",
]

type QualityDefect = Literal[
    "blur",
    "low_resolution",
    "poor_lighting",
    "color_cast",
    "occlusion",
    "compression_artifact",
    "tight_crop",
    "other",
]

type MissingDiscriminator = Literal[
    "duration_and_evolution",
    "symptoms",
    "palpation",
    "other_body_sites",
    "dermoscopy",
    "scale_or_ruler",
    "closer_image",
    "overview_image",
    "clinical_history",
    "histopathology",
]

_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid")


def _validate_complete_prose(value: str, *, field_name: str) -> str:
    """Reject multiline, padded, too-short, or visibly truncated prose."""
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{field_name} must be one paragraph")
    if len(value.split()) < 5:
        raise ValueError(f"{field_name} is too short")
    if not value.endswith((".", "?", "!")):
        raise ValueError(f"{field_name} must end at a sentence boundary")
    return value


class ImageAssessment(BaseModel):
    """What can and cannot be assessed from the supplied image."""

    model_config = _MODEL_CONFIG

    is_evaluable: bool
    image_modality: ImageModality
    views_available: list[str]
    quality_defects: list[QualityDefect]
    has_anatomic_overview: bool
    has_scale: bool
    has_lateral_profile: bool
    distribution_assessability: Literal["full", "partial", "not_assessable"]
    color_reliability: Literal["reliable", "limited", "not_assessable"]


class Observation(BaseModel):
    """One atomic, image-grounded clinical observation from Stage A."""

    model_config = _MODEL_CONFIG

    id: str = Field(
        pattern=r"^obs_[0-9]{3}$",
        description=(
            "Sequential observation identifier with exactly three digits; "
            "the tenth identifier is obs_010, never obs_0010."
        ),
    )
    concept_id: ObservationConcept
    value: str = Field(min_length=1)
    status: ObservationStatus
    scope: str = Field(min_length=1)
    confidence: ConfidenceLevel
    evidence_region: str | None = Field(min_length=1)

    @model_validator(mode="after")
    def _present_observation_requires_region(self) -> Observation:
        if self.status is ObservationStatus.PRESENT and self.evidence_region is None:
            raise ValueError("Present observations require an evidence_region")
        return self


class StageAMorphology(BaseModel):
    """Answer-blind Stage A target containing visual findings only."""

    model_config = _MODEL_CONFIG

    image_assessment: ImageAssessment
    dominant_visual_pattern: str = Field(min_length=1)
    observations: list[Observation]
    not_assessable_features: list[str]
    clinical_caption: str = Field(min_length=1)

    @field_validator("clinical_caption")
    @classmethod
    def _caption_must_be_complete(cls, value: str) -> str:
        return _validate_complete_prose(value, field_name="clinical_caption")

    @model_validator(mode="after")
    def _validate_observations(self) -> StageAMorphology:
        ids = tuple(item.id for item in self.observations)
        if len(ids) != len(set(ids)):
            raise ValueError("Observation IDs must be unique")

        duplicates = tuple(
            (item.concept_id, item.value.casefold(), item.status)
            for item in self.observations
        )
        if len(duplicates) != len(set(duplicates)):
            raise ValueError("Stage A contains duplicate observations")

        if self.image_assessment.is_evaluable and not self.observations:
            raise ValueError("Evaluable images require at least one observation")
        if not self.image_assessment.is_evaluable and any(
            item.status is ObservationStatus.PRESENT for item in self.observations
        ):
            raise ValueError(
                "A non-evaluable image cannot contain present observations"
            )
        return self


class DifferentialComparison(BaseModel):
    """Why the gold diagnosis is favoured over one plausible alternative."""

    model_config = _MODEL_CONFIG

    alternative: str = Field(min_length=1)
    features_favoring_diagnosis: list[str]
    features_favoring_alternative: list[str]
    comparison: str = Field(min_length=1)

    @field_validator("comparison")
    @classmethod
    def _comparison_must_be_complete(cls, value: str) -> str:
        return _validate_complete_prose(value, field_name="comparison")

    @model_validator(mode="after")
    def _reject_conflicting_evidence(self) -> DifferentialComparison:
        diagnosis = self.features_favoring_diagnosis
        alternative = self.features_favoring_alternative
        if len(diagnosis) != len(set(diagnosis)):
            raise ValueError("Diagnosis-favoring observation IDs must be unique")
        if len(alternative) != len(set(alternative)):
            raise ValueError("Alternative-favoring observation IDs must be unique")
        if set(diagnosis).intersection(alternative):
            raise ValueError(
                "An observation cannot favour both sides of one comparison"
            )
        return self


class StageBReasoning(BaseModel):
    """Gold-anchored facts and the teacher's student-facing justification."""

    model_config = _MODEL_CONFIG

    anchor_evidence_status: AnchorEvidenceStatus
    annotation_conflict: bool
    annotation_conflict_reason: str | None = Field(min_length=1)
    diagnostic_confidence: ConfidenceLevel
    diagnosis: str = Field(min_length=1)
    differential_comparisons: list[DifferentialComparison] = Field(max_length=4)
    limitations: list[MissingDiscriminator]
    response_policy: ResponsePolicy
    non_evaluable_reason: str | None = Field(min_length=1)
    clinical_reasoning: str = Field(min_length=1)

    @field_validator("clinical_reasoning")
    @classmethod
    def _clinical_reasoning_must_be_complete(cls, value: str) -> str:
        return _validate_complete_prose(value, field_name="clinical_reasoning")

    @model_validator(mode="after")
    def _validate_structure(self) -> StageBReasoning:
        diagnosis = self.diagnosis.strip().casefold()
        alternatives = tuple(
            item.alternative.strip().casefold()
            for item in self.differential_comparisons
        )
        if any(not alternative for alternative in alternatives):
            raise ValueError("Differential alternatives must not be blank")
        if diagnosis in alternatives:
            raise ValueError("The gold diagnosis cannot be its own alternative")
        if len(alternatives) != len(set(alternatives)):
            raise ValueError("Differential alternatives must be unique")

        if len(self.limitations) != len(set(self.limitations)):
            raise ValueError("Limitations must be unique")

        if self.annotation_conflict and self.annotation_conflict_reason is None:
            raise ValueError("annotation_conflict=true requires a reason")
        if not self.annotation_conflict and self.annotation_conflict_reason is not None:
            raise ValueError("annotation_conflict=false requires a null reason")

        if self.response_policy is ResponsePolicy.ANSWER_DIFFERENTIAL:
            if not self.differential_comparisons:
                raise ValueError("ANSWER_DIFFERENTIAL requires a comparison")
            if self.non_evaluable_reason is not None:
                raise ValueError("ANSWER_DIFFERENTIAL requires a null reason")
        else:
            if self.differential_comparisons:
                raise ValueError("REQUEST_NEW_IMAGE cannot contain comparisons")
            if self.non_evaluable_reason is None:
                raise ValueError("REQUEST_NEW_IMAGE requires a reason")
        return self


class ManifestRow(BaseModel):
    """One line of the optional local-file generation manifest."""

    model_config = _MODEL_CONFIG

    sample_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    gold_diagnosis: str = Field(min_length=1)

    @field_validator("sample_id", "gold_diagnosis")
    @classmethod
    def _strip_nonempty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class ImageSample(BaseModel):
    """Stage A input: identity and image only. No gold diagnosis."""

    model_config = _MODEL_CONFIG

    sample_id: str = Field(min_length=1)
    image_path: Path


class UsageInfo(BaseModel):
    """Provider token usage plus reported or locally estimated request cost."""

    model_config = _MODEL_CONFIG

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    thoughts_tokens: int | None = None
    request_attempts: int | None = Field(default=None, ge=1)
    cost: float | None = None
    cost_currency: Literal["USD"] | None = None
    cost_basis: Literal["provider_reported", "estimated_list_price"] | None = None

    @model_validator(mode="after")
    def _cost_metadata_is_complete(self) -> UsageInfo:
        if self.cost is None and (
            self.cost_currency is not None or self.cost_basis is not None
        ):
            raise ValueError("Cost metadata requires a numeric cost")
        return self


class ImagePreprocessingInfo(BaseModel):
    """Hash-addressed record of the exact image sent to the teacher."""

    model_config = _MODEL_CONFIG

    source_pixel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    source_mode: str = Field(min_length=1)
    exif_transposed: bool
    icc_profile_present: bool
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_width: int = Field(gt=0)
    output_height: int = Field(gt=0)
    output_media_type: Literal["image/jpeg"]
    max_side: int = Field(gt=0)
    jpeg_quality: int = Field(ge=1, le=100)


class GenerationProvenance(BaseModel):
    """Immutable identity of one teacher request, excluding secrets and images."""

    model_config = _MODEL_CONFIG

    attempt_id: str = Field(min_length=1)
    created_at: datetime
    provider: str = Field(min_length=1)
    teacher_name: str = Field(min_length=1)
    teacher_model: str = Field(min_length=1)
    seed: int
    max_output_tokens: int | None = Field(default=None, gt=0)
    reasoning_effort: Literal["high", "medium", "low"] | None = None
    reasoning_excluded: bool | None = None
    transport_retry_max_attempts: int | None = Field(default=None, ge=1)
    transport_retry_status_codes: tuple[int, ...] | None = None
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    finish_reason: str | None = None
    native_finish_reason: str | None = None


class StageAFileRow(BaseModel):
    """One Stage A JSONL record. It must never contain the gold diagnosis."""

    model_config = _MODEL_CONFIG

    sample_id: str
    status: RecordStatus
    morphology: StageAMorphology | None = None
    error: str | None = None
    usage: UsageInfo | None = None
    teacher: str
    image_path: str
    image_preprocessing: ImagePreprocessingInfo | None = None
    provenance: GenerationProvenance | None = None

    @model_validator(mode="after")
    def _validate_status_payload(self) -> StageAFileRow:
        if self.status is RecordStatus.OK and (
            self.morphology is None or self.error is not None
        ):
            raise ValueError("Stage A ok requires morphology and no error")
        if self.status is RecordStatus.ERROR and (
            self.morphology is not None or self.error is None
        ):
            raise ValueError("Stage A error requires an error and no morphology")
        return self


class StageBFileRow(BaseModel):
    """One Stage B JSONL record. Rejected rows remain auditable."""

    model_config = _MODEL_CONFIG

    sample_id: str
    status: RecordStatus
    reasoning: StageBReasoning | None = None
    reasons: tuple[str, ...] = ()
    error: str | None = None
    usage: UsageInfo | None = None
    teacher: str
    gold_diagnosis: str
    stage_a_sample_id: str
    image_path: str
    image_preprocessing: ImagePreprocessingInfo | None = None
    provenance: GenerationProvenance | None = None

    @model_validator(mode="after")
    def _validate_status_payload(self) -> StageBFileRow:
        if self.status is RecordStatus.OK:
            if self.reasoning is None or self.error is not None or self.reasons:
                raise ValueError("Stage B ok requires reasoning and no failures")
        elif self.status is RecordStatus.REJECTED:
            if self.reasoning is None or not self.reasons or self.error is not None:
                raise ValueError("Stage B rejected requires reasoning and reason codes")
        elif self.reasoning is not None or self.error is None:
            raise ValueError("Stage B error requires an error and no reasoning")
        return self


def parse_stage_a(raw: dict[str, object]) -> StageAMorphology:
    """Validate untrusted Stage A JSON from the teacher."""
    return StageAMorphology.model_validate(raw)


def parse_stage_b(raw: dict[str, object]) -> StageBReasoning:
    """Validate untrusted Stage B JSON from the teacher."""
    return StageBReasoning.model_validate(raw)


def teacher_output_schema(stage_key: str) -> dict[str, object]:
    """Return the canonical JSON Schema for one teacher stage."""
    models: dict[str, type[BaseModel]] = {
        "A": StageAMorphology,
        "B": StageBReasoning,
    }
    try:
        model = models[stage_key]
    except KeyError as exc:
        raise KeyError(f"Unknown teacher schema stage: {stage_key}") from exc
    return model.model_json_schema()


def image_sample_from_manifest(row: ManifestRow, *, project_root: Path) -> ImageSample:
    """Drop the gold label and resolve a local manifest image path."""
    path = Path(row.image_path)
    if not path.is_absolute():
        path = project_root / path
    return ImageSample(sample_id=row.sample_id, image_path=path)
