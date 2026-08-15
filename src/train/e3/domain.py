"""Strict external-data contract for E3 teacher clinical targets."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)


def _tuple_from_json(value: object) -> object:
    """Convert JSON arrays to immutable tuples before strict validation."""

    return tuple(value) if isinstance(value, list) else value


type StringTuple = Annotated[tuple[str, ...], BeforeValidator(_tuple_from_json)]


def _observation_status_from_json(value: object) -> object:
    if isinstance(value, str):
        return ObservationStatus(value)
    return value


def _confidence_from_json(value: object) -> object:
    if isinstance(value, str):
        return ConfidenceLevel(value)
    return value


def _risk_from_json(value: object) -> object:
    if isinstance(value, str):
        return RiskLevel(value)
    return value


def _action_from_json(value: object) -> object:
    if isinstance(value, str):
        return ClinicalAction(value)
    return value


class _StrictModel(BaseModel):
    """Immutable Pydantic boundary for generated or human-authored targets."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class ObservationStatus(StrEnum):
    """Permitted evidence states without inventing unobserved absence."""

    PRESENT = "present"
    ABSENT_IN_OBSERVED_SCOPE = "absent_in_observed_scope"
    UNCERTAIN = "uncertain"
    NOT_ASSESSABLE = "not_assessable"
    NOT_SHOWN = "not_shown"


type ObservationStatusValue = Annotated[
    ObservationStatus, BeforeValidator(_observation_status_from_json)
]


class ConfidenceLevel(StrEnum):
    """Closed confidence vocabulary for evidence and diagnoses."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


type ConfidenceValue = Annotated[
    ConfidenceLevel, BeforeValidator(_confidence_from_json)
]


class RiskLevel(StrEnum):
    """Closed risk vocabulary for a missed differential."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


type RiskValue = Annotated[RiskLevel, BeforeValidator(_risk_from_json)]


class ClinicalAction(StrEnum):
    """Actions that E3 may render without issuing uncontrolled advice."""

    DIAGNOSE_PROVISIONALLY = "DIAGNOSE_PROVISIONALLY"
    REQUEST_OVERVIEW_IMAGE = "REQUEST_OVERVIEW_IMAGE"
    REQUEST_CLOSEUP_IMAGE = "REQUEST_CLOSEUP_IMAGE"
    REQUEST_SCALE_OR_PROFILE = "REQUEST_SCALE_OR_PROFILE"
    REQUEST_CLINICAL_CONTEXT = "REQUEST_CLINICAL_CONTEXT"
    REQUEST_DERMOSCOPY = "REQUEST_DERMOSCOPY"
    REQUEST_IN_PERSON_EXAM = "REQUEST_IN_PERSON_EXAM"
    RECOMMEND_CONFIRMATORY_TEST = "RECOMMEND_CONFIRMATORY_TEST"
    ABSTAIN_POOR_QUALITY = "ABSTAIN_POOR_QUALITY"
    ABSTAIN_OUT_OF_DOMAIN = "ABSTAIN_OUT_OF_DOMAIN"


type ClinicalActionValue = Annotated[ClinicalAction, BeforeValidator(_action_from_json)]


class ImageAssessment(_StrictModel):
    """What can and cannot be assessed from the supplied image."""

    is_evaluable: bool
    views_available: StringTuple
    quality_defects: StringTuple
    has_anatomic_overview: bool
    has_scale: bool
    has_lateral_profile: bool
    distribution_assessability: str = Field(min_length=1)
    color_reliability: str = Field(min_length=1)


class Observation(_StrictModel):
    """One image-grounded clinical observation."""

    id: str = Field(min_length=1)
    concept: str = Field(min_length=1)
    status: ObservationStatusValue
    provenance: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    confidence: ConfidenceValue
    evidence_region: str | None = None


class MissingDiscriminator(_StrictModel):
    """Information required to distinguish a differential diagnosis."""

    feature: str = Field(min_length=1)
    required_source: str = Field(min_length=1)


class DifferentialDiagnosis(_StrictModel):
    """One ranked diagnosis tied explicitly to Stage-A observations."""

    rank: int = Field(ge=1)
    disease_id: str = Field(pattern=r"^D[0-9]{3}$")
    supporting_observation_ids: StringTuple
    contradicting_observation_ids: StringTuple
    missing_discriminators: Annotated[
        tuple[MissingDiscriminator, ...], BeforeValidator(_tuple_from_json)
    ] = ()
    diagnostic_confidence: ConfidenceValue
    clinical_risk_if_missed: RiskValue

    @model_validator(mode="after")
    def _reject_conflicting_evidence_links(self) -> DifferentialDiagnosis:
        overlap = set(self.supporting_observation_ids).intersection(
            self.contradicting_observation_ids
        )
        if overlap:
            raise ValueError(
                "An observation cannot both support and contradict a diagnosis"
            )
        return self


class StructuredClinicalTarget(_StrictModel):
    """Accepted Stage-A/B record used by E3 structured and open tasks."""

    image_assessment: ImageAssessment
    dominant_visual_pattern: str = Field(min_length=1)
    observations: Annotated[
        tuple[Observation, ...], BeforeValidator(_tuple_from_json)
    ] = Field(min_length=1)
    not_assessable_features: StringTuple = ()
    differential: Annotated[
        tuple[DifferentialDiagnosis, ...], BeforeValidator(_tuple_from_json)
    ] = Field(min_length=1, max_length=6)
    action: ClinicalActionValue
    action_urgency: str | None = None
    requested_information: str | None = None
    concise_clinical_rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_evidence_graph(self) -> StructuredClinicalTarget:
        observation_ids = tuple(observation.id for observation in self.observations)
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("Observation IDs must be unique")

        expected_ranks = tuple(range(1, len(self.differential) + 1))
        ranks = tuple(item.rank for item in self.differential)
        if ranks != expected_ranks:
            raise ValueError("Differential ranks must be contiguous and ordered")
        disease_ids = tuple(item.disease_id for item in self.differential)
        if len(disease_ids) != len(set(disease_ids)):
            raise ValueError("Differential disease IDs must be unique")

        known = set(observation_ids)
        for item in self.differential:
            linked = set(item.supporting_observation_ids).union(
                item.contradicting_observation_ids
            )
            missing = linked - known
            if missing:
                raise ValueError(
                    "Differential contains unknown observation links: "
                    + ", ".join(sorted(missing))
                )
        return self
