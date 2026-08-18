"""Strict Stage-A/Stage-B and review contracts for E3 hard distillation."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.train.e3.terminology import ImageModality


def _tuple_from_json(value: object) -> object:
    """Convert JSON arrays to immutable tuples before strict validation."""

    return tuple(value) if isinstance(value, list) else value


type StringTuple = Annotated[tuple[str, ...], BeforeValidator(_tuple_from_json)]


def _observation_status_from_json(value: object) -> object:
    if isinstance(value, str):
        return ObservationStatus(value)
    return value


def _image_modality_from_json(value: object) -> object:
    if isinstance(value, str):
        return ImageModality(value)
    return value


def _confidence_from_json(value: object) -> object:
    if isinstance(value, str):
        return ConfidenceLevel(value)
    return value


def _risk_from_json(value: object) -> object:
    if isinstance(value, str):
        return RiskLevel(value)
    return value


def _information_sufficiency_from_json(value: object) -> object:
    if isinstance(value, str):
        return InformationSufficiency(value)
    return value


def _response_policy_from_json(value: object) -> object:
    if isinstance(value, str):
        return ResponsePolicy(value)
    return value


def _review_status_from_json(value: object) -> object:
    if isinstance(value, str):
        return StageReviewStatus(value)
    return value


def _generation_status_from_json(value: object) -> object:
    if isinstance(value, str):
        return TeacherGenerationStatus(value)
    return value


class _StrictModel(BaseModel):
    """Immutable Pydantic boundary for generated or reviewed targets."""

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


class InformationSufficiency(StrEnum):
    """Whether the available image supports a grounded differential response."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


type InformationSufficiencyValue = Annotated[
    InformationSufficiency,
    BeforeValidator(_information_sufficiency_from_json),
]


class ResponsePolicy(StrEnum):
    """The mutually exclusive response behavior taught by the policy task."""

    ANSWER_DIFFERENTIAL = "ANSWER_DIFFERENTIAL"
    REQUEST_CONTEXT = "REQUEST_CONTEXT"


type ResponsePolicyValue = Annotated[
    ResponsePolicy,
    BeforeValidator(_response_policy_from_json),
]


class StageReviewStatus(StrEnum):
    """Scientific review state, separate from provider-call outcome."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NOT_APPLICABLE = "not_applicable"
    NOT_GENERATED = "not_generated"


type StageReviewStatusValue = Annotated[
    StageReviewStatus, BeforeValidator(_review_status_from_json)
]


class TeacherGenerationStatus(StrEnum):
    """Mutually exclusive outcome of one external teacher call."""

    SUCCEEDED = "succeeded"
    PROVIDER_SAFETY_REFUSAL = "provider_safety_refusal"
    TRANSPORT_ERROR = "transport_error"
    TIMEOUT = "timeout"
    EMPTY_RESPONSE = "empty_response"
    INVALID_SCHEMA = "invalid_schema"


type TeacherGenerationStatusValue = Annotated[
    TeacherGenerationStatus,
    BeforeValidator(_generation_status_from_json),
]


type ImageModalityValue = Annotated[
    ImageModality,
    BeforeValidator(_image_modality_from_json),
]


class ImageAssessment(_StrictModel):
    """What can and cannot be assessed from the supplied image."""

    is_evaluable: bool
    image_modality: ImageModalityValue
    views_available: StringTuple
    quality_defects: StringTuple
    has_anatomic_overview: bool
    has_scale: bool
    has_lateral_profile: bool
    distribution_assessability: str = Field(min_length=1)
    color_reliability: str = Field(min_length=1)


class Observation(_StrictModel):
    """One image-grounded clinical observation generated in Stage A."""

    id: str = Field(min_length=1)
    concept_id: str = Field(
        pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$"
    )
    concept_label: str = Field(min_length=1)
    concept_detail: str | None = Field(default=None, min_length=1)
    status: ObservationStatusValue
    provenance: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    confidence: ConfidenceValue
    evidence_region: str | None = None


_INCOMPLETE_PROSE_SUFFIXES = (
    " it is",
    " which is",
    " and",
    " or",
    " as",
    " because",
    " with",
    " for",
    " of",
    " to",
)


def _validate_complete_prose(value: str, *, field_name: str) -> str:
    """Reject padded, multiline, too-short, or visibly truncated prose."""

    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{field_name} must be one paragraph")
    if len(value.split()) < 5:
        raise ValueError(f"{field_name} is too short")
    if not value.endswith((".", "?", "!")):
        raise ValueError(f"{field_name} must end at a sentence boundary")
    normalized = value.rstrip(".?!").casefold()
    if normalized.endswith(_INCOMPLETE_PROSE_SUFFIXES):
        raise ValueError(f"{field_name} ends with an incomplete clause")
    return value


class StageATarget(_StrictModel):
    """Answer-blind perceptual record generated without a diagnosis label."""

    image_assessment: ImageAssessment
    dominant_visual_pattern: str = Field(min_length=1)
    observations: Annotated[
        tuple[Observation, ...], BeforeValidator(_tuple_from_json)
    ] = Field(min_length=1)
    not_assessable_features: StringTuple = ()
    clinical_caption: str = Field(min_length=1)

    @field_validator("clinical_caption")
    @classmethod
    def _caption_must_be_complete(cls, value: str) -> str:
        return _validate_complete_prose(value, field_name="clinical_caption")

    @model_validator(mode="after")
    def _validate_observation_ids(self) -> StageATarget:
        observation_ids = tuple(item.id for item in self.observations)
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("Observation IDs must be unique")
        return self


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


class DiagnosticAssessment(_StrictModel):
    """Stage-B diagnostic result, kept separate from response policy."""

    differential: Annotated[
        tuple[DifferentialDiagnosis, ...], BeforeValidator(_tuple_from_json)
    ] = Field(min_length=1, max_length=6)
    concise_clinical_rationale: str = Field(min_length=1)

    @field_validator("concise_clinical_rationale")
    @classmethod
    def _rationale_must_be_complete(cls, value: str) -> str:
        return _validate_complete_prose(
            value,
            field_name="concise_clinical_rationale",
        )

    @model_validator(mode="after")
    def _validate_differential(self) -> DiagnosticAssessment:
        expected_ranks = tuple(range(1, len(self.differential) + 1))
        ranks = tuple(item.rank for item in self.differential)
        if ranks != expected_ranks:
            raise ValueError("Differential ranks must be contiguous and ordered")
        disease_ids = tuple(item.disease_id for item in self.differential)
        if len(disease_ids) != len(set(disease_ids)):
            raise ValueError("Differential disease IDs must be unique")
        return self


class ContextRequest(_StrictModel):
    """One explicit question that resolves a named diagnostic ambiguity."""

    request_id: str = Field(min_length=1)
    priority: int = Field(ge=1)
    context_type: str = Field(min_length=1)
    required_source: str = Field(min_length=1)
    question: str = Field(min_length=1)
    discriminates_between: Annotated[
        tuple[str, ...], BeforeValidator(_tuple_from_json)
    ] = Field(min_length=2)
    rationale: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def _question_must_be_explicit(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("Context question must not contain surrounding whitespace")
        if "\n" in value or "\r" in value:
            raise ValueError("Context question must be one paragraph")
        if len(value.split()) < 5:
            raise ValueError("Context question is too short")
        if not value.endswith("?"):
            raise ValueError("Context question must end with a question mark")
        return value

    @field_validator("rationale")
    @classmethod
    def _request_rationale_must_be_complete(cls, value: str) -> str:
        return _validate_complete_prose(value, field_name="context_request.rationale")

    @model_validator(mode="after")
    def _validate_disease_ids(self) -> ContextRequest:
        if len(self.discriminates_between) != len(set(self.discriminates_between)):
            raise ValueError("Context request disease IDs must be unique")
        invalid = tuple(
            disease_id
            for disease_id in self.discriminates_between
            if not (
                len(disease_id) == 4
                and disease_id.startswith("D")
                and disease_id[1:].isdigit()
            )
        )
        if invalid:
            raise ValueError("Context request contains invalid disease IDs")
        return self


class ContextDecision(_StrictModel):
    """Stage-B sufficiency judgment and mutually exclusive response policy."""

    information_sufficiency: InformationSufficiencyValue
    response_policy: ResponsePolicyValue
    decision_rationale: str = Field(min_length=1)
    requests: Annotated[
        tuple[ContextRequest, ...], BeforeValidator(_tuple_from_json)
    ] = ()

    @field_validator("decision_rationale")
    @classmethod
    def _decision_rationale_must_be_complete(cls, value: str) -> str:
        return _validate_complete_prose(value, field_name="decision_rationale")

    @model_validator(mode="after")
    def _validate_policy_and_requests(self) -> ContextDecision:
        if self.information_sufficiency is InformationSufficiency.SUFFICIENT:
            if self.response_policy is not ResponsePolicy.ANSWER_DIFFERENTIAL:
                raise ValueError("Sufficient information requires ANSWER_DIFFERENTIAL")
            if self.requests:
                raise ValueError("Sufficient information cannot request context")
        else:
            if self.response_policy is not ResponsePolicy.REQUEST_CONTEXT:
                raise ValueError("Insufficient information requires REQUEST_CONTEXT")
            if not self.requests:
                raise ValueError(
                    "REQUEST_CONTEXT requires at least one explicit request"
                )

        expected_priorities = tuple(range(1, len(self.requests) + 1))
        priorities = tuple(item.priority for item in self.requests)
        if priorities != expected_priorities:
            raise ValueError(
                "Context request priorities must be contiguous and ordered"
            )
        request_ids = tuple(item.request_id for item in self.requests)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("Context request IDs must be unique")
        return self


class StageBCorrection(_StrictModel):
    """Explicit audit record when Stage B revises one Stage-A observation."""

    observation_id: str = Field(min_length=1)
    field: str = Field(min_length=1)
    previous_value: str | None = None
    corrected_value: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class StageBTarget(_StrictModel):
    """Diagnosis and context policy from the image plus frozen Stage A."""

    stage_b_corrections: Annotated[
        tuple[StageBCorrection, ...], BeforeValidator(_tuple_from_json)
    ] = ()
    diagnostic_assessment: DiagnosticAssessment
    context_decision: ContextDecision

    @model_validator(mode="after")
    def _validate_context_diagnostic_links(self) -> StageBTarget:
        disease_ids = {
            item.disease_id for item in self.diagnostic_assessment.differential
        }
        for request in self.context_decision.requests:
            unknown = set(request.discriminates_between) - disease_ids
            if unknown:
                raise ValueError(
                    "Context request references diseases outside the differential: "
                    + ", ".join(sorted(unknown))
                )
        return self


class ProviderSafetyCategory(_StrictModel):
    """Sanitized provider safety annotation without raw prompts or secrets."""

    category: str = Field(min_length=1)
    severity: str | None = None
    filtered: bool | None = None


class TeacherGenerationProvenance(_StrictModel):
    """Frozen identity and typed outcome of one Stage-A or Stage-B call."""

    generation_id: str = Field(min_length=1)
    generation_status: TeacherGenerationStatusValue
    provider: str = Field(min_length=1)
    teacher_model: str = Field(min_length=1)
    teacher_revision: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_response_id: str | None = None
    provider_request_id: str | None = None
    provider_model_reported: str | None = None
    finish_reason: str | None = None
    provider_error_code: str | None = None
    safety_categories: Annotated[
        tuple[ProviderSafetyCategory, ...], BeforeValidator(_tuple_from_json)
    ] = ()
    gold_visible_to_teacher: bool

    @model_validator(mode="after")
    def _validate_outcome_metadata(self) -> TeacherGenerationProvenance:
        categories = tuple(item.category for item in self.safety_categories)
        if len(categories) != len(set(categories)):
            raise ValueError("Provider safety categories must be unique")
        if (
            self.generation_status is TeacherGenerationStatus.SUCCEEDED
            and self.provider_error_code is not None
        ):
            raise ValueError("Successful generation cannot carry an error code")
        if (
            self.generation_status
            is not TeacherGenerationStatus.PROVIDER_SAFETY_REFUSAL
            and self.safety_categories
        ):
            raise ValueError("Safety categories require provider_safety_refusal status")
        return self


class TeacherTargetBundle(_StrictModel):
    """Reviewed teacher outputs with task-isolated Stage-B acceptance."""

    stage_a_status: StageReviewStatusValue
    stage_a_target: StageATarget | None = None
    stage_a_provenance: TeacherGenerationProvenance | None = None
    stage_a_rejection_reasons: StringTuple = ()
    stage_b_status: StageReviewStatusValue
    stage_b_target: StageBTarget | None = None
    stage_b_provenance: TeacherGenerationProvenance | None = None
    stage_b_rejection_reasons: StringTuple = ()
    stage_b_diagnostic_status: StageReviewStatusValue
    stage_b_diagnostic_rejection_reasons: StringTuple = ()
    stage_b_context_policy_status: StageReviewStatusValue
    stage_b_context_policy_rejection_reasons: StringTuple = ()

    @model_validator(mode="before")
    @classmethod
    def _backfill_legacy_stage_b_reviews(cls, value: object) -> object:
        """Read v1 bundles whose single Stage-B review gated both tasks."""

        if not isinstance(value, dict):
            return value
        document = dict(value)
        status = document.get("stage_b_status")
        reasons = document.get("stage_b_rejection_reasons", ())
        document.setdefault("stage_b_diagnostic_status", status)
        document.setdefault("stage_b_diagnostic_rejection_reasons", reasons)
        document.setdefault("stage_b_context_policy_status", status)
        document.setdefault("stage_b_context_policy_rejection_reasons", reasons)
        return document

    @model_validator(mode="after")
    def _validate_review_and_evidence_graph(self) -> TeacherTargetBundle:
        self._validate_stage(
            stage_name="Stage A",
            status=self.stage_a_status,
            target=self.stage_a_target,
            provenance=self.stage_a_provenance,
            rejection_reasons=self.stage_a_rejection_reasons,
            allow_gold_visible=False,
        )
        self._validate_stage(
            stage_name="Stage B",
            status=self.stage_b_status,
            target=self.stage_b_target,
            provenance=self.stage_b_provenance,
            rejection_reasons=self.stage_b_rejection_reasons,
            allow_gold_visible=True,
        )
        self._validate_stage_b_subtarget(
            name="diagnostic",
            status=self.stage_b_diagnostic_status,
            rejection_reasons=self.stage_b_diagnostic_rejection_reasons,
        )
        self._validate_stage_b_subtarget(
            name="context policy",
            status=self.stage_b_context_policy_status,
            rejection_reasons=self.stage_b_context_policy_rejection_reasons,
        )
        if (
            self.stage_b_status is not StageReviewStatus.NOT_GENERATED
            and self.stage_a_status is not StageReviewStatus.ACCEPTED
        ):
            raise ValueError("Attempted Stage B requires accepted Stage A")
        usable_subtargets = {
            self.stage_b_diagnostic_status,
            self.stage_b_context_policy_status,
        }
        if self.stage_b_status is StageReviewStatus.ACCEPTED and (
            StageReviewStatus.ACCEPTED not in usable_subtargets
        ):
            raise ValueError("Accepted Stage B requires an accepted subtarget")
        if self.stage_b_status is StageReviewStatus.REJECTED and (
            StageReviewStatus.ACCEPTED in usable_subtargets
        ):
            raise ValueError("Rejected Stage B cannot carry an accepted subtarget")
        if (
            self.stage_a_status is StageReviewStatus.ACCEPTED
            and self.stage_b_diagnostic_status is StageReviewStatus.ACCEPTED
        ):
            self._validate_cross_stage_links()
        return self

    def _validate_stage_b_subtarget(
        self,
        *,
        name: str,
        status: StageReviewStatus,
        rejection_reasons: tuple[str, ...],
    ) -> None:
        if self.stage_b_status in {
            StageReviewStatus.NOT_GENERATED,
            StageReviewStatus.NOT_APPLICABLE,
        }:
            if status is not self.stage_b_status or rejection_reasons:
                raise ValueError(
                    f"Stage-B {name} review must mirror an ungenerated/failed stage"
                )
            return
        if status not in {
            StageReviewStatus.ACCEPTED,
            StageReviewStatus.REJECTED,
        }:
            raise ValueError(f"Generated Stage-B {name} requires scientific review")
        if (
            status is StageReviewStatus.ACCEPTED
            and self.stage_b_target is None
        ):
            raise ValueError(f"Stage-B {name} review requires a parsed target")
        if status is StageReviewStatus.ACCEPTED and rejection_reasons:
            raise ValueError(f"Accepted Stage-B {name} cannot have rejection reasons")
        if status is StageReviewStatus.REJECTED and not rejection_reasons:
            raise ValueError(f"Rejected Stage-B {name} requires rejection reasons")

    @staticmethod
    def _validate_stage(
        *,
        stage_name: str,
        status: StageReviewStatus,
        target: StageATarget | StageBTarget | None,
        provenance: TeacherGenerationProvenance | None,
        rejection_reasons: tuple[str, ...],
        allow_gold_visible: bool,
    ) -> None:
        if status is StageReviewStatus.NOT_GENERATED:
            if target is not None or provenance is not None or rejection_reasons:
                raise ValueError(
                    f"{stage_name} not_generated cannot carry target or review data"
                )
            return
        if provenance is None:
            raise ValueError(f"{stage_name} attempt requires generation provenance")
        if provenance.gold_visible_to_teacher and not allow_gold_visible:
            raise ValueError(f"{stage_name} teacher call must be answer-blind")
        if status is StageReviewStatus.NOT_APPLICABLE:
            if provenance.generation_status is TeacherGenerationStatus.SUCCEEDED:
                raise ValueError(
                    f"{stage_name} not_applicable requires a failed generation"
                )
            if target is not None or rejection_reasons:
                raise ValueError(
                    f"{stage_name} not_applicable cannot carry target or review data"
                )
            return
        if provenance.generation_status is not TeacherGenerationStatus.SUCCEEDED:
            raise ValueError(
                f"{stage_name} scientific review requires successful generation"
            )
        if status is StageReviewStatus.ACCEPTED:
            if target is None:
                raise ValueError(f"Accepted {stage_name} requires a parsed target")
            if rejection_reasons:
                raise ValueError(f"Accepted {stage_name} cannot have rejection reasons")
        elif not rejection_reasons:
            raise ValueError(f"Rejected {stage_name} requires rejection reasons")

    def _validate_cross_stage_links(self) -> None:
        stage_a = self.stage_a_target
        stage_b = self.stage_b_target
        if stage_a is None or stage_b is None:
            raise ValueError("Accepted Stage A/B targets are missing")
        known = {item.id for item in stage_a.observations}
        for item in stage_b.diagnostic_assessment.differential:
            linked = set(item.supporting_observation_ids).union(
                item.contradicting_observation_ids
            )
            missing = linked - known
            if missing:
                raise ValueError(
                    "Stage B contains unknown Stage-A observation links: "
                    + ", ".join(sorted(missing))
                )
        for correction in stage_b.stage_b_corrections:
            if correction.observation_id not in known:
                raise ValueError(
                    "Stage B correction references an unknown Stage-A observation"
                )
