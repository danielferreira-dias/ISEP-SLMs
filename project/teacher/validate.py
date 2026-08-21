"""Deterministic gates binding compact Stage B facts to Stage A and gold."""

import re
from dataclasses import dataclass

from project.teacher.schemas import (
    AnchorEvidenceStatus,
    ObservationStatus,
    ResponsePolicy,
    StageAMorphology,
    StageBReasoning,
)

REASON_GOLD_MISMATCH = "gold_mismatch"
REASON_ANNOTATION_CONFLICT = "annotation_conflict"
REASON_UNSUPPORTED_ANCHOR = "unsupported_anchor"
REASON_ANSWER_ON_NON_EVALUABLE = "answer_on_non_evaluable_image"
REASON_NEW_IMAGE_ON_EVALUABLE = "new_image_on_evaluable_image"
REASON_NEW_IMAGE_ANCHOR_NOT_UNSUPPORTED = "new_image_anchor_not_unsupported"
REASON_DIAGNOSIS_MISSING_SUPPORT = "diagnosis_missing_support"
REASON_UNSUPPORTED_ANCHOR_HAS_SUPPORT = "unsupported_anchor_has_support"
REASON_SUPPORTING_NOT_IN_STAGE_A = "supporting_not_in_stage_a"
REASON_NON_EVIDENTIARY_OBSERVATION = "non_evidentiary_observation"
REASON_COMPARISON_MISSING_GOLD = "comparison_missing_gold"
REASON_COMPARISON_MISSING_ALTERNATIVE = "comparison_missing_alternative"
REASON_REASONING_MISSING_GOLD = "clinical_reasoning_missing_gold"
REASON_REASONING_MISSING_ALTERNATIVE = "clinical_reasoning_missing_alternative"
REASON_REASONING_REVEALS_GOLD = "clinical_reasoning_reveals_gold_on_new_image"

_WORD_BOUNDARY = r"(?<![a-z0-9]){name}(?![a-z0-9])"
_EVIDENTIARY_STATUSES = {
    ObservationStatus.PRESENT,
    ObservationStatus.ABSENT_IN_OBSERVED_SCOPE,
}


@dataclass(slots=True, kw_only=True, frozen=True)
class ValidationResult:
    """Outcome of checking Stage B against Stage A and the gold label."""

    ok: bool
    reasons: tuple[str, ...]


def validate_stage_b(
    morphology: StageAMorphology,
    reasoning: StageBReasoning,
    gold_diagnosis: str,
) -> ValidationResult:
    """Reject Stage B when it drifts, fabricates support, or uses a bad policy."""
    reasons: list[str] = []
    gold = gold_diagnosis.strip()

    if reasoning.diagnosis.strip() != gold:
        reasons.append(REASON_GOLD_MISMATCH)
    if reasoning.annotation_conflict:
        reasons.append(REASON_ANNOTATION_CONFLICT)

    is_evaluable = morphology.image_assessment.is_evaluable
    if reasoning.response_policy is ResponsePolicy.ANSWER_DIFFERENTIAL:
        if not is_evaluable:
            reasons.append(REASON_ANSWER_ON_NON_EVALUABLE)
        if reasoning.anchor_evidence_status is AnchorEvidenceStatus.UNSUPPORTED:
            reasons.append(REASON_UNSUPPORTED_ANCHOR)
    else:
        if is_evaluable:
            reasons.append(REASON_NEW_IMAGE_ON_EVALUABLE)
        if reasoning.anchor_evidence_status is not AnchorEvidenceStatus.UNSUPPORTED:
            reasons.append(REASON_NEW_IMAGE_ANCHOR_NOT_UNSUPPORTED)

    diagnosis_ids = {
        observation_id
        for item in reasoning.differential_comparisons
        for observation_id in item.features_favoring_diagnosis
    }
    alternative_ids = {
        observation_id
        for item in reasoning.differential_comparisons
        for observation_id in item.features_favoring_alternative
    }
    if (
        reasoning.response_policy is ResponsePolicy.ANSWER_DIFFERENTIAL
        and reasoning.anchor_evidence_status
        in {AnchorEvidenceStatus.SUPPORTED, AnchorEvidenceStatus.WEAK}
        and not diagnosis_ids
    ):
        reasons.append(REASON_DIAGNOSIS_MISSING_SUPPORT)
    if (
        reasoning.anchor_evidence_status is AnchorEvidenceStatus.UNSUPPORTED
        and diagnosis_ids
    ):
        reasons.append(REASON_UNSUPPORTED_ANCHOR_HAS_SUPPORT)

    known = {item.id: item for item in morphology.observations}
    cited_ids = diagnosis_ids | alternative_ids
    if cited_ids - known.keys():
        reasons.append(REASON_SUPPORTING_NOT_IN_STAGE_A)
    if any(
        known[item_id].status not in _EVIDENTIARY_STATUSES
        for item_id in cited_ids.intersection(known)
    ):
        reasons.append(REASON_NON_EVIDENTIARY_OBSERVATION)

    for item in reasoning.differential_comparisons:
        if not _phrase_in_text(gold, item.comparison):
            reasons.append(REASON_COMPARISON_MISSING_GOLD)
        if not _phrase_in_text(item.alternative, item.comparison):
            reasons.append(REASON_COMPARISON_MISSING_ALTERNATIVE)

    if reasoning.response_policy is ResponsePolicy.ANSWER_DIFFERENTIAL:
        if not _phrase_in_text(gold, reasoning.clinical_reasoning):
            reasons.append(REASON_REASONING_MISSING_GOLD)
        if any(
            not _phrase_in_text(item.alternative, reasoning.clinical_reasoning)
            for item in reasoning.differential_comparisons
        ):
            reasons.append(REASON_REASONING_MISSING_ALTERNATIVE)
    elif _phrase_in_text(gold, reasoning.clinical_reasoning):
        reasons.append(REASON_REASONING_REVEALS_GOLD)

    return ValidationResult(ok=not reasons, reasons=tuple(dict.fromkeys(reasons)))


def _phrase_in_text(phrase: str, text: str) -> bool:
    """Match a canonical label in natural prose as a whole phrase.

    Canonical dataset separators are interchangeable in prose, so
    ``contact_dermatitis`` matches ``contact dermatitis`` without accepting a
    different medical phrase.
    """
    tokens = tuple(
        token for token in re.split(r"[\s_-]+", phrase.strip().casefold()) if token
    )
    if not tokens:
        return False
    name = r"[\s_-]+".join(re.escape(token) for token in tokens)
    pattern = _WORD_BOUNDARY.format(name=name)
    return re.search(pattern, text.casefold()) is not None
