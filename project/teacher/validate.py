"""Deterministic gates that bind Stage B to frozen Stage A and gold."""

import re
from dataclasses import dataclass

from project.teacher.schemas import (
    DERMOSCOPIC_ONLY_FEATURES,
    CiteField,
    ImageModality,
    StageAMorphology,
    StageBReasoning,
)

REASON_GOLD_MISMATCH = "gold_mismatch"
REASON_RANK1_NOT_GOLD = "rank1_not_gold"
REASON_DDX_LENGTH = "ddx_length"
REASON_RANKS_NOT_CONTIGUOUS = "ranks_not_contiguous"
REASON_RANK1_MISSING_SUPPORT = "rank1_missing_support"
REASON_SUPPORTING_NOT_IN_STAGE_A = "supporting_not_in_stage_a"
REASON_CONTRADICTING_NOT_IN_STAGE_A = "contradicting_not_in_stage_a"
REASON_DERMOSCOPIC_ON_CLINICAL = "dermoscopic_on_clinical"
REASON_EMPTY_REASONING = "empty_reasoning"
REASON_GOLD_NOT_IN_REASONING = "gold_not_in_reasoning"
REASON_DISEASE_OUTSIDE_DDX = "disease_outside_ddx"

_WORD_BOUNDARY = r"(?<![a-z0-9]){name}(?![a-z0-9])"


@dataclass(slots=True, kw_only=True, frozen=True)
class ValidationResult:
    """Outcome of checking Stage B against Stage A and gold."""

    ok: bool
    reasons: tuple[str, ...]


def validate_stage_b(
    morphology: StageAMorphology,
    reasoning: StageBReasoning,
    gold_diagnosis: str,
) -> ValidationResult:
    """Reject Stage B when it drifts from frozen A or the gold anchor.

    Args:
        morphology: Frozen Stage A record.
        reasoning: Parsed Stage B JSON.
        gold_diagnosis: Canonical label from the manifest.

    Returns:
        ``ok=False`` with stable reason codes when any gate fails.
    """
    reasons: list[str] = []
    gold = gold_diagnosis.strip()

    if reasoning.diagnosis.strip() != gold:
        reasons.append(REASON_GOLD_MISMATCH)

    items = reasoning.differential_diagnosis
    if not 2 <= len(items) <= 5:
        reasons.append(REASON_DDX_LENGTH)

    ranks = tuple(item.rank for item in items)
    expected = tuple(range(1, len(items) + 1))
    if ranks != expected:
        reasons.append(REASON_RANKS_NOT_CONTIGUOUS)

    if items and items[0].disease.strip() != gold:
        reasons.append(REASON_RANK1_NOT_GOLD)

    if items and not items[0].supporting:
        reasons.append(REASON_RANK1_MISSING_SUPPORT)

    if _has_unknown_citation(morphology, reasoning, supporting=True):
        reasons.append(REASON_SUPPORTING_NOT_IN_STAGE_A)

    if _has_unknown_citation(morphology, reasoning, supporting=False):
        reasons.append(REASON_CONTRADICTING_NOT_IN_STAGE_A)

    if _has_dermoscopic_on_clinical(morphology, reasoning):
        reasons.append(REASON_DERMOSCOPIC_ON_CLINICAL)

    if not reasoning.reasoning.strip():
        reasons.append(REASON_EMPTY_REASONING)
    elif not _phrase_in_text(gold, reasoning.reasoning):
        reasons.append(REASON_GOLD_NOT_IN_REASONING)

    extra = _diseases_outside_ddx(reasoning)
    if extra:
        reasons.append(REASON_DISEASE_OUTSIDE_DDX)

    return ValidationResult(ok=not reasons, reasons=tuple(reasons))


def _has_unknown_citation(
    morphology: StageAMorphology,
    reasoning: StageBReasoning,
    *,
    supporting: bool,
) -> bool:
    """Return True if a citation value is not present on Stage A."""
    for item in reasoning.differential_diagnosis:
        citations = item.supporting if supporting else item.contradicting
        for citation in citations:
            allowed = _values_for_field(morphology, citation.field)
            if citation.value not in allowed:
                return True
    return False


def _values_for_field(morphology: StageAMorphology, field: CiteField) -> set[str]:
    """Collect string values Stage B may cite for one field."""
    raw = getattr(morphology, field.value)
    if isinstance(raw, list):
        return {str(item) for item in raw}
    return {str(raw)}


def _has_dermoscopic_on_clinical(
    morphology: StageAMorphology,
    reasoning: StageBReasoning,
) -> bool:
    """Return True if B cites a dermoscopy sign on a clinical image."""
    if morphology.modality is ImageModality.DERMOSCOPY:
        return False

    for item in reasoning.differential_diagnosis:
        for citation in (*item.supporting, *item.contradicting):
            if citation.value in DERMOSCOPIC_ONLY_FEATURES:
                return True
    return False


def _phrase_in_text(phrase: str, text: str) -> bool:
    """Return True if ``phrase`` appears in ``text`` as a whole phrase."""
    pattern = _WORD_BOUNDARY.format(name=re.escape(phrase.strip().casefold()))
    return re.search(pattern, text.casefold()) is not None


def _diseases_outside_ddx(reasoning: StageBReasoning) -> tuple[str, ...]:
    """Find DDx-like names in reasoning that are not in the differential list.

    Only checks the diagnosis field and each DDx name against the prose.
    Extra names cannot be listed without a closed taxonomy, so this gate
    only fires when ``diagnosis`` itself is missing from the DDx diseases.
    """
    ddx_names = {
        item.disease.strip().casefold()
        for item in reasoning.differential_diagnosis
    }
    diagnosis = reasoning.diagnosis.strip().casefold()
    if diagnosis and diagnosis not in ddx_names:
        return (reasoning.diagnosis.strip(),)
    return ()
