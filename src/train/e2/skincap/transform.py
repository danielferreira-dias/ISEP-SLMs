"""Deterministic high-precision extraction of visual SkinCAP observations."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

from src.train.e2.skincap.domain import (
    BoundaryKind,
    RejectionReason,
    SkinCapTransformPolicy,
    SkinCapTransformResult,
)

_WORD = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
_DIAGNOSTIC = re.compile(
    r"\b(?:diagnos(?:is|ed|tic)|differential(?:\s+diagnos(?:is|es))?|"
    r"suggest(?:s|ed|ive)?|suspect(?:ed|ion)?|consistent\s+with|"
    r"compatible\s+with|indicative\s+of|likely(?:\s+diagnos(?:is|es))?|"
    r"may\s+(?:be|indicate|suggest|represent)|consider(?:ed|ing|ation)?|"
    r"favor(?:s|ed)?|raises?\s+(?:the\s+)?(?:possibility|concern))\b",
    flags=re.IGNORECASE,
)
_TESTING = re.compile(
    r"\b(?:patholog(?:y|ical)|dermoscop(?:y|ic)|biops(?:y|ies)|"
    r"wood(?:'s)?\s+lamp|laboratory|histopatholog(?:y|ical)|"
    r"examination|testing?)\b",
    flags=re.IGNORECASE,
)
_MANAGEMENT = re.compile(
    r"\b(?:recommend(?:ed|ation)?|treat(?:ment|ed|ing)?|therap(?:y|ies)|"
    r"consult(?:ation|ed|ing)?|follow[- ]?up|medical\s+attention|"
    r"dermatologist|physician|doctor|should\s+(?:undergo|seek|receive))\b",
    flags=re.IGNORECASE,
)
DEFAULT_TRANSFORM_POLICY = SkinCapTransformPolicy()


@dataclass(frozen=True, slots=True)
class _Boundary:
    offset: int
    kind: BoundaryKind


def transform_caption(
    caption: str,
    disease_label: str,
    policy: SkinCapTransformPolicy = DEFAULT_TRANSFORM_POLICY,
) -> SkinCapTransformResult:
    """Remove diagnosis/testing/management suffixes from one SkinCAP caption.

    The operation is deliberately prefix-only. It never stitches later clauses
    back into the target after an unsafe boundary, which keeps the output
    auditable and prevents a later recommendation or label-conditioned clause
    from being mistaken for an independent observation.

    Args:
        caption: English SkinCAP source caption.
        disease_label: Upstream diagnosis used only as a leakage guard.
        policy: Frozen minimum-length and version contract.

    Returns:
        The extracted observation, removed suffix, boundary provenance, and
        deterministic acceptance reasons.
    """

    normalized = _normalize_text(caption)
    if not normalized:
        raise ValueError("SkinCAP source caption must not be empty")
    if not disease_label.strip():
        raise ValueError("SkinCAP disease label must not be empty")
    boundary = _first_boundary(normalized, disease_label)
    observation = _clean_fragment(
        normalized if boundary is None else normalized[: boundary.offset]
    )
    removed = "" if boundary is None else normalized[boundary.offset :].strip()
    reasons = _rejection_reasons(observation, disease_label, policy)
    return SkinCapTransformResult(
        source_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        observation_text=observation,
        removed_suffix=removed,
        boundary_kind=(BoundaryKind.NONE if boundary is None else boundary.kind),
        boundary_offset=(None if boundary is None else boundary.offset),
        word_count=len(_WORD.findall(observation)),
        character_count=len(observation),
        accepted=not reasons,
        rejection_reasons=reasons,
    )


def _first_boundary(text: str, disease_label: str) -> _Boundary | None:
    candidates: list[_Boundary] = []
    for kind, pattern in (
        (BoundaryKind.DIAGNOSTIC, _DIAGNOSTIC),
        (BoundaryKind.TESTING, _TESTING),
        (BoundaryKind.MANAGEMENT, _MANAGEMENT),
    ):
        match = pattern.search(text)
        if match is not None:
            candidates.append(_Boundary(match.start(), kind))
    gold_offset = _gold_diagnosis_offset(text, disease_label)
    if gold_offset is not None:
        candidates.append(_Boundary(gold_offset, BoundaryKind.GOLD_DIAGNOSIS))
    if not candidates:
        return None
    priority = {
        BoundaryKind.GOLD_DIAGNOSIS: 0,
        BoundaryKind.DIAGNOSTIC: 1,
        BoundaryKind.TESTING: 2,
        BoundaryKind.MANAGEMENT: 3,
        BoundaryKind.NONE: 4,
    }
    return min(candidates, key=lambda item: (item.offset, priority[item.kind]))


def _rejection_reasons(
    observation: str,
    disease_label: str,
    policy: SkinCapTransformPolicy,
) -> tuple[RejectionReason, ...]:
    reasons: list[RejectionReason] = []
    words = _WORD.findall(observation)
    if not observation:
        reasons.append(RejectionReason.EMPTY_OBSERVATION)
    if len(words) < policy.minimum_words:
        reasons.append(RejectionReason.TOO_FEW_WORDS)
    if len(observation) < policy.minimum_characters:
        reasons.append(RejectionReason.TOO_FEW_CHARACTERS)
    if _gold_diagnosis_offset(observation, disease_label) is not None:
        reasons.append(RejectionReason.RESIDUAL_GOLD_DIAGNOSIS)
    if _DIAGNOSTIC.search(observation):
        reasons.append(RejectionReason.RESIDUAL_DIAGNOSTIC_LANGUAGE)
    if _TESTING.search(observation):
        reasons.append(RejectionReason.RESIDUAL_TESTING_LANGUAGE)
    if _MANAGEMENT.search(observation):
        reasons.append(RejectionReason.RESIDUAL_MANAGEMENT_LANGUAGE)
    return tuple(reasons)


def _gold_diagnosis_offset(text: str, disease_label: str) -> int | None:
    label_tokens = _meaningful_label_tokens(disease_label)
    if not label_tokens:
        return None
    exact_phrase = r"[\s-]+".join(re.escape(token) for token in label_tokens)
    exact = re.search(rf"\b{exact_phrase}\b", text, flags=re.IGNORECASE)
    if exact is not None:
        return exact.start()
    required = 1 if len(label_tokens) == 1 else min(2, len(label_tokens))
    for sentence_match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text):
        sentence = sentence_match.group()
        offsets = [
            match.start()
            for token in label_tokens
            if (
                match := re.search(
                    rf"\b{re.escape(token)}\b", sentence, flags=re.IGNORECASE
                )
            )
            is not None
        ]
        if len(offsets) >= required:
            return sentence_match.start() + min(offsets)
    return None


def _meaningful_label_tokens(label: str) -> tuple[str, ...]:
    normalized = _normalize_for_matching(label)
    tokens = tuple(
        token
        for token in normalized.split()
        if token not in {"and", "of", "the", "in", "situ", "nm"}
    )
    return tokens


def _normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", value).strip()


def _normalize_for_matching(text: str) -> str:
    value = _normalize_text(text).lower().replace("-", " ")
    value = re.sub(r"[^\w\s']", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _clean_fragment(text: str) -> str:
    value = text.rstrip()
    value = re.sub(
        r"(?:\b(?:and|or|which|that|with|to|for|as|the|a|an|it|this|these))\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip(" \t\n,;:-()[]")
