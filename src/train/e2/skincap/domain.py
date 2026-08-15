"""Immutable domain objects for the SkinCAP observation transform."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BoundaryKind(StrEnum):
    """First unsafe semantic boundary detected in a source caption."""

    NONE = "none"
    GOLD_DIAGNOSIS = "gold_diagnosis"
    DIAGNOSTIC = "diagnostic"
    TESTING = "testing"
    MANAGEMENT = "management"


class RejectionReason(StrEnum):
    """Deterministic reason preventing an extracted target from admission."""

    EMPTY_OBSERVATION = "empty_observation"
    TOO_FEW_WORDS = "too_few_words"
    TOO_FEW_CHARACTERS = "too_few_characters"
    RESIDUAL_GOLD_DIAGNOSIS = "residual_gold_diagnosis"
    RESIDUAL_DIAGNOSTIC_LANGUAGE = "residual_diagnostic_language"
    RESIDUAL_TESTING_LANGUAGE = "residual_testing_language"
    RESIDUAL_MANAGEMENT_LANGUAGE = "residual_management_language"


@dataclass(frozen=True, slots=True)
class SkinCapTransformPolicy:
    """Frozen high-precision policy for observation-only candidate targets."""

    version: str = "skincap_observation_prefix_v1"
    caption_variant: str = "v240715_google_translate_en"
    minimum_words: int = 5
    minimum_characters: int = 20

    def __post_init__(self) -> None:
        """Reject policies that could silently admit empty targets."""

        if not self.version or not self.caption_variant:
            raise ValueError("SkinCAP transform identity must not be empty")
        if self.minimum_words < 1 or self.minimum_characters < 1:
            raise ValueError("SkinCAP minimum target lengths must be positive")


@dataclass(frozen=True, slots=True)
class SkinCapTransformResult:
    """One deterministic separation of observation and unsafe suffix text."""

    source_sha256: str
    observation_text: str
    removed_suffix: str
    boundary_kind: BoundaryKind
    boundary_offset: int | None
    word_count: int
    character_count: int
    accepted: bool
    rejection_reasons: tuple[RejectionReason, ...]


@dataclass(frozen=True, slots=True)
class SkinCapAuditReport:
    """Aggregate-only audit safe to retain outside the gated raw payload."""

    transform_version: str
    caption_variant: str
    metadata_sha256: str
    downloaded_rows: int
    author_excluded_rows: int
    usable_before_leakage_rows: int
    frozen_validation_overlap_rows: int
    frozen_internal_overlap_rows: int
    technical_candidate_rows: int
    technical_candidate_groups: int
    accepted_observation_rows: int
    rejected_observation_rows: int
    accepted_by_source: tuple[tuple[str, int], ...]
    boundary_counts: tuple[tuple[str, int], ...]
    rejection_counts: tuple[tuple[str, int], ...]
    observation_word_min: int
    observation_word_median: float
    observation_word_p95: float
    observation_word_max: int
    derivatives_materialized: bool = False

    def as_record(self) -> dict[str, object]:
        """Return a stable JSON-compatible aggregate without clinical text."""

        return {
            "transform_version": self.transform_version,
            "caption_variant": self.caption_variant,
            "metadata_sha256": self.metadata_sha256,
            "downloaded_rows": self.downloaded_rows,
            "author_excluded_rows": self.author_excluded_rows,
            "usable_before_leakage_rows": self.usable_before_leakage_rows,
            "frozen_validation_overlap_rows": self.frozen_validation_overlap_rows,
            "frozen_internal_overlap_rows": self.frozen_internal_overlap_rows,
            "technical_candidate_rows": self.technical_candidate_rows,
            "technical_candidate_groups": self.technical_candidate_groups,
            "accepted_observation_rows": self.accepted_observation_rows,
            "rejected_observation_rows": self.rejected_observation_rows,
            "accepted_by_source": dict(self.accepted_by_source),
            "boundary_counts": dict(self.boundary_counts),
            "rejection_counts": dict(self.rejection_counts),
            "observation_word_statistics": {
                "min": self.observation_word_min,
                "median": self.observation_word_median,
                "p95": self.observation_word_p95,
                "max": self.observation_word_max,
            },
            "derivatives_materialized": self.derivatives_materialized,
        }
