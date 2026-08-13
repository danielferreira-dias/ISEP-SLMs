"""Strict label parsing for closed-set dermatology evaluation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .models import PredictionInput, PredictionRecord

_WHITESPACE = re.compile(r"\s+")


def _normalized_exact(value: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value).strip()).casefold()


@dataclass(frozen=True, slots=True)
class LabelAlias:
    """An accepted exact surface form for a canonical label."""

    alias: str
    canonical_label: str


@dataclass(frozen=True, slots=True)
class LabelVocabulary:
    """Closed canonical label set with explicitly audited aliases."""

    labels: tuple[str, ...]
    aliases: tuple[LabelAlias, ...] = ()

    def __post_init__(self) -> None:
        """Validate non-empty, collision-free canonical and alias forms."""

        if not self.labels:
            raise ValueError("At least one canonical label is required")
        if any(not label.strip() for label in self.labels):
            raise ValueError("Canonical labels cannot be blank")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Canonical labels must be unique")

        normalized: dict[str, str] = {}
        for label in self.labels:
            key = _normalized_exact(label)
            existing = normalized.get(key)
            if existing is not None and existing != label:
                raise ValueError(
                    f"Canonical labels normalize to the same value: "
                    f"{existing!r} and {label!r}"
                )
            normalized[key] = label
        canonical_set = set(self.labels)
        for alias in self.aliases:
            if alias.canonical_label not in canonical_set:
                raise ValueError(
                    f"Alias target is not canonical: {alias.canonical_label}"
                )
            key = _normalized_exact(alias.alias)
            if not key:
                raise ValueError("Aliases cannot be blank")
            existing = normalized.get(key)
            if existing is not None and existing != alias.canonical_label:
                raise ValueError(f"Alias {alias.alias!r} collides with {existing!r}")
            normalized[key] = alias.canonical_label

    def parse(self, raw_output: str) -> str | None:
        """Resolve only an exact canonical label or registered alias.

        Surrounding whitespace, Unicode compatibility forms, case, and runs
        of whitespace are normalized. Punctuation, prose, JSON wrappers, and
        substring matches are deliberately rejected.
        """

        key = _normalized_exact(raw_output)
        if not key:
            return None
        lookup = {_normalized_exact(label): label for label in self.labels}
        lookup.update(
            {
                _normalized_exact(alias.alias): alias.canonical_label
                for alias in self.aliases
            }
        )
        return lookup.get(key)

    def require_canonical(self, label: str) -> None:
        """Raise if a gold label is not an exact canonical value."""

        if label not in set(self.labels):
            raise ValueError(f"Gold label is not canonical: {label!r}")


def canonicalize_predictions(
    predictions: tuple[PredictionInput, ...],
    vocabulary: LabelVocabulary,
) -> tuple[PredictionRecord, ...]:
    """Convert raw model outputs into auditable canonical records."""

    records: list[PredictionRecord] = []
    seen_samples: set[str] = set()
    for prediction in predictions:
        if not prediction.sample_id:
            raise ValueError("sample_id cannot be blank")
        if prediction.sample_id in seen_samples:
            raise ValueError(f"Duplicate sample_id: {prediction.sample_id}")
        seen_samples.add(prediction.sample_id)
        if not prediction.leakage_group_id:
            raise ValueError("leakage_group_id cannot be blank")
        vocabulary.require_canonical(prediction.true_label)
        parsed = vocabulary.parse(prediction.raw_output)
        records.append(
            PredictionRecord(
                sample_id=prediction.sample_id,
                leakage_group_id=prediction.leakage_group_id,
                true_label=prediction.true_label,
                raw_output=prediction.raw_output,
                predicted_label=parsed,
                is_valid=parsed is not None,
                checkpoint_id=prediction.checkpoint_id,
                seed=prediction.seed,
            )
        )
    return tuple(records)
