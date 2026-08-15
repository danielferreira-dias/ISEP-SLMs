"""Deterministic parsing and multilabel metrics for SKINCON morphology."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from src.train.e2.domain import SkinConOntology


@dataclass(frozen=True, slots=True)
class MorphologyPredictionInput:
    """Raw model output paired with one complete human SKINCON target."""

    sample_id: str
    leakage_group_id: str
    true_concepts: tuple[str, ...]
    raw_output: str
    checkpoint_id: str
    seed: int


@dataclass(frozen=True, slots=True)
class MorphologyPredictionRecord:
    """Canonicalized morphology prediction with validity provenance."""

    sample_id: str
    leakage_group_id: str
    true_concepts: tuple[str, ...]
    raw_output: str
    predicted_concepts: tuple[str, ...]
    is_valid: bool
    checkpoint_id: str
    seed: int


@dataclass(frozen=True, slots=True)
class ConceptMetrics:
    """Binary metrics for one SKINCON concept."""

    concept: str
    support: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class MorphologyMetrics:
    """Aggregate exact, micro, macro, and format metrics."""

    sample_count: int
    exact_match: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_f1: float
    hamming_loss: float
    invalid_output_rate: float
    per_concept: tuple[ConceptMetrics, ...]


def canonicalize_morphology_predictions(
    inputs: tuple[MorphologyPredictionInput, ...],
    ontology: SkinConOntology,
) -> tuple[MorphologyPredictionRecord, ...]:
    """Parse strict JSON outputs against the frozen SKINCON vocabulary."""

    return tuple(_canonicalize(item, ontology) for item in inputs)


def evaluate_morphology_predictions(
    records: tuple[MorphologyPredictionRecord, ...],
    ontology: SkinConOntology,
) -> MorphologyMetrics:
    """Compute multilabel metrics with invalid outputs kept in denominator."""

    if not records:
        raise ValueError("Morphology evaluation requires at least one prediction")
    per_concept: list[ConceptMetrics] = []
    total_tp = total_fp = total_fn = 0
    for concept in ontology.concepts:
        tp = fp = fn = 0
        for record in records:
            truth = concept in record.true_concepts
            predicted = concept in record.predicted_concepts
            tp += int(truth and predicted)
            fp += int(not truth and predicted)
            fn += int(truth and not predicted)
        support = tp + fn
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, support)
        f1 = _f1(precision, recall)
        per_concept.append(
            ConceptMetrics(concept, support, tp, fp, fn, precision, recall, f1)
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
    micro_precision = _ratio(total_tp, total_tp + total_fp)
    micro_recall = _ratio(total_tp, total_tp + total_fn)
    exact = sum(
        record.is_valid and set(record.true_concepts) == set(record.predicted_concepts)
        for record in records
    )
    mismatches = sum(
        len(set(record.true_concepts) ^ set(record.predicted_concepts))
        for record in records
    )
    return MorphologyMetrics(
        sample_count=len(records),
        exact_match=exact / len(records),
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=_f1(micro_precision, micro_recall),
        macro_f1=sum(item.f1 for item in per_concept) / len(per_concept),
        hamming_loss=mismatches / (len(records) * len(ontology.concepts)),
        invalid_output_rate=(
            sum(not record.is_valid for record in records) / len(records)
        ),
        per_concept=tuple(per_concept),
    )


def _canonicalize(
    item: MorphologyPredictionInput,
    ontology: SkinConOntology,
) -> MorphologyPredictionRecord:
    parsed = _parse_output(item.raw_output, ontology)
    return MorphologyPredictionRecord(
        sample_id=item.sample_id,
        leakage_group_id=item.leakage_group_id,
        true_concepts=item.true_concepts,
        raw_output=item.raw_output,
        predicted_concepts=parsed if parsed is not None else (),
        is_valid=parsed is not None,
        checkpoint_id=item.checkpoint_id,
        seed=item.seed,
    )


def _parse_output(text: str, ontology: SkinConOntology) -> tuple[str, ...] | None:
    try:
        value: object = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "positive_concepts",
        "all_concepts_annotated",
    }:
        return None
    if value.get("all_concepts_annotated") is not True:
        return None
    raw = value.get("positive_concepts")
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        return None
    strings = tuple(item for item in raw if isinstance(item, str))
    if len(strings) != len(set(strings)) or not set(strings) <= set(ontology.concepts):
        return None
    return tuple(concept for concept in ontology.concepts if concept in set(strings))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return (
        2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
