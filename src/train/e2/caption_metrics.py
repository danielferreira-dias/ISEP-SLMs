"""Deterministic, judge-free quality metrics for SkinCAP observation captions."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from src.train.e2.domain import SkinConOntology


@dataclass(frozen=True, slots=True)
class CaptionPredictionInput:
    """Raw candidate paired with its human reference and linked morphology."""

    sample_id: str
    leakage_group_id: str
    reference_text: str
    true_concepts: tuple[str, ...]
    raw_output: str
    checkpoint_id: str
    seed: int


@dataclass(frozen=True, slots=True)
class CaptionPredictionRecord:
    """Parsed caption evidence used for deterministic aggregate metrics."""

    sample_id: str
    leakage_group_id: str
    reference_text: str
    true_concepts: tuple[str, ...]
    raw_output: str
    normalized_output: str
    predicted_concepts: tuple[str, ...]
    is_nonempty: bool
    is_one_sentence: bool
    length_compliant: bool
    contains_prohibited_content: bool
    is_compliant: bool
    rouge_l_f1: float
    token_f1: float
    checkpoint_id: str
    seed: int


@dataclass(frozen=True, slots=True)
class CaptionMetrics:
    """Aggregate reference, concept, format, and unsupported-claim metrics."""

    sample_count: int
    valid_response_rate: float
    one_sentence_rate: float
    length_compliance_rate: float
    prohibited_content_rate: float
    clinical_compliance_rate: float
    rouge_l_f1_mean: float
    token_f1_mean: float
    reference_similarity_mean: float
    concept_precision: float
    concept_recall: float
    concept_f1: float
    unsupported_concept_rate: float
    caption_task_score: float


def canonicalize_caption_predictions(
    inputs: tuple[CaptionPredictionInput, ...],
    ontology: SkinConOntology,
    forbidden_labels: tuple[str, ...],
) -> tuple[CaptionPredictionRecord, ...]:
    """Parse captions with a frozen ontology and deterministic compliance rules."""

    return tuple(_canonicalize(item, ontology, forbidden_labels) for item in inputs)


def evaluate_caption_predictions(
    records: tuple[CaptionPredictionRecord, ...],
) -> CaptionMetrics:
    """Compute task-level SkinCAP metrics without an LLM-as-a-judge."""

    if not records:
        raise ValueError("Caption evaluation requires at least one prediction")
    true_positive = false_positive = false_negative = 0
    for record in records:
        truth = set(record.true_concepts)
        predicted = set(record.predicted_concepts)
        true_positive += len(truth & predicted)
        false_positive += len(predicted - truth)
        false_negative += len(truth - predicted)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    concept_f1 = _f1(precision, recall)
    rouge = _mean(tuple(item.rouge_l_f1 for item in records))
    token_f1 = _mean(tuple(item.token_f1 for item in records))
    reference_similarity = (rouge + token_f1) / 2.0
    compliance = _ratio(sum(item.is_compliant for item in records), len(records))
    return CaptionMetrics(
        sample_count=len(records),
        valid_response_rate=_ratio(
            sum(item.is_nonempty for item in records), len(records)
        ),
        one_sentence_rate=_ratio(
            sum(item.is_one_sentence for item in records), len(records)
        ),
        length_compliance_rate=_ratio(
            sum(item.length_compliant for item in records), len(records)
        ),
        prohibited_content_rate=_ratio(
            sum(item.contains_prohibited_content for item in records), len(records)
        ),
        clinical_compliance_rate=compliance,
        rouge_l_f1_mean=rouge,
        token_f1_mean=token_f1,
        reference_similarity_mean=reference_similarity,
        concept_precision=precision,
        concept_recall=recall,
        concept_f1=concept_f1,
        unsupported_concept_rate=_ratio(false_positive, true_positive + false_positive),
        caption_task_score=(compliance + concept_f1 + reference_similarity) / 3.0,
    )


def _canonicalize(
    item: CaptionPredictionInput,
    ontology: SkinConOntology,
    forbidden_labels: tuple[str, ...],
) -> CaptionPredictionRecord:
    output = item.raw_output.strip()
    words = _tokens(output)
    nonempty = bool(output)
    one_sentence = nonempty and _sentence_count(output) == 1
    length_compliant = 5 <= len(words) <= 60
    prohibited = _contains_prohibited(output, forbidden_labels)
    predicted = tuple(
        concept for concept in ontology.concepts if _contains_phrase(output, concept)
    )
    return CaptionPredictionRecord(
        sample_id=item.sample_id,
        leakage_group_id=item.leakage_group_id,
        reference_text=item.reference_text,
        true_concepts=item.true_concepts,
        raw_output=item.raw_output,
        normalized_output=output,
        predicted_concepts=predicted,
        is_nonempty=nonempty,
        is_one_sentence=one_sentence,
        length_compliant=length_compliant,
        contains_prohibited_content=prohibited,
        is_compliant=(
            nonempty and one_sentence and length_compliant and not prohibited
        ),
        rouge_l_f1=_rouge_l_f1(item.reference_text, output),
        token_f1=_token_f1(item.reference_text, output),
        checkpoint_id=item.checkpoint_id,
        seed=item.seed,
    )


def _contains_prohibited(text: str, labels: tuple[str, ...]) -> bool:
    normalized = _normalize(text)
    triggers = (
        "diagnosis",
        "differential diagnosis",
        "biopsy",
        "test for",
        "testing",
        "treatment",
        "management",
        "recommend",
        "prognosis",
        "follow up",
    )
    return any(_contains_phrase(normalized, item) for item in (*triggers, *labels))


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = _normalize(text)
    normalized_phrase = _normalize(phrase)
    return (
        re.search(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)", normalized_text)
        is not None
    )


def _sentence_count(text: str) -> int:
    segments = tuple(item.strip() for item in re.split(r"[.!?]+", text) if item.strip())
    return len(segments)


def _rouge_l_f1(reference: str, candidate: str) -> float:
    left = _tokens(reference)
    right = _tokens(candidate)
    if not left or not right:
        return 0.0
    common = _longest_common_subsequence(left, right)
    return _f1(common / len(right), common / len(left))


def _token_f1(reference: str, candidate: str) -> float:
    left = set(_tokens(reference))
    right = set(_tokens(candidate))
    if not left or not right:
        return 0.0
    common = len(left & right)
    return _f1(common / len(right), common / len(left))


def _longest_common_subsequence(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, 1):
            current.append(
                previous[index - 1] + 1
                if left_token == right_token
                else max(previous[index], current[index - 1])
            )
        previous = current
    return previous[-1]


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", _normalize(text)))


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join(
        "".join(char for char in decomposed if not unicodedata.combining(char)).split()
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return (
        2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    )


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)
