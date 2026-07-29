"""Deterministic metrics for evidence-grounded dermatology diagnosis."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any

from src.benchmark.evidence_validation import (
    extract_morphology_concepts,
)
from src.benchmark.runner import BenchmarkPrediction, ModelResponse


def compute_evidence_grounded_metrics(
    predictions: Iterable[BenchmarkPrediction],
    *,
    allowed_disease_ids: Sequence[str],
    allowed_concept_ids: Sequence[str],
    minimum_positive_cases_per_concept: int = 20,
    calibration_bins: int = 10,
) -> dict[str, float | int]:
    """Compute the benchmark's morphology, diagnosis, and grounding metrics.

    Cohorts are selected by the ``score_morphology``, ``score_description``,
    and ``score_diagnosis`` fields stored in each prediction's metadata.
    Structurally or semantically invalid final answers count as incorrect for
    task-performance metrics, while their parseable fields still contribute to
    explicit structure and evidence-link audit rates.
    """

    rows = list(predictions)
    if not rows:
        raise ValueError("At least one prediction is required")
    if minimum_positive_cases_per_concept <= 0:
        raise ValueError(
            "minimum_positive_cases_per_concept must be positive"
        )
    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive")

    disease_ids = [str(value) for value in allowed_disease_ids]
    concept_ids = [str(value) for value in allowed_concept_ids]
    disease_id_set = set(disease_ids)
    concept_id_set = set(concept_ids)

    morphology_rows = [
        row
        for row in rows
        if _cohort_enabled(
            row.metadata,
            "score_morphology",
            fallback="morphology_concept_ids" in row.metadata,
        )
    ]
    description_rows = [
        row
        for row in rows
        if _cohort_enabled(
            row.metadata,
            "score_description",
            fallback=False,
        )
    ]
    diagnosis_rows = [
        row
        for row in rows
        if _cohort_enabled(
            row.metadata,
            "score_diagnosis",
            fallback=bool(row.ground_truth_disease_id),
        )
    ]

    morphology = _morphology_metrics(
        morphology_rows,
        allowed_concept_ids=concept_ids,
        allowed_concept_id_set=concept_id_set,
        minimum_positive_cases_per_concept=(
            minimum_positive_cases_per_concept
        ),
    )
    description = _description_metrics(
        description_rows,
        allowed_concept_id_set=concept_id_set,
    )
    diagnosis = _diagnosis_metrics(
        diagnosis_rows,
        allowed_disease_ids=disease_ids,
        allowed_disease_id_set=disease_id_set,
    )
    grounding = _grounding_metrics(
        diagnosis_rows,
        allowed_concept_id_set=concept_id_set,
    )
    calibration = _calibration_metrics(
        diagnosis_rows,
        bins=calibration_bins,
        allowed_disease_id_set=disease_id_set,
    )
    structure = _structure_metrics(rows)

    return {
        "sample_count": len(rows),
        "morphology_sample_count": len(morphology_rows),
        "description_sample_count": len(description_rows),
        "diagnosis_sample_count": len(diagnosis_rows),
        **morphology,
        **description,
        **diagnosis,
        **grounding,
        **calibration,
        **structure,
    }


def _morphology_metrics(
    rows: list[BenchmarkPrediction],
    *,
    allowed_concept_ids: list[str],
    allowed_concept_id_set: set[str],
    minimum_positive_cases_per_concept: int,
) -> dict[str, float | int]:
    if not rows:
        return {
            "finding_precision": 0.0,
            "finding_recall": 0.0,
            "finding_f1": 0.0,
            "micro_f1_all_concepts": 0.0,
            "macro_f1_supported_concepts": 0.0,
            "supported_macro_concept_count": 0,
            "unsupported_finding_rate": 0.0,
        }

    sample_precision: list[float] = []
    sample_recall: list[float] = []
    sample_f1: list[float] = []
    reference_sets: list[set[str]] = []
    prediction_sets: list[set[str]] = []
    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0
    total_predicted = 0

    for row in rows:
        reference = _reference_concepts(
            row,
            allowed_concept_id_set=allowed_concept_id_set,
        )
        predicted = (
            _declared_concepts(row.response) & allowed_concept_id_set
            if _fully_valid(row.response)
            else set()
        )
        precision, recall, f1 = _set_scores(predicted, reference)
        sample_precision.append(precision)
        sample_recall.append(recall)
        sample_f1.append(f1)
        reference_sets.append(reference)
        prediction_sets.append(predicted)
        total_true_positive += len(predicted & reference)
        total_false_positive += len(predicted - reference)
        total_false_negative += len(reference - predicted)
        total_predicted += len(predicted)

    supported = [
        concept_id
        for concept_id in allowed_concept_ids
        if sum(concept_id in reference for reference in reference_sets)
        >= minimum_positive_cases_per_concept
    ]
    per_concept_f1 = [
        _binary_f1(
            truths=[
                concept_id in reference for reference in reference_sets
            ],
            predictions=[
                concept_id in predicted for predicted in prediction_sets
            ],
        )
        for concept_id in supported
    ]
    return {
        "finding_precision": _mean(sample_precision),
        "finding_recall": _mean(sample_recall),
        "finding_f1": _mean(sample_f1),
        "micro_f1_all_concepts": _f1_from_counts(
            total_true_positive,
            total_false_positive,
            total_false_negative,
        ),
        "macro_f1_supported_concepts": _mean(per_concept_f1),
        "supported_macro_concept_count": len(supported),
        "unsupported_finding_rate": (
            total_false_positive / total_predicted
            if total_predicted
            else 0.0
        ),
    }


def _description_metrics(
    rows: list[BenchmarkPrediction],
    *,
    allowed_concept_id_set: set[str],
) -> dict[str, float]:
    if not rows:
        return {
            "description_concept_precision": 0.0,
            "description_concept_recall": 0.0,
            "description_concept_f1": 0.0,
            "description_findings_consistency": 0.0,
            "description_unsupported_concept_rate": 0.0,
        }

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    f1_scores: list[float] = []
    consistency_scores: list[float] = []
    unsupported_count = 0
    described_count = 0
    for row in rows:
        reference = _reference_concepts(
            row,
            allowed_concept_id_set=allowed_concept_id_set,
        )
        if _fully_valid(row.response):
            described = _description_concepts(
                row.response,
                allowed_concept_id_set=allowed_concept_id_set,
            )
            declared = (
                _declared_concepts(row.response) & allowed_concept_id_set
            )
            consistency = _set_scores(described, declared)[2]
        else:
            described = set()
            consistency = 0.0
        precision, recall, f1 = _set_scores(described, reference)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        consistency_scores.append(consistency)
        unsupported_count += len(described - reference)
        described_count += len(described)
    return {
        "description_concept_precision": _mean(precision_scores),
        "description_concept_recall": _mean(recall_scores),
        "description_concept_f1": _mean(f1_scores),
        "description_findings_consistency": _mean(consistency_scores),
        "description_unsupported_concept_rate": (
            unsupported_count / described_count if described_count else 0.0
        ),
    }


def _diagnosis_metrics(
    rows: list[BenchmarkPrediction],
    *,
    allowed_disease_ids: list[str],
    allowed_disease_id_set: set[str],
) -> dict[str, float]:
    if not rows:
        return {
            "diagnosis_class_count": 0,
            "top_1_accuracy": 0.0,
            "top_3_accuracy": 0.0,
            "top_6_accuracy": 0.0,
            "mean_reciprocal_rank": 0.0,
            "macro_f1_top_1": 0.0,
        }

    hits = {1: 0, 3: 0, 6: 0}
    reciprocal_rank = 0.0
    top_one_predictions: list[str | None] = []
    truths: list[str] = []
    for row in rows:
        ranked = (
            [
                disease_id
                for disease_id in _ranked_disease_ids(row.response)
                if disease_id in allowed_disease_id_set
            ]
            if _fully_valid(row.response)
            else []
        )
        truth = row.ground_truth_disease_id
        truths.append(truth)
        top_one_predictions.append(ranked[0] if ranked else None)
        if truth in ranked:
            rank = ranked.index(truth) + 1
            reciprocal_rank += 1.0 / rank
            for k in hits:
                hits[k] += int(rank <= k)
    count = len(rows)
    covered_labels = [
        disease_id
        for disease_id in allowed_disease_ids
        if disease_id in set(truths)
    ]
    return {
        "diagnosis_class_count": len(covered_labels),
        "top_1_accuracy": hits[1] / count,
        "top_3_accuracy": hits[3] / count,
        "top_6_accuracy": hits[6] / count,
        "mean_reciprocal_rank": reciprocal_rank / count,
        "macro_f1_top_1": _macro_f1(
            truths=truths,
            predictions=top_one_predictions,
            labels=covered_labels,
        ),
    }


def _grounding_metrics(
    rows: list[BenchmarkPrediction],
    *,
    allowed_concept_id_set: set[str],
) -> dict[str, float]:
    if not rows:
        return {
            "visible_evidence_precision": 0.0,
            "valid_evidence_link_rate": 0.0,
            "grounded_top_1_success": 0.0,
            "correct_diagnosis_unsupported_evidence_rate": 0.0,
        }

    total_links = 0
    resolved_links = 0
    resolved_visible_links = 0
    grounded_top_one = 0
    structurally_correct_top_one = 0
    unsupported_correct_top_one = 0
    for row in rows:
        reference = _reference_concepts(
            row,
            allowed_concept_id_set=allowed_concept_id_set,
        )
        finding_map, differential = _evidence_fields(row.response)
        for diagnosis in differential:
            for support_id in _support_ids(diagnosis):
                total_links += 1
                concept_id = finding_map.get(support_id)
                if concept_id is None:
                    continue
                resolved_links += 1
                resolved_visible_links += int(concept_id in reference)

        top = differential[0] if differential else None
        top_disease = (
            top.get("disease_id") if isinstance(top, dict) else None
        )
        top_support_ids = _support_ids(top)
        top_concepts = [
            finding_map[support_id]
            for support_id in top_support_ids
            if support_id in finding_map
        ]
        structurally_correct = (
            row.response.json_valid
            and row.response.schema_valid
            and top_disease == row.ground_truth_disease_id
        )
        if structurally_correct:
            structurally_correct_top_one += 1
            unsupported_correct_top_one += int(
                not top_concepts
                or all(
                    concept_id not in reference
                    for concept_id in top_concepts
                )
            )
        grounded_top_one += int(
            _fully_valid(row.response)
            and top_disease == row.ground_truth_disease_id
            and bool(top_support_ids)
            and len(top_concepts) == len(top_support_ids)
            and all(concept_id in reference for concept_id in top_concepts)
        )
    return {
        "visible_evidence_precision": (
            resolved_visible_links / resolved_links
            if resolved_links
            else 0.0
        ),
        "valid_evidence_link_rate": (
            resolved_links / total_links if total_links else 0.0
        ),
        "grounded_top_1_success": grounded_top_one / len(rows),
        "correct_diagnosis_unsupported_evidence_rate": (
            unsupported_correct_top_one / structurally_correct_top_one
            if structurally_correct_top_one
            else 0.0
        ),
    }


def _calibration_metrics(
    rows: list[BenchmarkPrediction],
    *,
    bins: int,
    allowed_disease_id_set: set[str],
) -> dict[str, float | int]:
    confidence_outcomes: list[tuple[float, int]] = []
    for row in rows:
        differential = _differential(row.response)
        if not row.response.json_valid:
            continue
        if not differential or not isinstance(differential[0], dict):
            continue
        top = differential[0]
        confidence = top.get("confidence")
        if not _is_probability(confidence):
            continue
        disease_id = top.get("disease_id")
        correct = int(
            _fully_valid(row.response)
            and disease_id in allowed_disease_id_set
            and disease_id == row.ground_truth_disease_id
        )
        confidence_outcomes.append((float(confidence), correct))

    if not confidence_outcomes:
        return {
            "calibration_sample_count": 0,
            "top_1_expected_calibration_error": 0.0,
            "top_1_brier_score": 0.0,
        }

    binned: list[list[tuple[float, int]]] = [
        [] for _ in range(bins)
    ]
    for confidence, outcome in confidence_outcomes:
        index = min(int(confidence * bins), bins - 1)
        binned[index].append((confidence, outcome))
    sample_count = len(confidence_outcomes)
    expected_calibration_error = sum(
        (len(bucket) / sample_count)
        * abs(
            _mean([confidence for confidence, _ in bucket])
            - _mean([outcome for _, outcome in bucket])
        )
        for bucket in binned
        if bucket
    )
    brier_score = _mean(
        [
            (confidence - outcome) ** 2
            for confidence, outcome in confidence_outcomes
        ]
    )
    return {
        "calibration_sample_count": sample_count,
        "top_1_expected_calibration_error": expected_calibration_error,
        "top_1_brier_score": brier_score,
    }


def _structure_metrics(
    rows: list[BenchmarkPrediction],
) -> dict[str, float]:
    count = len(rows)
    json_valid = 0
    schema_valid = 0
    semantic_valid = 0
    audit_counts = Counter(
        {
            "invalid_disease_id": 0,
            "invalid_concept_id": 0,
            "broken_evidence_reference": 0,
            "duplicate_prediction": 0,
            "duplicate_finding": 0,
            "forbidden_description_content": 0,
        }
    )
    for row in rows:
        response = row.response
        json_valid += int(response.json_valid)
        schema_valid += int(response.schema_valid)
        semantic_valid += int(
            response.json_valid
            and response.schema_valid
            and bool(response.metadata.get("semantic_valid", False))
        )
        audit = response.metadata.get("audit")
        fallback = _audit_response(response)
        for key in audit_counts:
            value = (
                audit.get(key, fallback[key])
                if isinstance(audit, Mapping)
                else fallback[key]
            )
            audit_counts[key] += int(bool(value))
    return {
        "json_validity_rate": json_valid / count,
        "schema_compliance_rate": schema_valid / count,
        "semantic_compliance_rate": semantic_valid / count,
        "invalid_disease_id_rate": (
            audit_counts["invalid_disease_id"] / count
        ),
        "invalid_concept_id_rate": (
            audit_counts["invalid_concept_id"] / count
        ),
        "broken_evidence_reference_rate": (
            audit_counts["broken_evidence_reference"] / count
        ),
        "duplicate_prediction_rate": (
            audit_counts["duplicate_prediction"] / count
        ),
        "duplicate_finding_rate": (
            audit_counts["duplicate_finding"] / count
        ),
        "forbidden_description_content_rate": (
            audit_counts["forbidden_description_content"] / count
        ),
    }


def _audit_response(response: ModelResponse) -> dict[str, bool]:
    finding_map, differential = _evidence_fields(response)
    findings = _findings(response)
    finding_ids = [
        item.get("finding_id")
        for item in findings
        if isinstance(item, dict)
        and isinstance(item.get("finding_id"), str)
    ]
    concept_ids = [
        item.get("concept_id")
        for item in findings
        if isinstance(item, dict)
        and isinstance(item.get("concept_id"), str)
    ]
    disease_ids = [
        item.get("disease_id")
        for item in differential
        if isinstance(item, dict)
        and isinstance(item.get("disease_id"), str)
    ]
    support_ids = [
        support_id
        for item in differential
        for support_id in _support_ids(item)
    ]
    return {
        "invalid_disease_id": any(
            "disease_id_unknown" in error
            for error in response.validation_errors
        ),
        "invalid_concept_id": any(
            "concept_id_unknown" in error
            for error in response.validation_errors
        ),
        "broken_evidence_reference": any(
            support_id not in finding_map for support_id in support_ids
        ),
        "duplicate_prediction": len(disease_ids)
        != len(set(disease_ids)),
        "duplicate_finding": (
            len(finding_ids) != len(set(finding_ids))
            or len(concept_ids) != len(set(concept_ids))
        ),
        "forbidden_description_content": any(
            "clinical_description_contains_forbidden_content" in error
            for error in response.validation_errors
        ),
    }


def _cohort_enabled(
    metadata: Mapping[str, Any],
    key: str,
    *,
    fallback: bool,
) -> bool:
    return bool(metadata[key]) if key in metadata else fallback


def _fully_valid(response: ModelResponse) -> bool:
    return (
        response.json_valid
        and response.schema_valid
        and bool(response.metadata.get("semantic_valid", False))
    )


def _reference_concepts(
    row: BenchmarkPrediction,
    *,
    allowed_concept_id_set: set[str],
) -> set[str]:
    values = row.metadata.get("morphology_concept_ids", ())
    if isinstance(values, str):
        values = [values]
    try:
        return {
            str(value)
            for value in values
            if str(value) in allowed_concept_id_set
        }
    except TypeError:
        return set()


def _declared_concepts(response: ModelResponse) -> set[str]:
    return {
        str(item["concept_id"])
        for item in _findings(response)
        if isinstance(item, dict)
        and isinstance(item.get("concept_id"), str)
    }


def _description_concepts(
    response: ModelResponse,
    *,
    allowed_concept_id_set: set[str],
) -> set[str]:
    stored = response.metadata.get("description_concept_ids")
    if isinstance(stored, (list, tuple, set, frozenset)):
        return {
            str(value)
            for value in stored
            if str(value) in allowed_concept_id_set
        }
    output = response.parsed_output
    description = (
        output.get("clinical_description", "")
        if isinstance(output, dict)
        else ""
    )
    return extract_morphology_concepts(
        description,
        allowed_concept_ids=allowed_concept_id_set,
    )


def _findings(response: ModelResponse) -> list[Any]:
    output = response.parsed_output
    findings = output.get("findings", []) if isinstance(output, dict) else []
    return findings if isinstance(findings, list) else []


def _differential(response: ModelResponse) -> list[Any]:
    output = response.parsed_output
    differential = (
        output.get("differential", []) if isinstance(output, dict) else []
    )
    return differential if isinstance(differential, list) else []


def _ranked_disease_ids(response: ModelResponse) -> list[str]:
    return [
        str(item["disease_id"])
        for item in _differential(response)
        if isinstance(item, dict)
        and isinstance(item.get("disease_id"), str)
    ]


def _evidence_fields(
    response: ModelResponse,
) -> tuple[dict[str, str], list[Any]]:
    finding_map = {
        str(item["finding_id"]): str(item["concept_id"])
        for item in _findings(response)
        if isinstance(item, dict)
        and isinstance(item.get("finding_id"), str)
        and isinstance(item.get("concept_id"), str)
    }
    return finding_map, _differential(response)


def _support_ids(diagnosis: Any) -> list[str]:
    if not isinstance(diagnosis, dict):
        return []
    values = diagnosis.get("supporting_finding_ids", [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if isinstance(value, str)]


def _set_scores(
    predicted: set[str],
    reference: set[str],
) -> tuple[float, float, float]:
    true_positive = len(predicted & reference)
    precision = (
        true_positive / len(predicted)
        if predicted
        else float(not reference)
    )
    recall = (
        true_positive / len(reference)
        if reference
        else float(not predicted)
    )
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return precision, recall, f1


def _binary_f1(
    *,
    truths: list[bool],
    predictions: list[bool],
) -> float:
    true_positive = sum(
        truth and prediction
        for truth, prediction in zip(truths, predictions, strict=True)
    )
    false_positive = sum(
        not truth and prediction
        for truth, prediction in zip(truths, predictions, strict=True)
    )
    false_negative = sum(
        truth and not prediction
        for truth, prediction in zip(truths, predictions, strict=True)
    )
    return _f1_from_counts(
        true_positive,
        false_positive,
        false_negative,
    )


def _f1_from_counts(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> float:
    denominator = (
        2 * true_positive + false_positive + false_negative
    )
    return (
        0.0
        if denominator == 0
        else 2 * true_positive / denominator
    )


def _macro_f1(
    *,
    truths: list[str],
    predictions: list[str | None],
    labels: list[str],
) -> float:
    if not labels:
        return 0.0
    truth_counts = Counter(truths)
    prediction_counts = Counter(predictions)
    true_positive_counts = Counter(
        truth
        for truth, prediction in zip(truths, predictions, strict=True)
        if truth == prediction
    )
    scores = []
    for label in labels:
        true_positive = true_positive_counts[label]
        false_positive = prediction_counts[label] - true_positive
        false_negative = truth_counts[label] - true_positive
        scores.append(
            _f1_from_counts(
                true_positive,
                false_positive,
                false_negative,
            )
        )
    return _mean(scores)


def _is_probability(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _mean(values: Sequence[float | int]) -> float:
    return sum(values) / len(values) if values else 0.0
