"""Metrics for closed-set ranked dermatology predictions."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from src.benchmark.runner import BenchmarkPrediction


def compute_metrics(
    predictions: Iterable[BenchmarkPrediction],
    *,
    allowed_disease_ids: list[str],
) -> dict[str, float | int]:
    """Compute diagnostic and structured-output metrics."""

    rows = list(predictions)
    count = len(rows)
    if count == 0:
        raise ValueError("At least one prediction is required")

    top_hits = {1: 0, 3: 0, 6: 0}
    reciprocal_rank_sum = 0.0
    json_valid_count = 0
    schema_valid_count = 0
    invalid_disease_output_count = 0
    duplicate_output_count = 0
    top_one_predictions: list[str | None] = []

    for row in rows:
        response = row.response
        json_valid_count += int(response.json_valid)
        schema_valid_count += int(response.schema_valid)
        parsed_predictions = (
            response.parsed_output.get("predictions", [])
            if isinstance(response.parsed_output, dict)
            else []
        )
        disease_ids = [
            item.get("disease_id")
            for item in parsed_predictions
            if isinstance(item, dict)
            and isinstance(item.get("disease_id"), str)
        ]
        invalid_disease_output_count += int(
            any(value not in allowed_disease_ids for value in disease_ids)
        )
        duplicate_output_count += int(
            len(disease_ids) != len(set(disease_ids))
        )
        ranked = disease_ids if response.schema_valid else []
        top_one_predictions.append(ranked[0] if ranked else None)
        if row.ground_truth_disease_id in ranked:
            rank = ranked.index(row.ground_truth_disease_id) + 1
            reciprocal_rank_sum += 1.0 / rank
            for top_k in top_hits:
                top_hits[top_k] += int(rank <= top_k)

    return {
        "sample_count": count,
        "top_1_accuracy": top_hits[1] / count,
        "top_3_accuracy": top_hits[3] / count,
        "top_6_accuracy": top_hits[6] / count,
        "mean_reciprocal_rank": reciprocal_rank_sum / count,
        "macro_f1_top_1": _macro_f1(
            truths=[row.ground_truth_disease_id for row in rows],
            predictions=top_one_predictions,
            labels=allowed_disease_ids,
        ),
        "json_validity_rate": json_valid_count / count,
        "schema_compliance_rate": schema_valid_count / count,
        "invalid_disease_id_rate": invalid_disease_output_count / count,
        "duplicate_prediction_rate": duplicate_output_count / count,
    }


def _macro_f1(
    *,
    truths: list[str],
    predictions: list[str | None],
    labels: list[str],
) -> float:
    truth_counts = Counter(truths)
    prediction_counts = Counter(predictions)
    true_positive_counts = Counter(
        truth
        for truth, prediction in zip(truths, predictions, strict=True)
        if truth == prediction
    )
    scores: list[float] = []
    for label in labels:
        true_positive = true_positive_counts[label]
        false_positive = prediction_counts[label] - true_positive
        false_negative = truth_counts[label] - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(
            0.0
            if denominator == 0
            else (2 * true_positive) / denominator
        )
    return sum(scores) / len(scores)
