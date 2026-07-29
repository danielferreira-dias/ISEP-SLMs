"""Metrics for closed-set ranked dermatology predictions."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

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


def compute_confusion_set_metrics(
    predictions: Iterable[BenchmarkPrediction],
    *,
    allowed_disease_ids: list[str],
    bootstrap_resamples: int = 10000,
    bootstrap_seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Compute paired metrics for three-way disease confusion tasks."""

    rows = list(predictions)
    if not rows:
        raise ValueError("At least one prediction is required")
    if bootstrap_resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between zero and one")

    scored_rows: list[dict[str, Any]] = []
    json_valid_count = 0
    schema_valid_count = 0
    invalid_candidate_count = 0
    duplicate_count = 0
    reciprocal_rank_sum = 0.0
    top_one_predictions: list[str | None] = []
    top_one_hits = 0
    top_two_hits = 0
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
        candidates = {
            str(value)
            for value in row.metadata.get(
                "candidate_disease_ids",
                allowed_disease_ids,
            )
        }
        invalid_candidate_count += int(
            any(value not in candidates for value in disease_ids)
        )
        duplicate_count += int(
            len(disease_ids) != len(set(disease_ids))
        )
        ranked = disease_ids if response.schema_valid else []
        top_one = ranked[0] if ranked else None
        top_one_predictions.append(top_one)
        rank = (
            ranked.index(row.ground_truth_disease_id) + 1
            if row.ground_truth_disease_id in ranked
            else None
        )
        top_one_correct = int(rank == 1)
        top_two_correct = int(rank is not None and rank <= 2)
        top_one_hits += top_one_correct
        top_two_hits += top_two_correct
        if rank is not None:
            reciprocal_rank_sum += 1.0 / rank
        scored_rows.append(
            {
                "pair_id": str(row.metadata.get("pair_id", "")),
                "difficulty": str(
                    row.metadata.get("difficulty", "")
                ),
                "confusion_set_id": str(
                    row.metadata.get("confusion_set_id", "")
                ),
                "disease_id": row.ground_truth_disease_id,
                "top_one_correct": top_one_correct,
            }
        )

    count = len(rows)
    covered_labels = sorted(
        {
            row.ground_truth_disease_id
            for row in rows
        }
    )
    by_condition = _group_accuracy(
        scored_rows,
        group_key="difficulty",
    )
    by_set = _group_accuracy(
        scored_rows,
        group_key="confusion_set_id",
    )
    by_disease = _group_accuracy(
        scored_rows,
        group_key="disease_id",
    )
    low_accuracy = by_condition.get("low_confusability", 0.0)
    high_accuracy = by_condition.get("high_confusability", 0.0)
    paired_differences = _paired_condition_differences(scored_rows)
    gap = low_accuracy - high_accuracy
    ci_lower, ci_upper = _bootstrap_mean_interval(
        paired_differences,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    return {
        "task_count": count,
        "pair_count": len(paired_differences),
        "top_1_accuracy": top_one_hits / count,
        "top_2_accuracy": top_two_hits / count,
        "mean_reciprocal_rank": reciprocal_rank_sum / count,
        "macro_f1_top_1": _macro_f1(
            truths=[
                row.ground_truth_disease_id
                for row in rows
            ],
            predictions=top_one_predictions,
            labels=covered_labels,
        ),
        "macro_set_top_1_accuracy": (
            sum(by_set.values()) / len(by_set)
            if by_set
            else 0.0
        ),
        "low_confusability_accuracy": low_accuracy,
        "high_confusability_accuracy": high_accuracy,
        "confusability_accuracy_gap": gap,
        "confusability_accuracy_gap_ci_lower": ci_lower,
        "confusability_accuracy_gap_ci_upper": ci_upper,
        "json_validity_rate": json_valid_count / count,
        "schema_compliance_rate": schema_valid_count / count,
        "invalid_candidate_id_rate": invalid_candidate_count / count,
        "duplicate_prediction_rate": duplicate_count / count,
        "by_condition_top_1_accuracy": by_condition,
        "by_confusion_set_top_1_accuracy": by_set,
        "by_disease_top_1_accuracy": by_disease,
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


def _group_accuracy(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
) -> dict[str, float]:
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(int(row["top_one_correct"]))
    return {
        group: sum(values) / len(values)
        for group, values in sorted(grouped.items())
        if group
    }


def _paired_condition_differences(
    rows: list[dict[str, Any]],
) -> list[float]:
    paired: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        pair_id = str(row["pair_id"])
        difficulty = str(row["difficulty"])
        if not pair_id or not difficulty:
            raise ValueError(
                "Confusion-set metrics require pair_id and difficulty metadata"
            )
        if difficulty in paired[pair_id]:
            raise ValueError(
                f"Duplicate {difficulty} task for pair {pair_id}"
            )
        paired[pair_id][difficulty] = int(row["top_one_correct"])
    differences: list[float] = []
    for pair_id, conditions in paired.items():
        if set(conditions) != {
            "low_confusability",
            "high_confusability",
        }:
            raise ValueError(
                f"Pair {pair_id} does not contain both difficulty conditions"
            )
        differences.append(
            float(
                conditions["low_confusability"]
                - conditions["high_confusability"]
            )
        )
    return differences


def _bootstrap_mean_interval(
    values: list[float],
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float]:
    if not values:
        raise ValueError("Paired bootstrap requires at least one pair")
    if len(values) == 1:
        return values[0], values[0]
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        len(array),
        size=(resamples, len(array)),
    )
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lower), float(upper)
