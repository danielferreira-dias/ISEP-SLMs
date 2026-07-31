"""Metrics for closed-set ranked dermatology predictions."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable as IterableABC
import math
import re
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np

from src.benchmark.runner import BenchmarkPrediction


def compute_metrics(
    predictions: Iterable[BenchmarkPrediction],
    *,
    allowed_disease_ids: list[str],
    minimum_subgroup_unique_groups: int = 30,
    minimum_per_disease_unique_groups: int = 10,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Compute diagnostic and structured-output metrics."""

    rows = list(predictions)
    count = len(rows)
    if count == 0:
        raise ValueError("At least one prediction is required")
    if minimum_subgroup_unique_groups <= 0:
        raise ValueError("Minimum subgroup size must be positive")
    if minimum_per_disease_unique_groups <= 0:
        raise ValueError("Minimum per-disease subgroup size must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between zero and one")

    top_hits = {1: 0, 3: 0, 6: 0}
    canonical_top_hits = {1: 0, 3: 0, 6: 0}
    reciprocal_rank_sum = 0.0
    canonical_reciprocal_rank_sum = 0.0
    json_valid_count = 0
    recoverable_json_valid_count = 0
    schema_valid_count = 0
    canonical_schema_valid_count = 0
    invalid_disease_output_count = 0
    duplicate_output_count = 0
    top_one_predictions: list[str | None] = []
    canonical_top_one_predictions: list[str | None] = []
    scored_rows: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        response = row.response
        json_valid_count += int(response.json_valid)
        recoverable_json_valid_count += int(
            response.recoverable_json_valid
        )
        schema_valid_count += int(response.schema_valid)
        canonical_schema_valid_count += int(
            response.canonical_schema_valid
        )
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
        rank: int | None = None
        if row.ground_truth_disease_id in ranked:
            rank = ranked.index(row.ground_truth_disease_id) + 1
            reciprocal_rank_sum += 1.0 / rank
            for top_k in top_hits:
                top_hits[top_k] += int(rank <= top_k)
        canonical_ranked = _ranked_disease_ids(
            response.canonical_output
            if response.canonical_schema_valid
            else None
        )
        canonical_top_one_predictions.append(
            canonical_ranked[0] if canonical_ranked else None
        )
        if row.ground_truth_disease_id in canonical_ranked:
            canonical_rank = (
                canonical_ranked.index(row.ground_truth_disease_id) + 1
            )
            canonical_reciprocal_rank_sum += 1.0 / canonical_rank
            for top_k in canonical_top_hits:
                canonical_top_hits[top_k] += int(
                    canonical_rank <= top_k
                )
        exact_skin_tone = _skin_tone_group(row.metadata)
        scored_rows.append(
            {
                "skin_tone": exact_skin_tone,
                "skin_tone_aggregate": _aggregate_skin_tone_group(
                    exact_skin_tone
                ),
                "unique_group_id": _unique_group_id(
                    row.metadata,
                    fallback=f"{row.sample_id}:{row_index}",
                ),
                "disease_id": row.ground_truth_disease_id,
                "top_1_correct": int(rank == 1),
                "top_3_correct": int(rank is not None and rank <= 3),
                "top_6_correct": int(rank is not None and rank <= 6),
                "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
            }
        )

    exact_skin_tone = _skin_tone_performance(
        scored_rows,
        group_key="skin_tone",
        minimum_subgroup_unique_groups=minimum_subgroup_unique_groups,
        minimum_per_disease_unique_groups=(
            minimum_per_disease_unique_groups
        ),
        confidence_level=confidence_level,
    )
    aggregate_skin_tone = _skin_tone_performance(
        scored_rows,
        group_key="skin_tone_aggregate",
        minimum_subgroup_unique_groups=minimum_subgroup_unique_groups,
        minimum_per_disease_unique_groups=(
            minimum_per_disease_unique_groups
        ),
        confidence_level=confidence_level,
    )
    supported_exact = [
        values
        for group, values in exact_skin_tone.items()
        if group != "unknown" and values["statistically_supported"]
    ]
    supported_top_one = [
        float(values["top_1_accuracy"])
        for values in supported_exact
    ]
    missing_skin_tone_count = sum(
        row["skin_tone"] == "unknown"
        for row in scored_rows
    )

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
        "canonical_top_1_accuracy": canonical_top_hits[1] / count,
        "canonical_top_3_accuracy": canonical_top_hits[3] / count,
        "canonical_top_6_accuracy": canonical_top_hits[6] / count,
        "canonical_mean_reciprocal_rank": (
            canonical_reciprocal_rank_sum / count
        ),
        "canonical_macro_f1_top_1": _macro_f1(
            truths=[row.ground_truth_disease_id for row in rows],
            predictions=canonical_top_one_predictions,
            labels=allowed_disease_ids,
        ),
        "json_validity_rate": json_valid_count / count,
        "recoverable_json_validity_rate": (
            recoverable_json_valid_count / count
        ),
        "schema_compliance_rate": schema_valid_count / count,
        "canonical_schema_compliance_rate": (
            canonical_schema_valid_count / count
        ),
        "invalid_disease_id_rate": invalid_disease_output_count / count,
        "duplicate_prediction_rate": duplicate_output_count / count,
        "skin_tone_coverage_rate": (
            count - missing_skin_tone_count
        )
        / count,
        "skin_tone_missing_count": missing_skin_tone_count,
        "skin_tone_supported_group_count": len(supported_exact),
        "skin_tone_worst_group_top_1_accuracy": (
            min(supported_top_one) if supported_top_one else None
        ),
        "skin_tone_top_1_accuracy_gap": (
            max(supported_top_one) - min(supported_top_one)
            if len(supported_top_one) >= 2
            else None
        ),
        "by_skin_tone": exact_skin_tone,
        "by_skin_tone_aggregate": aggregate_skin_tone,
    }


def _ranked_disease_ids(output: Any) -> list[str]:
    """Return ranked disease IDs from one canonical output object."""

    predictions = (
        output.get("predictions", [])
        if isinstance(output, dict)
        else []
    )
    return [
        item["disease_id"]
        for item in predictions
        if isinstance(item, dict)
        and isinstance(item.get("disease_id"), str)
    ]


def _skin_tone_group(metadata: dict[str, Any]) -> str:
    """Return a scale-qualified skin-tone label or ``unknown``."""

    tone = _metadata_text(metadata.get("skin_tone"))
    system = _metadata_text(metadata.get("skin_tone_system"))
    if not tone:
        return "unknown"
    if not system:
        if tone.casefold().startswith("fst_"):
            system = "fitzpatrick"
        elif tone.casefold().startswith("mst_"):
            system = "monk"
        else:
            system = "unspecified"
    return f"{system.casefold()}:{tone.upper()}"


def _aggregate_skin_tone_group(group: str) -> str:
    """Map exact FST/MST values to prespecified within-scale bands."""

    if group == "unknown":
        return group
    system, _, tone = group.partition(":")
    fitzpatrick = re.fullmatch(r"FST_([1-6])", tone)
    if fitzpatrick:
        value = int(fitzpatrick.group(1))
        lower = 1 if value <= 2 else 3 if value <= 4 else 5
        return f"fitzpatrick:FST_{lower}-{lower + 1}"
    fitzpatrick_band = re.fullmatch(r"FST_(12|34|56)", tone)
    if fitzpatrick_band and system.startswith("fitzpatrick"):
        lower, upper = (int(value) for value in fitzpatrick_band.group(1))
        return f"fitzpatrick:FST_{lower}-{upper}"
    monk = re.fullmatch(r"MST_(10|[1-9])", tone)
    if monk:
        value = int(monk.group(1))
        if value <= 3:
            return "monk:MST_1-3"
        if value <= 6:
            return "monk:MST_4-6"
        return "monk:MST_7-10"
    return group


def _unique_group_id(metadata: dict[str, Any], *, fallback: str) -> str:
    for key in ("leakage_group_id", "group_id"):
        value = _metadata_text(metadata.get(key))
        if value:
            return value
    return fallback


def _metadata_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "<na>"} else text


def _skin_tone_performance(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    minimum_subgroup_unique_groups: int,
    minimum_per_disease_unique_groups: int,
    confidence_level: float,
) -> dict[str, dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    results = {
        group: _skin_tone_group_metrics(
            group_rows,
            minimum_subgroup_unique_groups=(
                minimum_subgroup_unique_groups
            ),
            minimum_per_disease_unique_groups=(
                minimum_per_disease_unique_groups
            ),
            confidence_level=confidence_level,
        )
        for group, group_rows in sorted(grouped.items())
    }
    if "unknown" in results:
        results["unknown"]["statistically_supported"] = False
    return results


def _skin_tone_group_metrics(
    rows: list[dict[str, Any]],
    *,
    minimum_subgroup_unique_groups: int,
    minimum_per_disease_unique_groups: int,
    confidence_level: float,
) -> dict[str, Any]:
    sample_count = len(rows)
    unique_group_count = len(
        {str(row["unique_group_id"]) for row in rows}
    )
    result: dict[str, Any] = {
        "sample_count": sample_count,
        "unique_group_count": unique_group_count,
        "statistically_supported": (
            unique_group_count >= minimum_subgroup_unique_groups
        ),
    }
    for top_k in (1, 3, 6):
        hits = sum(int(row[f"top_{top_k}_correct"]) for row in rows)
        lower, upper = _wilson_interval(
            hits,
            sample_count,
            confidence_level=confidence_level,
        )
        result[f"top_{top_k}_accuracy"] = hits / sample_count
        result[f"top_{top_k}_ci_lower"] = lower
        result[f"top_{top_k}_ci_upper"] = upper
    result["mean_reciprocal_rank"] = sum(
        float(row["reciprocal_rank"]) for row in rows
    ) / sample_count

    per_disease: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per_disease[str(row["disease_id"])].append(row)
    supported_disease_accuracies = [
        sum(int(row["top_1_correct"]) for row in disease_rows)
        / len(disease_rows)
        for disease_rows in per_disease.values()
        if len(
            {
                str(row["unique_group_id"])
                for row in disease_rows
            }
        )
        >= minimum_per_disease_unique_groups
    ]
    result["disease_adjusted_top_1_accuracy"] = (
        sum(supported_disease_accuracies)
        / len(supported_disease_accuracies)
        if supported_disease_accuracies
        else None
    )
    result["disease_adjusted_disease_count"] = len(
        supported_disease_accuracies
    )
    return result


def _wilson_interval(
    hits: int,
    count: int,
    *,
    confidence_level: float,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if count <= 0:
        raise ValueError("Wilson interval requires at least one sample")
    z_score = NormalDist().inv_cdf(
        0.5 + confidence_level / 2.0
    )
    proportion = hits / count
    squared = z_score**2
    denominator = 1.0 + squared / count
    centre = (proportion + squared / (2.0 * count)) / denominator
    margin = (
        z_score
        * math.sqrt(
            proportion * (1.0 - proportion) / count
            + squared / (4.0 * count**2)
        )
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


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
    recoverable_json_valid_count = 0
    schema_valid_count = 0
    canonical_schema_valid_count = 0
    invalid_candidate_count = 0
    duplicate_count = 0
    reciprocal_rank_sum = 0.0
    canonical_reciprocal_rank_sum = 0.0
    top_one_predictions: list[str | None] = []
    canonical_top_one_predictions: list[str | None] = []
    top_one_hits = 0
    top_two_hits = 0
    canonical_top_one_hits = 0
    canonical_top_two_hits = 0
    for row in rows:
        response = row.response
        json_valid_count += int(response.json_valid)
        recoverable_json_valid_count += int(
            response.recoverable_json_valid
        )
        schema_valid_count += int(response.schema_valid)
        canonical_schema_valid_count += int(
            response.canonical_schema_valid
        )
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
        candidates = _candidate_id_set(
            row.metadata.get("candidate_disease_ids"),
            fallback=allowed_disease_ids,
        )
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
        canonical_ranked = _ranked_disease_ids(
            response.canonical_output
            if response.canonical_schema_valid
            else None
        )
        canonical_top_one_predictions.append(
            canonical_ranked[0] if canonical_ranked else None
        )
        canonical_rank = (
            canonical_ranked.index(row.ground_truth_disease_id) + 1
            if row.ground_truth_disease_id in canonical_ranked
            else None
        )
        canonical_top_one_correct = int(canonical_rank == 1)
        canonical_top_two_correct = int(
            canonical_rank is not None and canonical_rank <= 2
        )
        canonical_top_one_hits += canonical_top_one_correct
        canonical_top_two_hits += canonical_top_two_correct
        if canonical_rank is not None:
            canonical_reciprocal_rank_sum += 1.0 / canonical_rank
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
                "canonical_top_one_correct": canonical_top_one_correct,
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
    canonical_by_condition = _group_accuracy(
        scored_rows,
        group_key="difficulty",
        value_key="canonical_top_one_correct",
    )
    canonical_by_set = _group_accuracy(
        scored_rows,
        group_key="confusion_set_id",
        value_key="canonical_top_one_correct",
    )
    canonical_by_disease = _group_accuracy(
        scored_rows,
        group_key="disease_id",
        value_key="canonical_top_one_correct",
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
        "canonical_top_1_accuracy": canonical_top_one_hits / count,
        "canonical_top_2_accuracy": canonical_top_two_hits / count,
        "canonical_mean_reciprocal_rank": (
            canonical_reciprocal_rank_sum / count
        ),
        "canonical_macro_f1_top_1": _macro_f1(
            truths=[
                row.ground_truth_disease_id
                for row in rows
            ],
            predictions=canonical_top_one_predictions,
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
        "recoverable_json_validity_rate": (
            recoverable_json_valid_count / count
        ),
        "schema_compliance_rate": schema_valid_count / count,
        "canonical_schema_compliance_rate": (
            canonical_schema_valid_count / count
        ),
        "invalid_candidate_id_rate": invalid_candidate_count / count,
        "duplicate_prediction_rate": duplicate_count / count,
        "by_condition_top_1_accuracy": by_condition,
        "by_confusion_set_top_1_accuracy": by_set,
        "by_disease_top_1_accuracy": by_disease,
        "canonical_by_condition_top_1_accuracy": (
            canonical_by_condition
        ),
        "canonical_by_confusion_set_top_1_accuracy": canonical_by_set,
        "canonical_by_disease_top_1_accuracy": canonical_by_disease,
    }


def _candidate_id_set(
    value: Any,
    *,
    fallback: Iterable[str],
) -> set[str]:
    """Normalize current and legacy candidate metadata representations."""

    if value is None:
        return {str(item) for item in fallback}
    if isinstance(value, str):
        # Older JSONL artifacts stored NumPy arrays as strings such as
        # "['D001' 'D002' 'D003']". Extracting quoted values supports both
        # that representation and conventional Python-list strings.
        quoted = re.findall(r"""['"]([^'"]+)['"]""", value)
        if quoted:
            return set(quoted)
        stripped = value.strip().strip("[]")
        tokens = [
            token.strip().strip("'\"")
            for token in stripped.split(",")
            if token.strip()
        ]
        return set(tokens) if tokens else {str(item) for item in fallback}
    if isinstance(value, dict):
        return {str(item) for item in fallback}
    if isinstance(value, IterableABC):
        return {str(item) for item in value}
    return {str(value)}


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
    value_key: str = "top_one_correct",
) -> dict[str, float]:
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(int(row[value_key]))
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
