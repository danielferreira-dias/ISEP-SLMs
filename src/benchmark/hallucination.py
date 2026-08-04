"""Parsers and deterministic metrics for visual hallucination audits.

The general audit scores answerability and premise grounding.  It deliberately
does not pretend that lexical matching against a free-text HaloQuest reference
is a reliable answer-correctness metric.  The dermatology audit scores two
counterfactual conditions: corrupted pixels should trigger abstention, while a
hard-negative image swap should make the prediction follow the replacement
image rather than the hidden source label.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
import math
from statistics import NormalDist
from typing import Any

from src.benchmark.json_parsing import parse_json_output
from src.benchmark.runner import BenchmarkPrediction, ModelResponse


GENERAL_STATUSES = {
    "answerable",
    "false_premise",
    "insufficient_visual_evidence",
}
IMAGE_STATUSES = {"not_evaluable", "evaluable"}
CONFIDENCE_LEVELS = {"low", "moderate", "high"}
DISEASE_RESPONSE_FIELDS = {
    "image_status",
    "visual_findings",
    "predictions",
    "confidence",
}


def parse_general_visual_hallucination_response(
    *,
    model_id: str,
    raw_text: str,
    reasoning_text: str | None,
) -> ModelResponse:
    """Parse the three-field general visual-grounding response."""

    parsed = parse_json_output(raw_text)
    reasoning_metadata = _reasoning_metadata(reasoning_text)
    if not parsed.recoverable_valid:
        return _invalid_json_response(
            model_id=model_id,
            raw_text=raw_text,
            error=parsed.error,
            metadata=reasoning_metadata,
        )
    value = parsed.decoded
    errors: list[str] = []
    if not isinstance(value, dict):
        errors.append("root_must_be_object")
        value = {}
    else:
        expected = {"question_status", "answer", "confidence"}
        _validate_root_fields(value, expected, errors)
    status = value.get("question_status")
    if status not in GENERAL_STATUSES:
        errors.append("invalid_question_status")
    answer = value.get("answer")
    if answer is not None and (
        not isinstance(answer, str) or not answer.strip()
    ):
        errors.append("answer_must_be_nonempty_string_or_null")
    confidence = value.get("confidence")
    if confidence not in CONFIDENCE_LEVELS:
        errors.append("invalid_confidence")
    schema_valid = not errors
    validation_errors = []
    if not parsed.raw_valid:
        validation_errors.append(f"invalid_json:{parsed.error}")
    validation_errors.extend(errors)
    return ModelResponse(
        model_id=model_id,
        raw_text=raw_text,
        parsed_output=value,
        json_valid=parsed.raw_valid,
        schema_valid=schema_valid,
        recoverable_json_valid=True,
        canonical_output=value,
        canonical_schema_valid=schema_valid,
        validation_errors=validation_errors,
        metadata=reasoning_metadata,
    )


def compute_general_visual_hallucination_metrics(
    predictions: Iterable[BenchmarkPrediction],
) -> dict[str, Any]:
    """Score status decisions without using brittle free-text similarity."""

    rows = list(predictions)
    if not rows:
        raise ValueError("At least one prediction is required")
    expected_by_condition = {
        "false_premises": "false_premise",
        "insufficient_context": "insufficient_visual_evidence",
        "visual_challenge": "answerable",
    }
    status_hits = 0
    unanswerable_rows = 0
    unanswerable_hits = 0
    unanswerable_hallucinations = 0
    unanswerable_overconfidence = 0
    condition_totals = {condition: 0 for condition in expected_by_condition}
    condition_hits = {condition: 0 for condition in expected_by_condition}
    challenge_nonempty_answers = 0
    for row in rows:
        condition = str(row.metadata.get("condition", ""))
        if condition not in expected_by_condition:
            raise ValueError(f"Unknown HaloQuest condition: {condition!r}")
        expected = expected_by_condition[condition]
        output = row.response.canonical_output or {}
        predicted = output.get("question_status")
        answer = output.get("answer")
        confidence = output.get("confidence")
        hit = row.response.schema_valid and predicted == expected
        status_hits += int(hit)
        condition_totals[condition] += 1
        condition_hits[condition] += int(hit)
        if condition == "visual_challenge":
            challenge_nonempty_answers += int(
                hit and isinstance(answer, str) and bool(answer.strip())
            )
            continue
        unanswerable_rows += 1
        clean_abstention = hit and answer is None
        unanswerable_hits += int(clean_abstention)
        unanswerable_hallucinations += int(
            predicted == "answerable"
            or (isinstance(answer, str) and bool(answer.strip()))
        )
        unanswerable_overconfidence += int(
            not clean_abstention and confidence == "high"
        )

    total = len(rows)
    result: dict[str, Any] = {
        "sample_count": total,
        "json_validity_rate": sum(row.response.json_valid for row in rows)
        / total,
        "recoverable_json_validity_rate": sum(
            row.response.recoverable_json_valid for row in rows
        )
        / total,
        "schema_compliance_rate": sum(
            row.response.schema_valid for row in rows
        )
        / total,
        "question_status_accuracy": status_hits / total,
        "unanswerable_detection_accuracy": (
            unanswerable_hits / unanswerable_rows
        ),
        "false_premise_rejection_accuracy": _rate(
            condition_hits["false_premises"],
            condition_totals["false_premises"],
        ),
        "insufficient_context_recognition_accuracy": _rate(
            condition_hits["insufficient_context"],
            condition_totals["insufficient_context"],
        ),
        "visual_challenge_answerability_accuracy": _rate(
            condition_hits["visual_challenge"],
            condition_totals["visual_challenge"],
        ),
        "visual_challenge_nonempty_answer_rate": _rate(
            challenge_nonempty_answers,
            condition_totals["visual_challenge"],
        ),
        "unanswerable_hallucination_rate": (
            unanswerable_hallucinations / unanswerable_rows
        ),
        "overconfidence_on_unanswerable_rate": (
            unanswerable_overconfidence / unanswerable_rows
        ),
        "free_text_answer_correctness_scored": False,
        "reference_answer_scoring_status": "not_scored_requires_semantic_judge",
    }
    _add_reasoning_metrics(result, rows)
    for name, hits, count in (
        ("question_status", status_hits, total),
        ("unanswerable_detection", unanswerable_hits, unanswerable_rows),
        (
            "unanswerable_hallucination",
            unanswerable_hallucinations,
            unanswerable_rows,
        ),
    ):
        lower, upper = _wilson_interval(hits, count)
        result[f"{name}_rate_ci_lower"] = lower
        result[f"{name}_rate_ci_upper"] = upper
    return result


def parse_dermatology_counterfactual_response(
    *,
    model_id: str,
    raw_text: str,
    reasoning_text: str | None,
    allowed_disease_ids: set[str],
) -> ModelResponse:
    """Validate a dermatology counterfactual response without gold leakage."""

    parsed = parse_json_output(raw_text)
    reasoning_metadata = _reasoning_metadata(reasoning_text)
    if not parsed.recoverable_valid:
        return _invalid_json_response(
            model_id=model_id,
            raw_text=raw_text,
            error=parsed.error,
            metadata=reasoning_metadata,
        )
    value = parsed.decoded
    errors: list[str] = []
    if not isinstance(value, dict):
        errors.append("root_must_be_object")
        value = {}
    else:
        _validate_root_fields(value, DISEASE_RESPONSE_FIELDS, errors)
    if value.get("image_status") not in IMAGE_STATUSES:
        errors.append("invalid_image_status")
    findings = value.get("visual_findings")
    if not isinstance(findings, list):
        errors.append("visual_findings_must_be_array")
        findings = []
    elif len(findings) > 8:
        errors.append("visual_findings_max_items_8")
    elif any(not isinstance(item, str) or not item.strip() for item in findings):
        errors.append("visual_findings_must_be_nonempty_strings")
    predictions = value.get("predictions")
    if not isinstance(predictions, list):
        errors.append("predictions_must_be_array")
        predictions = []
    elif len(predictions) > 3:
        errors.append("predictions_max_items_3")
    ranks: list[int] = []
    disease_ids: list[str] = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            errors.append(f"prediction_{index}_must_be_object")
            continue
        if set(prediction) != {"rank", "disease_id"}:
            errors.append(f"prediction_{index}_fields_invalid")
        rank = prediction.get("rank")
        disease_id = prediction.get("disease_id")
        if not isinstance(rank, int) or isinstance(rank, bool):
            errors.append(f"prediction_{index}_rank_invalid")
        else:
            ranks.append(rank)
        if not isinstance(disease_id, str) or disease_id not in allowed_disease_ids:
            errors.append(f"prediction_{index}_disease_id_invalid")
        else:
            disease_ids.append(disease_id)
    if ranks and ranks != list(range(1, len(predictions) + 1)):
        errors.append("prediction_ranks_must_be_consecutive")
    if len(disease_ids) != len(set(disease_ids)):
        errors.append("prediction_disease_ids_must_be_unique")
    if value.get("confidence") not in CONFIDENCE_LEVELS:
        errors.append("invalid_confidence")
    schema_valid = not errors
    validation_errors = []
    if not parsed.raw_valid:
        validation_errors.append(f"invalid_json:{parsed.error}")
    validation_errors.extend(errors)
    return ModelResponse(
        model_id=model_id,
        raw_text=raw_text,
        parsed_output=value,
        json_valid=parsed.raw_valid,
        schema_valid=schema_valid,
        recoverable_json_valid=True,
        canonical_output=value,
        canonical_schema_valid=schema_valid,
        validation_errors=validation_errors,
        metadata=reasoning_metadata,
    )


def compute_dermatology_counterfactual_metrics(
    predictions: Iterable[BenchmarkPrediction],
) -> dict[str, Any]:
    """Compute condition-aware grounding and diagnosis metrics."""

    rows = list(predictions)
    if not rows:
        raise ValueError("At least one prediction is required")
    shuffled = [
        row for row in rows if row.metadata.get("condition") == "pixel_shuffle"
    ]
    swaps = [
        row
        for row in rows
        if row.metadata.get("condition") == "hard_negative_image_swap"
    ]
    if len(shuffled) + len(swaps) != len(rows):
        raise ValueError("Counterfactual predictions contain unknown conditions")

    shuffled_abstentions = 0
    shuffled_findings = 0
    shuffled_diagnoses = 0
    swap_evaluable = 0
    swap_top_1 = 0
    swap_top_3 = 0
    source_persistence = 0
    full_success = 0
    for row in shuffled:
        output = row.response.canonical_output or {}
        findings = output.get("visual_findings")
        ranked = output.get("predictions")
        correct = (
            row.response.schema_valid
            and output.get("image_status") == "not_evaluable"
            and findings == []
            and ranked == []
            and output.get("confidence") == "low"
        )
        shuffled_abstentions += int(correct)
        shuffled_findings += int(isinstance(findings, list) and bool(findings))
        shuffled_diagnoses += int(isinstance(ranked, list) and bool(ranked))
        full_success += int(correct)
    for row in swaps:
        output = row.response.canonical_output or {}
        evaluable = row.response.schema_valid and output.get("image_status") == "evaluable"
        swap_evaluable += int(evaluable)
        ranked = _ranked_disease_ids(output)
        top_1 = bool(ranked and ranked[0] == row.ground_truth_disease_id)
        top_3 = row.ground_truth_disease_id in ranked[:3]
        swap_top_1 += int(top_1)
        swap_top_3 += int(top_3)
        provenance = _reference_payload(row)
        source_id = str(provenance.get("source_prompt_disease_id", ""))
        source_persistence += int(bool(ranked) and ranked[0] == source_id)
        full_success += int(evaluable and top_1)

    total = len(rows)
    grounding_hits = shuffled_abstentions + swap_evaluable
    result: dict[str, Any] = {
        "sample_count": total,
        "pixel_shuffle_sample_count": len(shuffled),
        "hard_negative_sample_count": len(swaps),
        "json_validity_rate": sum(row.response.json_valid for row in rows)
        / total,
        "recoverable_json_validity_rate": sum(
            row.response.recoverable_json_valid for row in rows
        )
        / total,
        "schema_compliance_rate": sum(
            row.response.schema_valid for row in rows
        )
        / total,
        "counterfactual_grounding_accuracy": grounding_hits / total,
        "full_counterfactual_success_rate": full_success / total,
        "pixel_shuffle_correct_abstention_rate": _rate(
            shuffled_abstentions, len(shuffled)
        ),
        "pixel_shuffle_hallucinated_visual_finding_rate": _rate(
            shuffled_findings, len(shuffled)
        ),
        "pixel_shuffle_hallucinated_diagnosis_rate": _rate(
            shuffled_diagnoses, len(shuffled)
        ),
        "hard_negative_evaluable_rate": _rate(swap_evaluable, len(swaps)),
        "hard_negative_top_1_accuracy": _rate(swap_top_1, len(swaps)),
        "hard_negative_top_3_accuracy": _rate(swap_top_3, len(swaps)),
        "hard_negative_source_label_persistence_rate": _rate(
            source_persistence, len(swaps)
        ),
    }
    _add_reasoning_metrics(result, rows)
    for name, hits, count in (
        ("counterfactual_grounding", grounding_hits, total),
        ("full_counterfactual_success", full_success, total),
        ("pixel_shuffle_correct_abstention", shuffled_abstentions, len(shuffled)),
        ("hard_negative_top_1", swap_top_1, len(swaps)),
    ):
        lower, upper = _wilson_interval(hits, count)
        result[f"{name}_rate_ci_lower"] = lower
        result[f"{name}_rate_ci_upper"] = upper
    return result


def _reference_payload(row: BenchmarkPrediction) -> dict[str, Any]:
    value = row.metadata.get("reference_diagnoses_json")
    if not isinstance(value, str) or not value:
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _ranked_disease_ids(output: dict[str, Any]) -> list[str]:
    ranked = output.get("predictions")
    if not isinstance(ranked, list):
        return []
    return [
        str(item.get("disease_id"))
        for item in ranked
        if isinstance(item, dict) and item.get("disease_id")
    ]


def _validate_root_fields(
    value: dict[str, Any],
    expected: set[str],
    errors: list[str],
) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing:
        errors.append("missing_root_fields:" + ",".join(sorted(missing)))
    if extra:
        errors.append("unexpected_root_fields:" + ",".join(sorted(extra)))


def _invalid_json_response(
    *,
    model_id: str,
    raw_text: str,
    error: str | None,
    metadata: dict[str, Any],
) -> ModelResponse:
    return ModelResponse(
        model_id=model_id,
        raw_text=raw_text,
        parsed_output=None,
        json_valid=False,
        schema_valid=False,
        recoverable_json_valid=False,
        canonical_output=None,
        canonical_schema_valid=False,
        validation_errors=[f"invalid_json:{error}"],
        metadata=metadata,
    )


def _reasoning_metadata(reasoning_text: str | None) -> dict[str, Any]:
    reasoning = (reasoning_text or "").strip()
    return {
        "reasoning_text_available": bool(reasoning),
        "reasoning_character_count": len(reasoning),
    }


def _add_reasoning_metrics(
    result: dict[str, Any],
    rows: list[BenchmarkPrediction],
) -> None:
    available = [
        row
        for row in rows
        if row.response.metadata.get("reasoning_text_available")
    ]
    tokens = [
        value
        for row in rows
        if (value := _execution_value(row, "reasoning_token_count")) is not None
    ]
    result["reasoning_text_availability_rate"] = len(available) / len(rows)
    result["reasoning_token_reporting_rate"] = len(tokens) / len(rows)
    result["mean_reasoning_tokens_reported"] = (
        sum(int(value) for value in tokens) / len(tokens) if tokens else 0.0
    )


def _execution_value(row: BenchmarkPrediction, key: str) -> Any | None:
    execution = row.response.metadata.get("_execution")
    return execution.get(key) if isinstance(execution, dict) else None


def _rate(hits: int, count: int) -> float:
    return hits / count if count else 0.0


def _wilson_interval(
    hits: int,
    count: int,
    *,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 0.0
    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
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
