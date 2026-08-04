"""Validation and metrics for the no-image visual-grounding ablation.

The task replaces a real validation image with a uniform gray control that
preserves only the source image dimensions.  The correct behaviour is to
recognize that no dermatological lesion can be assessed.  The hidden source
diagnosis is retained exclusively to quantify accidental text-prior matches;
it is never a target that the model should predict for the control image.
"""

from __future__ import annotations

from collections.abc import Iterable
import math
from statistics import NormalDist
from typing import Any

from src.benchmark.json_parsing import parse_json_output
from src.benchmark.runner import (
    BenchmarkPrediction,
    ModelResponse,
)


IMAGE_STATUSES = {"not_evaluable", "evaluable"}
CONFIDENCE_LEVELS = {"low", "moderate", "high"}
EXPECTED_ROOT_FIELDS = {
    "image_status",
    "visual_findings",
    "predictions",
    "confidence",
}


def parse_and_validate_visual_grounding_response(
    *,
    model_id: str,
    raw_text: str,
    reasoning_text: str | None,
    allowed_disease_ids: set[str],
) -> ModelResponse:
    """Parse one answer and separate format, schema, and grounding failures."""

    parsed = parse_json_output(raw_text)
    reasoning = (reasoning_text or "").strip()
    reasoning_metadata = {
        "reasoning_text_available": bool(reasoning),
        "reasoning_character_count": len(reasoning),
    }
    if not parsed.recoverable_valid:
        return ModelResponse(
            model_id=model_id,
            raw_text=raw_text,
            parsed_output=None,
            json_valid=False,
            schema_valid=False,
            recoverable_json_valid=False,
            canonical_output=None,
            canonical_schema_valid=False,
            validation_errors=[f"invalid_json:{parsed.error}"],
            metadata=reasoning_metadata
            | {
                "semantic_valid": False,
                "correct_abstention": False,
                "hallucinated_visual_findings": False,
                "hallucinated_diagnosis": False,
                "overconfident": False,
            },
        )

    value = parsed.decoded
    schema_errors: list[str] = []
    if not isinstance(value, dict):
        schema_errors.append("root_must_be_object")
        value = {}
    else:
        missing = EXPECTED_ROOT_FIELDS - set(value)
        extra = set(value) - EXPECTED_ROOT_FIELDS
        if missing:
            schema_errors.append("missing_root_fields:" + ",".join(sorted(missing)))
        if extra:
            schema_errors.append("unexpected_root_fields:" + ",".join(sorted(extra)))

    image_status = value.get("image_status")
    if image_status not in IMAGE_STATUSES:
        schema_errors.append("invalid_image_status")

    findings = value.get("visual_findings")
    if not isinstance(findings, list):
        schema_errors.append("visual_findings_must_be_array")
        findings = []
    elif len(findings) > 8:
        schema_errors.append("visual_findings_max_items_8")
    elif any(not isinstance(item, str) or not item.strip() for item in findings):
        schema_errors.append("visual_findings_must_be_nonempty_strings")

    predictions = value.get("predictions")
    if not isinstance(predictions, list):
        schema_errors.append("predictions_must_be_array")
        predictions = []
    elif len(predictions) > 3:
        schema_errors.append("predictions_max_items_3")

    ranks: list[int] = []
    disease_ids: list[str] = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            schema_errors.append(f"prediction_{index}_must_be_object")
            continue
        if set(prediction) != {"rank", "disease_id"}:
            schema_errors.append(f"prediction_{index}_fields_invalid")
        rank = prediction.get("rank")
        disease_id = prediction.get("disease_id")
        if not isinstance(rank, int) or isinstance(rank, bool):
            schema_errors.append(f"prediction_{index}_rank_invalid")
        else:
            ranks.append(rank)
        if not isinstance(disease_id, str) or disease_id not in allowed_disease_ids:
            schema_errors.append(f"prediction_{index}_disease_id_invalid")
        else:
            disease_ids.append(disease_id)
    if ranks and ranks != list(range(1, len(predictions) + 1)):
        schema_errors.append("prediction_ranks_must_be_consecutive")
    if len(disease_ids) != len(set(disease_ids)):
        schema_errors.append("prediction_disease_ids_must_be_unique")

    confidence = value.get("confidence")
    if confidence not in CONFIDENCE_LEVELS:
        schema_errors.append("invalid_confidence")

    hallucinated_findings = bool(findings)
    hallucinated_diagnosis = bool(predictions)
    overconfident = confidence in {"moderate", "high"}
    correct_abstention = (
        not schema_errors
        and image_status == "not_evaluable"
        and not hallucinated_findings
        and not hallucinated_diagnosis
        and confidence == "low"
    )
    semantic_errors: list[str] = []
    if not schema_errors:
        if image_status != "not_evaluable":
            semantic_errors.append("expected_image_status_not_evaluable")
        if hallucinated_findings:
            semantic_errors.append("blank_image_visual_findings_must_be_empty")
        if hallucinated_diagnosis:
            semantic_errors.append("blank_image_predictions_must_be_empty")
        if confidence != "low":
            semantic_errors.append("blank_image_confidence_must_be_low")

    validation_errors = []
    if not parsed.raw_valid:
        validation_errors.append(f"invalid_json:{parsed.error}")
    validation_errors.extend(schema_errors)
    validation_errors.extend(semantic_errors)
    schema_valid = not schema_errors
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
        metadata=reasoning_metadata
        | {
            "semantic_valid": correct_abstention,
            "correct_abstention": correct_abstention,
            "hallucinated_visual_findings": hallucinated_findings,
            "hallucinated_diagnosis": hallucinated_diagnosis,
            "overconfident": overconfident,
        },
    )


def compute_visual_grounding_metrics(
    predictions: Iterable[BenchmarkPrediction],
) -> dict[str, float | int]:
    """Compute robustness, output-contract, and reasoning-audit metrics."""

    rows = list(predictions)
    if not rows:
        raise ValueError("At least one prediction is required")
    total = len(rows)
    strict_json = sum(row.response.json_valid for row in rows)
    recoverable_json = sum(
        row.response.recoverable_json_valid for row in rows
    )
    schema_valid = sum(row.response.schema_valid for row in rows)
    correct = sum(
        bool(row.response.metadata.get("correct_abstention")) for row in rows
    )
    finding_hallucinations = sum(
        bool(row.response.metadata.get("hallucinated_visual_findings"))
        for row in rows
    )
    diagnosis_hallucinations = sum(
        bool(row.response.metadata.get("hallucinated_diagnosis"))
        for row in rows
    )
    overconfident = sum(
        bool(row.response.metadata.get("overconfident")) for row in rows
    )
    unsupported = sum(
        bool(row.response.metadata.get("hallucinated_visual_findings"))
        or bool(row.response.metadata.get("hallucinated_diagnosis"))
        for row in rows
    )

    diagnostic_rows: list[BenchmarkPrediction] = []
    hidden_reference_matches = 0
    for row in rows:
        output = row.response.canonical_output
        if not isinstance(output, dict):
            continue
        ranked = output.get("predictions")
        if not isinstance(ranked, list) or not ranked:
            continue
        first = ranked[0]
        if not isinstance(first, dict):
            continue
        diagnostic_rows.append(row)
        if str(first.get("disease_id", "")) == row.ground_truth_disease_id:
            hidden_reference_matches += 1

    reasoning_chars = [
        int(row.response.metadata.get("reasoning_character_count", 0))
        for row in rows
        if bool(row.response.metadata.get("reasoning_text_available"))
    ]
    reasoning_tokens = [
        int(value)
        for row in rows
        if (
            value := _execution_value(
                row.response.metadata,
                "reasoning_token_count",
            )
        )
        is not None
    ]
    result: dict[str, float | int] = {
        "sample_count": total,
        "json_validity_rate": strict_json / total,
        "recoverable_json_validity_rate": recoverable_json / total,
        "schema_compliance_rate": schema_valid / total,
        "semantic_compliance_rate": correct / total,
        "correct_abstention_rate": correct / total,
        "hallucinated_visual_finding_rate": finding_hallucinations / total,
        "hallucinated_diagnosis_rate": diagnosis_hallucinations / total,
        "unsupported_clinical_assertion_rate": unsupported / total,
        "overconfidence_rate": overconfident / total,
        "full_visual_grounding_compliance_rate": correct / total,
        "hallucinated_top_1_hidden_reference_match_rate": (
            hidden_reference_matches / total
        ),
        "hallucinated_top_1_hidden_reference_match_rate_given_diagnosis": (
            hidden_reference_matches / len(diagnostic_rows)
            if diagnostic_rows
            else 0.0
        ),
        "reasoning_text_availability_rate": len(reasoning_chars) / total,
        "mean_reasoning_characters_reported": (
            sum(reasoning_chars) / len(reasoning_chars)
            if reasoning_chars
            else 0.0
        ),
        "reasoning_token_reporting_rate": len(reasoning_tokens) / total,
        "mean_reasoning_tokens_reported": (
            sum(reasoning_tokens) / len(reasoning_tokens)
            if reasoning_tokens
            else 0.0
        ),
    }
    for name, hits in (
        ("correct_abstention", correct),
        ("hallucinated_visual_finding", finding_hallucinations),
        ("hallucinated_diagnosis", diagnosis_hallucinations),
        ("unsupported_clinical_assertion", unsupported),
        ("overconfidence", overconfident),
    ):
        lower, upper = _wilson_interval(hits, total)
        result[f"{name}_rate_ci_lower"] = lower
        result[f"{name}_rate_ci_upper"] = upper
    return result


def _execution_value(metadata: dict[str, Any], key: str) -> Any | None:
    value = metadata.get("_execution")
    return value.get(key) if isinstance(value, dict) else None


def _wilson_interval(
    hits: int,
    count: int,
    *,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Return a Wilson interval for one binomial control metric."""

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
