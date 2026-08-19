"""Validation-gate tests for Stage B versus frozen Stage A."""

from project.teacher.schemas import parse_stage_a, parse_stage_b
from project.teacher.validate import (
    REASON_DERMOSCOPIC_ON_CLINICAL,
    REASON_GOLD_MISMATCH,
    REASON_SUPPORTING_NOT_IN_STAGE_A,
    validate_stage_b,
)
from project.tests.fixtures import STAGE_A_PAYLOAD, STAGE_B_PAYLOAD


def test_validate_accepts_aligned_fixture() -> None:
    result = validate_stage_b(
        parse_stage_a(STAGE_A_PAYLOAD),
        parse_stage_b(STAGE_B_PAYLOAD),
        "melanoma",
    )
    assert result.ok is True
    assert result.reasons == ()


def test_validate_rejects_gold_mismatch() -> None:
    result = validate_stage_b(
        parse_stage_a(STAGE_A_PAYLOAD),
        parse_stage_b(STAGE_B_PAYLOAD),
        "psoriasis",
    )
    assert result.ok is False
    assert REASON_GOLD_MISMATCH in result.reasons


def test_validate_rejects_citation_missing_from_stage_a() -> None:
    payload = {
        **STAGE_B_PAYLOAD,
        "differential_diagnosis": [
            {
                "rank": 1,
                "disease": "melanoma",
                "supporting": [{"field": "shape", "value": "round"}],
                "contradicting": [],
                "missing": [],
            },
            {
                "rank": 2,
                "disease": "atypical nevus",
                "supporting": [{"field": "configuration", "value": "solitary"}],
                "contradicting": [],
                "missing": [],
            },
        ],
    }
    result = validate_stage_b(
        parse_stage_a(STAGE_A_PAYLOAD),
        parse_stage_b(payload),
        "melanoma",
    )
    assert REASON_SUPPORTING_NOT_IN_STAGE_A in result.reasons


def test_validate_rejects_dermoscopic_sign_on_clinical_image() -> None:
    payload = {
        **STAGE_B_PAYLOAD,
        "differential_diagnosis": [
            {
                "rank": 1,
                "disease": "melanoma",
                "supporting": [
                    {
                        "field": "additional_features",
                        "value": "blue-white veil",
                    }
                ],
                "contradicting": [],
                "missing": [],
            },
            {
                "rank": 2,
                "disease": "atypical nevus",
                "supporting": [{"field": "configuration", "value": "solitary"}],
                "contradicting": [],
                "missing": [],
            },
        ],
    }
    result = validate_stage_b(
        parse_stage_a(STAGE_A_PAYLOAD),
        parse_stage_b(payload),
        "melanoma",
    )
    assert REASON_DERMOSCOPIC_ON_CLINICAL in result.reasons
    assert REASON_SUPPORTING_NOT_IN_STAGE_A in result.reasons
