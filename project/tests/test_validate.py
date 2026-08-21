"""Validation-gate tests for compact Stage B versus frozen Stage A."""

from copy import deepcopy
from typing import cast

from project.teacher.schemas import parse_stage_a, parse_stage_b
from project.teacher.validate import (
    REASON_ANNOTATION_CONFLICT,
    REASON_COMPARISON_MISSING_ALTERNATIVE,
    REASON_DIAGNOSIS_MISSING_SUPPORT,
    REASON_GOLD_MISMATCH,
    REASON_NEW_IMAGE_ON_EVALUABLE,
    REASON_NON_EVIDENTIARY_OBSERVATION,
    REASON_REASONING_MISSING_ALTERNATIVE,
    REASON_REASONING_MISSING_GOLD,
    REASON_REASONING_REVEALS_GOLD,
    REASON_SUPPORTING_NOT_IN_STAGE_A,
    REASON_UNSUPPORTED_ANCHOR,
    REASON_UNSUPPORTED_ANCHOR_HAS_SUPPORT,
    validate_stage_b,
)
from project.tests.fixtures import STAGE_A_PAYLOAD, STAGE_B_PAYLOAD


def _validate(payload: dict[str, object], gold: str = "melanoma") -> tuple[str, ...]:
    result = validate_stage_b(
        parse_stage_a(STAGE_A_PAYLOAD),
        parse_stage_b(payload),
        gold,
    )
    return result.reasons


def _comparison(payload: dict[str, object]) -> dict[str, object]:
    comparisons = cast(
        list[dict[str, object]],
        payload["differential_comparisons"],
    )
    return comparisons[0]


def test_validate_accepts_aligned_fixture() -> None:
    assert _validate(STAGE_B_PAYLOAD) == ()


def test_validate_accepts_canonical_underscore_as_space_in_clinical_prose() -> None:
    payload = deepcopy(STAGE_B_PAYLOAD)
    payload["diagnosis"] = "contact_dermatitis"
    comparison = _comparison(payload)
    comparison["comparison"] = (
        "Contact dermatitis is favored over an atypical nevus by the visible "
        "distribution and border pattern."
    )
    payload["clinical_reasoning"] = (
        "The visible distribution and border pattern support contact dermatitis "
        "over an atypical nevus with moderate confidence."
    )

    assert _validate(payload, "contact_dermatitis") == ()


def test_validate_rejects_gold_mismatch() -> None:
    assert REASON_GOLD_MISMATCH in _validate(STAGE_B_PAYLOAD, "psoriasis")


def test_validate_rejects_unknown_supporting_observation() -> None:
    payload = deepcopy(STAGE_B_PAYLOAD)
    _comparison(payload)["features_favoring_diagnosis"] = ["obs_999"]
    assert REASON_SUPPORTING_NOT_IN_STAGE_A in _validate(payload)


def test_validate_rejects_uncertain_observation_as_evidence() -> None:
    payload = deepcopy(STAGE_B_PAYLOAD)
    _comparison(payload)["features_favoring_diagnosis"] = ["obs_005"]
    assert REASON_NON_EVIDENTIARY_OBSERVATION in _validate(payload)


def test_supported_anchor_requires_diagnosis_support() -> None:
    payload = deepcopy(STAGE_B_PAYLOAD)
    _comparison(payload)["features_favoring_diagnosis"] = []
    assert REASON_DIAGNOSIS_MISSING_SUPPORT in _validate(payload)


def test_unsupported_answer_is_excluded() -> None:
    payload = deepcopy(STAGE_B_PAYLOAD)
    payload["anchor_evidence_status"] = "unsupported"
    _comparison(payload)["features_favoring_diagnosis"] = []
    assert REASON_UNSUPPORTED_ANCHOR in _validate(payload)


def test_unsupported_anchor_cannot_claim_support() -> None:
    payload = deepcopy(STAGE_B_PAYLOAD)
    payload["anchor_evidence_status"] = "unsupported"
    assert REASON_UNSUPPORTED_ANCHOR_HAS_SUPPORT in _validate(payload)


def test_annotation_conflict_is_audited_but_excluded() -> None:
    payload = deepcopy(STAGE_B_PAYLOAD)
    payload["annotation_conflict"] = True
    payload["annotation_conflict_reason"] = (
        "The visible lesion pattern appears inconsistent with the supplied label."
    )
    assert REASON_ANNOTATION_CONFLICT in _validate(payload)


def test_comparison_must_name_the_alternative() -> None:
    payload = deepcopy(STAGE_B_PAYLOAD)
    _comparison(payload)["comparison"] = (
        "Melanoma is favored by the visible asymmetry and color variation."
    )
    assert REASON_COMPARISON_MISSING_ALTERNATIVE in _validate(payload)


def test_clinical_reasoning_must_name_gold() -> None:
    payload = deepcopy(STAGE_B_PAYLOAD)
    payload["clinical_reasoning"] = (
        "The asymmetric lesion has irregular borders and variable pigmentation. "
        "These findings are more concerning than an atypical nevus."
    )
    assert REASON_REASONING_MISSING_GOLD in _validate(payload)


def test_clinical_reasoning_must_name_every_alternative() -> None:
    payload = deepcopy(STAGE_B_PAYLOAD)
    payload["clinical_reasoning"] = (
        "The visible asymmetry, irregular border, and color variation support "
        "melanoma with moderate confidence. Evolution and dermoscopy are not "
        "available in this image."
    )
    assert REASON_REASONING_MISSING_ALTERNATIVE in _validate(payload)


def test_request_new_image_is_rejected_for_evaluable_input() -> None:
    payload = deepcopy(STAGE_B_PAYLOAD)
    payload.update(
        {
            "anchor_evidence_status": "unsupported",
            "diagnostic_confidence": "low",
            "differential_comparisons": [],
            "limitations": ["closer_image"],
            "response_policy": "REQUEST_NEW_IMAGE",
            "non_evaluable_reason": (
                "Severe blur prevents assessment of the lesion and its margins."
            ),
            "clinical_reasoning": (
                "Severe blur prevents reliable assessment of the lesion and its "
                "margins. Please provide a sharper replacement image."
            ),
        }
    )
    assert REASON_NEW_IMAGE_ON_EVALUABLE in _validate(payload)


def test_request_new_image_is_valid_only_for_non_evaluable_input() -> None:
    stage_a = deepcopy(STAGE_A_PAYLOAD)
    assessment = cast(dict[str, object], stage_a["image_assessment"])
    assessment["is_evaluable"] = False
    assessment["quality_defects"] = ["blur"]
    stage_a["observations"] = []
    stage_a["clinical_caption"] = (
        "Severe blur prevents visual assessment of the lesion and its margins."
    )

    stage_b = deepcopy(STAGE_B_PAYLOAD)
    stage_b.update(
        {
            "anchor_evidence_status": "unsupported",
            "diagnostic_confidence": "low",
            "differential_comparisons": [],
            "limitations": ["closer_image"],
            "response_policy": "REQUEST_NEW_IMAGE",
            "non_evaluable_reason": (
                "Severe blur prevents assessment of the lesion and its margins."
            ),
            "clinical_reasoning": (
                "Severe blur prevents reliable assessment of the lesion and its "
                "margins. Please provide a sharper replacement image."
            ),
        }
    )
    result = validate_stage_b(
        parse_stage_a(stage_a),
        parse_stage_b(stage_b),
        "melanoma",
    )

    assert result.ok
    assert result.reasons == ()


def test_request_new_image_reasoning_must_not_reveal_gold() -> None:
    stage_a = deepcopy(STAGE_A_PAYLOAD)
    assessment = cast(dict[str, object], stage_a["image_assessment"])
    assessment["is_evaluable"] = False
    assessment["quality_defects"] = ["blur"]
    stage_a["observations"] = []
    stage_a["clinical_caption"] = (
        "Severe blur prevents visual assessment of the lesion and its margins."
    )

    stage_b = deepcopy(STAGE_B_PAYLOAD)
    stage_b.update(
        {
            "anchor_evidence_status": "unsupported",
            "diagnostic_confidence": "low",
            "differential_comparisons": [],
            "limitations": ["closer_image"],
            "response_policy": "REQUEST_NEW_IMAGE",
            "non_evaluable_reason": (
                "Severe blur prevents assessment of the lesion and its margins."
            ),
            "clinical_reasoning": (
                "The image is too blurred to assess melanoma reliably. Please "
                "provide a sharper replacement image."
            ),
        }
    )
    result = validate_stage_b(
        parse_stage_a(stage_a),
        parse_stage_b(stage_b),
        "melanoma",
    )

    assert REASON_REASONING_REVEALS_GOLD in result.reasons
