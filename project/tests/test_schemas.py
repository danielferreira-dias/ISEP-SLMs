"""Parse tests for Stage A and Stage B Pydantic models."""

import json
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from project.teacher.schemas import parse_stage_a, parse_stage_b
from project.tests.fixtures import STAGE_A_PAYLOAD, STAGE_B_PAYLOAD

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_parse_stage_a_accepts_fixture() -> None:
    morphology = parse_stage_a(STAGE_A_PAYLOAD)
    assert morphology.observations[0].value == "macule"
    assert morphology.image_assessment.image_modality.value == "clinical"


def test_stage_a_allows_same_finding_for_distinct_visible_lesions() -> None:
    payload = deepcopy(STAGE_A_PAYLOAD)
    observations = cast(list[dict[str, object]], payload["observations"])
    repeated = deepcopy(observations[0])
    repeated.update(
        {
            "id": "obs_007",
            "scope": "second visible lesion",
            "evidence_region": "right side of the image",
        }
    )
    observations.append(repeated)

    parsed = parse_stage_a(payload)

    assert len(parsed.observations) == len(observations)


def test_stage_a_rejects_exact_semantic_duplicate_observation() -> None:
    payload = deepcopy(STAGE_A_PAYLOAD)
    observations = cast(list[dict[str, object]], payload["observations"])
    repeated = deepcopy(observations[0])
    repeated["id"] = "obs_007"
    observations.append(repeated)

    with pytest.raises(ValidationError, match="duplicate observations"):
        parse_stage_a(payload)


def test_parse_stage_a_rejects_extra_key() -> None:
    payload = dict(STAGE_A_PAYLOAD)
    payload["diagnosis"] = "melanoma"
    with pytest.raises(ValidationError):
        parse_stage_a(payload)


def test_parse_stage_a_rejects_bad_enum() -> None:
    payload = dict(STAGE_A_PAYLOAD)
    assessment = cast(dict[str, object], payload["image_assessment"])
    payload["image_assessment"] = {
        **assessment,
        "image_modality": "radiology",
    }
    with pytest.raises(ValidationError):
        parse_stage_a(payload)


def test_parse_stage_b_accepts_fixture() -> None:
    parsed = parse_stage_b(STAGE_B_PAYLOAD)
    assert parsed.diagnosis == "melanoma"
    assert parsed.differential_comparisons[0].alternative == "atypical nevus"
    assert "supports melanoma" in parsed.clinical_reasoning


def test_parse_stage_b_rejects_answer_without_comparison() -> None:
    payload = deepcopy(STAGE_B_PAYLOAD)
    payload["differential_comparisons"] = []
    with pytest.raises(ValidationError):
        parse_stage_b(payload)


def test_parse_stage_b_rejects_evidence_used_on_both_sides() -> None:
    payload = deepcopy(STAGE_B_PAYLOAD)
    comparisons = cast(
        list[dict[str, object]],
        payload["differential_comparisons"],
    )
    comparisons[0]["features_favoring_alternative"] = ["obs_002"]
    with pytest.raises(ValidationError):
        parse_stage_b(payload)


def test_parse_stage_b_accepts_new_image_policy() -> None:
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
    assert parse_stage_b(payload).response_policy.value == "REQUEST_NEW_IMAGE"


def test_stage_a_required_fields_match_on_disk_schema() -> None:
    schema_path = PROJECT_ROOT / "configs" / "schemas" / "stage_a_morphology.json"
    on_disk = json.loads(schema_path.read_text(encoding="utf-8"))
    from project.teacher.schemas import StageAMorphology

    assert on_disk == StageAMorphology.model_json_schema()


def test_stage_b_schema_snapshot_matches_model() -> None:
    schema_path = PROJECT_ROOT / "configs" / "schemas" / "stage_b_reasoning.json"
    on_disk = json.loads(schema_path.read_text(encoding="utf-8"))
    from project.teacher.schemas import StageBReasoning

    assert on_disk == StageBReasoning.model_json_schema()
