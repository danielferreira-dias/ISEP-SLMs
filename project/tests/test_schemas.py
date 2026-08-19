"""Parse tests for Stage A and Stage B Pydantic models."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from project.teacher.schemas import parse_stage_a, parse_stage_b
from project.tests.fixtures import STAGE_A_PAYLOAD, STAGE_B_PAYLOAD

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_parse_stage_a_accepts_fixture() -> None:
    morphology = parse_stage_a(STAGE_A_PAYLOAD)
    assert morphology.primary_lesion == "macule"
    assert morphology.modality.value == "clinical"


def test_parse_stage_a_rejects_extra_key() -> None:
    payload = dict(STAGE_A_PAYLOAD)
    payload["diagnosis"] = "melanoma"
    with pytest.raises(ValidationError):
        parse_stage_a(payload)


def test_parse_stage_a_rejects_bad_enum() -> None:
    payload = dict(STAGE_A_PAYLOAD)
    payload["shape"] = "square"
    with pytest.raises(ValidationError):
        parse_stage_a(payload)


def test_parse_stage_b_accepts_fixture() -> None:
    parsed = parse_stage_b(STAGE_B_PAYLOAD)
    assert parsed.diagnosis == "melanoma"
    assert parsed.differential_diagnosis[0].rank == 1


def test_parse_stage_b_rejects_short_ddx() -> None:
    payload = dict(STAGE_B_PAYLOAD)
    payload["differential_diagnosis"] = payload["differential_diagnosis"][:1]
    with pytest.raises(ValidationError):
        parse_stage_b(payload)


def test_stage_a_required_fields_match_on_disk_schema() -> None:
    schema_path = PROJECT_ROOT / "configs" / "schemas" / "stage_a_morphology.json"
    on_disk = json.loads(schema_path.read_text(encoding="utf-8"))
    from project.teacher.schemas import StageAMorphology

    dumped = StageAMorphology.model_json_schema()
    assert set(on_disk["required"]) == set(dumped["required"])
    assert on_disk["additionalProperties"] is False
