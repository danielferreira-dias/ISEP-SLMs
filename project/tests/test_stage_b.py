"""Stage B messages, merge-on-generate, and gold handling."""

import json
from pathlib import Path
from typing import Literal

import pytest
from PIL import Image

from project.dataset.examples import DistillExample
from project.stages.stage_b import (
    build_stage_b_messages,
    generate_reasoning,
    run_stage_b,
)
from project.teacher.client import TeacherResponse
from project.teacher.schemas import (
    RecordStatus,
    StageAFileRow,
    parse_stage_a,
)
from project.teacher.teacher import TeacherModel
from project.teacher.utils.jsonl import completed_stage_b_ids
from project.tests.fixtures import (
    STAGE_A_PAYLOAD,
    STAGE_B_PAYLOAD,
    fake_response,
    stage_a_morphology,
)


class _FakeCompleter:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.messages: list[dict[str, object]] | None = None

    def complete_stage(
        self,
        stage_key: Literal["A", "B"],
        messages: list[dict[str, object]],
    ) -> TeacherResponse:
        self.messages = messages
        assert stage_key == "B"
        return fake_response(self.payload)


def test_stage_b_messages_contain_gold_and_frozen_a() -> None:
    teacher = TeacherModel.from_yaml()
    morphology = stage_a_morphology()
    messages = build_stage_b_messages(
        teacher,
        "data:image/jpeg;base64,abc",
        morphology,
        "melanoma",
    )
    blob = json.dumps(messages)
    assert "melanoma" in blob
    assert "lesion.primary" in blob
    assert "macule" in blob
    assert "obs_003" in blob


def test_generate_reasoning_rejects_gold_mismatch(tmp_path: Path) -> None:
    teacher = TeacherModel.from_yaml()
    example = DistillExample(
        sample_id="s001",
        gold_diagnosis="psoriasis",
        image=Image.new("RGB", (8, 8), "red"),
        source_ref=str(tmp_path / "s001.jpg"),
    )
    stage_a = StageAFileRow(
        sample_id="s001",
        status=RecordStatus.OK,
        morphology=parse_stage_a(STAGE_A_PAYLOAD),
        error=None,
        usage=None,
        teacher=teacher.name,
        image_path=example.source_ref,
    )
    completer = _FakeCompleter(STAGE_B_PAYLOAD)
    result = generate_reasoning(
        completer,
        teacher,
        example,
        stage_a,
        "data:image/jpeg;base64,abc",
    )
    assert result.status is RecordStatus.REJECTED
    assert "gold_mismatch" in result.reasons


def test_run_stage_b_writes_ok_row(tmp_path: Path) -> None:
    teacher = TeacherModel.from_yaml()
    image = tmp_path / "s001.png"
    Image.new("RGB", (32, 32), "brown").save(image)

    example = DistillExample(
        sample_id="s001",
        gold_diagnosis="melanoma",
        image=Image.open(image).convert("RGB"),
        source_ref=str(image),
    )
    stage_a_path = tmp_path / "stage_a.jsonl"
    stage_a_path.write_text(
        json.dumps(
            {
                "sample_id": "s001",
                "status": "ok",
                "morphology": STAGE_A_PAYLOAD,
                "error": None,
                "usage": None,
                "teacher": teacher.name,
                "image_path": str(image),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "stage_b.jsonl"
    failures = run_stage_b(
        teacher=teacher,
        completer=_FakeCompleter(STAGE_B_PAYLOAD),
        examples=[example],
        stage_a_path=stage_a_path,
        output_path=output,
    )
    assert failures == 0
    written = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert written["status"] == "ok"
    assert written["gold_diagnosis"] == "melanoma"
    assert written["stage_a_sample_id"] == "s001"
    assert (
        written["reasoning"]["clinical_reasoning"]
        == (STAGE_B_PAYLOAD["clinical_reasoning"])
    )
    assert "student_target" not in written


def test_run_stage_b_records_missing_stage_a_as_error(tmp_path: Path) -> None:
    teacher = TeacherModel.from_yaml()
    example = DistillExample(
        sample_id="missing",
        gold_diagnosis="melanoma",
        image=Image.new("RGB", (8, 8), "brown"),
        source_ref="hf://diagnosis/sft_train/missing",
    )
    stage_a_path = tmp_path / "stage_a.jsonl"
    stage_a_path.write_text("", encoding="utf-8")
    output = tmp_path / "stage_b.jsonl"

    failures = run_stage_b(
        teacher=teacher,
        completer=_FakeCompleter(STAGE_B_PAYLOAD),
        examples=[example],
        stage_a_path=stage_a_path,
        output_path=output,
    )

    assert failures == 1
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["status"] == "error"
    assert written["error"] == "missing_ok_stage_a_record"


def test_run_stage_b_treats_rejected_output_as_terminal(tmp_path: Path) -> None:
    teacher = TeacherModel.from_yaml()
    example = DistillExample(
        sample_id="s001",
        gold_diagnosis="psoriasis",
        image=Image.new("RGB", (8, 8), "red"),
        source_ref="hf://diagnosis/sft_train/s001",
    )
    stage_a_path = tmp_path / "stage_a.jsonl"
    stage_a_path.write_text(
        StageAFileRow(
            sample_id="s001",
            status=RecordStatus.OK,
            morphology=parse_stage_a(STAGE_A_PAYLOAD),
            teacher=teacher.name,
            image_path=example.source_ref,
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "stage_b.jsonl"

    failures = run_stage_b(
        teacher=teacher,
        completer=_FakeCompleter(STAGE_B_PAYLOAD),
        examples=[example],
        stage_a_path=stage_a_path,
        output_path=output,
    )

    assert failures == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "rejected"
    assert completed_stage_b_ids(output) == {"s001"}


def test_resume_rejects_incomplete_legacy_success(
    tmp_path: Path,
) -> None:
    output = tmp_path / "legacy_stage_b.jsonl"
    output.write_text(
        json.dumps({"sample_id": "s001", "status": "ok"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid Stage B row"):
        completed_stage_b_ids(output)
