"""Stage B messages, merge-on-generate, and gold handling."""

import json
from pathlib import Path
from typing import Literal

from PIL import Image

from project.stages.stage_b import (
    build_stage_b_messages,
    generate_reasoning,
    run_stage_b,
)
from project.teacher.client import TeacherResponse
from project.teacher.schemas import (
    ManifestRow,
    RecordStatus,
    StageAFileRow,
    parse_stage_a,
)
from project.teacher.teacher import TeacherModel
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
    assert "primary_lesion" in blob
    assert "macule" in blob
    assert "asymmetric" in blob


def test_generate_reasoning_rejects_gold_mismatch(tmp_path: Path) -> None:
    teacher = TeacherModel.from_yaml()
    row = ManifestRow(
        sample_id="s001",
        image_path=str(tmp_path / "s001.jpg"),
        gold_diagnosis="psoriasis",
    )
    stage_a = StageAFileRow(
        sample_id="s001",
        status=RecordStatus.OK,
        morphology=parse_stage_a(STAGE_A_PAYLOAD),
        error=None,
        usage=None,
        teacher=teacher.name,
        image_path=row.image_path,
    )
    completer = _FakeCompleter(STAGE_B_PAYLOAD)
    result = generate_reasoning(
        completer,
        teacher,
        row,
        stage_a,
        "data:image/jpeg;base64,abc",
    )
    assert result.status is RecordStatus.REJECTED
    assert "gold_mismatch" in result.reasons


def test_run_stage_b_writes_ok_row(tmp_path: Path) -> None:
    teacher = TeacherModel.from_yaml()
    image = tmp_path / "s001.png"
    Image.new("RGB", (32, 32), "brown").save(image)

    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "s001",
                "image_path": str(image),
                "gold_diagnosis": "melanoma",
            }
        )
        + "\n",
        encoding="utf-8",
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
        manifest_path=manifest,
        stage_a_path=stage_a_path,
        output_path=output,
    )
    assert failures == 0
    written = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert written["status"] == "ok"
    assert written["gold_diagnosis"] == "melanoma"
    assert written["stage_a_sample_id"] == "s001"
