"""Stage A message building, generation, and resume."""

import json
from pathlib import Path
from typing import Literal

from PIL import Image

from project.dataset.examples import DistillExample
from project.stages.stage_a import (
    build_stage_a_messages,
    generate_morphology,
    run_stage_a,
)
from project.teacher.client import TeacherResponse
from project.teacher.schemas import ImageSample, RecordStatus
from project.teacher.teacher import TeacherModel
from project.teacher.utils.jsonl import completed_ids
from project.tests.fixtures import STAGE_A_PAYLOAD, fake_response


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
        assert stage_key == "A"
        return fake_response(self.payload)


def test_stage_a_messages_include_image_and_omit_gold() -> None:
    teacher = TeacherModel.from_yaml()
    messages = build_stage_a_messages(teacher, "data:image/jpeg;base64,abc")
    blob = json.dumps(messages)
    user = messages[1]
    assert "data:image/jpeg;base64,abc" in blob
    assert "gold_diagnosis" not in blob
    assert isinstance(user["content"], list)
    texts = [
        part["text"]
        for part in user["content"]
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    assert texts == [teacher.stage("A").prompt.user]


def test_generate_morphology_ok(tmp_path: Path) -> None:
    teacher = TeacherModel.from_yaml()
    sample = ImageSample(sample_id="s001", image_path=tmp_path / "s001.jpg")
    completer = _FakeCompleter(STAGE_A_PAYLOAD)
    row = generate_morphology(completer, teacher, sample, "data:image/jpeg;base64,abc")
    assert row.status is RecordStatus.OK
    assert row.morphology is not None
    assert "gold_diagnosis" not in row.model_dump()


def test_run_stage_a_resume_skips_ok(tmp_path: Path) -> None:
    teacher = TeacherModel.from_yaml()
    image = tmp_path / "s001.png"
    Image.new("RGB", (32, 32), "red").save(image)

    example = DistillExample(
        sample_id="s001",
        gold_diagnosis="melanoma",
        image=Image.open(image).convert("RGB"),
        source_ref=str(image),
    )
    output = tmp_path / "stage_a.jsonl"
    output.write_text(
        json.dumps(
            {
                "sample_id": "s001",
                "status": "ok",
                "morphology": STAGE_A_PAYLOAD,
                "error": None,
                "usage": None,
                "teacher": "gemini_3_7_flash",
                "image_path": str(image),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completer = _FakeCompleter(STAGE_A_PAYLOAD)
    failures = run_stage_a(
        teacher=teacher,
        completer=completer,
        examples=[example],
        output_path=output,
        resume=True,
    )
    assert failures == 0
    assert completer.messages is None
    assert completed_ids(output) == {"s001"}
