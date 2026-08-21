"""Sequential Stage A/B campaign and live progress tests."""

from __future__ import annotations

import json
from copy import deepcopy
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from typing import Literal

import pytest
from PIL import Image

from project.dataset.examples import DistillExample
from project.pipeline.generate import (
    CampaignBudgetExceeded,
    CampaignFailure,
    ExampleCohort,
    run_teacher_campaign,
)
from project.teacher.client import TeacherCompletionError, TeacherResponse
from project.teacher.schemas import UsageInfo
from project.teacher.teacher import TeacherModel
from project.tests.fixtures import (
    STAGE_A_PAYLOAD,
    STAGE_B_PAYLOAD,
    fake_response,
)


class _SequentialCompleter:
    def __init__(
        self,
        *,
        fail_stage_a: bool = False,
        usage: UsageInfo | None = None,
        stage_b_payload: dict[str, object] | None = None,
    ) -> None:
        self.fail_stage_a = fail_stage_a
        self.usage = usage
        self.stage_b_payload = stage_b_payload
        self.calls: list[Literal["A", "B"]] = []

    def complete_stage(
        self,
        stage_key: Literal["A", "B"],
        messages: list[dict[str, object]],
    ) -> TeacherResponse:
        del messages
        self.calls.append(stage_key)
        if self.fail_stage_a and stage_key == "A":
            raise TeacherCompletionError("provider_error:test")
        payload = (
            STAGE_A_PAYLOAD
            if stage_key == "A"
            else self.stage_b_payload or STAGE_B_PAYLOAD
        )
        response = fake_response(payload)
        if self.usage is None:
            return response
        return TeacherResponse(
            content_json=response.content_json,
            raw_content=response.raw_content,
            usage=self.usage,
            finish_reason=response.finish_reason,
            native_finish_reason=response.native_finish_reason,
        )


def _vertex_teacher() -> TeacherModel:
    root = Path(__file__).resolve().parents[1]
    return TeacherModel.from_yaml(
        root / "configs" / "teacher_configs" / "gemini_3_7_flash_vertex.yaml"
    )


def _cohort(count: int = 2) -> ExampleCohort:
    sample_ids = tuple(f"sample-{index:03d}" for index in range(count))

    def factory(selected: tuple[str, ...]) -> Iterator[DistillExample]:
        for sample_id in selected:
            yield DistillExample(
                sample_id=sample_id,
                gold_diagnosis="melanoma",
                image=Image.new("RGB", (12, 12), "brown"),
                source_ref=f"hf://test/{sample_id}",
            )

    return ExampleCohort(sample_ids=sample_ids, factory=factory)


def test_campaign_runs_a_then_b_and_reports_live_progress(tmp_path: Path) -> None:
    teacher = TeacherModel.from_yaml()
    completer = _SequentialCompleter()
    stream = StringIO()
    stage_a = tmp_path / "stage_a.jsonl"
    stage_b = tmp_path / "stage_b.jsonl"

    result = run_teacher_campaign(
        teacher=teacher,
        completer=completer,
        cohort=_cohort(),
        stage_a_output=stage_a,
        stage_b_output=stage_b,
        progress_stream=stream,
    )

    assert completer.calls == ["A", "A", "B", "B"]
    assert result.selected_samples == 2
    assert result.stage_a_completed == 2
    assert result.stage_b_completed == 2
    assert result.stage_b_ok == 2
    assert result.stage_b_rejected == 0
    assert len(stage_a.read_text(encoding="utf-8").splitlines()) == 2
    assert len(stage_b.read_text(encoding="utf-8").splitlines()) == 2
    progress = stream.getvalue()
    assert "Stage A" in progress
    assert "Stage B" in progress
    assert "2/2" in progress
    assert "left=0" in progress
    assert "failed=0" in progress

    resumed = _SequentialCompleter()
    resumed_result = run_teacher_campaign(
        teacher=teacher,
        completer=resumed,
        cohort=_cohort(),
        stage_a_output=stage_a,
        stage_b_output=stage_b,
        progress_stream=StringIO(),
    )
    assert resumed.calls == []
    assert resumed_result.stage_b_completed == 2


def test_campaign_preserves_rejected_stage_b_without_retry(tmp_path: Path) -> None:
    teacher = TeacherModel.from_yaml()
    rejected_payload = deepcopy(STAGE_B_PAYLOAD)
    rejected_payload["diagnosis"] = "psoriasis"
    stage_a = tmp_path / "stage_a.jsonl"
    stage_b = tmp_path / "stage_b.jsonl"
    stream = StringIO()

    first = _SequentialCompleter(stage_b_payload=rejected_payload)
    result = run_teacher_campaign(
        teacher=teacher,
        completer=first,
        cohort=_cohort(count=1),
        stage_a_output=stage_a,
        stage_b_output=stage_b,
        progress_stream=stream,
    )

    assert first.calls == ["A", "B"]
    assert result.stage_b_completed == 1
    assert result.stage_b_ok == 0
    assert result.stage_b_rejected == 1
    assert "rejected=1" in stream.getvalue()
    assert "failed=0" in stream.getvalue()
    assert len(stage_b.read_text(encoding="utf-8").splitlines()) == 1

    resumed = _SequentialCompleter()
    resumed_result = run_teacher_campaign(
        teacher=teacher,
        completer=resumed,
        cohort=_cohort(count=1),
        stage_a_output=stage_a,
        stage_b_output=stage_b,
        progress_stream=StringIO(),
    )

    assert resumed.calls == []
    assert resumed_result.stage_b_rejected == 1
    assert len(stage_b.read_text(encoding="utf-8").splitlines()) == 1


def test_campaign_stops_before_b_when_a_fails(tmp_path: Path) -> None:
    teacher = TeacherModel.from_yaml()
    completer = _SequentialCompleter(fail_stage_a=True)
    stream = StringIO()

    with pytest.raises(CampaignFailure, match="Stage B was not started"):
        run_teacher_campaign(
            teacher=teacher,
            completer=completer,
            cohort=_cohort(count=1),
            stage_a_output=tmp_path / "stage_a.jsonl",
            stage_b_output=tmp_path / "stage_b.jsonl",
            progress_stream=stream,
        )

    assert completer.calls == ["A"]
    assert not (tmp_path / "stage_b.jsonl").exists()
    assert "failed=1" in stream.getvalue()


def test_campaign_limit_selects_one_stable_a_b_sample(tmp_path: Path) -> None:
    teacher = TeacherModel.from_yaml()
    completer = _SequentialCompleter()

    result = run_teacher_campaign(
        teacher=teacher,
        completer=completer,
        cohort=_cohort(count=3),
        stage_a_output=tmp_path / "stage_a.jsonl",
        stage_b_output=tmp_path / "stage_b.jsonl",
        limit=1,
        progress_stream=StringIO(),
    )

    assert result.selected_samples == 1
    assert completer.calls == ["A", "B"]


def test_campaign_refuses_resume_across_prompt_or_schema_change(
    tmp_path: Path,
) -> None:
    teacher = TeacherModel.from_yaml()
    stage_a = tmp_path / "stage_a.jsonl"
    stage_b = tmp_path / "stage_b.jsonl"
    run_teacher_campaign(
        teacher=teacher,
        completer=_SequentialCompleter(),
        cohort=_cohort(count=1),
        stage_a_output=stage_a,
        stage_b_output=stage_b,
        progress_stream=StringIO(),
    )
    payload = json.loads(stage_a.read_text(encoding="utf-8"))
    payload["provenance"]["prompt_sha256"] = "0" * 64
    stage_a.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    resumed = _SequentialCompleter()

    with pytest.raises(CampaignFailure, match="resume provenance mismatch"):
        run_teacher_campaign(
            teacher=teacher,
            completer=resumed,
            cohort=_cohort(count=1),
            stage_a_output=stage_a,
            stage_b_output=stage_b,
            progress_stream=StringIO(),
        )

    assert resumed.calls == []


def test_campaign_refuses_resume_across_reasoning_effort_change(
    tmp_path: Path,
) -> None:
    teacher = _vertex_teacher()
    stage_a = tmp_path / "stage_a.jsonl"
    stage_b = tmp_path / "stage_b.jsonl"
    run_teacher_campaign(
        teacher=teacher,
        completer=_SequentialCompleter(),
        cohort=_cohort(count=1),
        stage_a_output=stage_a,
        stage_b_output=stage_b,
        progress_stream=StringIO(),
    )
    payload = json.loads(stage_a.read_text(encoding="utf-8"))
    assert payload["provenance"]["reasoning_effort"] == "medium"
    payload["provenance"]["reasoning_effort"] = "high"
    stage_a.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    resumed = _SequentialCompleter()

    with pytest.raises(CampaignFailure, match="resume provenance mismatch"):
        run_teacher_campaign(
            teacher=teacher,
            completer=resumed,
            cohort=_cohort(count=1),
            stage_a_output=stage_a,
            stage_b_output=stage_b,
            progress_stream=StringIO(),
        )

    assert resumed.calls == []


def test_vertex_campaign_displays_cumulative_estimated_cost(tmp_path: Path) -> None:
    stream = StringIO()
    result = run_teacher_campaign(
        teacher=_vertex_teacher(),
        completer=_SequentialCompleter(
            usage=UsageInfo(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=200,
            )
        ),
        cohort=_cohort(count=1),
        stage_a_output=tmp_path / "stage_a.jsonl",
        stage_b_output=tmp_path / "stage_b.jsonl",
        progress_stream=stream,
    )

    assert result.estimated_cost_usd == pytest.approx(0.0009)
    assert "est_cost=$0.0009" in stream.getvalue()


def test_local_cost_guard_stops_without_starting_next_stage(tmp_path: Path) -> None:
    completer = _SequentialCompleter(
        usage=UsageInfo(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=200,
        )
    )
    with pytest.raises(CampaignBudgetExceeded, match="configured ceiling"):
        run_teacher_campaign(
            teacher=_vertex_teacher(),
            completer=completer,
            cohort=_cohort(count=2),
            stage_a_output=tmp_path / "stage_a.jsonl",
            stage_b_output=tmp_path / "stage_b.jsonl",
            progress_stream=StringIO(),
            max_estimated_cost_usd=0.0005,
        )

    assert completer.calls == ["A", "A"]
    assert not (tmp_path / "stage_b.jsonl").exists()


def test_stage_a_prompt_contains_operational_terminology_not_full_examples() -> None:
    system = TeacherModel.from_yaml().stage("A").prompt.system
    normalized = " ".join(system.split())

    assert "Distinguish border demarcation" in normalized
    assert "Reserve `hyperkeratotic`" in normalized
    assert "Do not list symptoms, history, systemic involvement" in normalized
    assert "The tenth ID is `obs_010`, never `obs_0010`" in normalized
    assert "Never use tactile wording" in normalized
    assert "A single time point cannot establish temporal behaviour" in normalized
    assert "Do not infer cause or exposure" in normalized
    assert "Example output" not in normalized


def test_stage_b_prompt_calibrates_anchor_and_forbids_new_clinical_facts() -> None:
    system = TeacherModel.from_yaml().stage("B").prompt.system
    normalized = " ".join(system.split())

    assert "Never upgrade compatibility into discrimination" in normalized
    assert "silence is not negative evidence" in normalized
    assert "do not recommend treatment, biopsy, excision" in normalized
    assert "never use ranges such as `moderate-to-high`" in normalized
    assert "Do not manufacture certainty" in " ".join(
        TeacherModel.from_yaml().stage("B").prompt.user.split()
    )
