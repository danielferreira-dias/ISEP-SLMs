"""Tests for the durable E3 teacher-generation progress interface."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.train.e3 import (
    E3CampaignSpec,
    E3CampaignState,
    E3GenerationStage,
    E3ProgressEvent,
    E3ProgressStore,
    ResponsePolicy,
    StageReviewStatus,
    TeacherGenerationStatus,
    progress_cli,
    render_terminal,
)
from src.train.e3.progress_cli import main as progress_main


def test_progress_store_publishes_live_and_final_interfaces(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "pilot"
    store = E3ProgressStore.start(output, _spec(total_samples=2))

    initial = store.read_snapshot()
    assert initial["status"] == "running"
    assert initial["progress"]["samples_finished"] == 0
    assert 'http-equiv="refresh"' in store.paths.report.read_text()

    store.record(_accepted_stage_a("sample-001"))
    running = store.record(_accepted_stage_b("sample-001"))
    assert running["progress"]["samples_finished"] == 1
    assert running["private_validation"] == {
        "evaluable_stage_b": 1,
        "gold_in_top3_matches": 1,
        "gold_in_top3_rate": 1.0,
        "gold_was_visible_to_teacher": False,
        "stage_a_gold_was_visible_to_teacher": False,
        "stage_b_gold_was_visible_to_teacher": False,
        "leading_label_metric_interpretation": "answer_blind_accuracy",
        "leading_label_match_rate": 1.0,
        "leading_label_matches": 1,
    }
    assert running["response_policy"]["REQUEST_CONTEXT"] == 1
    assert running["stage_b"]["diagnostic_review"]["accepted"] == 1
    assert running["stage_b"]["context_policy_review"]["accepted"] == 1
    assert running["materializable_rows"]["grounded_differential"] == 1
    assert "leading-label match: 1/1 (100.0%)" in render_terminal(running)

    completed_progress = store.record(_stage_a_safety_refusal("sample-002"))
    assert completed_progress["progress"]["samples_finished"] == 2
    assert completed_progress["failures"] == {"provider_safety_refusal": 1}
    assert completed_progress["stage_b"]["review"]["not_generated"] == 1

    final = store.finalize(E3CampaignState.COMPLETED)
    assert final["status"] == "completed"
    assert final["finished_at"] is not None
    assert 'http-equiv="refresh"' not in store.paths.report.read_text()
    assert len(store.read_events()) == 3

    assert progress_main([str(output), "--once"]) == 0
    captured = capsys.readouterr()
    assert "E3 TEACHER GENERATION  [completed]" in captured.out
    assert "Qwen/Qwen3.6-27B" in captured.out


def test_progress_counts_independent_stage_b_targets(tmp_path: Path) -> None:
    store = E3ProgressStore.start(tmp_path / "pilot", _spec(total_samples=1))
    store.record(_accepted_stage_a("sample-001"))
    stage_b = _accepted_stage_b("sample-001").model_copy(
        update={
            "leading_label_match": False,
            "diagnostic_review_status": StageReviewStatus.REJECTED,
            "context_policy_review_status": StageReviewStatus.ACCEPTED,
        }
    )

    snapshot = store.record(stage_b)

    assert snapshot["stage_b"]["diagnostic_review"]["rejected"] == 1
    assert snapshot["stage_b"]["context_policy_review"]["accepted"] == 1
    assert snapshot["materializable_rows"]["grounded_differential"] == 0
    assert snapshot["materializable_rows"]["context_policy"] == 1


def test_progress_labels_gold_conditioned_stage_b_as_anchor_compliance(
    tmp_path: Path,
) -> None:
    spec = _spec(total_samples=1).model_copy(
        update={"stage_b_gold_visible_to_teacher": True}
    )
    store = E3ProgressStore.start(tmp_path / "gold-conditioned", spec)
    store.record(_accepted_stage_a("sample-001"))

    snapshot = store.record(_accepted_stage_b("sample-001"))

    validation = snapshot["private_validation"]
    assert validation["gold_was_visible_to_teacher"] is True
    assert validation["stage_a_gold_was_visible_to_teacher"] is False
    assert validation["stage_b_gold_was_visible_to_teacher"] is True
    assert validation["leading_label_metric_interpretation"] == "anchor_compliance"
    assert "Stage-B anchor compliance" in render_terminal(snapshot)


def test_progress_store_forbids_silent_retry_and_stage_b_without_a(
    tmp_path: Path,
) -> None:
    store = E3ProgressStore.start(tmp_path / "pilot", _spec(total_samples=2))
    first = _accepted_stage_a("sample-001")
    store.record(first)
    with pytest.raises(ValueError, match="silent retries are forbidden"):
        store.record(first.model_copy(update={"event_id": "event-a-retry"}))
    with pytest.raises(ValueError, match="requires an accepted Stage-A"):
        store.record(_accepted_stage_b("sample-002"))


def test_completed_campaign_requires_every_sample_to_finish(tmp_path: Path) -> None:
    store = E3ProgressStore.start(tmp_path / "pilot", _spec(total_samples=2))
    store.record(_stage_a_safety_refusal("sample-001"))
    with pytest.raises(ValueError, match="every sample"):
        store.finalize(E3CampaignState.COMPLETED)
    interrupted = store.finalize(E3CampaignState.INTERRUPTED)
    assert interrupted["status"] == "interrupted"


def test_resume_requires_exact_campaign_identity(tmp_path: Path) -> None:
    output = tmp_path / "pilot"
    spec = _spec(total_samples=2)
    E3ProgressStore.start(output, spec)
    reopened = E3ProgressStore.start(output, spec, resume=True)
    assert reopened.spec == spec

    changed = spec.model_copy(update={"teacher_revision": "different-revision"})
    with pytest.raises(ValueError, match="identity mismatch"):
        E3ProgressStore.start(output, changed, resume=True)


def test_progress_event_rejects_gold_fields_and_incomplete_b_audit() -> None:
    payload = _accepted_stage_b("sample-001").model_dump(mode="json")
    payload["gold_disease_id"] = "D001"
    with pytest.raises(ValidationError, match="Extra inputs"):
        E3ProgressEvent.model_validate(payload)

    payload.pop("gold_disease_id")
    payload["leading_label_match"] = None
    with pytest.raises(ValidationError, match="gold audit booleans"):
        E3ProgressEvent.model_validate(payload)


def test_manifest_hash_detects_tampering(tmp_path: Path) -> None:
    output = tmp_path / "pilot"
    store = E3ProgressStore.start(output, _spec(total_samples=1))
    manifest = json.loads(store.paths.manifest.read_text())
    manifest["campaign"]["teacher_revision"] = "tampered"
    store.paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="identity hash mismatch"):
        E3ProgressStore.open(output)


def test_continuous_watcher_waits_for_campaign_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "future-pilot"
    calls = 0

    def create_campaign(_: float) -> None:
        nonlocal calls
        calls += 1
        store = E3ProgressStore.start(output, _spec(total_samples=1))
        store.record(_stage_a_safety_refusal("sample-001"))
        store.finalize(E3CampaignState.COMPLETED)

    monkeypatch.setattr(progress_cli.time, "sleep", create_campaign)

    assert progress_main([str(output), "--no-clear"]) == 0
    captured = capsys.readouterr()
    assert calls == 1
    assert "waiting for campaign to start" in captured.err
    assert "E3 TEACHER GENERATION  [completed]" in captured.out


def _spec(*, total_samples: int) -> E3CampaignSpec:
    return E3CampaignSpec(
        campaign_id="e3-pilot-qwen-3-6-27b",
        total_samples=total_samples,
        provider="modal",
        backend="vllm_endpoint",
        teacher_model="Qwen/Qwen3.6-27B",
        teacher_revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        stage_a_prompt_id="e3-stage-a-v1",
        stage_b_prompt_id="e3-stage-b-v1",
    )


def _accepted_stage_a(sample_id: str) -> E3ProgressEvent:
    return E3ProgressEvent(
        event_id=f"event-a-{sample_id}",
        recorded_at=datetime.now(UTC),
        sample_id=sample_id,
        stage=E3GenerationStage.STAGE_A,
        generation_status=TeacherGenerationStatus.SUCCEEDED,
        review_status=StageReviewStatus.ACCEPTED,
        latency_seconds=2.5,
        input_tokens=100,
        output_tokens=250,
    )


def _accepted_stage_b(sample_id: str) -> E3ProgressEvent:
    return E3ProgressEvent(
        event_id=f"event-b-{sample_id}",
        recorded_at=datetime.now(UTC),
        sample_id=sample_id,
        stage=E3GenerationStage.STAGE_B,
        generation_status=TeacherGenerationStatus.SUCCEEDED,
        review_status=StageReviewStatus.ACCEPTED,
        response_policy=ResponsePolicy.REQUEST_CONTEXT,
        leading_label_match=True,
        gold_in_top3=True,
        latency_seconds=3.5,
        input_tokens=350,
        output_tokens=500,
    )


def _stage_a_safety_refusal(sample_id: str) -> E3ProgressEvent:
    return E3ProgressEvent(
        event_id=f"event-a-{sample_id}",
        recorded_at=datetime.now(UTC),
        sample_id=sample_id,
        stage=E3GenerationStage.STAGE_A,
        generation_status=TeacherGenerationStatus.PROVIDER_SAFETY_REFUSAL,
        review_status=StageReviewStatus.NOT_APPLICABLE,
        provider_error_code="content_filter",
    )
