"""Vertex Batch transport tests for frozen Stage B."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from project.dataset.examples import DistillExample
from project.pipeline.generate import ExampleCohort
from project.pipeline.stage_b_batch import (
    build_batch_request,
    freeze_stage_b,
    ingest_results,
    normalize_overlap_errors,
    prepare_batch,
    submit_batch,
    upload_batch,
)
from project.teacher.provenance import generation_provenance
from project.teacher.schemas import RecordStatus, StageAFileRow, parse_stage_a
from project.teacher.teacher import TeacherModel
from project.teacher.utils.images import prepare_pil_image
from project.teacher.utils.jsonl import (
    append_jsonl,
    completed_stage_b_ids,
    load_stage_b_rows,
)
from project.tests.fixtures import STAGE_A_PAYLOAD, STAGE_B_PAYLOAD


def _teacher() -> TeacherModel:
    root = Path(__file__).resolve().parents[1]
    return TeacherModel.from_yaml(
        root / "configs" / "teacher_configs" / "gemini_3_7_flash_vertex.yaml"
    )


def _image() -> Image.Image:
    return Image.new("RGB", (20, 12), "brown")


def _cohort() -> ExampleCohort:
    ids = ("sample-000", "sample-001")

    def factory(selected: tuple[str, ...]) -> Iterator[DistillExample]:
        for sample_id in selected:
            yield DistillExample(
                sample_id=sample_id,
                gold_diagnosis="melanoma",
                image=_image(),
                source_ref=f"hf://test/{sample_id}",
            )

    return ExampleCohort(sample_ids=ids, factory=factory)


def _stage_a_row(teacher: TeacherModel, sample_id: str) -> StageAFileRow:
    return StageAFileRow(
        sample_id=sample_id,
        status=RecordStatus.OK,
        morphology=parse_stage_a(STAGE_A_PAYLOAD),
        error=None,
        usage=None,
        teacher=teacher.name,
        image_path=f"hf://test/{sample_id}",
        image_preprocessing=prepare_pil_image(_image()).info,
        provenance=generation_provenance(teacher, "A"),
    )


def _write_stage_a(path: Path, teacher: TeacherModel) -> None:
    for sample_id in _cohort().sample_ids:
        append_jsonl(path, _stage_a_row(teacher, sample_id))


def _batch_result(
    item: dict[str, object],
    payload: dict[str, object],
) -> dict[str, object]:
    return {
        "status": "",
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "private prompt"},
                        {
                            "fileData": {
                                "fileUri": item["image_gcs_uri"],
                                "mimeType": "image/jpeg",
                            }
                        },
                    ],
                }
            ]
        },
        "response": {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": json.dumps(payload)}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 200,
                "totalTokenCount": 300,
            },
        },
    }


def _mark_batch_succeeded(work_dir: Path) -> None:
    path = work_dir / "campaign_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = "batch_succeeded"
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_batch_request_contains_gold_and_frozen_stage_a() -> None:
    teacher = _teacher()
    request = build_batch_request(
        teacher=teacher,
        image_gcs_uri="gs://private/e3/images/one.jpg",
        stage_a=_stage_a_row(teacher, "sample-000"),
        gold_diagnosis="melanoma",
    )
    serialized = json.dumps(request)

    assert "melanoma" in serialized
    assert "obs_003" in serialized
    assert "lesion.symmetry" in serialized
    body = request["request"]
    assert isinstance(body, dict)
    generation = body["generationConfig"]
    assert isinstance(generation, dict)
    assert generation["seed"] == 42
    assert generation["thinkingConfig"] == {
        "thinkingLevel": "MEDIUM",
        "includeThoughts": False,
    }
    assert "responseSchema" in generation
    assert "responseJsonSchema" not in generation


def test_prepare_batch_records_private_gold_and_stage_a_hashes(tmp_path: Path) -> None:
    teacher = _teacher()
    stage_a = tmp_path / "stage_a.jsonl"
    _write_stage_a(stage_a, teacher)
    work_dir = tmp_path / "batch"

    manifest = prepare_batch(
        teacher=teacher,
        cohort=_cohort(),
        stage_a_path=stage_a,
        stage_b_output=tmp_path / "stage_b.jsonl",
        work_dir=work_dir,
        gcs_prefix="gs://private/e3/stage-b-test",
        pending_limit=1,
    )

    assert manifest["stage"] == "B"
    assert manifest["pending_request_count"] == 1
    assert manifest["gold_sent_to_teacher"] is True
    assert manifest["contains_private_gold"] is True
    assert manifest["external_transfer_authorized"] is False
    teacher_manifest = manifest["teacher"]
    assert isinstance(teacher_manifest, dict)
    assert teacher_manifest["protocol_frozen"] is True
    item = json.loads((work_dir / "items.jsonl").read_text().splitlines()[0])
    assert item["gold_diagnosis"] == "melanoma"
    assert item["stage_a_sample_id"] == "sample-000"
    assert len(item["stage_a_morphology_sha256"]) == 64
    assert "melanoma" in (work_dir / "requests.jsonl").read_text()


def test_upload_requires_private_gold_acknowledgement(tmp_path: Path) -> None:
    work_dir = tmp_path / "batch"
    work_dir.mkdir()
    (work_dir / "campaign_manifest.json").write_text(
        json.dumps({"stage": "B", "contains_private_gold": True}),
        encoding="utf-8",
    )

    with pytest.raises(PermissionError, match="private gold labels"):
        upload_batch(work_dir, authorize_private_gold_upload=False)


class _FakeJob:
    def model_dump(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "name": "projects/p/locations/global/batchPredictionJobs/2",
            "state": "JOB_STATE_PENDING",
        }


class _FakeBatches:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeJob:
        self.calls.append(kwargs)
        return _FakeJob()


class _FakeClient:
    def __init__(self) -> None:
        self.batches = _FakeBatches()


def test_submit_uses_stage_b_display_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher = _teacher()
    stage_a = tmp_path / "stage_a.jsonl"
    _write_stage_a(stage_a, teacher)
    work_dir = tmp_path / "batch"
    prepare_batch(
        teacher=teacher,
        cohort=_cohort(),
        stage_a_path=stage_a,
        stage_b_output=tmp_path / "stage_b.jsonl",
        work_dir=work_dir,
        gcs_prefix="gs://private/e3/stage-b-test",
        pending_limit=1,
    )
    manifest_path = work_dir / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "uploaded"
    manifest["external_transfer_authorized"] = True
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        "project.pipeline.stage_b_batch._upload_manifest_if_possible",
        lambda *args: None,
    )
    client = _FakeClient()

    submitted = submit_batch(teacher=teacher, work_dir=work_dir, client=client)

    assert submitted["status"] == "submitted"
    call = client.batches.calls[0]
    assert call["model"] == "gemini-3.7-flash"
    config = cast(Any, call["config"])
    assert "stage-b" in config.display_name


def test_ingest_reuses_parser_and_clinical_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher = _teacher()
    stage_a = tmp_path / "stage_a.jsonl"
    _write_stage_a(stage_a, teacher)
    stage_b = tmp_path / "stage_b.jsonl"
    work_dir = tmp_path / "batch"
    prepare_batch(
        teacher=teacher,
        cohort=_cohort(),
        stage_a_path=stage_a,
        stage_b_output=stage_b,
        work_dir=work_dir,
        gcs_prefix="gs://private/e3/stage-b-test",
    )
    items = [
        json.loads(line) for line in (work_dir / "items.jsonl").read_text().splitlines()
    ]
    results = work_dir / "results"
    results.mkdir()
    with (results / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, item in enumerate(reversed(items)):
            payload = dict(STAGE_B_PAYLOAD)
            if index == 1:
                payload["diagnosis"] = "psoriasis"
            handle.write(json.dumps(_batch_result(item, payload)) + "\n")
    _mark_batch_succeeded(work_dir)
    monkeypatch.setattr(
        "project.pipeline.stage_b_batch._upload_manifest_if_possible",
        lambda *args: None,
    )

    counts = ingest_results(
        teacher=teacher,
        work_dir=work_dir,
        stage_a_path=stage_a,
        stage_b_output=stage_b,
    )

    assert counts == {
        "ok": 1,
        "rejected": 1,
        "errors": 0,
        "skipped_existing_terminal": 0,
        "missing_batch_outputs": 0,
    }
    rows = load_stage_b_rows(stage_b)
    assert {row.status for row in rows} == {
        RecordStatus.OK,
        RecordStatus.REJECTED,
    }
    assert completed_stage_b_ids(stage_b) == {"sample-000", "sample-001"}
    assert any("gold_mismatch" in row.reasons for row in rows)
    assert all(row.usage is not None and row.usage.cost is not None for row in rows)

    freeze_dir = tmp_path / "frozen"
    manifest = freeze_stage_b(
        teacher=teacher,
        cohort=_cohort(),
        stage_a_path=stage_a,
        stage_b_output=stage_b,
        freeze_dir=freeze_dir,
    )
    accepted = load_stage_b_rows(freeze_dir / "stage_b.jsonl")
    rejected = load_stage_b_rows(freeze_dir / "rejected.jsonl")
    assert [row.status for row in accepted] == [RecordStatus.OK]
    assert [row.status for row in rejected] == [RecordStatus.REJECTED]
    assert manifest["terminal_coverage"] == {
        "rows": 2,
        "unique_sample_ids": 2,
        "missing_sample_ids": 0,
        "status_counts": {"ok": 1, "rejected": 1},
    }
    assert manifest["normalization"] == {
        "normalized_rows": 0,
        "artifacts": [],
    }
    assert (freeze_dir / "stage_b.md").is_file()
    assert (freeze_dir / "stage_b_reasoning.json").is_file()
    assert (freeze_dir / "gemini_3_7_flash_vertex.yaml").is_file()
    assert (freeze_dir / "freeze_manifest.json").is_file()


def test_ingest_keeps_batch_parse_error_retryable_and_auditable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher = _teacher()
    stage_a = tmp_path / "stage_a.jsonl"
    _write_stage_a(stage_a, teacher)
    stage_b = tmp_path / "stage_b.jsonl"
    work_dir = tmp_path / "batch"
    prepare_batch(
        teacher=teacher,
        cohort=_cohort(),
        stage_a_path=stage_a,
        stage_b_output=stage_b,
        work_dir=work_dir,
        gcs_prefix="gs://private/e3/stage-b-test",
        pending_limit=1,
    )
    item = json.loads((work_dir / "items.jsonl").read_text().splitlines()[0])
    results = work_dir / "results"
    results.mkdir()
    bad = _batch_result(item, STAGE_B_PAYLOAD)
    response = cast(dict[str, Any], bad["response"])
    response["candidates"][0]["content"]["parts"] = [{"text": "not-json"}]
    (results / "predictions.jsonl").write_text(json.dumps(bad) + "\n")
    _mark_batch_succeeded(work_dir)
    monkeypatch.setattr(
        "project.pipeline.stage_b_batch._upload_manifest_if_possible",
        lambda *args: None,
    )

    counts = ingest_results(
        teacher=teacher,
        work_dir=work_dir,
        stage_a_path=stage_a,
        stage_b_output=stage_b,
    )

    assert counts["errors"] == 1
    row = load_stage_b_rows(stage_b)[0]
    assert row.status is RecordStatus.ERROR
    assert completed_stage_b_ids(stage_b) == set()
    saved = json.loads((work_dir / "campaign_manifest.json").read_text())
    assert saved["status"] == "ingested_with_retryable_errors"


def test_overlap_normalization_is_dry_run_then_audited_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher = _teacher()
    stage_a = tmp_path / "stage_a.jsonl"
    _write_stage_a(stage_a, teacher)
    stage_b = tmp_path / "stage_b.jsonl"
    work_dir = tmp_path / "batch"
    prepare_batch(
        teacher=teacher,
        cohort=_cohort(),
        stage_a_path=stage_a,
        stage_b_output=stage_b,
        work_dir=work_dir,
        gcs_prefix="gs://private/e3/stage-b-test",
        pending_limit=1,
    )
    item = json.loads((work_dir / "items.jsonl").read_text().splitlines()[0])
    payload = json.loads(json.dumps(STAGE_B_PAYLOAD))
    comparisons = cast(list[dict[str, Any]], payload["differential_comparisons"])
    alternative_ids = cast(list[str], comparisons[0]["features_favoring_alternative"])
    alternative_ids.append("obs_003")
    results = work_dir / "results"
    results.mkdir()
    (results / "predictions.jsonl").write_text(
        json.dumps(_batch_result(item, payload)) + "\n",
        encoding="utf-8",
    )
    _mark_batch_succeeded(work_dir)
    monkeypatch.setattr(
        "project.pipeline.stage_b_batch._upload_manifest_if_possible",
        lambda *args: None,
    )
    ingested = ingest_results(
        teacher=teacher,
        work_dir=work_dir,
        stage_a_path=stage_a,
        stage_b_output=stage_b,
    )
    assert ingested["errors"] == 1

    dry_run = normalize_overlap_errors(
        teacher=teacher,
        work_dir=work_dir,
        stage_a_path=stage_a,
        stage_b_output=stage_b,
    )
    assert dry_run["apply"] is False
    assert dry_run["status_counts"] == {"ok": 1, "rejected": 0}
    assert len(load_stage_b_rows(stage_b)) == 1

    applied = normalize_overlap_errors(
        teacher=teacher,
        work_dir=work_dir,
        stage_a_path=stage_a,
        stage_b_output=stage_b,
        apply=True,
    )
    assert applied["status"] == "completed"
    assert applied["diagnosis_unchanged"] is True
    assert applied["clinical_reasoning_unchanged"] is True
    rows = load_stage_b_rows(stage_b)
    assert [row.status for row in rows] == [RecordStatus.ERROR, RecordStatus.OK]
    assert completed_stage_b_ids(stage_b) == {"sample-000"}
    record = json.loads(
        (work_dir / "overlap_normalization.jsonl").read_text(encoding="utf-8")
    )
    assert record["changes"] == [
        {"comparison_index": 0, "removed_observation_ids": ["obs_003"]}
    ]
    assert record["original_content_sha256"] != record["normalized_content_sha256"]
