"""Vertex Batch transport tests for frozen Stage A."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

from project.dataset.examples import DistillExample
from project.pipeline.generate import CampaignFailure, ExampleCohort
from project.pipeline.stage_a_batch import (
    _gcs_object_lines,
    _inline_local_json_schema_refs,
    _json_schema_to_vertex_openapi,
    _validate_batch_request_file,
    build_batch_request,
    freeze_stage_a,
    ingest_results,
    prepare_batch,
    reprice_batch_rows,
    submit_batch,
)
from project.teacher.provenance import generation_provenance
from project.teacher.schemas import RecordStatus, StageAFileRow, parse_stage_a
from project.teacher.teacher import TeacherModel
from project.teacher.utils.jsonl import append_jsonl, load_stage_a_rows
from project.tests.fixtures import STAGE_A_PAYLOAD


def _teacher() -> TeacherModel:
    root = Path(__file__).resolve().parents[1]
    return TeacherModel.from_yaml(
        root / "configs" / "teacher_configs" / "gemini_3_7_flash_vertex.yaml"
    )


def _cohort() -> ExampleCohort:
    ids = ("sample-000", "sample-001")

    def factory(selected: tuple[str, ...]) -> Iterator[DistillExample]:
        for sample_id in selected:
            yield DistillExample(
                sample_id=sample_id,
                gold_diagnosis="must-not-leak",
                image=Image.new("RGB", (20, 12), "brown"),
                source_ref=f"hf://test/{sample_id}",
            )

    return ExampleCohort(sample_ids=ids, factory=factory)


def _existing_row(teacher: TeacherModel, sample_id: str) -> StageAFileRow:
    return StageAFileRow(
        sample_id=sample_id,
        status=RecordStatus.OK,
        morphology=parse_stage_a(STAGE_A_PAYLOAD),
        error=None,
        usage=None,
        teacher=teacher.name,
        image_path=f"hf://test/{sample_id}",
        provenance=generation_provenance(teacher, "A"),
    )


def test_batch_request_preserves_frozen_stage_a_and_excludes_gold() -> None:
    teacher = _teacher()
    request = build_batch_request(
        teacher=teacher,
        image_gcs_uri="gs://private/e3/images/one.jpg",
    )
    serialized = json.dumps(request)

    assert "must-not-leak" not in serialized
    body = request["request"]
    assert isinstance(body, dict)
    generation = body["generationConfig"]
    assert isinstance(generation, dict)
    assert generation["seed"] == 42
    assert generation["maxOutputTokens"] == 16384
    assert generation["responseMimeType"] == "application/json"
    assert "responseJsonSchema" not in generation
    assert generation["responseSchema"] == _json_schema_to_vertex_openapi(
        teacher.stage("A").json_schema.schema
    )
    assert '"$ref"' not in json.dumps(generation["responseSchema"])
    assert '"$defs"' not in json.dumps(generation["responseSchema"])
    assert generation["thinkingConfig"] == {
        "thinkingLevel": "MEDIUM",
        "includeThoughts": False,
    }


def test_batch_schema_expansion_preserves_nested_contract() -> None:
    schema = {
        "$defs": {
            "Nested": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            }
        },
        "type": "object",
        "properties": {"nested": {"$ref": "#/$defs/Nested"}},
        "required": ["nested"],
    }

    expanded = _inline_local_json_schema_refs(schema)

    assert "$defs" not in expanded
    assert expanded["properties"] == {
        "nested": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }
    }


def test_batch_schema_uses_openapi_nullable_without_heterogeneous_anyof() -> None:
    converted = _json_schema_to_vertex_openapi(
        {
            "type": "object",
            "properties": {
                "evidence_region": {
                    "anyOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "null"},
                    ],
                    "title": "Evidence Region",
                }
            },
        }
    )

    properties = converted["properties"]
    assert isinstance(properties, dict)
    assert properties["evidence_region"] == {
        "type": "STRING",
        "minLength": 1,
        "title": "Evidence Region",
        "nullable": True,
    }


def test_batch_request_validation_rejects_dollar_prefixed_schema_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "requests.jsonl"
    path.write_text(
        json.dumps(
            {
                "request": {
                    "generationConfig": {"responseSchema": {"$ref": "#/$defs/Bad"}}
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CampaignFailure, match="dollar-prefixed schema keys"):
        _validate_batch_request_file(path, expected_count=1)


def test_recursive_gcs_listing_does_not_count_directory_header() -> None:
    output = "\n".join(
        (
            "gs://bucket/prefix/images/:",
            "gs://bucket/prefix/images/001.jpg",
            "gs://bucket/prefix/images/002.jpg",
        )
    )

    assert _gcs_object_lines(output) == [
        "gs://bucket/prefix/images/001.jpg",
        "gs://bucket/prefix/images/002.jpg",
    ]


def test_prepare_batch_skips_only_compatible_ok_rows(tmp_path: Path) -> None:
    teacher = _teacher()
    stage_a = tmp_path / "stage_a.jsonl"
    append_jsonl(stage_a, _existing_row(teacher, "sample-000"))
    work_dir = tmp_path / "batch"

    manifest = prepare_batch(
        teacher=teacher,
        cohort=_cohort(),
        stage_a_output=stage_a,
        work_dir=work_dir,
        gcs_prefix="gs://private/e3/stage-a-test",
    )

    assert manifest["source_sample_count"] == 2
    assert manifest["existing_compatible_ok_count"] == 1
    assert manifest["pending_request_count"] == 1
    assert manifest["gold_sent_to_teacher"] is False
    items = [
        json.loads(line) for line in (work_dir / "items.jsonl").read_text().splitlines()
    ]
    assert [item["sample_id"] for item in items] == ["sample-001"]
    assert "gold_diagnosis" not in items[0]
    request_text = (work_dir / "requests.jsonl").read_text(encoding="utf-8")
    assert "must-not-leak" not in request_text
    assert len(tuple((work_dir / "images").glob("*.jpg"))) == 1


def test_prepare_batch_pending_limit_creates_one_image_canary(tmp_path: Path) -> None:
    teacher = _teacher()
    work_dir = tmp_path / "canary"

    manifest = prepare_batch(
        teacher=teacher,
        cohort=_cohort(),
        stage_a_output=tmp_path / "missing.jsonl",
        work_dir=work_dir,
        gcs_prefix="gs://private/e3/stage-a-canary",
        pending_limit=1,
    )

    assert manifest["pending_source_count_before_limit"] == 2
    assert manifest["pending_request_count"] == 1
    assert manifest["pending_limit"] == 1
    items = [
        json.loads(line) for line in (work_dir / "items.jsonl").read_text().splitlines()
    ]
    assert [item["sample_id"] for item in items] == ["sample-000"]


def test_freeze_stage_a_writes_one_ordered_ok_row_per_source_id(
    tmp_path: Path,
) -> None:
    teacher = _teacher()
    stage_a = tmp_path / "stage_a.jsonl"
    append_jsonl(stage_a, _existing_row(teacher, "sample-001"))
    append_jsonl(stage_a, _existing_row(teacher, "sample-000"))
    freeze_dir = tmp_path / "freeze"

    manifest = freeze_stage_a(
        teacher=teacher,
        cohort=_cohort(),
        stage_a_output=stage_a,
        freeze_dir=freeze_dir,
    )

    frozen = load_stage_a_rows(freeze_dir / "stage_a.jsonl")
    assert [row.sample_id for row in frozen] == ["sample-000", "sample-001"]
    assert all(row.status is RecordStatus.OK for row in frozen)
    assert manifest["status"] == "completed"
    assert manifest["accepted_release"]["rows"] == 2
    assert (freeze_dir / "stage_a.md").read_bytes() == teacher.stage(
        "A"
    ).prompt.source_path.read_bytes()


class _FakeJob:
    def model_dump(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "name": "projects/p/locations/global/batchPredictionJobs/1",
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


def test_submit_records_one_job_without_resubmission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher = _teacher()
    work_dir = tmp_path / "batch"
    prepare_batch(
        teacher=teacher,
        cohort=_cohort(),
        stage_a_output=tmp_path / "missing.jsonl",
        work_dir=work_dir,
        gcs_prefix="gs://private/e3/stage-a-test",
    )
    manifest_path = work_dir / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "uploaded"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    client = _FakeClient()

    from project.pipeline import stage_a_batch

    monkeypatch.setattr(
        stage_a_batch,
        "_upload_manifest_if_possible",
        lambda *args: None,
    )
    submitted = submit_batch(teacher=teacher, work_dir=work_dir, client=client)

    assert submitted["status"] == "submitted"
    assert len(client.batches.calls) == 1
    assert client.batches.calls[0]["model"] == "gemini-3.7-flash"


def test_ingest_correlates_output_by_echoed_image_uri(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher = _teacher()
    work_dir = tmp_path / "batch"
    prepare_batch(
        teacher=teacher,
        cohort=_cohort(),
        stage_a_output=tmp_path / "missing.jsonl",
        work_dir=work_dir,
        gcs_prefix="gs://private/e3/stage-a-test",
    )
    items = [
        json.loads(line) for line in (work_dir / "items.jsonl").read_text().splitlines()
    ]
    results = work_dir / "results"
    results.mkdir()
    output = results / "predictions.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for item in reversed(items):
            payload = {
                "status": "",
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": "prompt"},
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
                                "parts": [{"text": json.dumps(STAGE_A_PAYLOAD)}],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 20,
                        "totalTokenCount": 30,
                    },
                },
            }
            handle.write(json.dumps(payload) + "\n")
    manifest_path = work_dir / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "batch_succeeded"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    stage_a = tmp_path / "stage_a.jsonl"

    from project.pipeline import stage_a_batch

    monkeypatch.setattr(
        stage_a_batch,
        "_upload_manifest_if_possible",
        lambda *args: None,
    )
    counts = ingest_results(
        teacher=teacher,
        work_dir=work_dir,
        stage_a_output=stage_a,
    )

    assert counts == {
        "ok": 2,
        "errors": 0,
        "skipped_existing_ok": 0,
        "missing_batch_outputs": 0,
    }
    assert {row.sample_id for row in load_stage_a_rows(stage_a)} == {
        "sample-000",
        "sample-001",
    }
    for row in load_stage_a_rows(stage_a):
        assert row.usage is not None
        assert row.usage.cost == pytest.approx(0.00004125)


def test_ingest_quarantines_invalid_rows_without_polluting_canonical_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher = _teacher()
    work_dir = tmp_path / "batch"
    prepare_batch(
        teacher=teacher,
        cohort=_cohort(),
        stage_a_output=tmp_path / "missing.jsonl",
        work_dir=work_dir,
        gcs_prefix="gs://private/e3/stage-a-test",
        pending_limit=1,
    )
    item = json.loads((work_dir / "items.jsonl").read_text().splitlines()[0])
    results = work_dir / "results"
    results.mkdir()
    payload = {
        "status": "",
        "request": {
            "contents": [{"parts": [{"fileData": {"fileUri": item["image_gcs_uri"]}}]}]
        },
        "response": {
            "candidates": [
                {
                    "content": {"parts": [{"text": ""}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30,
            },
        },
    }
    (results / "predictions.jsonl").write_text(json.dumps(payload) + "\n")
    manifest_path = work_dir / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "batch_succeeded"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        "project.pipeline.stage_a_batch._upload_manifest_if_possible",
        lambda *args: None,
    )
    stage_a = tmp_path / "stage_a.jsonl"

    counts = ingest_results(
        teacher=teacher,
        work_dir=work_dir,
        stage_a_output=stage_a,
    )

    assert counts == {
        "ok": 0,
        "errors": 1,
        "skipped_existing_ok": 0,
        "missing_batch_outputs": 0,
    }
    assert not stage_a.exists()
    quarantine = load_stage_a_rows(work_dir / "ingestion_errors.jsonl")
    assert len(quarantine) == 1
    assert quarantine[0].status is RecordStatus.ERROR
    assert quarantine[0].usage is not None
    assert quarantine[0].usage.cost == pytest.approx(0.00004125)
    saved = json.loads(manifest_path.read_text())
    assert saved["status"] == "ingested_with_quarantine"
    assert saved["ingestion"]["accepted_rows_only"] is True


def test_reprice_batch_rows_changes_only_usage_and_preserves_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher = _teacher()
    work_dir = tmp_path / "batch"
    prepare_batch(
        teacher=teacher,
        cohort=_cohort(),
        stage_a_output=tmp_path / "missing.jsonl",
        work_dir=work_dir,
        gcs_prefix="gs://private/e3/stage-a-test",
        pending_limit=1,
    )
    item = json.loads((work_dir / "items.jsonl").read_text().splitlines()[0])
    results = work_dir / "results"
    results.mkdir()
    payload = {
        "status": "",
        "request": {
            "contents": [{"parts": [{"fileData": {"fileUri": item["image_gcs_uri"]}}]}]
        },
        "response": {
            "candidates": [
                {
                    "content": {"parts": [{"text": json.dumps(STAGE_A_PAYLOAD)}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30,
            },
        },
    }
    (results / "predictions.jsonl").write_text(json.dumps(payload) + "\n")
    from project.pipeline.stage_a_batch import _stage_a_row_from_result

    batch_row = _stage_a_row_from_result(teacher, item, payload)
    assert batch_row.usage is not None
    standard_usage = teacher.pricing.estimate_usage(
        batch_row.usage.model_copy(
            update={"cost": None, "cost_currency": None, "cost_basis": None}
        )
    )
    canonical = batch_row.model_copy(update={"usage": standard_usage})
    stage_a = tmp_path / "stage_a.jsonl"
    append_jsonl(stage_a, canonical)
    before_target = canonical.morphology
    backup = tmp_path / "stage_a.before.jsonl"
    monkeypatch.setattr(
        "project.pipeline.stage_a_batch._upload_manifest_if_possible",
        lambda *args: None,
    )

    result = reprice_batch_rows(
        teacher=teacher,
        work_dir=work_dir,
        stage_a_output=stage_a,
        backup_path=backup,
    )

    corrected = load_stage_a_rows(stage_a)[0]
    original = load_stage_a_rows(backup)[0]
    assert result["changed_rows"] == 1
    assert corrected.morphology == before_target == original.morphology
    assert corrected.usage is not None
    assert corrected.usage.cost == pytest.approx(0.00004125)
    assert original.usage is not None
    assert original.usage.cost == pytest.approx(0.0000825)
