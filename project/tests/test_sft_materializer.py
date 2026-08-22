"""Contract tests for the E3 multitask SFT materializer."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from project.pipeline.materialize_sft import (
    CAPTION_PROMPT,
    CLINICAL_ASSESSMENT_PROMPT,
    MORPHOLOGY_PROMPT,
    MaterializationSource,
    SFTTask,
    materialize_multitask_rows,
    source_from_hub_row,
    write_multitask_release,
)
from project.teacher.schemas import (
    RecordStatus,
    StageAFileRow,
    StageBFileRow,
    parse_stage_a,
    parse_stage_b,
)
from project.tests.fixtures import STAGE_A_PAYLOAD, STAGE_B_PAYLOAD

_DIAGNOSIS_PROMPT = "Classify the dermatology image.\n\n/no_think"


def test_legacy_sft_module_is_only_a_compatible_materializer_facade() -> None:
    from project.pipeline import materialize_sft
    from project.pipeline import sft as legacy

    assert legacy.MaterializedSFTRow is materialize_sft.MaterializedSFTRow
    assert (
        legacy.materialize_multitask_rows is materialize_sft.materialize_multitask_rows
    )
    assert legacy.SCHEMA_VERSION == materialize_sft.SCHEMA_VERSION


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "brown").save(buffer, format="PNG")
    return buffer.getvalue()


def _source(*, image: bytes | None = None) -> MaterializationSource:
    encoded = image or _png_bytes()
    return MaterializationSource(
        sample_id="s001",
        image={"bytes": encoded, "path": None},
        source_ref="hf://datasets/test/revision/diagnosis/sft_train/s001",
        split="sft_train",
        leakage_group_id="group-001",
        disease_id="melanoma",
        gold_diagnosis="melanoma",
        source_dataset="test-source",
        source_sample_id="source-001",
        image_sha256=hashlib.sha256(encoded).hexdigest(),
        diagnosis_prompt=_DIAGNOSIS_PROMPT,
        diagnosis_prompt_sha256=hashlib.sha256(
            _DIAGNOSIS_PROMPT.encode("utf-8")
        ).hexdigest(),
    )


def _stage_a(
    source: MaterializationSource,
    payload: dict[str, object] | None = None,
) -> StageAFileRow:
    return StageAFileRow(
        sample_id=source.sample_id,
        status=RecordStatus.OK,
        morphology=parse_stage_a(payload or STAGE_A_PAYLOAD),
        teacher="test-teacher",
        image_path=source.source_ref,
    )


def _stage_b(
    source: MaterializationSource,
    payload: dict[str, object] | None = None,
) -> StageBFileRow:
    return StageBFileRow(
        sample_id=source.sample_id,
        status=RecordStatus.OK,
        reasoning=parse_stage_b(payload or STAGE_B_PAYLOAD),
        teacher="test-teacher",
        gold_diagnosis=source.gold_diagnosis,
        stage_a_sample_id=source.sample_id,
        image_path=source.source_ref,
    )


def test_evaluable_sample_expands_to_four_non_contradictory_tasks() -> None:
    source = _source()
    result = materialize_multitask_rows(
        [source],
        [_stage_a(source)],
        [_stage_b(source)],
    )

    rows = {row.task: row for row in result.rows}
    assert set(rows) == {
        SFTTask.DIAGNOSIS,
        SFTTask.MORPHOLOGY,
        SFTTask.CAPTION,
        SFTTask.GROUNDED_DIFFERENTIAL,
    }
    assert result.task_counts == {
        "caption": 1,
        "diagnosis": 1,
        "grounded_differential": 1,
        "morphology": 1,
    }
    assert rows[SFTTask.DIAGNOSIS].prompt == _DIAGNOSIS_PROMPT
    assert rows[SFTTask.DIAGNOSIS].target_text == "melanoma"
    assert rows[SFTTask.MORPHOLOGY].prompt == MORPHOLOGY_PROMPT
    morphology = json.loads(rows[SFTTask.MORPHOLOGY].target_text)
    assert "clinical_caption" not in morphology
    assert "melanoma" not in rows[SFTTask.MORPHOLOGY].target_text.casefold()
    assert rows[SFTTask.CAPTION].prompt == CAPTION_PROMPT
    assert rows[SFTTask.CAPTION].target_text == STAGE_A_PAYLOAD["clinical_caption"]
    assert (
        rows[SFTTask.GROUNDED_DIFFERENTIAL].target_text
        == STAGE_B_PAYLOAD["clinical_reasoning"]
    )
    assert rows[SFTTask.GROUNDED_DIFFERENTIAL].prompt == CLINICAL_ASSESSMENT_PROMPT
    assert len({row.row_id for row in result.rows}) == 4
    for row in result.rows:
        assert row.messages[0]["role"] == "user"
        assert row.messages[1]["role"] == "assistant"


def test_non_evaluable_sample_teaches_request_new_image_instead() -> None:
    source = _source()
    stage_a_payload = deepcopy(STAGE_A_PAYLOAD)
    assessment = cast(dict[str, object], stage_a_payload["image_assessment"])
    assessment["is_evaluable"] = False
    assessment["quality_defects"] = ["blur"]
    stage_a_payload["observations"] = []
    stage_a_payload["clinical_caption"] = (
        "Severe blur prevents visual assessment of the lesion and its margins."
    )
    stage_b_payload = deepcopy(STAGE_B_PAYLOAD)
    stage_b_payload.update(
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

    result = materialize_multitask_rows(
        [source],
        [_stage_a(source, stage_a_payload)],
        [_stage_b(source, stage_b_payload)],
    )

    tasks = {row.task for row in result.rows}
    assert SFTTask.REQUEST_NEW_IMAGE in tasks
    assert SFTTask.GROUNDED_DIFFERENTIAL not in tasks
    request = next(row for row in result.rows if row.task is SFTTask.REQUEST_NEW_IMAGE)
    assert "melanoma" not in request.target_text.casefold()
    assert request.prompt == CLINICAL_ASSESSMENT_PROMPT


def test_teacher_failure_preserves_human_diagnosis_and_stage_a_targets() -> None:
    source = _source()
    failed_b = StageBFileRow(
        sample_id=source.sample_id,
        status=RecordStatus.ERROR,
        error="provider_error",
        teacher="test-teacher",
        gold_diagnosis=source.gold_diagnosis,
        stage_a_sample_id=source.sample_id,
        image_path=source.source_ref,
    )

    result = materialize_multitask_rows(
        [source],
        [_stage_a(source)],
        [failed_b],
    )

    assert {row.task for row in result.rows} == {
        SFTTask.DIAGNOSIS,
        SFTTask.MORPHOLOGY,
        SFTTask.CAPTION,
    }
    assert result.source_ids_without_stage_b == ("s001",)
    assert result.stage_b_status_counts == {
        "error": 1,
        "missing_attempt": 0,
        "not_eligible_stage_a": 0,
        "ok": 0,
        "rejected": 0,
    }
    assert len(result.stage_b_error_attempts) == 1
    assert result.stage_b_error_attempts[0].error == "provider_error"


def test_rejected_stage_b_is_audited_and_excluded_from_training() -> None:
    source = _source()
    rejected_b = StageBFileRow(
        sample_id=source.sample_id,
        status=RecordStatus.REJECTED,
        reasoning=parse_stage_b(STAGE_B_PAYLOAD),
        reasons=("anchor_not_discriminative",),
        teacher="test-teacher",
        gold_diagnosis=source.gold_diagnosis,
        stage_a_sample_id=source.sample_id,
        image_path=source.source_ref,
    )

    result = materialize_multitask_rows(
        [source],
        [_stage_a(source)],
        [rejected_b],
    )

    assert {row.task for row in result.rows} == {
        SFTTask.DIAGNOSIS,
        SFTTask.MORPHOLOGY,
        SFTTask.CAPTION,
    }
    assert result.stage_b_status_counts["rejected"] == 1
    assert result.source_ids_without_stage_b == ("s001",)
    assert result.stage_b_rejected_attempts[0].reasons == ("anchor_not_discriminative",)


def test_missing_generation_attempts_fail_closed_unless_partial() -> None:
    source = _source()
    with pytest.raises(ValueError, match="Generation coverage is incomplete"):
        materialize_multitask_rows([source], [], [])

    result = materialize_multitask_rows(
        [source],
        [],
        [],
        require_complete=False,
    )
    assert [row.task for row in result.rows] == [SFTTask.DIAGNOSIS]
    assert result.source_ids_without_stage_a == ("s001",)
    assert result.stage_b_status_counts["not_eligible_stage_a"] == 1
    assert result.stage_b_missing_attempt_ids == ()


def test_stage_a_only_sample_is_tracked_as_missing_stage_b_attempt() -> None:
    source = _source()

    result = materialize_multitask_rows(
        [source],
        [_stage_a(source)],
        [],
        require_complete=False,
    )

    assert {row.task for row in result.rows} == {
        SFTTask.DIAGNOSIS,
        SFTTask.MORPHOLOGY,
        SFTTask.CAPTION,
    }
    assert result.stage_b_status_counts["missing_attempt"] == 1
    assert result.stage_b_missing_attempt_ids == ("s001",)
    assert result.source_ids_without_stage_b == ("s001",)


def test_stage_a_gold_leak_is_rejected() -> None:
    source = _source()
    payload = deepcopy(STAGE_A_PAYLOAD)
    payload["clinical_caption"] = (
        "A melanoma appears as an asymmetric pigmented lesion in this image."
    )

    with pytest.raises(ValueError, match="Stage A caption leaks gold"):
        materialize_multitask_rows(
            [source],
            [_stage_a(source, payload)],
            [_stage_b(source)],
        )


def test_hub_source_adapter_rejects_image_hash_drift() -> None:
    source = _source()
    raw: dict[str, object] = {
        "sample_id": source.sample_id,
        "split": source.split,
        "image": source.image,
        "image_sha256": "0" * 64,
        "leakage_group_id": source.leakage_group_id,
        "disease_id": source.disease_id,
        "gold_diagnosis": source.gold_diagnosis,
        "source_dataset": source.source_dataset,
        "source_sample_id": source.source_sample_id,
        "prompt": source.diagnosis_prompt,
        "prompt_sha256": source.diagnosis_prompt_sha256,
    }

    with pytest.raises(ValueError, match="image SHA-256 differs"):
        source_from_hub_row(
            raw,
            repo_id="test",
            revision="revision",
            config="diagnosis",
            split="sft_train",
        )


def test_release_writes_image_aware_parquet_and_integrity_manifest(
    tmp_path: Path,
) -> None:
    datasets = pytest.importorskip("datasets")
    source = _source()
    result = materialize_multitask_rows(
        [source],
        [_stage_a(source)],
        [_stage_b(source)],
    )
    output = tmp_path / "sft_train.parquet"

    manifest = write_multitask_release(result, output)

    manifest_path = output.with_suffix(".manifest.json")
    assert output.is_file()
    assert manifest_path.is_file()
    assert manifest["rows"] == 4
    assert manifest["stage_b_coverage"]["ok"] == 1
    assert manifest["stage_b_rejected_attempts"] == []
    assert manifest["stage_b_error_attempts"] == []
    assert manifest["stage_b_missing_attempt_ids"] == []
    assert manifest["stage_b_duplicate_attempt_counts"] == {}
    assert manifest["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    loaded = datasets.Dataset.from_parquet(str(output))
    assert loaded.num_rows == 4
    assert set(loaded["task"]) == {
        "diagnosis",
        "morphology",
        "caption",
        "grounded_differential",
    }
    with pytest.raises(FileExistsError):
        write_multitask_release(result, output)
