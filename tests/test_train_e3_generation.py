"""Offline tests for the fail-closed E3 two-stage teacher runner."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from PIL import Image

from src.inference.base import (
    InferenceBackend,
    InferenceRequest,
    InferenceResult,
    InferenceSafetyRefusal,
    InferenceTransportError,
    ReasoningTrace,
    TokenUsage,
)
from src.train.domain import Taxonomy, TaxonomyClass
from src.train.e3 import (
    E3Candidate,
    E3Selection,
    E3TeacherGenerationRunner,
    E3TeacherSample,
    StageATarget,
    load_e3_teacher_generation_config,
    load_selected_images,
    select_e3_samples,
)
from src.train.e3.generation_cli import _resolve_selection_limit
from src.train.e3.prompts import (
    load_stage_a_prompt,
    load_stage_b_prompt,
    render_stage_a_prompt,
    render_stage_b_prompt,
    stage_a_output_schema,
)
from src.train.e3.terminology import ImageModality

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/e3_teacher_generation_gpt_5_6_sol_medium.yaml"
GOLD_CONFIG = ROOT / (
    "configs/training/"
    "e3_teacher_generation_gpt_5_6_luna_high_gold_stage_b.yaml"
)


class _QueueBackend(InferenceBackend):
    def __init__(self, outcomes: list[dict[str, Any] | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[InferenceRequest] = []

    @property
    def model_id(self) -> str:
        return "gpt_5_6_sol"

    def complete(self, request: InferenceRequest) -> InferenceResult:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return InferenceResult(
            model_id=self.model_id,
            final_text=json.dumps(outcome),
            reasoning=ReasoningTrace(capture_mode="none"),
            usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            request_id=request.request_id,
            provider_response_id=f"response-{len(self.requests)}",
            finish_reason="completed",
            metadata={"provider_model": "gpt-5.6-sol"},
        )


def test_two_stage_runner_records_accepted_bundle_without_raw_prompts(
    tmp_path: Path,
) -> None:
    backend = _QueueBackend([_stage_a_payload(), _stage_b_payload()])
    output = tmp_path / "accepted"
    snapshot = _runner(output, backend).run()

    assert snapshot["status"] == "completed"
    assert snapshot["progress"]["samples_finished"] == 1
    assert snapshot["private_validation"]["leading_label_match_rate"] == 1.0
    assert snapshot["response_policy"]["REQUEST_CONTEXT"] == 1
    assert len(backend.requests) == 2
    assert backend.requests[0].generation == {
        "reasoning_effort": "medium",
        "max_output_tokens": 2400,
    }
    assert backend.requests[1].generation == {
        "reasoning_effort": "medium",
        "max_output_tokens": 3600,
    }
    assert "melanoma" not in backend.requests[0].user_prompt.casefold()
    assert "Closed taxonomy" in backend.requests[1].user_prompt
    assert "obs_1" in backend.requests[1].user_prompt

    bundles = _read_jsonl(output / "teacher_bundles.jsonl")
    assert len(bundles) == 1
    assert bundles[0]["teacher_targets"]["stage_a_status"] == "accepted"
    assert bundles[0]["teacher_targets"]["stage_b_status"] == "accepted"
    artifacts_text = (output / "stage_results.jsonl").read_text()
    assert "system_prompt" not in artifacts_text
    assert "user_prompt" not in artifacts_text
    manifest = json.loads((output / "campaign_manifest.json").read_text())
    campaign = manifest["campaign"]
    assert campaign["terminology_lexicon_id"] == (
        "e3_dermatology_terminology_v1"
    )
    assert len(campaign["terminology_resource_sha256"]) == 64
    assert len(campaign["stage_a_rendered_prompt_sha256"]) == 64


def test_stage_b_reviews_diagnostic_and_context_policy_independently(
    tmp_path: Path,
) -> None:
    stage_b = _stage_b_payload()
    assessment = stage_b["diagnostic_assessment"]
    assert isinstance(assessment, dict)
    differential = assessment["differential"]
    assert isinstance(differential, list)
    lead = differential[0]
    alternative = differential[1]
    assert isinstance(lead, dict)
    assert isinstance(alternative, dict)
    lead["disease_id"], alternative["disease_id"] = (
        alternative["disease_id"],
        lead["disease_id"],
    )

    output = tmp_path / "independent-stage-b-reviews"
    snapshot = _runner(
        output,
        _QueueBackend([_stage_a_payload(), stage_b]),
    ).run()

    artifact = _read_jsonl(output / "stage_results.jsonl")[1]
    assert artifact["review_status"] == "accepted"
    assert artifact["diagnostic_review_status"] == "rejected"
    assert artifact["diagnostic_rejection_reasons"] == [
        "leading_diagnosis_does_not_match_private_gold"
    ]
    assert artifact["context_policy_review_status"] == "accepted"
    assert artifact["context_policy_rejection_reasons"] == []
    bundle = _read_jsonl(output / "teacher_bundles.jsonl")[0]["teacher_targets"]
    assert bundle["stage_b_diagnostic_status"] == "rejected"
    assert bundle["stage_b_context_policy_status"] == "accepted"
    assert snapshot["materializable_rows"]["grounded_differential"] == 0
    assert snapshot["materializable_rows"]["context_policy"] == 1


def test_gold_conditioned_stage_b_receives_anchor_but_stage_a_does_not(
    tmp_path: Path,
) -> None:
    backend = _QueueBackend([_stage_a_payload(), _stage_b_payload()])
    output = tmp_path / "gold-conditioned"

    snapshot = _runner(output, backend, config_path=GOLD_CONFIG).run()

    assert snapshot["status"] == "completed"
    assert backend.requests[0].generation["reasoning_effort"] == "high"
    assert "melanoma" not in backend.requests[0].user_prompt.casefold()
    assert '"diagnosis":"melanoma"' in backend.requests[1].user_prompt
    assert '"disease_id":"D001"' in backend.requests[1].user_prompt
    assert snapshot["private_validation"]["gold_was_visible_to_teacher"] is True
    assert snapshot["private_validation"]["leading_label_metric_interpretation"] == (
        "anchor_compliance"
    )
    artifacts = _read_jsonl(output / "stage_results.jsonl")
    assert artifacts[0]["provenance"]["gold_visible_to_teacher"] is False
    assert artifacts[1]["provenance"]["gold_visible_to_teacher"] is True


def test_gold_conditioned_prompt_rejects_taxonomy_mismatch() -> None:
    config = load_e3_teacher_generation_config(GOLD_CONFIG, root=ROOT)
    resource = load_stage_b_prompt(config.path(config.prompts.stage_b))
    selection, _ = _selection_and_sample()
    stage_a = _stage_a_payload()

    with pytest.raises(ValueError, match="does not match the taxonomy"):
        render_stage_b_prompt(
            resource,
            taxonomy=selection.taxonomy,
            stage_a=StageATarget.model_validate(stage_a),
            gold_disease_id="D001",
            gold_diagnosis="melanocytic_nevus",
        )


def test_stage_a_v2_injects_frozen_diagnosis_free_terminology() -> None:
    config = load_e3_teacher_generation_config(CONFIG, root=ROOT)
    terminology = config.load_terminology()
    resource = load_stage_a_prompt(config.path(config.prompts.stage_a))

    rendered = render_stage_a_prompt(resource, terminology=terminology)
    assert resource.prompt_id == "e3_teacher_stage_a_answer_blind_terminology_v2"
    assert terminology.lexicon_id in rendered.user_prompt
    assert '"concept_id":"border.irregular"' in rendered.user_prompt
    assert "melanoma" not in rendered.user_prompt.casefold()
    assert "psoriasis" not in rendered.user_prompt.casefold()

    schema = stage_a_output_schema(terminology)
    concept_schema = schema["$defs"]["Observation"]["properties"]["concept_id"]
    assert concept_schema["enum"] == list(terminology.concept_ids)


def test_terminology_audit_is_fail_closed_for_label_and_modality() -> None:
    config = load_e3_teacher_generation_config(CONFIG, root=ROOT)
    terminology = config.load_terminology()

    assert terminology.audit_observation(
        concept_id="border.irregular",
        concept_label="irregular border",
        image_modality=ImageModality.CLINICAL_PHOTO,
    ) == ()
    assert terminology.audit_observation(
        concept_id="border.irregular",
        concept_label="jagged edge",
        image_modality=ImageModality.CLINICAL_PHOTO,
    ) == ("stage_a_terminology_label_mismatch",)
    assert terminology.audit_observation(
        concept_id="dermoscopy_element.dot",
        concept_label="dermoscopic dot",
        image_modality=ImageModality.UNKNOWN,
    ) == ("stage_a_dermoscopy_concept_requires_confirmed_modality",)


def test_stage_a_terminology_mismatch_is_rejected_before_stage_b(
    tmp_path: Path,
) -> None:
    payload = _stage_a_payload()
    observations = payload["observations"]
    assert isinstance(observations, list)
    first = observations[0]
    assert isinstance(first, dict)
    first["concept_label"] = "jagged edge"
    backend = _QueueBackend([payload])

    snapshot = _runner(tmp_path / "terminology-mismatch", backend).run()

    assert snapshot["status"] == "completed"
    assert len(backend.requests) == 1
    artifact = _read_jsonl(
        tmp_path / "terminology-mismatch" / "stage_results.jsonl"
    )[0]
    assert artifact["review_status"] == "rejected"
    assert artifact["rejection_reasons"] == [
        "stage_a_terminology_label_mismatch"
    ]


def test_invalid_schema_is_terminal_without_repair_or_retry(tmp_path: Path) -> None:
    backend = _QueueBackend([{}])
    output = tmp_path / "invalid"
    snapshot = _runner(output, backend).run()

    assert snapshot["status"] == "completed"
    assert snapshot["failures"] == {"invalid_schema": 1}
    assert len(backend.requests) == 1
    artifact = _read_jsonl(output / "stage_results.jsonl")[0]
    assert artifact["provenance"]["generation_status"] == "invalid_schema"
    assert artifact["review_status"] == "not_applicable"
    assert artifact["target"] is None


def test_quality_slice_stops_after_failed_first_case_gate(tmp_path: Path) -> None:
    backend = _QueueBackend([{}])
    output = tmp_path / "quality-gate"
    snapshot = _runner(output, backend, gate_first_case=True).run()

    assert snapshot["status"] == "failed"
    assert snapshot["error"] == "quality_gate_failed_no_retry"
    assert len(backend.requests) == 1


def test_provider_guardrail_has_separate_typed_safety_annotation(
    tmp_path: Path,
) -> None:
    refusal = InferenceSafetyRefusal(
        "blocked",
        details={
            "code": "content_filter",
            "content_filter": {"medical": {"filtered": True, "severity": "medium"}},
        },
    )
    backend = _QueueBackend([refusal])
    output = tmp_path / "refusal"
    snapshot = _runner(output, backend).run()

    assert snapshot["status"] == "completed"
    assert snapshot["failures"] == {"provider_safety_refusal": 1}
    assert len(backend.requests) == 1
    provenance = _read_jsonl(output / "stage_results.jsonl")[0]["provenance"]
    assert provenance["provider_error_code"] == "content_filter"
    assert provenance["safety_categories"] == [
        {"category": "medical", "filtered": True, "severity": "medium"}
    ]


def test_transport_error_stops_campaign_without_retry(tmp_path: Path) -> None:
    backend = _QueueBackend([InferenceTransportError("network unavailable")])
    output = tmp_path / "transport"
    snapshot = _runner(output, backend).run()

    assert snapshot["status"] == "failed"
    assert snapshot["error"] == "teacher_transport_failure_no_retry"
    assert snapshot["failures"] == {"transport_error": 1}
    assert len(backend.requests) == 1


def test_dataset_selection_is_deterministic_and_checks_selected_shards(
    tmp_path: Path,
) -> None:
    config_path, shard_path = _synthetic_generation_config(tmp_path)
    config = load_e3_teacher_generation_config(config_path, root=tmp_path)

    first = select_e3_samples(config, limit=4)
    second = select_e3_samples(config, limit=4)
    assert first.selection_sha256 == second.selection_sha256
    assert [item.sample_id for item in first.candidates] == [
        item.sample_id for item in second.candidates
    ]
    assert [item.disease_id for item in first.candidates] == [
        "D001",
        "D002",
        "D001",
        "D002",
    ]
    loaded = load_selected_images(
        first,
        verify_shard_sha256=True,
        verify_image_sha256=True,
    )
    assert len(loaded) == 4
    assert all(item.image_width == 24 and item.image_height == 16 for item in loaded)

    with shard_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="shard SHA-256 mismatch"):
        load_selected_images(
            first,
            verify_shard_sha256=True,
            verify_image_sha256=True,
        )


def test_dry_run_limit_cannot_change_smoke_or_pilot_size() -> None:
    assert _resolve_selection_limit(
        mode="dry-run", requested_limit=25, pilot_samples=100
    ) == 25
    assert _resolve_selection_limit(
        mode="dry-run", requested_limit=None, pilot_samples=100
    ) == 100
    assert _resolve_selection_limit(
        mode="smoke", requested_limit=None, pilot_samples=100
    ) == 1
    assert _resolve_selection_limit(
        mode="pilot", requested_limit=None, pilot_samples=100
    ) == 100
    assert _resolve_selection_limit(
        mode="quality", requested_limit=None, pilot_samples=25
    ) == 25

    with pytest.raises(ValueError, match="only for dry-run"):
        _resolve_selection_limit(
            mode="pilot", requested_limit=25, pilot_samples=100
        )
    with pytest.raises(ValueError, match="between 1 and 100"):
        _resolve_selection_limit(
            mode="dry-run", requested_limit=0, pilot_samples=100
        )


def _runner(
    output: Path,
    backend: InferenceBackend,
    *,
    gate_first_case: bool = False,
    config_path: Path = CONFIG,
) -> E3TeacherGenerationRunner:
    config = load_e3_teacher_generation_config(config_path, root=ROOT)
    selection, sample = _selection_and_sample()
    return E3TeacherGenerationRunner(
        config=config,
        selection=selection,
        samples=(sample,),
        backend=backend,
        stage_a_prompt=load_stage_a_prompt(config.path(config.prompts.stage_a)),
        stage_b_prompt=load_stage_b_prompt(config.path(config.prompts.stage_b)),
        output_directory=output,
        campaign_id="e3-generation-unit-test",
        gate_first_case=gate_first_case,
    )


def _selection_and_sample() -> tuple[E3Selection, E3TeacherSample]:
    image_buffer = io.BytesIO()
    Image.new("RGB", (24, 16), "brown").save(image_buffer, format="JPEG")
    image_bytes = image_buffer.getvalue()
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    candidate = E3Candidate(
        sample_id="sample-001",
        leakage_group_id="group-001",
        disease_id="D001",
        gold_diagnosis="melanoma",
        split="sft_train",
        image_sha256=image_sha,
        shard_path=ROOT / "unused-test-shard.parquet",
        shard_manifest_sha256="a" * 64,
        row_index=0,
    )
    taxonomy = Taxonomy(
        taxonomy_id="unit-taxonomy",
        classes=(
            TaxonomyClass("D001", "melanoma"),
            TaxonomyClass("D002", "melanocytic_nevus"),
        ),
    )
    selection = E3Selection(
        taxonomy=taxonomy,
        candidates=(candidate,),
        selection_sha256="b" * 64,
        release_id="unit-release",
        release_manifest_sha256="c" * 64,
    )
    sample = E3TeacherSample(
        candidate=candidate,
        image_bytes=image_bytes,
        image_mime_type="image/jpeg",
        image_width=24,
        image_height=16,
    )
    return selection, sample


def _stage_a_payload() -> dict[str, object]:
    return {
        "image_assessment": {
            "is_evaluable": True,
            "image_modality": "clinical_photo",
            "views_available": ["close_up"],
            "quality_defects": [],
            "has_anatomic_overview": False,
            "has_scale": False,
            "has_lateral_profile": False,
            "distribution_assessability": "within_frame_only",
            "color_reliability": "uncertain",
        },
        "dominant_visual_pattern": "pigmented asymmetric lesion",
        "observations": [
            {
                "id": "obs_1",
                "concept_id": "border.irregular",
                "concept_label": "irregular border",
                "concept_detail": None,
                "status": "present",
                "provenance": "visible_image",
                "scope": "index_lesion",
                "confidence": "moderate",
                "evidence_region": "lesion_periphery",
            },
            {
                "id": "obs_2",
                "concept_id": "color.multicolored",
                "concept_label": "multicolored",
                "concept_detail": "visible color variation",
                "status": "present",
                "provenance": "visible_image",
                "scope": "index_lesion",
                "confidence": "moderate",
                "evidence_region": "whole_lesion",
            },
        ],
        "not_assessable_features": ["recent_evolution"],
        "clinical_caption": (
            "The image shows a pigmented asymmetric lesion with an irregular "
            "border and visible color variation."
        ),
    }


def _stage_b_payload() -> dict[str, object]:
    return {
        "stage_b_corrections": [],
        "diagnostic_assessment": {
            "differential": [
                {
                    "rank": 1,
                    "disease_id": "D001",
                    "supporting_observation_ids": ["obs_1", "obs_2"],
                    "contradicting_observation_ids": [],
                    "missing_discriminators": [
                        {
                            "feature": "recent_evolution",
                            "required_source": "history",
                        }
                    ],
                    "diagnostic_confidence": "moderate",
                    "clinical_risk_if_missed": "high",
                },
                {
                    "rank": 2,
                    "disease_id": "D002",
                    "supporting_observation_ids": ["obs_2"],
                    "contradicting_observation_ids": ["obs_1"],
                    "missing_discriminators": [],
                    "diagnostic_confidence": "low",
                    "clinical_risk_if_missed": "low",
                },
            ],
            "concise_clinical_rationale": (
                "The irregular border and color variation support an atypical "
                "pigmented lesion, while recent evolution is not visible."
            ),
        },
        "context_decision": {
            "information_sufficiency": "insufficient",
            "response_policy": "REQUEST_CONTEXT",
            "decision_rationale": (
                "Image-only evidence leaves recent evolution unresolved between "
                "the leading diagnoses."
            ),
            "requests": [
                {
                    "request_id": "ctx_1",
                    "priority": 1,
                    "context_type": "lesion_evolution",
                    "required_source": "clinical_history",
                    "question": (
                        "Has the lesion changed in size, color, or shape recently?"
                    ),
                    "discriminates_between": ["D001", "D002"],
                    "rationale": (
                        "Recent evolution would help distinguish these competing "
                        "pigmented diagnoses."
                    ),
                }
            ],
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _synthetic_generation_config(tmp_path: Path) -> tuple[Path, Path]:
    release_root = tmp_path / "release"
    shard_path = release_root / "data/diagnosis/sft_train-00000.parquet"
    shard_path.parent.mkdir(parents=True)
    rows = []
    for index, disease_id in enumerate(("D001", "D002", "D001", "D002")):
        image_buffer = io.BytesIO()
        Image.new("RGB", (24, 16), (index * 30, 20, 10)).save(
            image_buffer,
            format="PNG",
        )
        image_bytes = image_buffer.getvalue()
        rows.append(
            {
                "image": {"bytes": image_bytes, "path": f"image-{index}.png"},
                "sample_id": f"sample-{index}",
                "leakage_group_id": f"group-{index}",
                "disease_id": disease_id,
                "gold_diagnosis": (
                    "melanoma" if disease_id == "D001" else "melanocytic_nevus"
                ),
                "split": "sft_train",
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), shard_path)
    shard_sha = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    manifest_path = release_root / "metadata/release.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "release_id": "synthetic-e3-release",
                "shards": [
                    {
                        "bytes": shard_path.stat().st_size,
                        "config": "diagnosis",
                        "path": "data/diagnosis/sft_train-00000.parquet",
                        "rows": 4,
                        "sha256": shard_sha,
                        "split": "sft_train",
                    }
                ],
            }
        )
    )
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(
        json.dumps(
            {
                "taxonomy_id": "synthetic-taxonomy",
                "classes": [
                    {"disease_id": "D001", "label": "melanoma"},
                    {"disease_id": "D002", "label": "melanocytic_nevus"},
                ],
            }
        )
    )
    config_path = tmp_path / "e3.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "campaign": {"id": "synthetic"},
                "model": {
                    "config": "model.yaml",
                    "required_model_id": "gpt_5_6_sol",
                    "reasoning_effort": "medium",
                    "structured_output_mode": "json_schema",
                },
                "dataset": {
                    "release_root": "release",
                    "release_manifest": "release/metadata/release.json",
                    "taxonomy": "taxonomy.json",
                    "config": "diagnosis",
                    "splits": ["sft_train"],
                    "selection": {
                        "strategy": "stratified_round_robin",
                        "seed": 42,
                        "pilot_samples": 4,
                        "max_per_leakage_group": 1,
                    },
                },
                "prompts": {"stage_a": "a.yaml", "stage_b": "b.yaml"},
                "terminology": {
                    "resource": "terminology.yaml",
                    "required_lexicon_id": "synthetic_terminology_v1",
                },
                "generation": {
                    "stage_a_max_output_tokens": 100,
                    "stage_b_max_output_tokens": 100,
                    "sequential": True,
                    "retries": 0,
                    "stop_on_transport_error": True,
                },
                "integrity": {
                    "verify_selected_shard_sha256": True,
                    "verify_image_sha256": True,
                },
            }
        )
    )
    return config_path, shard_path
