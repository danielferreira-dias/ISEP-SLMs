"""Prepare, submit, monitor, and ingest Vertex Batch Stage B jobs.

Stage B is a frozen protocol over three joined inputs: the original image, one
accepted Stage A morphology record, and the private gold diagnosis.  This
module changes only the transport from synchronous Vertex calls to Vertex
Batch.  It deliberately reuses the same prompt, schema, image preprocessing,
provenance, parser, and deterministic clinical validation gate.

Unlike Stage A, the prepared request and item files contain private gold
labels.  Upload therefore requires an explicit acknowledgement flag; merely
preparing a local batch never authorizes or performs an external transfer.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from project.pipeline.generate import CampaignFailure, ExampleCohort, load_hub_cohort
from project.pipeline.stage_a_batch import (
    ITEMS_NAME,
    MANIFEST_NAME,
    REQUESTS_NAME,
    _artifact_manifest,
    _display_name,
    _fsync_path,
    _image_filename,
    _iter_jsonl_objects,
    _job_payload,
    _json_schema_to_vertex_openapi,
    _load_items,
    _load_manifest,
    _mapping,
    _normalize_gcs_prefix,
    _pricing_manifest,
    _provenance_signature,
    _required_str,
    _result_image_uri,
    _sha256_json,
    _sha256_path,
    _teacher_response_from_batch,
    _upload_manifest_if_possible,
    _validate_batch_request_file,
    _vertex_client,
    _write_jsonl,
    _write_manifest,
    _write_named_manifest,
)
from project.pipeline.stage_a_batch import (
    download_results as _download_results,
)
from project.pipeline.stage_a_batch import (
    refresh_status as _refresh_status,
)
from project.pipeline.stage_a_batch import (
    upload_batch as _upload_batch,
)
from project.teacher.provenance import generation_provenance
from project.teacher.schemas import (
    ImagePreprocessingInfo,
    RecordStatus,
    StageAFileRow,
    StageBFileRow,
    UsageInfo,
    parse_stage_b,
)
from project.teacher.teacher import (
    PROJECT_ROOT,
    TeacherModel,
    TeacherProvider,
    VertexAPI,
)
from project.teacher.utils.images import prepare_pil_image
from project.teacher.utils.jsonl import (
    append_jsonl,
    completed_stage_b_ids,
    index_ok_stage_a,
    load_stage_a_rows,
    load_stage_b_rows,
)
from project.teacher.validate import validate_stage_b
from project.teacher.vertex import jpeg_bytes_from_data_url

LOGGER = logging.getLogger("project.pipeline.stage_b_batch")

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "teacher_configs" / "gemini_3_7_flash_vertex.yaml"
)
DEFAULT_STAGE_A = (
    PROJECT_ROOT
    / "data"
    / "morphology"
    / "frozen"
    / "e3_stage_a_v1_20260822"
    / "stage_a.jsonl"
)
DEFAULT_STAGE_B = PROJECT_ROOT / "data" / "reasoning" / "stage_b.jsonl"
OVERLAP_NORMALIZATION_DATA_NAME = "overlap_normalization.jsonl"
OVERLAP_NORMALIZATION_MANIFEST_NAME = "overlap_normalization_manifest.json"
OVERLAP_NORMALIZATION_VERSION = "stage_b_overlap_normalization_v1"
STAGE_B_FREEZE_DATA_NAME = "stage_b.jsonl"
STAGE_B_REJECTED_DATA_NAME = "rejected.jsonl"
STAGE_B_FREEZE_MANIFEST_NAME = "freeze_manifest.json"


def prepare_batch(
    *,
    teacher: TeacherModel,
    cohort: ExampleCohort,
    stage_a_path: Path,
    stage_b_output: Path,
    work_dir: Path,
    gcs_prefix: str,
    dataset_config: str = "diagnosis",
    dataset_split: str = "sft_train",
    pending_limit: int | None = None,
) -> dict[str, object]:
    """Prepare one local Stage B batch without uploading private inputs."""
    api = _require_vertex_teacher(teacher)
    prefix = _normalize_gcs_prefix(gcs_prefix)
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(f"Batch work directory is not empty: {work_dir}")
    if pending_limit is not None and pending_limit <= 0:
        raise ValueError("Pending batch limit must be greater than zero")

    selected = cohort.sample_ids
    selected_set = set(selected)
    stage_a = _compatible_stage_a_index(
        stage_a_path,
        teacher=teacher,
        selected=selected_set,
    )
    completed = _compatible_stage_b_ids(
        stage_b_output,
        teacher=teacher,
        selected=selected_set,
    )
    all_pending = tuple(
        sample_id for sample_id in selected if sample_id not in completed
    )
    if not all_pending:
        raise CampaignFailure(
            "Stage B already has a compatible terminal row for every ID"
        )
    pending = all_pending if pending_limit is None else all_pending[:pending_limit]

    work_dir.mkdir(parents=True, exist_ok=True)
    images_dir = work_dir / "images"
    images_dir.mkdir()
    requests_path = work_dir / REQUESTS_NAME
    items_path = work_dir / ITEMS_NAME
    image_bytes = 0
    started = datetime.now(UTC)

    with (
        requests_path.open("x", encoding="utf-8") as requests_handle,
        items_path.open("x", encoding="utf-8") as items_handle,
    ):
        for index, example in enumerate(cohort.iter_selected(pending), start=1):
            stage_a_row = stage_a[example.sample_id]
            morphology = stage_a_row.morphology
            if morphology is None:  # defensive; accepted Stage A forbids this
                raise CampaignFailure(
                    f"Accepted Stage A row has no morphology: {example.sample_id}"
                )
            prepared = prepare_pil_image(example.image)
            _require_same_stage_a_image(stage_a_row, prepared.info)
            encoded = jpeg_bytes_from_data_url(prepared.data_url)
            filename = _image_filename(index, example.sample_id)
            image_path = images_dir / filename
            image_path.write_bytes(encoded)
            image_bytes += len(encoded)
            image_uri = f"{prefix}/images/{filename}"

            request = build_batch_request(
                teacher=teacher,
                image_gcs_uri=image_uri,
                stage_a=stage_a_row,
                gold_diagnosis=example.gold_diagnosis,
            )
            item = {
                "sample_id": example.sample_id,
                "source_ref": example.source_ref,
                "image_filename": filename,
                "image_gcs_uri": image_uri,
                "image_preprocessing": prepared.info.model_dump(mode="json"),
                "gold_diagnosis": example.gold_diagnosis,
                "stage_a_sample_id": stage_a_row.sample_id,
                "stage_a_attempt_id": (
                    stage_a_row.provenance.attempt_id
                    if stage_a_row.provenance is not None
                    else None
                ),
                "stage_a_morphology_sha256": _sha256_json(
                    morphology.model_dump(mode="json")
                ),
            }
            _write_jsonl(requests_handle, request)
            _write_jsonl(items_handle, item)
            if index == 1 or index % 100 == 0 or index == len(pending):
                LOGGER.info(
                    "Prepared Batch B %d/%d (%.2f%%)",
                    index,
                    len(pending),
                    100 * index / len(pending),
                )

    _fsync_path(requests_path)
    _fsync_path(items_path)
    _validate_batch_request_file(requests_path, expected_count=len(pending))
    provenance = generation_provenance(teacher, "B")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "campaign_id": work_dir.name,
        "stage": "B",
        "transport": "vertex_batch_gcs",
        "status": "prepared",
        "created_at": started.isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "work_dir": str(work_dir.resolve()),
        "stage_a_input": {
            "path": str(stage_a_path.resolve()),
            "rows": len(stage_a),
            "bytes": stage_a_path.stat().st_size,
            "sha256": _sha256_path(stage_a_path),
            "accepted_ok_only": True,
        },
        "stage_b_output": str(stage_b_output.resolve()),
        "stage_b_output_sha256_before": (
            _sha256_path(stage_b_output) if stage_b_output.is_file() else None
        ),
        "source_sample_count": len(selected),
        "existing_compatible_terminal_count": len(completed),
        "pending_request_count": len(pending),
        "pending_source_count_before_limit": len(all_pending),
        "pending_limit": pending_limit,
        "gold_sent_to_teacher": True,
        "contains_private_gold": True,
        "submission_requires_explicit_private_gold_authorization": True,
        "external_transfer_authorized": False,
        "dataset": {
            "repo_id": "danielfdias98/ISEPDistillDataset",
            "revision": "b215f0474e4931b5951da768e79a0d579d26919d",
            "config": dataset_config,
            "split": dataset_split,
        },
        "teacher": {
            "provider": teacher.provider.value,
            "name": teacher.name,
            "model": teacher.model.id,
            "project": api.project,
            "location": api.location,
            "seed": teacher.generation.seed,
            "max_output_tokens": teacher.generation.max_tokens,
            "reasoning_effort": teacher.reasoning.effort,
            "reasoning_excluded": teacher.reasoning.exclude,
            "prompt_version": teacher.stage("B").prompt.version,
            "prompt_sha256": provenance.prompt_sha256,
            "schema_sha256": provenance.schema_sha256,
            "protocol_frozen": True,
            "batch_schema_transport": "responseSchema_openapi_nullable",
            "batch_schema_sha256": _sha256_json(
                _json_schema_to_vertex_openapi(
                    cast(
                        dict[str, object],
                        teacher.vertex_generate_config("B")["response_json_schema"],
                    )
                )
            ),
            "config_path": str(teacher.config_path),
            "batch_pricing": _pricing_manifest(teacher),
        },
        "local": {
            "requests_path": str(requests_path.resolve()),
            "requests_bytes": requests_path.stat().st_size,
            "requests_sha256": _sha256_path(requests_path),
            "items_path": str(items_path.resolve()),
            "items_bytes": items_path.stat().st_size,
            "items_sha256": _sha256_path(items_path),
            "images_dir": str(images_dir.resolve()),
            "image_count": len(pending),
            "image_bytes": image_bytes,
        },
        "gcs": {
            "prefix": prefix,
            "images_prefix": f"{prefix}/images",
            "requests_uri": f"{prefix}/input/{REQUESTS_NAME}",
            "items_uri": f"{prefix}/input/{ITEMS_NAME}",
            "manifest_uri": f"{prefix}/input/{MANIFEST_NAME}",
            "output_prefix": f"{prefix}/output",
        },
        "batch_job": None,
        "ingestion": None,
    }
    _write_manifest(work_dir, manifest)
    return manifest


def build_batch_request(
    *,
    teacher: TeacherModel,
    image_gcs_uri: str,
    stage_a: StageAFileRow,
    gold_diagnosis: str,
) -> dict[str, object]:
    """Build one Vertex request using the frozen Stage B prompt and schema."""
    _require_vertex_teacher(teacher)
    if not image_gcs_uri.startswith("gs://"):
        raise ValueError("Batch image URI must use gs://")
    if stage_a.status is not RecordStatus.OK or stage_a.morphology is None:
        raise ValueError("Stage B Batch requires an accepted Stage A row")
    gold = gold_diagnosis.strip()
    if not gold:
        raise ValueError("Stage B Batch requires a non-empty gold diagnosis")

    stage = teacher.stage("B")
    generation = teacher.vertex_generate_config("B")
    thinking = cast(dict[str, object], generation["thinking_config"])
    response_schema = _json_schema_to_vertex_openapi(
        cast(dict[str, object], generation["response_json_schema"])
    )
    user_text = stage.prompt.render_user(
        gold_diagnosis=gold,
        stage_a_json=stage_a.morphology.model_dump_json(),
    )
    return {
        "request": {
            "systemInstruction": {"parts": [{"text": stage.prompt.system}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_text},
                        {
                            "fileData": {
                                "fileUri": image_gcs_uri,
                                "mimeType": "image/jpeg",
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": generation["max_output_tokens"],
                "seed": generation["seed"],
                "responseMimeType": generation["response_mime_type"],
                "responseSchema": response_schema,
                "thinkingConfig": {
                    "thinkingLevel": thinking["thinking_level"],
                    "includeThoughts": thinking["include_thoughts"],
                },
            },
        }
    }


def upload_batch(
    work_dir: Path,
    *,
    authorize_private_gold_upload: bool,
    resume: bool = False,
) -> dict[str, object]:
    """Upload images and gold-bearing inputs only after explicit acknowledgement."""
    if not authorize_private_gold_upload:
        raise PermissionError(
            "Stage B requests contain private gold labels; pass "
            "--authorize-private-gold-upload only after explicit authorization"
        )
    manifest = _load_manifest(work_dir)
    if (
        manifest.get("stage") != "B"
        or manifest.get("contains_private_gold") is not True
    ):
        raise CampaignFailure("This is not a private-gold Stage B batch manifest")
    manifest["external_transfer_authorized"] = True
    manifest["external_transfer_authorized_at"] = datetime.now(UTC).isoformat()
    _write_manifest(work_dir, manifest)
    return _upload_batch(work_dir, resume=resume)


def submit_batch(
    *,
    teacher: TeacherModel,
    work_dir: Path,
    client: object | None = None,
) -> dict[str, object]:
    """Submit exactly one uploaded Stage B batch and record the job resource."""
    _require_vertex_teacher(teacher)
    manifest = _load_manifest(work_dir)
    if manifest.get("stage") != "B":
        raise CampaignFailure("Batch manifest is not Stage B")
    if manifest.get("external_transfer_authorized") is not True:
        raise PermissionError("Stage B private-gold transfer was not authorized")
    if manifest.get("status") != "uploaded":
        raise CampaignFailure("Only an uploaded, unsubmitted batch can be submitted")
    if manifest.get("batch_job") is not None:
        raise CampaignFailure("Batch manifest already contains a submitted job")
    gcs = _mapping(manifest.get("gcs"), "manifest.gcs")
    sdk = cast(Any, client) if client is not None else _vertex_client(teacher)
    batches = getattr(sdk, "batches", None)
    if batches is None:
        raise TypeError("Vertex client has no batches API")

    from google.genai import types

    generic_name = _display_name(str(manifest["campaign_id"]))
    display_name = generic_name.replace("stage-a", "stage-b", 1)
    job = batches.create(
        model=teacher.model.id,
        src=_required_str(gcs.get("requests_uri"), "gcs.requests_uri"),
        config=types.CreateBatchJobConfig(
            display_name=display_name,
            dest=_required_str(gcs.get("output_prefix"), "gcs.output_prefix"),
        ),
    )
    job_payload = _job_payload(job)
    name = job_payload.get("name")
    if not isinstance(name, str) or not name:
        raise CampaignFailure("Vertex returned a batch job without a resource name")
    manifest["status"] = "submitted"
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    manifest["batch_job"] = job_payload
    _write_manifest(work_dir, manifest)
    _upload_manifest_if_possible(work_dir, manifest)
    return manifest


def refresh_status(
    *,
    teacher: TeacherModel,
    work_dir: Path,
    client: object | None = None,
) -> dict[str, object]:
    """Refresh the Stage B job using the shared Vertex Batch state machine."""
    return _refresh_status(teacher=teacher, work_dir=work_dir, client=client)


def download_results(work_dir: Path) -> Path:
    """Download Stage B results without deleting any remote object."""
    return _download_results(work_dir)


def ingest_results(
    *,
    teacher: TeacherModel,
    work_dir: Path,
    stage_a_path: Path,
    stage_b_output: Path,
    results_dir: Path | None = None,
    resume_ingestion: bool = False,
) -> dict[str, int]:
    """Parse, clinically validate, and append auditable Stage B attempts."""
    manifest = _load_manifest(work_dir)
    previous = manifest.get("ingestion")
    if previous is None:
        if manifest.get("status") not in {
            "batch_succeeded",
            "batch_completed_with_failures",
            "batch_failed",
        }:
            raise CampaignFailure("Refresh a terminal batch state before ingestion")
    else:
        if not resume_ingestion:
            raise CampaignFailure("This batch manifest has already been ingested")
        if manifest.get("status") not in {
            "ingested_incomplete",
            "ingested_with_retryable_errors",
        }:
            raise CampaignFailure(
                "Only an incomplete/error Stage B ingestion can be resumed"
            )

    expected_stage_a = _mapping(manifest.get("stage_a_input"), "stage_a_input")
    expected_stage_a_sha = _required_str(
        expected_stage_a.get("sha256"), "stage_a_input.sha256"
    )
    if _sha256_path(stage_a_path) != expected_stage_a_sha:
        raise CampaignFailure("Stage A input hash differs from the prepared batch")
    stage_a = index_ok_stage_a(load_stage_a_rows(stage_a_path))

    local = _mapping(manifest.get("local"), "manifest.local")
    items = _load_items(
        Path(_required_str(local.get("items_path"), "manifest.local.items_path"))
    )
    by_uri = {item["image_gcs_uri"]: item for item in items}
    if len(by_uri) != len(items):
        raise CampaignFailure("Batch items contain duplicate image GCS URIs")

    result_root = results_dir or work_dir / "results"
    result_files = tuple(sorted(result_root.rglob("*.jsonl")))
    if not result_files:
        raise FileNotFoundError(f"No batch result JSONL found under {result_root}")
    terminal_before = completed_stage_b_ids(stage_b_output)
    seen: set[str] = set()
    rows: list[StageBFileRow] = []
    ok = rejected = errors = skipped = 0
    for payload in _iter_jsonl_objects(result_files):
        image_uri = _result_image_uri(payload)
        try:
            item = by_uri[image_uri]
        except KeyError as exc:
            raise CampaignFailure(
                f"Batch output references an unknown image URI: {image_uri}"
            ) from exc
        sample_id = _required_str(item.get("sample_id"), "item.sample_id")
        if sample_id in seen:
            raise CampaignFailure(f"Duplicate batch output for sample_id={sample_id}")
        seen.add(sample_id)
        if sample_id in terminal_before:
            skipped += 1
            continue
        try:
            stage_a_row = stage_a[sample_id]
        except KeyError as exc:
            raise CampaignFailure(
                f"Prepared Stage A sample is missing during ingestion: {sample_id}"
            ) from exc
        row = _stage_b_row_from_result(teacher, item, stage_a_row, payload)
        rows.append(row)
        if row.status is RecordStatus.OK:
            ok += 1
        elif row.status is RecordStatus.REJECTED:
            rejected += 1
        else:
            errors += 1

    missing = len(items) - len(seen)
    for row in rows:
        append_jsonl(stage_b_output, row)

    counts = {
        "ok": ok,
        "rejected": rejected,
        "errors": errors,
        "skipped_existing_terminal": skipped,
        "missing_batch_outputs": missing,
    }
    record = {
        **counts,
        "completed_at": datetime.now(UTC).isoformat(),
        "stage_b_output": str(stage_b_output.resolve()),
        "stage_b_output_sha256_after": (
            _sha256_path(stage_b_output) if stage_b_output.is_file() else None
        ),
        "audit_rows_include_retryable_errors": True,
        "rejected_rows_are_terminal": True,
        "batch_pricing": _pricing_manifest(teacher),
    }
    if resume_ingestion:
        recoveries = manifest.get("ingestion_recoveries")
        if not isinstance(recoveries, list):
            recoveries = []
        recoveries.append(record)
        manifest["ingestion_recoveries"] = recoveries
    else:
        manifest["ingestion"] = record
    if missing:
        manifest["status"] = "ingested_incomplete"
    elif errors:
        manifest["status"] = "ingested_with_retryable_errors"
    else:
        manifest["status"] = "ingested"
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    _write_manifest(work_dir, manifest)
    _upload_manifest_if_possible(work_dir, manifest)
    return counts


def normalize_overlap_errors(
    *,
    teacher: TeacherModel,
    work_dir: Path,
    stage_a_path: Path,
    stage_b_output: Path,
    results_dir: Path | None = None,
    apply: bool = False,
) -> dict[str, object]:
    """Normalize evidence IDs cited on both sides of one comparison.

    An observation used for both diagnoses is non-discriminative. This narrow
    normalization removes the intersecting ID from both evidence lists without
    changing the diagnosis, alternatives, confidence, policy, limitations, or
    clinical prose. The canonical parser and clinical gate are then rerun.

    The operation is a dry run unless ``apply=True``. Applying writes a
    hash-addressed sidecar before appending new terminal attempts to the Stage B
    audit log. Raw provider outputs and prior error attempts remain unchanged.
    """
    manifest = _load_manifest(work_dir)
    if manifest.get("status") != "ingested_with_retryable_errors":
        raise CampaignFailure(
            "Overlap normalization requires an ingested retryable-error batch"
        )

    expected_stage_a = _mapping(manifest.get("stage_a_input"), "stage_a_input")
    expected_stage_a_sha = _required_str(
        expected_stage_a.get("sha256"), "stage_a_input.sha256"
    )
    if _sha256_path(stage_a_path) != expected_stage_a_sha:
        raise CampaignFailure("Stage A input hash differs from the prepared batch")
    stage_a = index_ok_stage_a(load_stage_a_rows(stage_a_path))

    local = _mapping(manifest.get("local"), "manifest.local")
    items = _load_items(
        Path(_required_str(local.get("items_path"), "manifest.local.items_path"))
    )
    by_uri = {item["image_gcs_uri"]: item for item in items}
    if len(by_uri) != len(items):
        raise CampaignFailure("Batch items contain duplicate image GCS URIs")

    result_root = results_dir or work_dir / "results"
    result_files = tuple(sorted(result_root.rglob("*.jsonl")))
    if not result_files:
        raise FileNotFoundError(f"No batch result JSONL found under {result_root}")

    audit_rows = load_stage_b_rows(stage_b_output)
    terminal_before = completed_stage_b_ids(stage_b_output)
    error_attempts: dict[str, list[str]] = {}
    for row in audit_rows:
        if row.status is not RecordStatus.ERROR:
            continue
        attempt_id = row.provenance.attempt_id if row.provenance else "unknown"
        error_attempts.setdefault(row.sample_id, []).append(attempt_id)

    seen: set[str] = set()
    normalized_rows: list[StageBFileRow] = []
    audit_entries: list[dict[str, object]] = []
    status_counts = {"ok": 0, "rejected": 0}
    skipped_terminal = 0
    removed_unique_ids = 0
    for payload in _iter_jsonl_objects(result_files):
        image_uri = _result_image_uri(payload)
        try:
            item = by_uri[image_uri]
        except KeyError as exc:
            raise CampaignFailure(
                f"Batch output references an unknown image URI: {image_uri}"
            ) from exc
        sample_id = _required_str(item.get("sample_id"), "item.sample_id")
        if sample_id in seen:
            raise CampaignFailure(f"Duplicate batch output for sample_id={sample_id}")
        seen.add(sample_id)
        if sample_id in terminal_before:
            skipped_terminal += 1
            continue
        if sample_id not in error_attempts:
            raise CampaignFailure(
                f"No retryable Stage B error attempt exists for {sample_id}"
            )

        try:
            stage_a_row = stage_a[sample_id]
        except KeyError as exc:
            raise CampaignFailure(
                f"Prepared Stage A sample is missing during normalization: {sample_id}"
            ) from exc
        _validate_stage_a_join(item, stage_a_row)
        response_payload = payload.get("response")
        response = _teacher_response_from_batch(response_payload, teacher)
        try:
            parse_stage_b(response.content_json)
        except ValidationError as exc:
            if not _only_overlap_validation_errors(exc):
                raise CampaignFailure(
                    f"Stage B error is not overlap-only for {sample_id}"
                ) from exc
        else:
            raise CampaignFailure(
                f"Stage B output no longer requires normalization: {sample_id}"
            )

        normalized, changes = _remove_conflicting_evidence(response.content_json)
        if not changes:
            raise CampaignFailure(f"No overlapping evidence found for {sample_id}")
        reasoning = parse_stage_b(normalized)
        if stage_a_row.morphology is None:
            raise CampaignFailure(f"Accepted Stage A has no morphology: {sample_id}")
        gold = _required_str(item.get("gold_diagnosis"), "item.gold_diagnosis")
        check = validate_stage_b(stage_a_row.morphology, reasoning, gold)
        status = RecordStatus.OK if check.ok else RecordStatus.REJECTED
        provenance = generation_provenance(teacher, "B", response=response)
        row = StageBFileRow(
            sample_id=sample_id,
            status=status,
            reasoning=reasoning,
            reasons=check.reasons,
            error=None,
            usage=response.usage,
            teacher=teacher.name,
            gold_diagnosis=gold,
            stage_a_sample_id=_required_str(
                item.get("stage_a_sample_id"), "item.stage_a_sample_id"
            ),
            image_path=_required_str(item.get("source_ref"), "item.source_ref"),
            image_preprocessing=ImagePreprocessingInfo.model_validate(
                item["image_preprocessing"]
            ),
            provenance=provenance,
        )
        normalized_rows.append(row)
        status_counts[status.value] += 1
        removed_unique_ids += sum(
            len(change["removed_observation_ids"]) for change in changes
        )
        response_mapping = _mapping(response_payload, "batch.response")
        response_id = response_mapping.get("responseId")
        audit_entries.append(
            {
                "schema_version": 1,
                "normalization_version": OVERLAP_NORMALIZATION_VERSION,
                "sample_id": sample_id,
                "source_error_attempt_ids": error_attempts[sample_id],
                "source_response_id": (
                    response_id if isinstance(response_id, str) else None
                ),
                "original_content_sha256": _sha256_json(response.content_json),
                "normalized_content_sha256": _sha256_json(normalized),
                "changes": changes,
                "diagnosis_unchanged": True,
                "clinical_reasoning_unchanged": True,
                "terminal_status": status.value,
                "gate_reasons": list(check.reasons),
                "normalized_attempt_id": provenance.attempt_id,
            }
        )

    missing = len(items) - len(seen)
    if missing:
        raise CampaignFailure(f"Missing batch outputs during normalization: {missing}")
    report: dict[str, object] = {
        "normalization_version": OVERLAP_NORMALIZATION_VERSION,
        "apply": apply,
        "source_items": len(items),
        "normalized_rows": len(normalized_rows),
        "skipped_existing_terminal": skipped_terminal,
        "removed_unique_observation_ids": removed_unique_ids,
        "status_counts": status_counts,
        "diagnosis_unchanged": True,
        "clinical_reasoning_unchanged": True,
    }
    if not apply:
        return report

    data_path = work_dir / OVERLAP_NORMALIZATION_DATA_NAME
    normalization_manifest_path = work_dir / OVERLAP_NORMALIZATION_MANIFEST_NAME
    if data_path.exists() or normalization_manifest_path.exists():
        raise FileExistsError("Overlap normalization artifacts already exist")
    stage_b_sha_before = _sha256_path(stage_b_output)
    with data_path.open("x", encoding="utf-8") as handle:
        for entry in audit_entries:
            _write_jsonl(handle, entry)
    _fsync_path(data_path)
    for row in normalized_rows:
        append_jsonl(stage_b_output, row)

    normalization_record: dict[str, object] = {
        **report,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "policy": "remove_intersection_from_both_evidence_lists",
        "raw_provider_outputs_preserved": True,
        "source_results": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256_path(path),
            }
            for path in result_files
        ],
        "audit_sidecar": {
            "path": str(data_path.resolve()),
            "rows": len(audit_entries),
            "bytes": data_path.stat().st_size,
            "sha256": _sha256_path(data_path),
        },
        "stage_b_output": {
            "path": str(stage_b_output.resolve()),
            "sha256_before": stage_b_sha_before,
            "sha256_after": _sha256_path(stage_b_output),
        },
    }
    _write_named_manifest(normalization_manifest_path, normalization_record)
    manifest["normalization"] = normalization_record
    manifest["status"] = "normalized_terminal_rows"
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    _write_manifest(work_dir, manifest)
    _upload_manifest_if_possible(work_dir, manifest)
    return normalization_record


def _only_overlap_validation_errors(exc: ValidationError) -> bool:
    errors = exc.errors()
    return bool(errors) and all(
        error.get("type") == "value_error"
        and "An observation cannot favour both sides of one comparison"
        in str(error.get("msg", ""))
        for error in errors
    )


def _remove_conflicting_evidence(
    raw: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Return a deep-copied payload with bilateral evidence removed."""
    normalized = cast(dict[str, object], json.loads(json.dumps(raw)))
    comparisons = normalized.get("differential_comparisons")
    if not isinstance(comparisons, list):
        raise TypeError("differential_comparisons is not a list")
    changes: list[dict[str, object]] = []
    for index, value in enumerate(comparisons):
        comparison = _mapping(value, f"differential_comparisons.{index}")
        diagnosis = comparison.get("features_favoring_diagnosis")
        alternative = comparison.get("features_favoring_alternative")
        if not isinstance(diagnosis, list) or not all(
            isinstance(item, str) for item in diagnosis
        ):
            raise TypeError("features_favoring_diagnosis is not a string list")
        if not isinstance(alternative, list) or not all(
            isinstance(item, str) for item in alternative
        ):
            raise TypeError("features_favoring_alternative is not a string list")
        overlap = set(cast(list[str], diagnosis)).intersection(
            cast(list[str], alternative)
        )
        if not overlap:
            continue
        comparison["features_favoring_diagnosis"] = [
            item for item in diagnosis if item not in overlap
        ]
        comparison["features_favoring_alternative"] = [
            item for item in alternative if item not in overlap
        ]
        comparisons[index] = comparison
        changes.append(
            {
                "comparison_index": index,
                "removed_observation_ids": sorted(overlap),
            }
        )
    return normalized, changes


def freeze_stage_b(
    *,
    teacher: TeacherModel,
    cohort: ExampleCohort,
    stage_a_path: Path,
    stage_b_output: Path,
    freeze_dir: Path,
    normalization_sidecars: tuple[Path, ...] = (),
    dataset_config: str = "diagnosis",
    dataset_split: str = "sft_train",
) -> dict[str, object]:
    """Materialize a complete, immutable Stage B release with audit artifacts."""
    if freeze_dir.exists() and any(freeze_dir.iterdir()):
        raise FileExistsError(f"Stage B freeze directory is not empty: {freeze_dir}")

    source_ids = cohort.sample_ids
    selected = set(source_ids)
    _compatible_stage_a_index(
        stage_a_path,
        teacher=teacher,
        selected=selected,
    )
    rows = load_stage_b_rows(stage_b_output)
    if not rows:
        raise CampaignFailure("Stage B freeze requires a non-empty audit log")

    expected_signature = _provenance_signature(generation_provenance(teacher, "B"))
    terminal: dict[str, StageBFileRow] = {}
    attempt_ids: set[str] = set()
    for row in rows:
        if row.sample_id not in selected:
            raise CampaignFailure(
                f"Stage B audit log contains an unknown sample: {row.sample_id}"
            )
        if row.stage_a_sample_id != row.sample_id:
            raise CampaignFailure(f"Stage B/Stage A sample mismatch: {row.sample_id}")
        if row.provenance is None:
            raise CampaignFailure(f"Stage B provenance is missing for {row.sample_id}")
        if _provenance_signature(row.provenance) != expected_signature:
            raise CampaignFailure(
                f"Stage B protocol mismatch for sample_id={row.sample_id}"
            )
        if row.provenance.attempt_id in attempt_ids:
            raise CampaignFailure(
                f"Duplicate Stage B attempt_id={row.provenance.attempt_id}"
            )
        attempt_ids.add(row.provenance.attempt_id)
        if row.status not in {RecordStatus.OK, RecordStatus.REJECTED}:
            continue
        if row.sample_id in terminal:
            raise CampaignFailure(
                f"Multiple terminal Stage B rows for sample_id={row.sample_id}"
            )
        terminal[row.sample_id] = row

    missing = selected.difference(terminal)
    if missing:
        raise CampaignFailure(
            f"Stage B freeze requires exact terminal coverage: missing={len(missing)}"
        )

    normalized_attempt_ids: set[str] = set()
    normalized_samples: set[str] = set()
    normalization_entries: list[tuple[Path, list[dict[str, object]]]] = []
    for sidecar in normalization_sidecars:
        entries = list(_iter_jsonl_objects((sidecar,)))
        if not entries:
            raise CampaignFailure(f"Normalization sidecar is empty: {sidecar}")
        for entry in entries:
            if entry.get("normalization_version") != OVERLAP_NORMALIZATION_VERSION:
                raise CampaignFailure(f"Unknown normalization protocol in {sidecar}")
            if entry.get("diagnosis_unchanged") is not True:
                raise CampaignFailure(f"Normalization changed diagnosis in {sidecar}")
            if entry.get("clinical_reasoning_unchanged") is not True:
                raise CampaignFailure(
                    f"Normalization changed clinical reasoning in {sidecar}"
                )
            sample_id = _required_str(entry.get("sample_id"), "sample_id")
            normalized_attempt_id = _required_str(
                entry.get("normalized_attempt_id"), "normalized_attempt_id"
            )
            if sample_id in normalized_samples:
                raise CampaignFailure(f"Duplicate normalized sample_id={sample_id}")
            normalized_samples.add(sample_id)
            try:
                terminal_row = terminal[sample_id]
            except KeyError as exc:
                raise CampaignFailure(
                    f"Normalized sample has no terminal Stage B row: {sample_id}"
                ) from exc
            if terminal_row.provenance is None:  # guarded above
                raise AssertionError("Terminal provenance unexpectedly missing")
            if terminal_row.provenance.attempt_id != normalized_attempt_id:
                raise CampaignFailure(
                    f"Normalized attempt does not match terminal row: {sample_id}"
                )
            if terminal_row.reasoning is None:  # terminal rows always have reasoning
                raise AssertionError("Terminal reasoning unexpectedly missing")
            normalized_sha = _required_str(
                entry.get("normalized_content_sha256"),
                "normalized_content_sha256",
            )
            actual_sha = _sha256_json(terminal_row.reasoning.model_dump(mode="json"))
            if actual_sha != normalized_sha:
                raise CampaignFailure(
                    f"Normalized content hash mismatch for {sample_id}"
                )
            normalized_attempt_ids.add(normalized_attempt_id)
            source_error_ids = entry.get("source_error_attempt_ids")
            if not isinstance(source_error_ids, list) or not all(
                isinstance(value, str) for value in source_error_ids
            ):
                raise CampaignFailure(
                    f"Invalid source error attempts for normalized sample {sample_id}"
                )
            unknown_attempts = set(source_error_ids).difference(attempt_ids)
            if unknown_attempts:
                raise CampaignFailure(
                    f"Normalization references unknown source attempts: {sample_id}"
                )
        normalization_entries.append((sidecar, entries))

    freeze_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = freeze_dir / STAGE_B_FREEZE_DATA_NAME
    rejected_path = freeze_dir / STAGE_B_REJECTED_DATA_NAME
    accepted_rows = [
        terminal[sample_id]
        for sample_id in source_ids
        if terminal[sample_id].status is RecordStatus.OK
    ]
    rejected_rows = [
        terminal[sample_id]
        for sample_id in source_ids
        if terminal[sample_id].status is RecordStatus.REJECTED
    ]
    _write_stage_b_rows_exclusive(accepted_path, accepted_rows)
    _write_stage_b_rows_exclusive(rejected_path, rejected_rows)

    stage = teacher.stage("B")
    prompt_path = freeze_dir / stage.prompt.source_path.name
    schema_path = freeze_dir / stage.json_schema.source_path.name
    config_path = freeze_dir / teacher.config_path.name
    shutil.copy2(stage.prompt.source_path, prompt_path)
    shutil.copy2(stage.json_schema.source_path, schema_path)
    shutil.copy2(teacher.config_path, config_path)

    copied_normalizations: list[dict[str, object]] = []
    if normalization_entries:
        normalization_dir = freeze_dir / "normalization"
        normalization_dir.mkdir()
        for index, (source, entries) in enumerate(normalization_entries, start=1):
            destination = normalization_dir / f"{index:02d}_{source.name}"
            shutil.copy2(source, destination)
            copied: dict[str, object] = {
                **_artifact_manifest(destination),
                "source_path": str(source.resolve()),
                "rows": len(entries),
            }
            source_manifest = source.with_name(OVERLAP_NORMALIZATION_MANIFEST_NAME)
            if source_manifest.is_file():
                manifest_destination = normalization_dir / (
                    f"{index:02d}_{source_manifest.name}"
                )
                shutil.copy2(source_manifest, manifest_destination)
                copied["manifest"] = _artifact_manifest(manifest_destination)
            copied_normalizations.append(copied)

    attempts = Counter(row.sample_id for row in rows)
    status_counts = Counter(row.status.value for row in rows)
    terminal_status_counts = Counter(row.status.value for row in terminal.values())
    expected = generation_provenance(teacher, "B")
    provider_cost = sum(
        row.usage.cost
        for row in rows
        if row.provenance is not None
        and row.provenance.attempt_id not in normalized_attempt_ids
        and row.usage is not None
        and row.usage.cost is not None
    )
    terminal_cost = sum(
        row.usage.cost
        for row in terminal.values()
        if row.usage is not None and row.usage.cost is not None
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "freeze_id": freeze_dir.name,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "repo_id": "danielfdias98/ISEPDistillDataset",
            "revision": "b215f0474e4931b5951da768e79a0d579d26919d",
            "config": dataset_config,
            "split": dataset_split,
            "source_sample_count": len(source_ids),
        },
        "teacher": {
            "provider": teacher.provider.value,
            "name": teacher.name,
            "model": teacher.model.id,
            "seed": teacher.generation.seed,
            "max_output_tokens": teacher.generation.max_tokens,
            "reasoning_effort": teacher.reasoning.effort,
            "reasoning_excluded": teacher.reasoning.exclude,
        },
        "protocol": {
            "prompt_version": stage.prompt.version,
            "prompt_sha256": expected.prompt_sha256,
            "schema_sha256": expected.schema_sha256,
            "normalization_version": (
                OVERLAP_NORMALIZATION_VERSION if normalization_entries else None
            ),
            "normalization_policy": (
                "remove_intersection_from_both_evidence_lists"
                if normalization_entries
                else None
            ),
            "diagnosis_changed_by_normalization": False,
            "clinical_reasoning_changed_by_normalization": False,
        },
        "stage_a_input": {
            **_artifact_manifest(stage_a_path),
            "rows": len(source_ids),
        },
        "source_audit_log": {
            **_artifact_manifest(stage_b_output),
            "rows": len(rows),
            "unique_sample_ids": len({row.sample_id for row in rows}),
            "status_counts": dict(sorted(status_counts.items())),
            "duplicate_attempt_counts": {
                sample_id: count
                for sample_id, count in sorted(attempts.items())
                if count > 1
            },
        },
        "terminal_coverage": {
            "rows": len(terminal),
            "unique_sample_ids": len(terminal),
            "missing_sample_ids": 0,
            "status_counts": dict(sorted(terminal_status_counts.items())),
        },
        "accepted_release": {
            **_artifact_manifest(accepted_path),
            "rows": len(accepted_rows),
            "unique_sample_ids": len({row.sample_id for row in accepted_rows}),
            "status": RecordStatus.OK.value,
        },
        "rejected_release": {
            **_artifact_manifest(rejected_path),
            "rows": len(rejected_rows),
            "unique_sample_ids": len({row.sample_id for row in rejected_rows}),
            "status": RecordStatus.REJECTED.value,
        },
        "normalization": {
            "normalized_rows": len(normalized_samples),
            "artifacts": copied_normalizations,
        },
        "cost": {
            "currency": "USD",
            "provider_cost_usd_all_requests": provider_cost,
            "terminal_release_source_cost_usd": terminal_cost,
            "normalized_terminal_usage_not_double_counted": True,
        },
        "frozen_artifacts": {
            "prompt": _artifact_manifest(prompt_path),
            "schema": _artifact_manifest(schema_path),
            "teacher_config": _artifact_manifest(config_path),
        },
    }
    _write_named_manifest(freeze_dir / STAGE_B_FREEZE_MANIFEST_NAME, manifest)
    return manifest


def _write_stage_b_rows_exclusive(
    path: Path,
    rows: Iterable[StageBFileRow],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")
        handle.flush()
        _fsync_path(path)


def _stage_b_row_from_result(
    teacher: TeacherModel,
    item: Mapping[str, object],
    stage_a: StageAFileRow,
    payload: Mapping[str, object],
) -> StageBFileRow:
    sample_id = _required_str(item.get("sample_id"), "item.sample_id")
    source_ref = _required_str(item.get("source_ref"), "item.source_ref")
    gold = _required_str(item.get("gold_diagnosis"), "item.gold_diagnosis")
    stage_a_sample_id = _required_str(
        item.get("stage_a_sample_id"), "item.stage_a_sample_id"
    )
    image_info = ImagePreprocessingInfo.model_validate(item["image_preprocessing"])
    _validate_stage_a_join(item, stage_a)

    response_payload = payload.get("response")
    usage = _batch_usage(response_payload, teacher)
    provider_status = payload.get("status")
    if isinstance(provider_status, str) and provider_status.strip():
        return _batch_error_row(
            teacher,
            sample_id=sample_id,
            source_ref=source_ref,
            gold=gold,
            stage_a_sample_id=stage_a_sample_id,
            image_info=image_info,
            error=f"provider_batch_error: {provider_status[:500]}",
            usage=usage,
        )
    try:
        response = _teacher_response_from_batch(response_payload, teacher)
        reasoning = parse_stage_b(response.content_json)
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        return _batch_error_row(
            teacher,
            sample_id=sample_id,
            source_ref=source_ref,
            gold=gold,
            stage_a_sample_id=stage_a_sample_id,
            image_info=image_info,
            error=f"batch_output_error:{type(exc).__name__}: {exc}",
            usage=usage,
        )
    if stage_a.morphology is None:
        raise CampaignFailure(f"Accepted Stage A has no morphology: {sample_id}")
    check = validate_stage_b(stage_a.morphology, reasoning, gold)
    status = RecordStatus.OK if check.ok else RecordStatus.REJECTED
    return StageBFileRow(
        sample_id=sample_id,
        status=status,
        reasoning=reasoning,
        reasons=check.reasons,
        error=None,
        usage=response.usage,
        teacher=teacher.name,
        gold_diagnosis=gold,
        stage_a_sample_id=stage_a_sample_id,
        image_path=source_ref,
        image_preprocessing=image_info,
        provenance=generation_provenance(teacher, "B", response=response),
    )


def _batch_usage(raw: object, teacher: TeacherModel) -> UsageInfo | None:
    if not isinstance(raw, Mapping):
        return None
    metadata = raw.get("usageMetadata")
    if not isinstance(metadata, Mapping):
        return None
    usage = UsageInfo(
        prompt_tokens=_optional_int(metadata.get("promptTokenCount")),
        completion_tokens=_optional_int(metadata.get("candidatesTokenCount")),
        total_tokens=_optional_int(metadata.get("totalTokenCount")),
        thoughts_tokens=_optional_int(metadata.get("thoughtsTokenCount")),
        request_attempts=1,
    )
    return (
        teacher.batch_pricing.estimate_usage(usage)
        if teacher.batch_pricing is not None
        else usage
    )


def _batch_error_row(
    teacher: TeacherModel,
    *,
    sample_id: str,
    source_ref: str,
    gold: str,
    stage_a_sample_id: str,
    image_info: ImagePreprocessingInfo,
    error: str,
    usage: UsageInfo | None,
) -> StageBFileRow:
    return StageBFileRow(
        sample_id=sample_id,
        status=RecordStatus.ERROR,
        reasoning=None,
        reasons=(),
        error=error,
        usage=usage,
        teacher=teacher.name,
        gold_diagnosis=gold,
        stage_a_sample_id=stage_a_sample_id,
        image_path=source_ref,
        image_preprocessing=image_info,
        provenance=generation_provenance(teacher, "B"),
    )


def _compatible_stage_a_index(
    path: Path,
    *,
    teacher: TeacherModel,
    selected: set[str],
) -> dict[str, StageAFileRow]:
    indexed = index_ok_stage_a(load_stage_a_rows(path))
    missing = selected.difference(indexed)
    unknown = set(indexed).difference(selected)
    if missing or unknown:
        raise CampaignFailure(
            "Stage B Batch requires exact frozen Stage A coverage: "
            f"missing={len(missing)}, unknown={len(unknown)}"
        )
    expected = _provenance_signature(generation_provenance(teacher, "A"))
    for row in indexed.values():
        if row.provenance is None:
            raise CampaignFailure("Stage A provenance is missing; do not mix protocols")
        if _provenance_signature(row.provenance) != expected:
            raise CampaignFailure("Stage A provenance differs from the frozen protocol")
    return indexed


def _compatible_stage_b_ids(
    path: Path,
    *,
    teacher: TeacherModel,
    selected: set[str],
) -> set[str]:
    if not path.is_file():
        return set()
    expected = _provenance_signature(generation_provenance(teacher, "B"))
    terminal: set[str] = set()
    for row in load_stage_b_rows(path):
        if row.sample_id not in selected or row.status not in {
            RecordStatus.OK,
            RecordStatus.REJECTED,
        }:
            continue
        if row.provenance is None:
            raise CampaignFailure("Stage B provenance is missing; use a new protocol")
        if _provenance_signature(row.provenance) != expected:
            raise CampaignFailure("Stage B output contains a different protocol")
        terminal.add(row.sample_id)
    return terminal


def _require_same_stage_a_image(
    stage_a: StageAFileRow,
    current: ImagePreprocessingInfo,
) -> None:
    if stage_a.image_preprocessing is None:
        raise CampaignFailure(
            f"Frozen Stage A image hash is missing: {stage_a.sample_id}"
        )
    if stage_a.image_preprocessing != current:
        raise CampaignFailure(
            "Stage B image preprocessing differs from frozen Stage A for "
            f"sample_id={stage_a.sample_id}"
        )


def _validate_stage_a_join(
    item: Mapping[str, object],
    stage_a: StageAFileRow,
) -> None:
    sample_id = _required_str(item.get("sample_id"), "item.sample_id")
    if stage_a.sample_id != sample_id:
        raise CampaignFailure(f"Stage A join mismatch for sample_id={sample_id}")
    if stage_a.morphology is None:
        raise CampaignFailure(f"Stage A morphology missing for sample_id={sample_id}")
    expected_hash = _required_str(
        item.get("stage_a_morphology_sha256"), "item.stage_a_morphology_sha256"
    )
    if _sha256_json(stage_a.morphology.model_dump(mode="json")) != expected_hash:
        raise CampaignFailure(f"Stage A morphology hash mismatch for {sample_id}")
    expected_attempt = item.get("stage_a_attempt_id")
    actual_attempt = stage_a.provenance.attempt_id if stage_a.provenance else None
    if expected_attempt != actual_attempt:
        raise CampaignFailure(f"Stage A attempt mismatch for {sample_id}")


def _require_vertex_teacher(teacher: TeacherModel) -> VertexAPI:
    if teacher.provider is not TeacherProvider.VERTEX or not isinstance(
        teacher.api, VertexAPI
    ):
        raise TypeError("Stage B batch requires a provider=vertex teacher config")
    return teacher.api


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vertex Batch transport for E3 Stage B"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--stage-a", type=Path, default=DEFAULT_STAGE_A)
    prepare.add_argument("--stage-b-output", type=Path, default=DEFAULT_STAGE_B)
    prepare.add_argument("--work-dir", type=Path, required=True)
    prepare.add_argument("--gcs-prefix", required=True)
    prepare.add_argument("--hub-config", default="diagnosis")
    prepare.add_argument("--hub-split", default="sft_train")
    prepare.add_argument(
        "--pending-limit",
        type=int,
        default=None,
        help="Prepare only the first N unfinished IDs as a local/canary batch.",
    )

    upload = subparsers.add_parser("upload")
    upload.add_argument("--work-dir", type=Path, required=True)
    upload.add_argument("--resume", action="store_true")
    upload.add_argument(
        "--authorize-private-gold-upload",
        action="store_true",
        help="Acknowledge that requests/items contain private gold labels.",
    )

    for command in ("submit", "status", "download", "ingest"):
        child = subparsers.add_parser(command)
        child.add_argument("--work-dir", type=Path, required=True)
        if command == "ingest":
            child.add_argument("--stage-a", type=Path, default=DEFAULT_STAGE_A)
            child.add_argument("--stage-b-output", type=Path, default=DEFAULT_STAGE_B)
            child.add_argument("--results-dir", type=Path, default=None)
            child.add_argument("--resume-ingestion", action="store_true")
    normalize = subparsers.add_parser("normalize-overlap")
    normalize.add_argument("--work-dir", type=Path, required=True)
    normalize.add_argument("--stage-a", type=Path, default=DEFAULT_STAGE_A)
    normalize.add_argument("--stage-b-output", type=Path, default=DEFAULT_STAGE_B)
    normalize.add_argument("--results-dir", type=Path, default=None)
    normalize.add_argument(
        "--apply",
        action="store_true",
        help="Append normalized terminal rows; otherwise perform a dry run.",
    )
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--stage-a", type=Path, default=DEFAULT_STAGE_A)
    freeze.add_argument("--stage-b-output", type=Path, default=DEFAULT_STAGE_B)
    freeze.add_argument("--freeze-dir", type=Path, required=True)
    freeze.add_argument(
        "--normalization-sidecar",
        type=Path,
        action="append",
        default=[],
        help="Audited overlap-normalization JSONL; may be repeated.",
    )
    freeze.add_argument("--hub-config", default="diagnosis")
    freeze.add_argument("--hub-split", default="sft_train")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    teacher = TeacherModel.from_yaml(args.config)
    if args.command == "prepare":
        result: object = prepare_batch(
            teacher=teacher,
            cohort=load_hub_cohort(config=args.hub_config, split=args.hub_split),
            stage_a_path=args.stage_a,
            stage_b_output=args.stage_b_output,
            work_dir=args.work_dir,
            gcs_prefix=args.gcs_prefix,
            dataset_config=args.hub_config,
            dataset_split=args.hub_split,
            pending_limit=args.pending_limit,
        )
    elif args.command == "upload":
        result = upload_batch(
            args.work_dir,
            authorize_private_gold_upload=args.authorize_private_gold_upload,
            resume=args.resume,
        )
    elif args.command == "submit":
        result = submit_batch(teacher=teacher, work_dir=args.work_dir)
    elif args.command == "status":
        result = refresh_status(teacher=teacher, work_dir=args.work_dir)
    elif args.command == "download":
        print(download_results(args.work_dir))
        return
    elif args.command == "ingest":
        result = ingest_results(
            teacher=teacher,
            work_dir=args.work_dir,
            stage_a_path=args.stage_a,
            stage_b_output=args.stage_b_output,
            results_dir=args.results_dir,
            resume_ingestion=args.resume_ingestion,
        )
    elif args.command == "normalize-overlap":
        result = normalize_overlap_errors(
            teacher=teacher,
            work_dir=args.work_dir,
            stage_a_path=args.stage_a,
            stage_b_output=args.stage_b_output,
            results_dir=args.results_dir,
            apply=args.apply,
        )
    else:
        result = freeze_stage_b(
            teacher=teacher,
            cohort=load_hub_cohort(config=args.hub_config, split=args.hub_split),
            stage_a_path=args.stage_a,
            stage_b_output=args.stage_b_output,
            freeze_dir=args.freeze_dir,
            normalization_sidecars=tuple(args.normalization_sidecar),
            dataset_config=args.hub_config,
            dataset_split=args.hub_split,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
