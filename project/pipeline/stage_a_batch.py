"""Prepare, submit, monitor, and ingest Vertex Batch Stage A jobs.

The batch path is deliberately an alternative transport for the frozen Stage A
protocol. It uses the same teacher YAML, prompt, JSON Schema, seed, thinking
configuration, image preprocessing, source revision, and final ``stage_a.jsonl``
contract as the synchronous generator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import shutil
import subprocess
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from project.pipeline.generate import CampaignFailure, ExampleCohort, load_hub_cohort
from project.teacher.client import TeacherResponse
from project.teacher.provenance import generation_provenance
from project.teacher.schemas import (
    GenerationProvenance,
    ImagePreprocessingInfo,
    RecordStatus,
    StageAFileRow,
    UsageInfo,
    parse_stage_a,
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
    index_ok_stage_a,
    load_stage_a_rows,
)
from project.teacher.vertex import jpeg_bytes_from_data_url

LOGGER = logging.getLogger("project.pipeline.stage_a_batch")

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "teacher_configs" / "gemini_3_7_flash_vertex.yaml"
)
DEFAULT_STAGE_A = PROJECT_ROOT / "data" / "morphology" / "stage_a.jsonl"
MANIFEST_NAME = "campaign_manifest.json"
ITEMS_NAME = "items.jsonl"
REQUESTS_NAME = "requests.jsonl"
FREEZE_DATA_NAME = "stage_a.jsonl"
FREEZE_MANIFEST_NAME = "freeze_manifest.json"
_TERMINAL_JOB_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
    "JOB_STATE_EXPIRED",
}


def prepare_batch(
    *,
    teacher: TeacherModel,
    cohort: ExampleCohort,
    stage_a_output: Path,
    work_dir: Path,
    gcs_prefix: str,
    dataset_config: str = "diagnosis",
    dataset_split: str = "sft_train",
    pending_limit: int | None = None,
) -> dict[str, object]:
    """Materialize one GCS-backed batch containing only unfinished Stage A IDs."""
    _require_vertex_teacher(teacher)
    prefix = _normalize_gcs_prefix(gcs_prefix)
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(f"Batch work directory is not empty: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    images_dir = work_dir / "images"
    images_dir.mkdir()

    selected = cohort.sample_ids
    selected_set = set(selected)
    completed = _compatible_stage_a_ids(
        stage_a_output,
        teacher=teacher,
        selected=selected_set,
    )
    all_pending = tuple(
        sample_id for sample_id in selected if sample_id not in completed
    )
    if not all_pending:
        raise CampaignFailure("Stage A already has a compatible ok row for every ID")
    if pending_limit is not None and pending_limit <= 0:
        raise ValueError("Pending batch limit must be greater than zero")
    pending = all_pending if pending_limit is None else all_pending[:pending_limit]

    requests_path = work_dir / REQUESTS_NAME
    items_path = work_dir / ITEMS_NAME
    image_bytes = 0
    started = datetime.now(UTC)
    with (
        requests_path.open("x", encoding="utf-8") as requests_handle,
        items_path.open("x", encoding="utf-8") as items_handle,
    ):
        for index, example in enumerate(cohort.iter_selected(pending), start=1):
            prepared = prepare_pil_image(example.image)
            encoded = jpeg_bytes_from_data_url(prepared.data_url)
            filename = _image_filename(index, example.sample_id)
            image_path = images_dir / filename
            image_path.write_bytes(encoded)
            image_bytes += len(encoded)
            image_uri = f"{prefix}/images/{filename}"

            request = build_batch_request(
                teacher=teacher,
                image_gcs_uri=image_uri,
            )
            item = {
                "sample_id": example.sample_id,
                "source_ref": example.source_ref,
                "image_filename": filename,
                "image_gcs_uri": image_uri,
                "image_preprocessing": prepared.info.model_dump(mode="json"),
            }
            _write_jsonl(requests_handle, request)
            _write_jsonl(items_handle, item)
            if index == 1 or index % 100 == 0 or index == len(pending):
                LOGGER.info(
                    "Prepared Batch A %d/%d (%.2f%%)",
                    index,
                    len(pending),
                    100 * index / len(pending),
                )

    _fsync_path(requests_path)
    _fsync_path(items_path)
    _validate_batch_request_file(requests_path, expected_count=len(pending))
    provenance = generation_provenance(teacher, "A")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "campaign_id": work_dir.name,
        "stage": "A",
        "transport": "vertex_batch_gcs",
        "status": "prepared",
        "created_at": started.isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "work_dir": str(work_dir.resolve()),
        "stage_a_output": str(stage_a_output.resolve()),
        "stage_a_output_sha256_before": (
            _sha256_path(stage_a_output) if stage_a_output.is_file() else None
        ),
        "source_sample_count": len(selected),
        "existing_compatible_ok_count": len(completed.intersection(selected_set)),
        "pending_request_count": len(pending),
        "pending_source_count_before_limit": len(all_pending),
        "pending_limit": pending_limit,
        "gold_sent_to_teacher": False,
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
            "project": cast(VertexAPI, teacher.api).project,
            "location": cast(VertexAPI, teacher.api).location,
            "seed": teacher.generation.seed,
            "max_output_tokens": teacher.generation.max_tokens,
            "reasoning_effort": teacher.reasoning.effort,
            "reasoning_excluded": teacher.reasoning.exclude,
            "prompt_sha256": provenance.prompt_sha256,
            "schema_sha256": provenance.schema_sha256,
            "batch_schema_transport": "responseSchema_openapi_nullable",
            "batch_schema_sha256": _sha256_json(
                _json_schema_to_vertex_openapi(
                    cast(
                        dict[str, object],
                        teacher.vertex_generate_config("A")["response_json_schema"],
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
) -> dict[str, object]:
    """Build one documented Vertex JSONL request for frozen Stage A."""
    _require_vertex_teacher(teacher)
    if not image_gcs_uri.startswith("gs://"):
        raise ValueError("Batch image URI must use gs://")
    stage = teacher.stage("A")
    generation = teacher.vertex_generate_config("A")
    thinking = cast(dict[str, object], generation["thinking_config"])
    response_schema = _json_schema_to_vertex_openapi(
        cast(dict[str, object], generation["response_json_schema"])
    )
    return {
        "request": {
            "systemInstruction": {"parts": [{"text": stage.prompt.system}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": stage.prompt.user},
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
                # The GCS batch importer tabularizes every JSONL request. Its
                # schema inference corrupts heterogeneous JSON-Schema ``anyOf``
                # arrays such as ``[string constraints, null]``. Vertex's
                # OpenAPI ``nullable`` form preserves that logical contract.
                "responseSchema": response_schema,
                "thinkingConfig": {
                    "thinkingLevel": thinking["thinking_level"],
                    "includeThoughts": thinking["include_thoughts"],
                },
            },
        }
    }


def _json_schema_to_vertex_openapi(
    schema: Mapping[str, object],
) -> dict[str, object]:
    """Convert the frozen JSON Schema to Vertex's OpenAPI Schema subset."""
    flattened = _inline_local_json_schema_refs(schema)

    def convert(value: object) -> object:
        if isinstance(value, list):
            return [convert(item) for item in value]
        if not isinstance(value, Mapping):
            return value

        node = copy.deepcopy(dict(value))
        raw_any_of = node.pop("anyOf", None)
        if raw_any_of is not None:
            if not isinstance(raw_any_of, list):
                raise TypeError("JSON Schema anyOf must be an array")
            nullable = False
            non_null: list[object] = []
            for option in raw_any_of:
                if isinstance(option, Mapping) and option.get("type") == "null":
                    nullable = True
                else:
                    non_null.append(option)
            if nullable and len(non_null) == 1:
                only = non_null[0]
                if not isinstance(only, Mapping):
                    raise TypeError("Nullable JSON Schema option must be an object")
                merged = copy.deepcopy(dict(only))
                merged.update(node)
                node = merged
                node["nullable"] = True
            else:
                node["anyOf"] = non_null
                if nullable:
                    node["nullable"] = True

        schema_type = node.get("type")
        if schema_type == "null":
            node.pop("type")
            node["nullable"] = True
        elif isinstance(schema_type, str):
            node["type"] = schema_type.upper()

        converted = {key: convert(child) for key, child in node.items()}
        return converted

    converted = convert(flattened)
    if not isinstance(converted, dict):
        raise TypeError("Vertex OpenAPI schema must be an object")
    _validate_vertex_openapi_schema(converted, path="responseSchema")
    return converted


def _inline_local_json_schema_refs(
    schema: Mapping[str, object],
) -> dict[str, object]:
    """Expand local ``#/$defs`` references for Vertex GCS batch ingestion.

    The logical schema is unchanged. Only its transport representation is
    flattened so the batch JSONL contains no dollar-prefixed schema keys that
    the Vertex/BigQuery import path can interpret as column names.
    """
    root = copy.deepcopy(dict(schema))
    raw_defs = root.get("$defs", {})
    if not isinstance(raw_defs, Mapping):
        raise TypeError("JSON Schema $defs must be an object")
    definitions = dict(raw_defs)

    def expand(value: object, stack: tuple[str, ...] = ()) -> object:
        if isinstance(value, list):
            return [expand(item, stack) for item in value]
        if not isinstance(value, Mapping):
            return value

        node = dict(value)
        reference = node.get("$ref")
        if reference is not None:
            if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
                raise ValueError(f"Unsupported JSON Schema reference: {reference!r}")
            name = (
                reference.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
            )
            if name not in definitions:
                raise ValueError(f"Unknown JSON Schema reference: {reference}")
            if reference in stack:
                chain = " -> ".join((*stack, reference))
                raise ValueError(
                    f"Cyclic JSON Schema reference cannot be flattened: {chain}"
                )
            resolved = expand(definitions[name], (*stack, reference))
            if not isinstance(resolved, Mapping):
                raise TypeError(f"Referenced schema must be an object: {reference}")
            merged = dict(resolved)
            for key, sibling in node.items():
                if key != "$ref":
                    merged[key] = expand(sibling, stack)
            return merged

        return {
            key: expand(child, stack) for key, child in node.items() if key != "$defs"
        }

    expanded = expand(root)
    if not isinstance(expanded, dict):
        raise TypeError("Expanded JSON Schema must be an object")
    return expanded


def upload_batch(
    work_dir: Path,
    *,
    resume: bool = False,
) -> dict[str, object]:
    """Upload a prepared batch without deleting or overwriting remote objects."""
    manifest = _load_manifest(work_dir)
    if manifest.get("status") != "prepared":
        raise CampaignFailure("Only a prepared batch can be uploaded")
    gcs = _mapping(manifest.get("gcs"), "manifest.gcs")
    local = _mapping(manifest.get("local"), "manifest.local")
    prefix = _required_str(gcs.get("prefix"), "manifest.gcs.prefix")
    _validate_batch_request_file(
        Path(_required_str(local.get("requests_path"), "manifest.local.requests_path")),
        expected_count=_required_int(
            manifest.get("pending_request_count"),
            "manifest.pending_request_count",
        ),
    )

    existing = _run_gcloud(
        ["storage", "ls", "--recursive", prefix],
        check=False,
    )
    existing_objects = _gcs_object_lines(existing.stdout)
    if existing_objects and not resume:
        raise FileExistsError(f"GCS batch prefix is not empty: {prefix}")
    if existing_objects:
        allowed_exact = {
            _required_str(gcs.get("requests_uri"), "gcs.requests_uri"),
            _required_str(gcs.get("items_uri"), "gcs.items_uri"),
            _required_str(gcs.get("manifest_uri"), "gcs.manifest_uri"),
        }
        images_root = _required_str(gcs.get("images_prefix"), "gcs.images_prefix") + "/"
        unexpected = [
            uri
            for uri in existing_objects
            if uri not in allowed_exact and not uri.startswith(images_root)
        ]
        if unexpected:
            raise CampaignFailure(
                "Cannot resume upload because the GCS prefix contains "
                f"{len(unexpected)} unexpected objects"
            )

    images_dir = Path(_required_str(local.get("images_dir"), "local.images_dir"))
    images_prefix = _required_str(gcs.get("images_prefix"), "gcs.images_prefix")
    _run_gcloud_streaming(
        ["storage", "rsync", str(images_dir), images_prefix, "--recursive"]
    )
    for filename, uri_key in (
        (REQUESTS_NAME, "requests_uri"),
        (ITEMS_NAME, "items_uri"),
    ):
        _run_gcloud_streaming(
            [
                "storage",
                "cp",
                str(work_dir / filename),
                _required_str(gcs.get(uri_key), f"gcs.{uri_key}"),
            ]
        )

    listing = _run_gcloud(
        ["storage", "ls", "--recursive", images_prefix]
    ).stdout.splitlines()
    remote_count = len(_gcs_object_lines("\n".join(listing)))
    expected_count = _required_int(
        manifest.get("pending_request_count"), "pending_request_count"
    )
    if remote_count != expected_count:
        raise CampaignFailure(
            "GCS image count mismatch: "
            f"remote={remote_count}, expected={expected_count}"
        )

    manifest["status"] = "uploaded"
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    manifest["upload"] = {
        "completed_at": datetime.now(UTC).isoformat(),
        "remote_image_count": remote_count,
        "remote_prefix_previously_empty": not existing_objects,
        "resumed_upload": bool(existing_objects),
        "integrity": "gcloud_crc32c_or_md5_transport_validation",
    }
    _write_manifest(work_dir, manifest)
    _run_gcloud_streaming(
        [
            "storage",
            "cp",
            str(work_dir / MANIFEST_NAME),
            _required_str(gcs.get("manifest_uri"), "gcs.manifest_uri"),
        ]
    )
    return manifest


def submit_batch(
    *,
    teacher: TeacherModel,
    work_dir: Path,
    client: object | None = None,
) -> dict[str, object]:
    """Submit exactly one uploaded batch and durably record its resource name."""
    _require_vertex_teacher(teacher)
    manifest = _load_manifest(work_dir)
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

    job = batches.create(
        model=teacher.model.id,
        src=_required_str(gcs.get("requests_uri"), "gcs.requests_uri"),
        config=types.CreateBatchJobConfig(
            display_name=_display_name(str(manifest["campaign_id"])),
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
    """Refresh the durable job state and completion counters."""
    _require_vertex_teacher(teacher)
    manifest = _load_manifest(work_dir)
    batch_job = _mapping(manifest.get("batch_job"), "manifest.batch_job")
    name = _required_str(batch_job.get("name"), "batch_job.name")
    sdk = cast(Any, client) if client is not None else _vertex_client(teacher)
    job = sdk.batches.get(name=name)
    payload = _job_payload(job)
    manifest["batch_job"] = payload
    state = str(payload.get("state") or "")
    completion = payload.get("completion_stats")
    completion_stats = dict(completion) if isinstance(completion, Mapping) else {}
    failed_count = _optional_int(completion_stats.get("failed_count")) or 0
    if state == "JOB_STATE_SUCCEEDED" and failed_count == 0:
        manifest["status"] = "batch_succeeded"
    elif state in _TERMINAL_JOB_STATES:
        manifest["status"] = "batch_completed_with_failures"
    else:
        manifest["status"] = "running"
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    _write_manifest(work_dir, manifest)
    _upload_manifest_if_possible(work_dir, manifest)
    return manifest


def download_results(work_dir: Path) -> Path:
    """Download successful or partial JSONL output without deleting remote data."""
    manifest = _load_manifest(work_dir)
    batch_job = _mapping(manifest.get("batch_job"), "manifest.batch_job")
    output_info = _mapping(batch_job.get("output_info"), "batch_job.output_info")
    source = _required_str(
        output_info.get("gcs_output_directory"),
        "batch_job.output_info.gcs_output_directory",
    )
    results_dir = work_dir / "results"
    results_dir.mkdir(exist_ok=True)
    _run_gcloud_streaming(["storage", "rsync", source, str(results_dir), "--recursive"])
    return results_dir


def ingest_results(
    *,
    teacher: TeacherModel,
    work_dir: Path,
    stage_a_output: Path,
    results_dir: Path | None = None,
    resume_ingestion: bool = False,
) -> dict[str, int]:
    """Validate batch output and append Stage A rows, correlated by image URI."""
    manifest = _load_manifest(work_dir)
    previous_ingestion = manifest.get("ingestion")
    if previous_ingestion is None:
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
            "ingested_with_quarantine",
            "ingested_incomplete",
        }:
            raise CampaignFailure(
                "Only an incomplete or quarantined ingestion can be resumed"
            )
    local = _mapping(manifest.get("local"), "manifest.local")
    item_path = Path(_required_str(local.get("items_path"), "local.items_path"))
    items = _load_items(item_path)
    by_uri = {item["image_gcs_uri"]: item for item in items}
    if len(by_uri) != len(items):
        raise CampaignFailure("Batch items contain duplicate image GCS URIs")

    result_root = results_dir or work_dir / "results"
    result_files = tuple(sorted(result_root.rglob("*.jsonl")))
    if not result_files:
        raise FileNotFoundError(f"No batch result JSONL found under {result_root}")
    already_ok = (
        set(index_ok_stage_a(load_stage_a_rows(stage_a_output)))
        if stage_a_output.is_file()
        else set()
    )
    seen: set[str] = set()
    accepted_rows: list[StageAFileRow] = []
    quarantined_rows: list[StageAFileRow] = []
    ok = 0
    errors = 0
    skipped = 0
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
        if sample_id in already_ok:
            skipped += 1
            continue
        row = _stage_a_row_from_result(teacher, item, payload)
        if row.status is RecordStatus.OK:
            accepted_rows.append(row)
            ok += 1
        else:
            quarantined_rows.append(row)
            errors += 1

    missing = len(items) - len(seen)

    # Only schema-valid rows belong to the canonical target file. Invalid
    # provider/schema outputs remain durable in a separate audit sidecar and
    # therefore stay retryable without contaminating accepted targets.
    for row in accepted_rows:
        append_jsonl(stage_a_output, row)
    recoveries = manifest.get("ingestion_recoveries")
    if not isinstance(recoveries, list):
        recoveries = []
    recovery_number = len(recoveries) + 1
    quarantine_path = work_dir / (
        f"ingestion_errors_recovery_{recovery_number:02d}.jsonl"
        if resume_ingestion
        else "ingestion_errors.jsonl"
    )
    if quarantined_rows:
        _write_stage_a_rows_exclusive(quarantine_path, quarantined_rows)

    ingestion = {
        "ok": ok,
        "errors": errors,
        "skipped_existing_ok": skipped,
        "missing_batch_outputs": missing,
    }
    ingestion_record = {
        **ingestion,
        "completed_at": datetime.now(UTC).isoformat(),
        "stage_a_output": str(stage_a_output.resolve()),
        "stage_a_output_sha256_after": (
            _sha256_path(stage_a_output) if stage_a_output.is_file() else None
        ),
        "accepted_rows_only": True,
        "quarantine_path": (
            str(quarantine_path.resolve()) if quarantined_rows else None
        ),
        "quarantine_sha256": (
            _sha256_path(quarantine_path) if quarantined_rows else None
        ),
        "batch_pricing": _pricing_manifest(teacher),
    }
    if resume_ingestion:
        recoveries.append(ingestion_record)
        manifest["ingestion_recoveries"] = recoveries
    else:
        manifest["ingestion"] = ingestion_record
    if missing:
        manifest["status"] = "ingested_incomplete"
    elif errors:
        manifest["status"] = "ingested_with_quarantine"
    else:
        manifest["status"] = "ingested"
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    _write_manifest(work_dir, manifest)
    _upload_manifest_if_possible(work_dir, manifest)
    return ingestion


def freeze_stage_a(
    *,
    teacher: TeacherModel,
    cohort: ExampleCohort,
    stage_a_output: Path,
    freeze_dir: Path,
    dataset_config: str = "diagnosis",
    dataset_split: str = "sft_train",
) -> dict[str, object]:
    """Materialize one immutable, accepted-only Stage A release with hashes."""
    if freeze_dir.exists() and any(freeze_dir.iterdir()):
        raise FileExistsError(f"Stage A freeze directory is not empty: {freeze_dir}")
    rows = load_stage_a_rows(stage_a_output)
    indexed = index_ok_stage_a(rows)
    source_ids = cohort.sample_ids
    selected = set(source_ids)
    accepted = set(indexed)
    missing = selected.difference(accepted)
    unknown = accepted.difference(selected)
    if missing or unknown:
        raise CampaignFailure(
            "Stage A freeze requires exact source coverage: "
            f"missing={len(missing)}, unknown={len(unknown)}"
        )
    _compatible_stage_a_ids(
        stage_a_output,
        teacher=teacher,
        selected=selected,
    )

    freeze_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = freeze_dir / FREEZE_DATA_NAME
    accepted_rows = [indexed[sample_id] for sample_id in source_ids]
    _write_stage_a_rows_exclusive(accepted_path, accepted_rows)

    stage = teacher.stage("A")
    prompt_path = freeze_dir / stage.prompt.source_path.name
    schema_path = freeze_dir / stage.json_schema.source_path.name
    config_path = freeze_dir / teacher.config_path.name
    shutil.copy2(stage.prompt.source_path, prompt_path)
    shutil.copy2(stage.json_schema.source_path, schema_path)
    shutil.copy2(teacher.config_path, config_path)

    attempts = Counter(row.sample_id for row in rows)
    status_counts = Counter(row.status.value for row in rows)
    expected = generation_provenance(teacher, "A")
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
        },
        "source_audit_log": {
            "path": str(stage_a_output.resolve()),
            "rows": len(rows),
            "unique_sample_ids": len({row.sample_id for row in rows}),
            "status_counts": dict(sorted(status_counts.items())),
            "duplicate_attempt_counts": {
                sample_id: count
                for sample_id, count in sorted(attempts.items())
                if count > 1
            },
            "bytes": stage_a_output.stat().st_size,
            "sha256": _sha256_path(stage_a_output),
        },
        "accepted_release": {
            "path": str(accepted_path.resolve()),
            "rows": len(accepted_rows),
            "unique_sample_ids": len(indexed),
            "status": "ok",
            "bytes": accepted_path.stat().st_size,
            "sha256": _sha256_path(accepted_path),
            "estimated_cost_usd_in_accepted_rows": sum(
                row.usage.cost
                for row in accepted_rows
                if row.usage is not None and row.usage.cost is not None
            ),
        },
        "frozen_artifacts": {
            "prompt": _artifact_manifest(prompt_path),
            "schema": _artifact_manifest(schema_path),
            "teacher_config": _artifact_manifest(config_path),
        },
    }
    _write_named_manifest(freeze_dir / FREEZE_MANIFEST_NAME, manifest)
    return manifest


def reprice_batch_rows(
    *,
    teacher: TeacherModel,
    work_dir: Path,
    stage_a_output: Path,
    backup_path: Path,
) -> dict[str, object]:
    """Correct stored Batch costs from original results without changing targets."""
    manifest = _load_manifest(work_dir)
    if manifest.get("repricing") is not None:
        raise CampaignFailure("This batch manifest has already been repriced")
    if backup_path.exists():
        raise FileExistsError(f"Repricing backup already exists: {backup_path}")
    local = _mapping(manifest.get("local"), "manifest.local")
    items = _load_items(
        Path(_required_str(local.get("items_path"), "manifest.local.items_path"))
    )
    by_uri = {item["image_gcs_uri"]: item for item in items}
    result_files = tuple(sorted((work_dir / "results").rglob("*.jsonl")))
    if not result_files:
        raise FileNotFoundError(
            f"No batch result JSONL found under {work_dir / 'results'}"
        )

    batch_rows: dict[str, StageAFileRow] = {}
    for payload in _iter_jsonl_objects(result_files):
        item = by_uri[_result_image_uri(payload)]
        row = _stage_a_row_from_result(teacher, item, payload)
        if row.status is RecordStatus.OK:
            batch_rows[row.sample_id] = row

    rows = load_stage_a_rows(stage_a_output)
    before_sha256 = _sha256_path(stage_a_output)
    matched: set[str] = set()
    changed = 0
    rewritten: list[StageAFileRow] = []
    for row in rows:
        batch_row = batch_rows.get(row.sample_id)
        if (
            batch_row is None
            or row.status is not RecordStatus.OK
            or row.morphology != batch_row.morphology
            or row.image_preprocessing != batch_row.image_preprocessing
        ):
            rewritten.append(row)
            continue
        matched.add(row.sample_id)
        if row.usage == batch_row.usage:
            rewritten.append(row)
            continue
        rewritten.append(row.model_copy(update={"usage": batch_row.usage}))
        changed += 1

    if batch_rows and not matched:
        raise CampaignFailure("No canonical Stage A row matched this batch output")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stage_a_output, backup_path)
    _rewrite_stage_a_rows_atomic(stage_a_output, rewritten)
    repricing = {
        "completed_at": datetime.now(UTC).isoformat(),
        "matched_batch_sample_ids": len(matched),
        "changed_rows": changed,
        "unmatched_valid_batch_sample_ids": len(set(batch_rows).difference(matched)),
        "backup_path": str(backup_path.resolve()),
        "backup_sha256": _sha256_path(backup_path),
        "stage_a_sha256_before": before_sha256,
        "stage_a_sha256_after": _sha256_path(stage_a_output),
        "batch_pricing": _pricing_manifest(teacher),
        "targets_changed": False,
    }
    manifest["repricing"] = repricing
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    _write_manifest(work_dir, manifest)
    _upload_manifest_if_possible(work_dir, manifest)
    return repricing


def _stage_a_row_from_result(
    teacher: TeacherModel,
    item: Mapping[str, object],
    payload: Mapping[str, object],
) -> StageAFileRow:
    sample_id = _required_str(item.get("sample_id"), "item.sample_id")
    source_ref = _required_str(item.get("source_ref"), "item.source_ref")
    image_info = ImagePreprocessingInfo.model_validate(item["image_preprocessing"])
    response_payload = payload.get("response")
    batch_usage = (
        _usage_from_batch(response_payload.get("usageMetadata"), teacher)
        if isinstance(response_payload, Mapping)
        else None
    )
    status = payload.get("status")
    if isinstance(status, str) and status.strip():
        return _batch_error_row(
            teacher,
            sample_id,
            source_ref,
            image_info,
            f"provider_batch_error: {status[:500]}",
            usage=batch_usage,
        )
    try:
        response = _teacher_response_from_batch(response_payload, teacher)
        morphology = parse_stage_a(response.content_json)
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        return _batch_error_row(
            teacher,
            sample_id,
            source_ref,
            image_info,
            f"batch_output_error:{type(exc).__name__}: {exc}",
            usage=batch_usage,
        )
    return StageAFileRow(
        sample_id=sample_id,
        status=RecordStatus.OK,
        morphology=morphology,
        error=None,
        usage=response.usage,
        teacher=teacher.name,
        image_path=source_ref,
        image_preprocessing=image_info,
        provenance=generation_provenance(teacher, "A", response=response),
    )


def _teacher_response_from_batch(
    raw: object,
    teacher: TeacherModel,
) -> TeacherResponse:
    response = _mapping(raw, "batch.response")
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("batch response has no candidates")
    candidate = _mapping(candidates[0], "batch.response.candidates[0]")
    finish = candidate.get("finishReason")
    finish_name = finish if isinstance(finish, str) else None
    if finish_name in {"MAX_TOKENS", "LENGTH"}:
        raise ValueError("batch response was truncated")
    content = _mapping(candidate.get("content"), "candidate.content")
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise TypeError("candidate.content.parts is not a list")
    texts = [
        part.get("text")
        for part in parts
        if isinstance(part, Mapping)
        and part.get("thought") is not True
        and isinstance(part.get("text"), str)
    ]
    raw_content = "".join(cast(list[str], texts)).strip()
    if not raw_content:
        raise ValueError("batch response text is empty")
    parsed = json.loads(raw_content)
    if not isinstance(parsed, dict):
        raise TypeError("batch response JSON is not an object")
    usage = _usage_from_batch(response.get("usageMetadata"), teacher)
    return TeacherResponse(
        content_json=cast(dict[str, object], parsed),
        raw_content=raw_content,
        usage=usage,
        finish_reason=finish_name.lower() if finish_name else None,
        native_finish_reason=finish_name,
    )


def _usage_from_batch(raw: object, teacher: TeacherModel) -> UsageInfo | None:
    if not isinstance(raw, Mapping):
        return None
    usage = UsageInfo(
        prompt_tokens=_optional_int(raw.get("promptTokenCount")),
        completion_tokens=_optional_int(raw.get("candidatesTokenCount")),
        total_tokens=_optional_int(raw.get("totalTokenCount")),
        thoughts_tokens=_optional_int(raw.get("thoughtsTokenCount")),
        request_attempts=1,
    )
    # Batch has a separate list-price class. Falling back to Standard pricing
    # would silently double the stored estimate, so missing Batch pricing means
    # tokens are retained but cost is deliberately left unset.
    return (
        teacher.batch_pricing.estimate_usage(usage)
        if teacher.batch_pricing is not None
        else usage
    )


def _batch_error_row(
    teacher: TeacherModel,
    sample_id: str,
    source_ref: str,
    image_info: ImagePreprocessingInfo,
    error: str,
    *,
    usage: UsageInfo | None = None,
) -> StageAFileRow:
    return StageAFileRow(
        sample_id=sample_id,
        status=RecordStatus.ERROR,
        morphology=None,
        error=error,
        usage=usage,
        teacher=teacher.name,
        image_path=source_ref,
        image_preprocessing=image_info,
        provenance=generation_provenance(teacher, "A"),
    )


def _result_image_uri(payload: Mapping[str, object]) -> str:
    request = _mapping(payload.get("request"), "batch.request")
    contents = request.get("contents")
    if not isinstance(contents, list):
        raise TypeError("batch.request.contents is not a list")
    uris: list[str] = []
    for content_raw in contents:
        content = _mapping(content_raw, "batch.request.contents[]")
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part_raw in parts:
            if not isinstance(part_raw, Mapping):
                continue
            file_data = part_raw.get("fileData")
            if not isinstance(file_data, Mapping):
                continue
            uri = file_data.get("fileUri")
            if isinstance(uri, str) and uri.startswith("gs://"):
                uris.append(uri)
    if len(uris) != 1:
        raise CampaignFailure(
            f"Expected exactly one GCS image URI in batch result, found {len(uris)}"
        )
    return uris[0]


def _compatible_stage_a_ids(
    path: Path,
    *,
    teacher: TeacherModel,
    selected: set[str],
) -> set[str]:
    if not path.is_file():
        return set()
    indexed = index_ok_stage_a(load_stage_a_rows(path))
    expected = generation_provenance(teacher, "A")
    expected_signature = _provenance_signature(expected)
    for sample_id, row in indexed.items():
        if sample_id not in selected:
            continue
        if row.provenance is None:
            raise CampaignFailure(
                f"Stage A resume provenance is missing in {path}; use a new protocol"
            )
        if _provenance_signature(row.provenance) != expected_signature:
            raise CampaignFailure(
                f"Stage A resume provenance mismatch in {path}; do not mix protocols"
            )
    return set(indexed).intersection(selected)


def _provenance_signature(value: GenerationProvenance) -> tuple[object, ...]:
    return (
        value.provider,
        value.teacher_name,
        value.teacher_model,
        value.seed,
        value.max_output_tokens,
        value.reasoning_effort,
        value.reasoning_excluded,
        value.prompt_sha256,
        value.schema_sha256,
    )


def _vertex_client(teacher: TeacherModel) -> Any:
    from google import genai
    from google.genai import types

    api = cast(VertexAPI, teacher.api)
    return genai.Client(
        enterprise=True,
        project=api.project,
        location=api.location,
        http_options=types.HttpOptions(api_version="v1"),
    )


def _job_payload(job: object) -> dict[str, object]:
    dumper = getattr(job, "model_dump", None)
    if not callable(dumper):
        raise TypeError("Vertex batch job is not serializable")
    raw = dumper(mode="json", by_alias=False, exclude_none=True)
    if not isinstance(raw, dict):
        raise TypeError("Vertex batch job did not serialize to an object")
    return cast(dict[str, object], raw)


def _upload_manifest_if_possible(
    work_dir: Path,
    manifest: Mapping[str, object],
) -> None:
    gcs = _mapping(manifest.get("gcs"), "manifest.gcs")
    uri = _required_str(gcs.get("manifest_uri"), "gcs.manifest_uri")
    _run_gcloud(["storage", "cp", str(work_dir / MANIFEST_NAME), uri])


def _load_items(path: Path) -> list[dict[str, object]]:
    return list(_iter_jsonl_objects((path,)))


def _iter_jsonl_objects(paths: Iterable[Path]) -> Iterable[dict[str, object]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}:{line_number}: expected an object")
                yield cast(dict[str, object], payload)


def _load_manifest(work_dir: Path) -> dict[str, object]:
    path = work_dir / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Batch manifest not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Batch manifest must be a JSON object")
    return cast(dict[str, object], raw)


def _write_manifest(work_dir: Path, manifest: Mapping[str, object]) -> None:
    path = work_dir / MANIFEST_NAME
    _write_named_manifest(path, manifest)


def _write_named_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_jsonl(handle: Any, payload: Mapping[str, object]) -> None:
    handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def _fsync_path(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _run_gcloud(
    args: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gcloud", *args],
        check=check,
        text=True,
        capture_output=True,
    )


def _run_gcloud_streaming(args: list[str]) -> None:
    subprocess.run(
        ["gcloud", *args],
        check=True,
        text=True,
    )


def _gcs_object_lines(output: str) -> list[str]:
    """Remove recursive-listing directory headers and return object URIs."""
    return [
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith("gs://")
        and not line.strip().endswith("/")
        and not line.strip().endswith(":")
    ]


def _validate_batch_request_file(path: Path, *, expected_count: int) -> None:
    """Fail locally before upload if Vertex's GCS importer would reject input."""
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise CampaignFailure(f"Blank batch request at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CampaignFailure(
                    f"Invalid batch JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, Mapping):
                raise CampaignFailure(
                    f"Batch request line {line_number} must be a JSON object"
                )
            request = row.get("request")
            if not isinstance(request, Mapping):
                raise CampaignFailure(
                    f"Batch request line {line_number} is missing request"
                )
            generation = request.get("generationConfig")
            if not isinstance(generation, Mapping):
                raise CampaignFailure(
                    f"Batch request line {line_number} is missing generationConfig"
                )
            if "responseJsonSchema" in generation:
                raise CampaignFailure(
                    "Vertex GCS batch requests must use responseSchema, not "
                    f"responseJsonSchema, at line {line_number}"
                )
            schema = generation.get("responseSchema")
            if not isinstance(schema, Mapping):
                raise CampaignFailure(
                    f"Batch request line {line_number} is missing responseSchema"
                )
            forbidden = next(_dollar_prefixed_keys(schema), None)
            if forbidden is not None:
                raise CampaignFailure(
                    "Vertex GCS batch input cannot contain dollar-prefixed schema "
                    f"keys; found {forbidden!r} at line {line_number}"
                )
            _validate_vertex_openapi_schema(
                schema,
                path=f"line {line_number}.responseSchema",
            )
            count += 1
    if count != expected_count:
        raise CampaignFailure(
            f"Batch request count mismatch: file={count}, expected={expected_count}"
        )


def _dollar_prefixed_keys(value: object) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.startswith("$"):
                yield key
            yield from _dollar_prefixed_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _dollar_prefixed_keys(child)


def _validate_vertex_openapi_schema(value: object, *, path: str) -> None:
    """Reject schema shapes known to be corrupted by Vertex's batch importer."""
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_vertex_openapi_schema(child, path=f"{path}[{index}]")
        return
    if not isinstance(value, Mapping):
        return

    schema_type = value.get("type")
    if schema_type is not None:
        if not isinstance(schema_type, str) or schema_type != schema_type.upper():
            raise CampaignFailure(f"{path}.type must be an uppercase OpenAPI type")
        if schema_type == "NULL":
            raise CampaignFailure(f"{path} must use nullable=true instead of NULL")
    nullable = value.get("nullable")
    if nullable is not None and not isinstance(nullable, bool):
        raise CampaignFailure(f"{path}.nullable must be a boolean")
    for constraint in ("minLength", "maxLength", "minItems", "maxItems"):
        raw = value.get(constraint)
        if raw is not None and (isinstance(raw, bool) or not isinstance(raw, int)):
            raise CampaignFailure(f"{path}.{constraint} must be an integer")
    for key, child in value.items():
        _validate_vertex_openapi_schema(child, path=f"{path}.{key}")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _pricing_manifest(teacher: TeacherModel) -> dict[str, object] | None:
    pricing = teacher.batch_pricing
    if pricing is None:
        return None
    return {
        "input_per_million_tokens_usd": pricing.input_per_million_tokens_usd,
        "output_per_million_tokens_usd": pricing.output_per_million_tokens_usd,
        "traffic_type": pricing.traffic_type,
        "effective_through": pricing.effective_through,
        "source_url": pricing.source_url,
    }


def _write_stage_a_rows_exclusive(
    path: Path,
    rows: Iterable[StageAFileRow],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _rewrite_stage_a_rows_atomic(path: Path, rows: Iterable[StageAFileRow]) -> None:
    temporary = path.with_suffix(path.suffix + ".rewrite.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stage A rewrite temporary already exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _artifact_manifest(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }


def _image_filename(index: int, sample_id: str) -> str:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:20]
    return f"{index:06d}-{digest}.jpg"


def _normalize_gcs_prefix(value: str) -> str:
    prefix = value.strip().rstrip("/")
    if not prefix.startswith("gs://") or prefix.count("/") < 3:
        raise ValueError("GCS prefix must include a bucket and object prefix")
    return prefix


def _display_name(campaign_id: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in "-_" else "-" for char in campaign_id
    )
    return f"isep-e3-stage-a-{safe}"[:128]


def _require_vertex_teacher(teacher: TeacherModel) -> VertexAPI:
    if teacher.provider is not TeacherProvider.VERTEX or not isinstance(
        teacher.api, VertexAPI
    ):
        raise TypeError("Stage A batch requires a provider=vertex teacher config")
    return teacher.api


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    return dict(value)


def _required_str(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path} must be a non-empty string")
    return value


def _required_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    return value


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vertex Batch transport for E3 Stage A"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--stage-a-output", type=Path, default=DEFAULT_STAGE_A)
    prepare.add_argument("--work-dir", type=Path, required=True)
    prepare.add_argument("--gcs-prefix", required=True)
    prepare.add_argument("--hub-config", default="diagnosis")
    prepare.add_argument("--hub-split", default="sft_train")
    prepare.add_argument(
        "--pending-limit",
        type=int,
        default=None,
        help="Prepare only the first N unfinished IDs (for a batch canary).",
    )

    upload = subparsers.add_parser("upload")
    upload.add_argument("--work-dir", type=Path, required=True)
    upload.add_argument(
        "--resume",
        action="store_true",
        help="Resume this exact prepared prefix after an interrupted upload.",
    )

    for command in ("submit", "status", "download", "ingest"):
        child = subparsers.add_parser(command)
        child.add_argument("--work-dir", type=Path, required=True)
        if command == "ingest":
            child.add_argument("--stage-a-output", type=Path, default=DEFAULT_STAGE_A)
            child.add_argument("--results-dir", type=Path, default=None)
            child.add_argument(
                "--resume-ingestion",
                action="store_true",
                help=(
                    "Re-validate a previously quarantined/incomplete batch and "
                    "append only IDs that are still missing a valid row."
                ),
            )
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--stage-a-output", type=Path, default=DEFAULT_STAGE_A)
    freeze.add_argument("--freeze-dir", type=Path, required=True)
    freeze.add_argument("--hub-config", default="diagnosis")
    freeze.add_argument("--hub-split", default="sft_train")
    reprice = subparsers.add_parser("reprice")
    reprice.add_argument("--work-dir", type=Path, required=True)
    reprice.add_argument("--stage-a-output", type=Path, default=DEFAULT_STAGE_A)
    reprice.add_argument("--backup-path", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    teacher = TeacherModel.from_yaml(args.config)
    if args.command == "prepare":
        manifest = prepare_batch(
            teacher=teacher,
            cohort=load_hub_cohort(config=args.hub_config, split=args.hub_split),
            stage_a_output=args.stage_a_output,
            work_dir=args.work_dir,
            gcs_prefix=args.gcs_prefix,
            dataset_config=args.hub_config,
            dataset_split=args.hub_split,
            pending_limit=args.pending_limit,
        )
    elif args.command == "upload":
        manifest = upload_batch(args.work_dir, resume=args.resume)
    elif args.command == "submit":
        manifest = submit_batch(teacher=teacher, work_dir=args.work_dir)
    elif args.command == "status":
        manifest = refresh_status(teacher=teacher, work_dir=args.work_dir)
    elif args.command == "download":
        path = download_results(args.work_dir)
        print(path)
        return
    elif args.command == "ingest":
        counts = ingest_results(
            teacher=teacher,
            work_dir=args.work_dir,
            stage_a_output=args.stage_a_output,
            results_dir=args.results_dir,
            resume_ingestion=args.resume_ingestion,
        )
        print(json.dumps(counts, sort_keys=True))
        return
    elif args.command == "freeze":
        manifest = freeze_stage_a(
            teacher=teacher,
            cohort=load_hub_cohort(config=args.hub_config, split=args.hub_split),
            stage_a_output=args.stage_a_output,
            freeze_dir=args.freeze_dir,
            dataset_config=args.hub_config,
            dataset_split=args.hub_split,
        )
    else:
        manifest = reprice_batch_rows(
            teacher=teacher,
            work_dir=args.work_dir,
            stage_a_output=args.stage_a_output,
            backup_path=args.backup_path,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
