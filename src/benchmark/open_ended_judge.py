"""Single-judge evaluation for open-ended dermatology responses."""

from __future__ import annotations

import asyncio
import base64
from collections import Counter
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from PIL import Image, ImageOps
import pandas as pd
import pyarrow.parquet as pq
import yaml

from src.benchmark.json_parsing import parse_json_output
from src.benchmark.results import canonical_hash, file_sha256, read_jsonl
from src.config import load_model_config
from src.inference.base import (
    InferenceBackend,
    InferenceRequest,
    InferenceSafetyRefusal,
)
from src.inference.factory import create_backend


JUDGE_MODEL_ID = "gpt_5_6_luna"
FALLBACK_JUDGE_MODEL_ID = "qwen_3_7_flash_openrouter"
JUDGE_MAX_OUTPUT_TOKENS = 2048
JUDGE_RETRIES = 3
BENCHMARK_ID = "open_ended_diagnosis"


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _read_split(directory: Path, split: str) -> pd.DataFrame:
    paths = sorted(directory.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"Missing judge split: {directory}/{split}")
    return pd.concat(
        [pq.read_table(path).to_pandas() for path in paths],
        ignore_index=True,
    )


def _image_bytes(value: Any) -> bytes:
    if not isinstance(value, dict):
        raise ValueError("Benchmark image value must be a mapping")
    raw = value.get("bytes")
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    if isinstance(raw, bytearray):
        raw = bytes(raw)
    if not isinstance(raw, bytes) or not raw:
        raise ValueError("Benchmark image bytes are unavailable")
    return raw


def _sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value]


def _render(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{ {key} }}}}", value)
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def _judge_paths(
    run_directory: Path,
    judge_model_id: str = JUDGE_MODEL_ID,
    fallback_judge_model_id: str | None = None,
) -> dict[str, Path]:
    if fallback_judge_model_id:
        protocol_id = (
            f"{judge_model_id}__fallback_{fallback_judge_model_id}"
        )
        directory = run_directory / "judges" / protocol_id
    else:
        directory = (
            run_directory
            if judge_model_id == JUDGE_MODEL_ID
            else run_directory / "judges" / judge_model_id
        )
    return {
        "manifest": directory / "judge_manifest.yaml",
        "judgments": directory / "judgments.jsonl",
        "metrics": directory / "judge_metrics.json",
        "report": directory / "judge_report.html",
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _latest_by_task(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["task_id"]): record for record in records}


def _generation(model: Any) -> dict[str, Any]:
    configured = model.generation
    if is_dataclass(configured) and not isinstance(configured, type):
        values = asdict(configured)
    elif isinstance(configured, dict):
        values = dict(configured)
    else:
        values = {}
    values["max_output_tokens"] = JUDGE_MAX_OUTPUT_TOKENS
    return values


def _judge_request(
    *,
    task: dict[str, Any],
    reference: dict[str, Any],
    prediction: dict[str, Any],
    prompt: dict[str, Any],
    schema: dict[str, Any],
    generation: dict[str, Any],
) -> InferenceRequest:
    response = prediction.get("response", {})
    final_text = (
        str(response.get("final_text", ""))
        if isinstance(response, dict)
        else ""
    )
    morphology = _sequence(reference.get("morphology_concept_ids"))
    description = reference.get("reference_clinical_description")
    values = {
        "reference_diagnosis_name": str(reference["reference_disease_name"]),
        "reference_diagnosis_id": str(reference["reference_disease_id"]),
        "reference_morphology": ", ".join(morphology) if morphology else "Not available.",
        "reference_description": str(description) if description else "Not available.",
        "assistant_response": final_text,
    }
    return InferenceRequest(
        system_prompt=str(prompt["system_prompt"]),
        user_prompt=_render(str(prompt["user_template"]), values),
        image_bytes=_image_bytes(task["image"]),
        schema=schema,
        generation=generation,
        request_id=str(task["task_id"]),
        image_mime_type="image/jpeg",
    )


def _validate_judgment_semantics(judgment: dict[str, Any]) -> None:
    """Reject internally contradictory judge outputs before scoring them."""

    rank = int(judgment["reference_diagnosis_rank"])
    diagnosis_score = int(judgment["diagnosis_correctness"])
    verdict = str(judgment["overall_verdict"])
    unsupported_count = int(judgment["unsupported_claim_count"])
    unsupported_examples = list(judgment["unsupported_claim_examples"])
    errors: list[str] = []

    if rank == 0 and diagnosis_score == 4:
        errors.append(
            "rank 0 cannot be combined with diagnosis_correctness 4"
        )
    if rank == 0 and verdict == "correct":
        errors.append("rank 0 cannot be combined with verdict 'correct'")
    if rank == 1 and diagnosis_score < 3:
        errors.append(
            "rank 1 requires diagnosis_correctness of at least 3"
        )
    if rank == 1 and verdict in {"partially_correct", "incorrect"}:
        errors.append(
            "rank 1 cannot be combined with a partially_correct or "
            "incorrect verdict"
        )
    if unsupported_count == 0 and unsupported_examples:
        errors.append(
            "unsupported_claim_examples must be empty when the count is 0"
        )
    if unsupported_count > 0 and not unsupported_examples:
        errors.append(
            "at least one unsupported claim example is required when the "
            "count is positive"
        )
    if errors:
        raise ValueError(
            "Judge semantic validation failed: " + "; ".join(errors)
        )


def _parse_judgment(raw_text: str, schema: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_json_output(raw_text)
    if not parsed.raw_valid or not isinstance(parsed.decoded, dict):
        raise ValueError(f"Judge returned invalid strict JSON: {parsed.error}")
    errors = sorted(
        Draft202012Validator(schema).iter_errors(parsed.decoded),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        detail = "; ".join(error.message for error in errors[:5])
        raise ValueError(f"Judge schema validation failed: {detail}")
    judgment = dict(parsed.decoded)
    _validate_judgment_semantics(judgment)
    return judgment


def _correction_request(
    request: InferenceRequest,
    error: Exception,
) -> InferenceRequest:
    return replace(
        request,
        user_prompt=(
            request.user_prompt
            + "\n\nYour previous JSON judgment was invalid: "
            + str(error)
            + "\nRe-evaluate the original assistant response and return a "
            "fresh, internally consistent JSON object. In particular, rank "
            "0 means that the reference diagnosis is absent from the explicit "
            "top three; rank 1 means it is the primary diagnosis."
        ),
    )


async def _judge_one(
    *,
    backend: InferenceBackend,
    request: InferenceRequest,
    schema: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    last_error: Exception | None = None
    request_for_attempt = request
    for attempt in range(1, JUDGE_RETRIES + 1):
        try:
            async with semaphore:
                result = await backend.acomplete(request_for_attempt)
            judgment = _parse_judgment(result.final_text, schema)
            return {
                "status": "ok",
                "attempts": attempt,
                "judgment": judgment,
                "judge_response": {
                    "final_text": result.final_text,
                    "finish_reason": result.finish_reason,
                    "provider_response_id": result.provider_response_id,
                    "usage": asdict(result.usage),
                },
            }
        except InferenceSafetyRefusal as exc:
            return {
                "status": "judge_safety_refusal",
                "attempts": attempt,
                "error": f"{type(exc).__name__}: {exc}",
                "safety_code": exc.details.get("code"),
                "safety_details": dict(exc.details),
            }
        except ValueError as exc:
            last_error = exc
            request_for_attempt = _correction_request(request, exc)
        except Exception as exc:
            last_error = exc
    if isinstance(last_error, ValueError):
        return {
            "status": "judge_invalid",
            "attempts": JUDGE_RETRIES,
            "error": f"{type(last_error).__name__}: {last_error}",
        }
    return {
        "status": "judge_error",
        "attempts": JUDGE_RETRIES,
        "error": f"{type(last_error).__name__}: {last_error}",
    }


def _is_content_policy_violation(result: dict[str, Any]) -> bool:
    code = str(result.get("safety_code") or "").casefold()
    return any(
        token in code for token in ("content_policy", "content_filter")
    )


def _model_failure_judgment(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": prediction["task_id"],
        "sample_id": prediction.get("sample_id"),
        "status": "model_failure",
        "target_status": prediction.get("status"),
        "attempts": 0,
        "primary_judge": None,
        "judge_used": None,
        "fallback_used": False,
        "fallback_reason": None,
        "judgment": {
            "reference_diagnosis_rank": 0,
            "diagnosis_correctness": 0,
            "visual_findings_correctness": 0,
            "evidence_grounding": 0,
            "clinical_rationale_quality": 0,
            "differential_quality": 0,
            "unsupported_claim_count": 0,
            "unsupported_claim_examples": [],
            "overall_verdict": "incorrect",
            "judge_summary": "No evaluable model response was produced.",
        },
    }


def compute_judge_metrics(
    judgments: list[dict[str, Any]],
    *,
    judge_model_id: str = JUDGE_MODEL_ID,
    fallback_judge_model_id: str | None = None,
) -> dict[str, Any]:
    latest = list(_latest_by_task(judgments).values())
    terminal_statuses = {
        "ok",
        "model_failure",
        "judge_safety_refusal",
        "judge_invalid",
    }
    if any(item.get("status") not in terminal_statuses for item in latest):
        raise ValueError("Judge metrics require every task to have a terminal judgment")
    valid = [item for item in latest if item.get("status") in {"ok", "model_failure"}]
    unavailable = [
        item for item in latest
        if item.get("status") in {"judge_safety_refusal", "judge_invalid"}
    ]
    payloads = [dict(item["judgment"]) for item in valid]
    ranks = [int(item["reference_diagnosis_rank"]) for item in payloads]
    total = len(payloads)
    score_fields = (
        "diagnosis_correctness",
        "visual_findings_correctness",
        "evidence_grounding",
        "clinical_rationale_quality",
        "differential_quality",
    )
    verdicts = Counter(str(item["overall_verdict"]) for item in payloads)
    judge_usage = Counter(
        str(item["judge_used"])
        for item in valid
        if item.get("status") == "ok" and item.get("judge_used")
    )
    fallback_reasons = Counter(
        str(item["fallback_reason"])
        for item in latest
        if item.get("fallback_used") and item.get("fallback_reason")
    )
    fallback_used_count = sum(
        bool(item.get("fallback_used")) for item in latest
    )
    metrics_by_judge: dict[str, dict[str, Any]] = {}
    for used_judge in sorted(judge_usage):
        judge_payloads = [
            dict(item["judgment"])
            for item in valid
            if item.get("status") == "ok"
            and item.get("judge_used") == used_judge
        ]
        judge_ranks = [
            int(item["reference_diagnosis_rank"])
            for item in judge_payloads
        ]
        judge_total = len(judge_payloads)
        metrics_by_judge[used_judge] = {
            "evaluated_total": judge_total,
            "top_1_accuracy": (
                sum(rank == 1 for rank in judge_ranks) / judge_total
                if judge_total
                else 0.0
            ),
            "top_3_accuracy": (
                sum(rank in {1, 2, 3} for rank in judge_ranks)
                / judge_total
                if judge_total
                else 0.0
            ),
            **{
                f"mean_{field}": (
                    sum(float(item[field]) for item in judge_payloads)
                    / judge_total
                    if judge_total
                    else 0.0
                )
                for field in score_fields
            },
            "unsupported_claim_rate": (
                sum(
                    int(item["unsupported_claim_count"]) > 0
                    for item in judge_payloads
                )
                / judge_total
                if judge_total
                else 0.0
            ),
        }
    return {
        "total": len(latest),
        "evaluated_total": total,
        "judge_coverage": total / len(latest) if latest else 0.0,
        "judge_safety_refusal_count": sum(
            item.get("status") == "judge_safety_refusal"
            for item in unavailable
        ),
        "judge_invalid_count": sum(
            item.get("status") == "judge_invalid" for item in unavailable
        ),
        "judge_model_id": judge_model_id,
        "fallback_judge_model_id": fallback_judge_model_id,
        "fallback_used_count": fallback_used_count,
        "fallback_used_rate": (
            fallback_used_count / len(latest) if latest else 0.0
        ),
        "judge_usage_distribution": dict(sorted(judge_usage.items())),
        "metrics_by_judge": metrics_by_judge,
        "fallback_reason_distribution": dict(
            sorted(fallback_reasons.items())
        ),
        "judge_top_1_accuracy": sum(rank == 1 for rank in ranks) / total if total else 0.0,
        "judge_top_3_accuracy": sum(rank in {1, 2, 3} for rank in ranks) / total if total else 0.0,
        "judge_mean_reciprocal_rank": sum((1 / rank) if rank else 0.0 for rank in ranks) / total if total else 0.0,
        **{
            f"mean_{field}": sum(float(item[field]) for item in payloads) / total if total else 0.0
            for field in score_fields
        },
        "unsupported_claim_rate": sum(int(item["unsupported_claim_count"]) > 0 for item in payloads) / total if total else 0.0,
        "mean_unsupported_claim_count": sum(int(item["unsupported_claim_count"]) for item in payloads) / total if total else 0.0,
        "overall_verdict_distribution": dict(sorted(verdicts.items())),
        "model_failure_count": sum(item.get("status") == "model_failure" for item in valid),
    }


def _thumbnail(value: Any) -> str:
    with Image.open(BytesIO(_image_bytes(value))) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((240, 180), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=70, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _write_report(
    *,
    path: Path,
    metrics: dict[str, Any],
    judgments: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    references: dict[str, dict[str, Any]],
    judge_model_id: str = JUDGE_MODEL_ID,
    fallback_judge_model_id: str | None = None,
) -> None:
    import html

    latest = _latest_by_task(judgments)
    cards = "".join(
        "<div>"
        f"<strong>{html.escape(key.replace('_', ' '))}</strong>"
        f"<span>{html.escape(json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value))}</span>"
        "</div>"
        for key, value in metrics.items()
    )
    rows: list[str] = []
    for task_id in sorted(latest):
        item = latest[task_id]
        prediction = predictions[task_id]
        reference = references[task_id]
        response = prediction.get("response", {})
        judgment = item.get("judgment", {})
        provenance = {
            "primary_judge": item.get("primary_judge"),
            "judge_used": item.get("judge_used"),
            "fallback_used": item.get("fallback_used", False),
            "fallback_reason": item.get("fallback_reason"),
            "status": item.get("status"),
        }
        rows.append(
            "<article>"
            f"<img src='{_thumbnail(tasks[task_id]['image'])}' alt='benchmark image'>"
            "<section>"
            f"<h2>{html.escape(str(task_id))}</h2>"
            f"<p><b>Reference:</b> {html.escape(str(reference['reference_disease_name']))}</p>"
            f"<h3>Model response</h3><pre>{html.escape(str(response.get('final_text', '')))}</pre>"
            f"<h3>Judge provenance</h3><pre>{html.escape(json.dumps(provenance, ensure_ascii=False, indent=2))}</pre>"
            f"<h3>Judgment</h3><pre>{html.escape(json.dumps(judgment, ensure_ascii=False, indent=2))}</pre>"
            "</section></article>"
        )
    document = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<title>Open-ended diagnosis judge report</title><style>
body{{font:14px system-ui;margin:0;background:#f4f6f8;color:#17212b}}header,main{{padding:24px;max-width:1400px;margin:auto}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}}.metrics div,article{{background:white;border:1px solid #dce2e7;border-radius:10px;padding:14px}}
.metrics strong,.metrics span{{display:block}}.metrics span{{font-size:20px;margin-top:6px}}article{{display:grid;grid-template-columns:240px 1fr;gap:18px;margin:16px 0}}img{{width:240px;height:180px;object-fit:contain;background:#eee}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f9fa;padding:12px;border-radius:8px}}@media(max-width:700px){{article{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Open-ended diagnosis — {html.escape(judge_model_id)}{html.escape(f' with {fallback_judge_model_id} safety fallback' if fallback_judge_model_id else ' single judge')}</h1><div class='metrics'>{cards}</div></header><main>{''.join(rows)}</main></body></html>"""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(document, encoding="utf-8")
    os.replace(temporary, path)


async def judge_run(
    *,
    root: Path,
    run_directory: Path,
    batch_size: int = 8,
    backend: InferenceBackend | None = None,
    dry_run: bool = False,
    judge_model_id: str = JUDGE_MODEL_ID,
    judge_backend_profile: str | None = None,
    fallback_judge_model_id: str | None = None,
    fallback_judge_backend_profile: str | None = None,
    fallback_backend: InferenceBackend | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    run_directory = run_directory.resolve()
    paths = _judge_paths(
        run_directory,
        judge_model_id,
        fallback_judge_model_id,
    )
    run_manifest = _load_yaml(run_directory / "run_manifest.yaml")
    benchmark = run_manifest.get("benchmark", {})
    if not isinstance(benchmark, dict) or benchmark.get("id") != BENCHMARK_ID:
        raise ValueError("The run is not an open_ended_diagnosis benchmark")
    evaluation = run_manifest.get("evaluation", {})
    split = str(evaluation["evaluation_set"])
    predictions_list = read_jsonl(run_directory / "predictions.jsonl")
    predictions = {str(item["task_id"]): item for item in predictions_list}
    if not predictions:
        raise ValueError("The run has no predictions to judge")

    release = root / "data/benchmarks/ISEPDermaBench"
    task_frame = _read_split(release / "tasks/open_ended_diagnosis", split)
    reference_frame = _read_split(release / "references/open_ended_diagnosis", split)
    tasks = {str(row["task_id"]): row for row in task_frame.to_dict(orient="records")}
    references = {str(row["task_id"]): row for row in reference_frame.to_dict(orient="records")}
    missing = set(predictions) - (set(tasks) & set(references))
    if missing:
        raise ValueError("Run tasks are missing from the frozen judge release")

    prompt_path = release / "artifacts/judges/open_ended_diagnosis_judge.yaml"
    schema_path = release / "artifacts/schemas/open_ended_diagnosis_judge.schema.json"
    prompt = _load_yaml(prompt_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    model = load_model_config(
        judge_model_id,
        root=root,
        backend_profile=judge_backend_profile,
    )
    fallback_model = (
        load_model_config(
            fallback_judge_model_id,
            root=root,
            backend_profile=fallback_judge_backend_profile,
        )
        if fallback_judge_model_id
        else None
    )
    identity = {
        "judge_model_id": judge_model_id,
        "judge_backend_profile": model.backend.active_profile.name,
        "judge_model_config_sha256": file_sha256(model.config_path),
        "fallback_judge_model_id": fallback_judge_model_id,
        "fallback_judge_backend_profile": (
            fallback_model.backend.active_profile.name
            if fallback_model
            else None
        ),
        "fallback_judge_model_config_sha256": (
            file_sha256(fallback_model.config_path)
            if fallback_model
            else None
        ),
        "judge_prompt_sha256": file_sha256(prompt_path),
        "judge_schema_sha256": file_sha256(schema_path),
        "target_predictions_sha256": file_sha256(run_directory / "predictions.jsonl"),
        "evaluation_set": split,
        "batch_size": str(batch_size),
        "judge_max_output_tokens": str(JUDGE_MAX_OUTPUT_TOKENS),
    }
    manifest_document = {
        "schema_version": 1,
        "status": "dry_run" if dry_run else "running",
        "judge": {
            "model_id": judge_model_id,
            "backend_profile": model.backend.active_profile.name,
            "fallback_model_id": fallback_judge_model_id,
            "fallback_backend_profile": (
                fallback_model.backend.active_profile.name
                if fallback_model
                else None
            ),
            "fallback_trigger": (
                "content_policy_violation"
                if fallback_model
                else None
            ),
            "single_judge": fallback_model is None,
        },
        "target_run": str(run_directory),
        "identity": identity,
        "identity_sha256": canonical_hash(identity),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if paths["manifest"].exists():
        existing = _load_yaml(paths["manifest"])
        if existing.get("identity") != identity:
            raise ValueError("Judge resume identity mismatch")

    if dry_run:
        return {
            "status": "dry_run_valid",
            "judge_model_id": judge_model_id,
            "fallback_judge_model_id": fallback_judge_model_id,
            "evaluation_set": split,
            "responses": len(predictions),
            "network_or_model_called": False,
        }

    if not paths["manifest"].exists():
        _atomic_yaml(paths["manifest"], manifest_document)

    prior = _latest_by_task(read_jsonl(paths["judgments"]))
    completed = {
        task_id
        for task_id, item in prior.items()
        if item.get("status") in {
            "ok",
            "model_failure",
            "judge_safety_refusal",
            "judge_invalid",
        }
    }
    generation = _generation(model)
    owns_backend = backend is None
    judge_backend = backend or create_backend(
        model,
        reasoning_capture="none",
        use_json_schema=True,
    )
    owns_fallback_backend = (
        fallback_model is not None and fallback_backend is None
    )
    active_fallback_backend = (
        fallback_backend
        or (
            create_backend(
                fallback_model,
                reasoning_capture="none",
                use_json_schema=True,
            )
            if fallback_model
            else None
        )
    )
    fallback_generation = (
        _generation(fallback_model) if fallback_model else None
    )
    semaphore = asyncio.Semaphore(batch_size)
    pending_tasks: list[asyncio.Task[tuple[str, dict[str, Any]]]] = []

    async def execute(task_id: str) -> tuple[str, dict[str, Any]]:
        request = _judge_request(
            task=tasks[task_id],
            reference=references[task_id],
            prediction=predictions[task_id],
            prompt=prompt,
            schema=schema,
            generation=generation,
        )
        primary_result = await _judge_one(
            backend=judge_backend,
            request=request,
            schema=schema,
            semaphore=semaphore,
        )
        if (
            primary_result.get("status") == "judge_safety_refusal"
            and _is_content_policy_violation(primary_result)
            and active_fallback_backend is not None
            and fallback_judge_model_id is not None
        ):
            fallback_result = await _judge_one(
                backend=active_fallback_backend,
                request=replace(
                    request,
                    generation=fallback_generation,
                ),
                schema=schema,
                semaphore=semaphore,
            )
            return task_id, {
                **fallback_result,
                "attempts": int(primary_result.get("attempts", 0))
                + int(fallback_result.get("attempts", 0)),
                "primary_attempts": primary_result.get("attempts", 0),
                "fallback_attempts": fallback_result.get("attempts", 0),
                "primary_judge": judge_model_id,
                "judge_used": fallback_judge_model_id,
                "fallback_used": True,
                "fallback_reason": "content_policy_violation",
                "primary_judge_error": primary_result.get("error"),
            }
        return task_id, {
            **primary_result,
            "primary_attempts": primary_result.get("attempts", 0),
            "fallback_attempts": 0,
            "primary_judge": judge_model_id,
            "judge_used": judge_model_id,
            "fallback_used": False,
            "fallback_reason": None,
        }

    try:
        for task_id, prediction in predictions.items():
            if task_id in completed:
                continue
            response = prediction.get("response", {})
            final_text = (
                response.get("final_text", "")
                if isinstance(response, dict)
                else ""
            )
            if prediction.get("status") != "ok" or not str(final_text).strip():
                _append_jsonl(paths["judgments"], _model_failure_judgment(prediction))
            else:
                pending_tasks.append(asyncio.create_task(execute(task_id)))
        for future in asyncio.as_completed(pending_tasks):
            task_id, result = await future
            _append_jsonl(
                paths["judgments"],
                {
                    "task_id": task_id,
                    "sample_id": predictions[task_id].get("sample_id"),
                    "target_status": predictions[task_id].get("status"),
                    **result,
                },
            )
    finally:
        if owns_backend:
            await judge_backend.aclose()
        if owns_fallback_backend and active_fallback_backend is not None:
            await active_fallback_backend.aclose()

    judgments = read_jsonl(paths["judgments"])
    latest = _latest_by_task(judgments)
    failed = [item for item in latest.values() if item.get("status") == "judge_error"]
    if failed or len(latest) != len(predictions):
        document = _load_yaml(paths["manifest"])
        document["status"] = "failed"
        document["judge_error_count"] = len(failed)
        document["finished_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_yaml(paths["manifest"], document)
        raise RuntimeError(
            f"Judging incomplete: {len(failed)} judge errors; rerun the command to resume"
        )

    metrics = compute_judge_metrics(
        judgments,
        judge_model_id=judge_model_id,
        fallback_judge_model_id=fallback_judge_model_id,
    )
    _atomic_json(paths["metrics"], metrics)
    _write_report(
        path=paths["report"],
        metrics=metrics,
        judgments=judgments,
        predictions=predictions,
        tasks=tasks,
        references=references,
        judge_model_id=judge_model_id,
        fallback_judge_model_id=fallback_judge_model_id,
    )
    document = _load_yaml(paths["manifest"])
    document["status"] = "completed"
    document["finished_at"] = datetime.now(timezone.utc).isoformat()
    document["counts"] = dict(
        Counter(str(item.get("status")) for item in latest.values())
    ) | {"total": len(latest)}
    _atomic_yaml(paths["manifest"], document)
    return {
        "status": "completed",
        "judge_model_id": judge_model_id,
        "fallback_judge_model_id": fallback_judge_model_id,
        "judgments_path": str(paths["judgments"]),
        "metrics_path": str(paths["metrics"]),
        "report_path": str(paths["report"]),
        "metrics": metrics,
    }
