"""Single-judge evaluation for open-ended dermatology responses."""

from __future__ import annotations

import asyncio
import base64
from collections import Counter
from dataclasses import asdict, is_dataclass
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
from src.inference.base import InferenceBackend, InferenceRequest
from src.inference.factory import create_backend


JUDGE_MODEL_ID = "gpt_5_6_luna"
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


def _judge_paths(run_directory: Path) -> dict[str, Path]:
    return {
        "manifest": run_directory / "judge_manifest.yaml",
        "judgments": run_directory / "judgments.jsonl",
        "metrics": run_directory / "judge_metrics.json",
        "report": run_directory / "judge_report.html",
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_yaml(path: Path, value: Any) -> None:
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
    return dict(parsed.decoded)


async def _judge_one(
    *,
    backend: InferenceBackend,
    request: InferenceRequest,
    schema: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, JUDGE_RETRIES + 1):
        try:
            async with semaphore:
                result = await backend.acomplete(request)
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
        except Exception as exc:
            last_error = exc
    return {
        "status": "judge_error",
        "attempts": JUDGE_RETRIES,
        "error": f"{type(last_error).__name__}: {last_error}",
    }


def _model_failure_judgment(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": prediction["task_id"],
        "sample_id": prediction.get("sample_id"),
        "status": "model_failure",
        "target_status": prediction.get("status"),
        "attempts": 0,
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
) -> dict[str, Any]:
    latest = list(_latest_by_task(judgments).values())
    valid = [item for item in latest if item.get("status") in {"ok", "model_failure"}]
    if len(valid) != len(latest):
        raise ValueError("Judge metrics require every task to have a terminal judgment")
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
    return {
        "total": total,
        "judge_model_id": JUDGE_MODEL_ID,
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
) -> None:
    latest = _latest_by_task(judgments)
    cards = "".join(
        f"<div><strong>{key.replace('_', ' ')}</strong><span>{value}</span></div>"
        for key, value in metrics.items()
        if not isinstance(value, dict)
    )
    rows: list[str] = []
    import html
    for task_id in sorted(latest):
        item = latest[task_id]
        prediction = predictions[task_id]
        reference = references[task_id]
        response = prediction.get("response", {})
        judgment = item.get("judgment", {})
        rows.append(
            "<article>"
            f"<img src='{_thumbnail(tasks[task_id]['image'])}' alt='benchmark image'>"
            "<section>"
            f"<h2>{html.escape(str(task_id))}</h2>"
            f"<p><b>Reference:</b> {html.escape(str(reference['reference_disease_name']))}</p>"
            f"<h3>Model response</h3><pre>{html.escape(str(response.get('final_text', '')))}</pre>"
            f"<h3>Judge</h3><pre>{html.escape(json.dumps(judgment, ensure_ascii=False, indent=2))}</pre>"
            "</section></article>"
        )
    document = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<title>Open-ended diagnosis judge report</title><style>
body{{font:14px system-ui;margin:0;background:#f4f6f8;color:#17212b}}header,main{{padding:24px;max-width:1400px;margin:auto}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}}.metrics div,article{{background:white;border:1px solid #dce2e7;border-radius:10px;padding:14px}}
.metrics strong,.metrics span{{display:block}}.metrics span{{font-size:20px;margin-top:6px}}article{{display:grid;grid-template-columns:240px 1fr;gap:18px;margin:16px 0}}img{{width:240px;height:180px;object-fit:contain;background:#eee}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f9fa;padding:12px;border-radius:8px}}@media(max-width:700px){{article{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Open-ended diagnosis — single Luna judge</h1><div class='metrics'>{cards}</div></header><main>{''.join(rows)}</main></body></html>"""
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
) -> dict[str, Any]:
    root = root.resolve()
    run_directory = run_directory.resolve()
    paths = _judge_paths(run_directory)
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
    model = load_model_config(JUDGE_MODEL_ID, root=root)
    identity = {
        "judge_model_id": JUDGE_MODEL_ID,
        "judge_model_config_sha256": file_sha256(model.config_path),
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
        "judge": {"model_id": JUDGE_MODEL_ID, "single_judge": True},
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
            "judge_model_id": JUDGE_MODEL_ID,
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
        if item.get("status") in {"ok", "model_failure"}
    }
    generation = _generation(model)
    owns_backend = backend is None
    judge_backend = backend or create_backend(
        model,
        reasoning_capture="none",
        use_json_schema=True,
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
        return task_id, await _judge_one(
            backend=judge_backend,
            request=request,
            schema=schema,
            semaphore=semaphore,
        )

    try:
        for task_id, prediction in predictions.items():
            if task_id in completed:
                continue
            response = prediction.get("response", {})
            final_text = response.get("final_text", "") if isinstance(response, dict) else ""
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

    metrics = compute_judge_metrics(judgments)
    _atomic_json(paths["metrics"], metrics)
    _write_report(
        path=paths["report"],
        metrics=metrics,
        judgments=judgments,
        predictions=predictions,
        tasks=tasks,
        references=references,
    )
    document = _load_yaml(paths["manifest"])
    document["status"] = "completed"
    document["finished_at"] = datetime.now(timezone.utc).isoformat()
    document["counts"] = {"total": len(latest), "ok": len(latest)}
    _atomic_yaml(paths["manifest"], document)
    return {
        "status": "completed",
        "judge_model_id": JUDGE_MODEL_ID,
        "judgments_path": str(paths["judgments"]),
        "metrics_path": str(paths["metrics"]),
        "report_path": str(paths["report"]),
        "metrics": metrics,
    }
