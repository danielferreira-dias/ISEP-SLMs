#!/usr/bin/env python3
"""Run the paired E2 ISEPDermaBench campaign without external judging.

The controller selects only each condition's sft_dev best checkpoint, merges it
with the pinned base into a standalone BF16 model, starts a loopback-only vLLM
endpoint, applies dry/1/10-case gates, and then runs the four frozen internal
cohorts.  It never retries or resumes a failed stage and never invokes a judge.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
BASE_URL = "http://127.0.0.1:8002/v1"
CAMPAIGN_ATTEMPT = os.environ.get("ISEP_E2_BENCHMARK_ATTEMPT", "attempt-1")
OUTPUT_ROOT = Path(
    os.environ.get(
        "ISEP_E2_BENCHMARK_OUTPUT_ROOT",
        "outputs/e2_internal_benchmark_historical_t06",
    )
)
MERGED_ROOT = Path(
    os.environ.get("ISEP_E2_MERGED_ROOT", "outputs/merged_models")
)
EXPECTED_COUNTS = {
    "visual_top_k_closed_set": 1000,
    "visual_disease_confusion_sets": 828,
    "evidence_grounded_diagnosis": 134,
    "open_ended_diagnosis": 300,
}


@dataclass(frozen=True, slots=True)
class Condition:
    """One paired E2 training condition and its serving identity."""

    key: str
    model_id: str
    request_model: str
    environment_name: str
    run_directory: Path
    merged_directory: Path


CONDITIONS = (
    Condition(
        key="frozen",
        model_id="qwen_3_5_4b_e2_frozen_vision_t06",
        request_model="qwen_3_5_4b_e2_frozen_vision",
        environment_name="ISEP_E2_FROZEN_REQUEST_MODEL",
        run_directory=Path(
            "outputs/training/e2_skincon_skincap_frozen_vision/"
            "full-l40s-e2-skincap-frozen-seed42-20260816"
        ),
        merged_directory=(
            MERGED_ROOT / "qwen_3_5_4b_e2_frozen_vision_epoch3"
        ),
    ),
    Condition(
        key="vision",
        model_id="qwen_3_5_4b_e2_vision_lora_t06",
        request_model="qwen_3_5_4b_e2_vision_lora",
        environment_name="ISEP_E2_VISION_REQUEST_MODEL",
        run_directory=Path(
            "outputs/training/e2_skincon_skincap_unsloth_all/"
            "full-l40s-e2-skincap-vision-seed42-20260816"
        ),
        merged_directory=(MERGED_ROOT / "qwen_3_5_4b_e2_vision_lora_epoch3"),
    ),
)


def utc_now() -> str:
    """Return an auditable UTC timestamp."""

    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_status(root: Path, *, status: str, detail: str) -> None:
    """Persist campaign state without clinical content."""

    write_json(
        root / OUTPUT_ROOT / "campaign_status.json",
        {"status": status, "detail": detail, "updated_at_utc": utc_now()},
    )


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object or fail closed."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def file_sha256(path: Path) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(
    directory: Path,
    *,
    excluded_relative_paths: set[str] | None = None,
) -> str:
    """Hash relative paths, sizes, and contents in stable order."""

    excluded = excluded_relative_paths or set()
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative in excluded:
            continue
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(path.stat().st_size).encode("ascii") + b"\0")
        digest.update(file_sha256(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def validate_best_checkpoint(root: Path, condition: Condition) -> Path:
    """Require the sft_dev-selected third-epoch checkpoint."""

    run = root / condition.run_directory
    status = read_json(run / "manifests/status.json")
    best = read_json(run / "manifests/best_checkpoint.json")
    uploads = json.loads(
        (run / "manifests/checkpoint_uploads.json").read_text(encoding="utf-8")
    )
    if status.get("status") != "completed":
        raise RuntimeError(f"Training run is not completed: {run}")
    if best.get("selection_metric") != "diagnosis_macro_f1":
        raise RuntimeError(f"Unexpected selection metric for {condition.key}")
    if best.get("checkpoint_id") != "checkpoint-4557" or best.get("epoch") != 3.0:
        raise RuntimeError(f"Unexpected best checkpoint for {condition.key}: {best}")
    if not isinstance(uploads, list) or len(uploads) != 3:
        raise RuntimeError(f"Expected three uploads for {condition.key}")
    if {item.get("epoch") for item in uploads} != {1, 2, 3}:
        raise RuntimeError(f"Incomplete checkpoint uploads for {condition.key}")
    checkpoint = run / "checkpoints/checkpoint-4557"
    if not (checkpoint / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing adapter weights: {checkpoint}")
    return checkpoint


def gpu_memory_used_mib() -> int:
    """Return current GPU memory allocation."""

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def require_free_gpu() -> None:
    """Refuse to start when another workload owns the GPU."""

    used = gpu_memory_used_mib()
    if used >= 2_000:
        raise RuntimeError(f"GPU is not free: {used} MiB allocated")


def canonical_manifest_is_valid(payload: dict[str, Any]) -> bool:
    """Return whether canonical weights exactly match the pinned base keyset."""

    return (
        payload.get("status") == "validated"
        and payload.get("base_model") == BASE_MODEL
        and payload.get("base_revision") == BASE_REVISION
        and payload.get("official_tensor_count") == 738
        and payload.get("stored_tensor_count") == 738
        and payload.get("official_keyset_sha256")
        == payload.get("stored_keyset_sha256")
    )


def validate_standalone(
    root: Path,
    condition: Condition,
    adapter: Path,
    output: Path,
) -> None:
    """Revalidate an immutable standalone model and its full provenance."""

    if not output.is_dir():
        raise FileNotFoundError(f"Standalone model does not exist: {output}")
    canonical_path = output / "canonical_weight_manifest.json"
    provenance_path = output / "isep_e2_standalone_manifest.json"
    canonical = read_json(canonical_path)
    provenance = read_json(provenance_path)
    if not canonical_manifest_is_valid(canonical):
        raise RuntimeError(f"Invalid canonical weights in {output}: {canonical}")
    expected_provenance = {
        "condition": condition.key,
        "model_id": condition.model_id,
        "request_model": condition.request_model,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "adapter_path": str(adapter.relative_to(root)),
        "adapter_tree_sha256": tree_sha256(adapter),
        "canonical_weight_manifest_sha256": file_sha256(canonical_path),
    }
    mismatches = {
        key: {"expected": value, "actual": provenance.get(key)}
        for key, value in expected_provenance.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Standalone provenance mismatch: {mismatches}")
    weights = provenance.get("weight_files")
    if not isinstance(weights, list) or not weights:
        raise RuntimeError(f"Missing weight-file provenance in {provenance_path}")
    expected_names: set[str] = set()
    for item in weights:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise RuntimeError(f"Invalid weight-file provenance: {item}")
        path = output / item["path"]
        expected_names.add(path.name)
        if (
            not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or file_sha256(path) != item.get("sha256")
        ):
            raise RuntimeError(f"Standalone weight integrity failed: {path}")
    actual_names = {path.name for path in output.glob("*.safetensors")}
    if actual_names != expected_names:
        raise RuntimeError(
            f"Standalone weight set mismatch: {actual_names} != {expected_names}"
        )
    actual_tree = tree_sha256(
        output,
        excluded_relative_paths={"isep_e2_standalone_manifest.json"},
    )
    if actual_tree != provenance.get("standalone_tree_sha256_before_manifest"):
        raise RuntimeError(f"Standalone tree integrity failed: {output}")


def standalone_override(root: Path, condition: Condition) -> Path | None:
    """Resolve an explicitly supplied, repository-local standalone model."""

    value = os.environ.get(f"ISEP_E2_{condition.key.upper()}_STANDALONE")
    if not value:
        return None
    output = (root / value).resolve()
    if not output.is_relative_to(root):
        raise RuntimeError(f"Standalone override escapes repository: {output}")
    return output


def merge_checkpoint(root: Path, condition: Condition, adapter: Path) -> Path:
    """Create one immutable standalone model and provenance manifest."""

    reused = standalone_override(root, condition)
    if reused is not None:
        validate_standalone(root, condition, adapter, reused)
        return reused
    output = root / condition.merged_directory
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite standalone model: {output}")
    command = [
        str(root / ".venv-training/bin/python"),
        "scripts/merge_e1_adapter.py",
        "--base-model",
        BASE_MODEL,
        "--revision",
        BASE_REVISION,
        "--adapter",
        str(adapter),
        "--output",
        str(output),
    ]
    subprocess.run(command, cwd=root, check=True)
    weights = sorted(output.glob("*.safetensors"))
    if not weights or not (output / "config.json").is_file():
        raise RuntimeError(f"Standalone merge is incomplete: {output}")
    canonical = read_json(output / "canonical_weight_manifest.json")
    if not canonical_manifest_is_valid(canonical):
        raise RuntimeError(
            f"Standalone canonical-weight validation failed: {canonical}"
        )
    manifest = {
        "campaign_attempt": CAMPAIGN_ATTEMPT,
        "condition": condition.key,
        "model_id": condition.model_id,
        "request_model": condition.request_model,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "adapter_path": str(adapter.relative_to(root)),
        "adapter_tree_sha256": tree_sha256(adapter),
        "standalone_tree_sha256_before_manifest": tree_sha256(output),
        "canonical_weight_manifest_sha256": file_sha256(
            output / "canonical_weight_manifest.json"
        ),
        "weight_files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in weights
        ],
        "created_at_utc": utc_now(),
    }
    write_json(output / "isep_e2_standalone_manifest.json", manifest)
    validate_standalone(root, condition, adapter, output)
    return output


def process_exists(pid: int) -> bool:
    """Return whether a non-zombie process is alive."""

    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file() and stat.read_text().split()[2] == "Z":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def endpoint_model() -> str | None:
    """Return the sole advertised endpoint model when healthy."""

    try:
        with urlopen(BASE_URL.removesuffix("/v1") + "/health", timeout=5) as reply:
            if not 200 <= reply.status < 300:
                return None
        with urlopen(BASE_URL + "/models", timeout=5) as reply:
            payload = json.load(reply)
    except (URLError, TimeoutError, json.JSONDecodeError):
        return None
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list) or len(models) != 1:
        return None
    item = models[0]
    return item.get("id") if isinstance(item, dict) else None


def start_server(root: Path, condition: Condition, model_path: Path) -> int:
    """Start one loopback-only vLLM server and await exact identity."""

    require_free_gpu()
    runtime_bin = root / ".venv/bin"
    ninja = runtime_bin / "ninja"
    if not ninja.is_file() or not os.access(ninja, os.X_OK):
        raise RuntimeError(f"Pinned ninja executable is unavailable: {ninja}")
    condition_root = root / OUTPUT_ROOT / condition.key
    log_path = condition_root / "vllm.log"
    command = [
        str(root / ".venv/bin/vllm"),
        "serve",
        str(model_path),
        "--served-model-name",
        condition.request_model,
        "--host",
        "127.0.0.1",
        "--port",
        "8002",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.90",
        "--limit-mm-per-prompt",
        '{"image":1}',
        "--reasoning-parser",
        "qwen3",
        "--max-num-seqs",
        "8",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
        "--gdn-prefill-backend",
        "triton",
    ]
    environment = os.environ.copy()
    environment["PATH"] = f"{runtime_bin}:{environment.get('PATH', '')}"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (condition_root / "vllm.pid").write_text(f"{process.pid}\n")
    for _ in range(90):
        if process.poll() is not None:
            raise RuntimeError(
                f"{condition.key} vLLM exited with {process.returncode}"
            )
        if endpoint_model() == condition.request_model:
            return process.pid
        time.sleep(10)
    raise RuntimeError(f"{condition.key} vLLM was not ready within 15 minutes")


def stop_server(pid: int, model_path: Path) -> None:
    """Stop only the exact vLLM server started for this standalone model."""

    if not process_exists(pid):
        return
    command_path = Path(f"/proc/{pid}/cmdline")
    command = command_path.read_bytes().replace(b"\0", b" ").decode()
    if "vllm serve" not in command or str(model_path) not in command:
        raise RuntimeError(f"Refusing to stop unexpected PID {pid}: {command}")
    os.kill(pid, signal.SIGTERM)
    for _ in range(90):
        if not process_exists(pid):
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"vLLM PID {pid} did not stop after SIGTERM")
    for _ in range(60):
        if gpu_memory_used_mib() < 2_000:
            return
        time.sleep(2)
    raise RuntimeError("GPU allocation was not released")


def benchmark_command(
    root: Path,
    condition: Condition,
    benchmark: str,
    output: Path,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    batch_size: int = 8,
) -> list[str]:
    """Build one auditable benchmark invocation."""

    command = [
        str(root / ".venv/bin/python"),
        "-m",
        "src.benchmark.cli",
        "run",
        "--model",
        condition.model_id,
        "--benchmark",
        benchmark,
        "--evaluation-set",
        "internal_benchmark",
        "--benchmark-source",
        "local",
        "--base-url",
        BASE_URL,
        "--thinking-mode",
        "disabled",
        "--batch-size",
        str(batch_size),
        "--reasoning-capture",
        "available",
        "--output-root",
        str(output),
    ]
    if limit is not None:
        command.extend(("--limit", str(limit)))
    if dry_run:
        command.append("--dry-run")
    return command


def run_once(
    root: Path,
    condition: Condition,
    benchmark: str,
    output: Path,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    batch_size: int = 8,
) -> None:
    """Execute exactly one benchmark command with no retry."""

    environment = os.environ.copy()
    environment[condition.environment_name] = condition.request_model
    subprocess.run(
        benchmark_command(
            root,
            condition,
            benchmark,
            output,
            limit=limit,
            dry_run=dry_run,
            batch_size=batch_size,
        ),
        cwd=root,
        env=environment,
        check=True,
    )


def only_run_directory(output: Path, benchmark: str, model_id: str) -> Path:
    """Require exactly one generated run directory."""

    parent = output / benchmark / model_id
    runs = sorted(path for path in parent.iterdir() if path.is_dir())
    if len(runs) != 1:
        raise RuntimeError(f"Expected one run below {parent}, found {len(runs)}")
    return runs[0]


def validate_gate(output: Path, condition: Condition, expected: int) -> None:
    """Require image-bearing, parsed, schema-valid gate responses."""

    run = only_run_directory(output, "visual_top_k_closed_set", condition.model_id)
    rows = [
        json.loads(line)
        for line in (run / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != expected or not (run / "metrics.json").is_file():
        raise RuntimeError(f"Gate count mismatch: {len(rows)} != {expected}")
    for row in rows:
        response = row.get("response")
        if row.get("status") != "ok" or not isinstance(response, dict):
            raise RuntimeError("Gate contains a failed response")
        if not row.get("image_uri") or not response.get("final_text"):
            raise RuntimeError("Gate did not preserve image ingestion/output")
        if not response.get("json_valid") or not response.get("schema_valid"):
            raise RuntimeError("Gate contains invalid JSON/schema output")
        if response.get("validation_errors"):
            raise RuntimeError("Gate contains parser validation errors")


def run_gates(root: Path, condition: Condition) -> None:
    """Run dry, one-case, and ten-case gates before the full cohorts."""

    gate_root = root / OUTPUT_ROOT / "_gates" / condition.key
    if gate_root.exists():
        raise FileExistsError(f"Refusing to mix existing gates: {gate_root}")
    run_once(
        root,
        condition,
        "visual_top_k_closed_set",
        gate_root / "dry_run",
        limit=10,
        dry_run=True,
    )
    for expected, name in ((1, "one"), (10, "ten")):
        output = gate_root / name
        run_once(
            root,
            condition,
            "visual_top_k_closed_set",
            output,
            limit=expected,
            batch_size=1 if expected == 1 else 8,
        )
        validate_gate(output, condition, expected)


def validate_full(condition_root: Path, condition: Condition) -> dict[str, Any]:
    """Validate 2,262 preserved outputs and four complete manifests."""

    counts: dict[str, int] = {}
    for benchmark, expected in EXPECTED_COUNTS.items():
        run = only_run_directory(condition_root, benchmark, condition.model_id)
        predictions = run / "predictions.jsonl"
        metrics = run / "metrics.json"
        manifest = run / "run_manifest.yaml"
        if not metrics.is_file() or not manifest.is_file():
            raise RuntimeError(f"Incomplete artifacts for {benchmark}")
        count = sum(1 for line in predictions.open() if line.strip())
        if count != expected:
            raise RuntimeError(f"{benchmark}: {count} != {expected}")
        counts[benchmark] = count
        if benchmark == "open_ended_diagnosis":
            payload = read_json(metrics)
            if payload.get("total") != 300 or payload.get("judging_status") != "pending":
                raise RuntimeError("Open-ended outputs are not complete/pending judge")
    if sum(counts.values()) != 2262:
        raise RuntimeError(f"Unexpected total: {sum(counts.values())}")
    return {
        "condition": condition.key,
        "status": "completed",
        "counts": counts,
        "total": sum(counts.values()),
        "external_judge": "pending_not_submitted",
        "validated_at_utc": utc_now(),
    }


def run_full(root: Path, condition: Condition) -> dict[str, Any]:
    """Run the four frozen cohorts exactly once."""

    output = root / OUTPUT_ROOT / condition.key
    for benchmark in EXPECTED_COUNTS:
        run_once(root, condition, benchmark, output)
    result = validate_full(output, condition)
    write_json(output / "benchmark_validation.json", result)
    return result


def main() -> int:
    """Execute Frozen then Vision, failing closed at the first anomaly."""

    root = Path(__file__).resolve().parents[1]
    campaign_root = root / OUTPUT_ROOT
    if campaign_root.exists():
        raise FileExistsError(f"Refusing to mix existing campaign: {campaign_root}")
    write_status(root, status="running", detail="validating training artifacts")
    server_pid: int | None = None
    server_model: Path | None = None
    try:
        for condition in CONDITIONS:
            write_status(root, status="running", detail=f"preparing {condition.key}")
            adapter = validate_best_checkpoint(root, condition)
            condition_root = campaign_root / condition.key
            condition_root.mkdir(parents=True, exist_ok=False)
            merged = merge_checkpoint(root, condition, adapter)
            server_model = merged
            reused = standalone_override(root, condition)
            standalone_provenance = read_json(
                merged / "isep_e2_standalone_manifest.json"
            )
            write_json(
                condition_root / "benchmark_protocol.json",
                {
                    "campaign_attempt": CAMPAIGN_ATTEMPT,
                    "condition": condition.key,
                    "model_id": condition.model_id,
                    "request_model": condition.request_model,
                    "base_model": BASE_MODEL,
                    "base_revision": BASE_REVISION,
                    "adapter": str(adapter.relative_to(root)),
                    "standalone_model": str(merged.relative_to(root)),
                    "standalone_reused": reused is not None,
                    "standalone_origin_attempt": standalone_provenance.get(
                        "campaign_attempt"
                    ),
                    "hardware": "NVIDIA L40S",
                    "backend": "vllm",
                    "vllm_executable": str(root / ".venv/bin/vllm"),
                    "ninja_executable": str(root / ".venv/bin/ninja"),
                    "runtime_path_prepend": str(root / ".venv/bin"),
                    "dtype": "bfloat16",
                    "batch_size": 8,
                    "thinking": "disabled",
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "presence_penalty": 1.5,
                    "seed": 42,
                    "invalid_responses_in_denominator": True,
                    "regeneration": False,
                    "answer_repair": False,
                    "external_judge": "not_authorized_not_submitted",
                },
            )
            write_status(root, status="running", detail=f"gating {condition.key}")
            server_pid = start_server(root, condition, merged)
            run_gates(root, condition)
            write_status(root, status="running", detail=f"benchmarking {condition.key}")
            run_full(root, condition)
            stop_server(server_pid, merged)
            server_pid = None
            server_model = None
        write_status(
            root,
            status="completed",
            detail="frozen and vision each completed 2262 internal cases",
        )
    except Exception as error:
        write_status(root, status="failed", detail=f"{type(error).__name__}: {error}")
        if server_pid is not None and server_model is not None:
            stop_server(server_pid, server_model)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
