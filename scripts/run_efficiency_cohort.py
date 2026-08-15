#!/usr/bin/env python3
"""Run one model through the fixed same-hardware efficiency cohort.

The script deliberately separates unmeasured 1-case and 10-case gates from
the 400 measured requests. Full execution is refused until both gates exist
and pass integrity, transport, image, and streamed-timing checks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from src.benchmark.efficiency import BenchmarkResourceMonitor
from src.benchmark.efficiency_report import build_efficiency_report
from src.benchmark.results import file_sha256
from src.config import ModelConfig, load_model_config


@dataclass(frozen=True, slots=True)
class CohortTask:
    """One measured task and its exact paired task-ID file."""

    selector: str
    benchmark_id: str
    task_ids_file: str


COHORT_TASKS = (
    CohortTask(
        "visual_top_k_closed_set",
        "visual_top_k_closed_set",
        "visual_top_k_100_cases.task_ids.txt",
    ),
    CohortTask(
        "visual_disease_confusion_sets",
        "visual_disease_confusion_sets",
        "visual_confusion_sets_50_pairs.task_ids.txt",
    ),
    CohortTask(
        "evidence_grounded_diagnosis",
        "evidence_grounded_diagnosis",
        "evidence_grounded_diagnosis_100_cases.task_ids.txt",
    ),
    CohortTask(
        "open_ended_diagnosis",
        "open_ended_diagnosis",
        "open_ended_diagnosis_100_cases.task_ids.txt",
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse the public experiment interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--parameters-billions", type=float, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--cohort-root",
        type=Path,
        default=Path("data/benchmarks/ISEPDermaBench/metadata/efficiency_cohort_v1"),
    )
    parser.add_argument(
        "--mode", choices=("dry-run", "smoke-1", "smoke-10", "full"), required=True
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--server-pid-file", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument(
        "--server-model-path",
        type=Path,
        help="Local weight directory used by vLLM, recorded for provenance only.",
    )
    parser.add_argument("--resource-sample-interval", type=float, default=0.5)
    parser.add_argument("--idle-baseline-seconds", type=float, default=10.0)
    parser.add_argument(
        "--hardware", default="NVIDIA RTX PRO 6000 Blackwell Server Edition"
    )
    parser.add_argument("--gpu-hourly-cost-usd", type=float)
    return parser.parse_args()


def endpoint_model(base_url: str) -> set[str]:
    """Return the served model IDs after health and registry validation."""

    try:
        with urlopen(base_url.removesuffix("/v1") + "/health", timeout=5) as response:
            if not 200 <= response.status < 300:
                return set()
        with urlopen(base_url.rstrip("/") + "/models", timeout=10) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, ValueError):
        return set()
    data = payload.get("data", []) if isinstance(payload, dict) else []
    return {
        str(item["id"])
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


def validate_inputs(
    model: ModelConfig, cohort_root: Path, parameters_billions: float
) -> tuple[dict[str, object], str]:
    """Fail closed on mutable model or cohort inputs."""

    if parameters_billions <= 0:
        raise ValueError("--parameters-billions must be positive")
    revisions = [model.source.revision]
    if model.processor is not None:
        revisions.append(model.processor.revision)
    if any(not _is_commit_sha(value) for value in revisions):
        raise ValueError("model and processor revisions must be 40-character SHAs")
    manifest_path = cohort_root / "manifest.json"
    manifest = _read_object(manifest_path)
    if manifest.get("measured_request_count") != 400:
        raise ValueError("cohort manifest must declare exactly 400 measured requests")
    for section_name in ("warmup", "smoke_gate"):
        section = _mapping(manifest.get(section_name))
        _validate_manifest_file(cohort_root, section)
    for value in _mapping(manifest.get("tasks")).values():
        _validate_manifest_file(cohort_root, _mapping(value))
    return manifest, file_sha256(manifest_path)


def run_benchmark(
    *,
    root: Path,
    model_path: Path,
    model_id: str,
    output_root: Path,
    base_url: str,
    task: CohortTask,
    task_ids_path: Path,
    batch_size: int,
    dry_run: bool,
) -> list[str]:
    """Execute or resume one exact paired benchmark invocation."""

    state, existing = _existing_run(output_root, task.benchmark_id, model_id)
    if state == "complete" and not dry_run:
        print(f"SKIP complete: {task.benchmark_id} / {existing}", flush=True)
        return []
    command = [
        sys.executable,
        "-m",
        "src.benchmark.cli",
        "run",
        "--model",
        str(model_path),
        "--benchmark",
        task.selector,
        "--evaluation-set",
        "internal_benchmark",
        "--benchmark-source",
        "local",
        "--base-url",
        base_url,
        "--thinking-mode",
        "disabled",
        "--temperature",
        "0",
        "--batch-size",
        str(batch_size),
        "--seed",
        "42",
        "--reasoning-capture",
        "available",
        "--task-ids-file",
        str(task_ids_path),
    ]
    if dry_run:
        command.extend(("--dry-run", "--output-root", str(output_root)))
    elif state == "resume" and existing is not None:
        command.extend(("--resume", str(existing)))
    else:
        command.extend(("--output-root", str(output_root)))
    print("COMMAND " + json.dumps(command), flush=True)
    subprocess.run(command, cwd=root, check=True)
    return command


def validate_smoke(output_root: Path, model_id: str, expected: int) -> None:
    """Require complete image inference and streamed timing for every gate case."""

    _, run = _existing_run(output_root, "visual_top_k_closed_set", model_id)
    if run is None or not (run / "metrics.json").is_file():
        raise RuntimeError("smoke run did not produce metrics.json")
    records = [
        json.loads(line)
        for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != expected:
        raise RuntimeError(f"smoke expected {expected} records, found {len(records)}")
    failures: list[str] = []
    for record in records:
        status = record.get("status")
        response = _mapping(record.get("response"))
        timing = _mapping(_mapping(response.get("provider_metadata")).get("timing"))
        if status in {"backend_error", "image_error"}:
            failures.append(str(record.get("task_id")))
        if not isinstance(timing.get("time_to_first_token_seconds"), (int, float)):
            failures.append(str(record.get("task_id")) + ":missing_ttft")
    if failures:
        raise RuntimeError("smoke gate failed: " + ", ".join(failures))


def main() -> int:
    """Run a dry-run, smoke gate, or fully measured cohort for one model."""

    args = parse_args()
    if args.batch_size < 1 or args.resource_sample_interval <= 0:
        raise ValueError("batch size and sample interval must be positive")
    if args.idle_baseline_seconds < 0:
        raise ValueError("idle baseline cannot be negative")
    root = args.project_root.resolve()
    cohort_root = (root / args.cohort_root).resolve()
    output_root = (root / args.output_root).resolve()
    model = load_model_config(args.model, root=root)
    manifest, cohort_sha = validate_inputs(model, cohort_root, args.parameters_billions)
    source_repo = model.source.repo_id
    if source_repo is None:
        raise ValueError("efficiency cohort requires a Hugging Face model source")
    model_env = model.backend.active_profile.model_env
    if model_env:
        os.environ[model_env] = source_repo
    if args.mode != "dry-run" and source_repo not in endpoint_model(args.base_url):
        raise RuntimeError(f"endpoint does not serve pinned source {source_repo!r}")

    model_path = model.config_path.resolve()
    commands: list[list[str]] = []
    if args.mode in {"smoke-1", "smoke-10"}:
        count = 1 if args.mode == "smoke-1" else 10
        section = "warmup" if count == 1 else "smoke_gate"
        task_ids = cohort_root / str(_mapping(manifest[section])["task_ids_file"])
        commands.append(
            run_benchmark(
                root=root,
                model_path=model_path,
                model_id=model.model_id,
                output_root=output_root,
                base_url=args.base_url,
                task=COHORT_TASKS[0],
                task_ids_path=task_ids,
                batch_size=1,
                dry_run=False,
            )
        )
        validate_smoke(output_root, model.model_id, count)
        _write_status(output_root, "completed", model.model_id, [args.mode])
        return 0

    if args.mode == "full":
        gates_root = output_root.parent / "_gates" / model.model_id
        validate_smoke(gates_root / "smoke-1", model.model_id, 1)
        validate_smoke(gates_root / "smoke-10", model.model_id, 10)

    server_pid = _read_pid(args.server_pid_file) if args.server_pid_file else None
    monitor: BenchmarkResourceMonitor | None = None
    completed: list[str] = []
    _write_provenance(
        output_root,
        model,
        manifest,
        cohort_sha,
        args,
        server_pid,
    )
    if args.mode == "full":
        monitor = BenchmarkResourceMonitor(
            output_root=output_root,
            server_pid=server_pid,
            interval_seconds=args.resource_sample_interval,
        )
        monitor.start()
        monitor.set_phase("idle_baseline")
        time.sleep(args.idle_baseline_seconds)
    _write_status(output_root, "running", model.model_id, completed)
    try:
        for task in COHORT_TASKS:
            if monitor is not None:
                monitor.set_phase(task.benchmark_id)
            commands.append(
                run_benchmark(
                    root=root,
                    model_path=model_path,
                    model_id=model.model_id,
                    output_root=output_root,
                    base_url=args.base_url,
                    task=task,
                    task_ids_path=cohort_root / task.task_ids_file,
                    batch_size=args.batch_size,
                    dry_run=args.mode == "dry-run",
                )
            )
            completed.append(task.benchmark_id)
            _write_status(output_root, "running", model.model_id, completed)
            if monitor is not None:
                monitor.set_phase("between_tasks")
    except Exception:
        _write_status(output_root, "failed", model.model_id, completed)
        raise
    finally:
        if monitor is not None:
            monitor.stop()
    if args.mode == "full":
        build_efficiency_report(
            output_root,
            model_id=model.model_id,
            model=model.model.display_name,
            parameters_billions=args.parameters_billions,
            dtype=model.dtype.upper(),
            hardware=args.hardware,
            concurrency=args.batch_size,
            gpu_hourly_cost_usd=args.gpu_hourly_cost_usd,
            cohort_manifest_sha256=cohort_sha,
        )
        if args.server_log and args.server_log.is_file():
            destination = output_root / "manifests" / "vllm_server.log"
            shutil.copy2(args.server_log, destination)
    _write_commands(output_root, commands)
    _write_status(output_root, "completed", model.model_id, completed)
    return 0


def _write_provenance(
    output_root: Path,
    model: ModelConfig,
    cohort: dict[str, object],
    cohort_sha: str,
    args: argparse.Namespace,
    server_pid: int | None,
) -> None:
    manifests = output_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model.config_path, manifests / "model_config.yaml")
    (manifests / "cohort_manifest.json").write_text(
        json.dumps(cohort, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    server_model_manifest_sha256: str | None = None
    if args.server_model_path is not None:
        model_files_path = manifests / "server_model_files.json"
        _write_server_model_manifest(args.server_model_path.resolve(), model_files_path)
        server_model_manifest_sha256 = file_sha256(model_files_path)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "model_id": model.model_id,
        "model_repo": model.source.repo_id,
        "model_revision": model.source.revision,
        "processor_revision": model.processor.revision if model.processor else None,
        "cohort_manifest_sha256": cohort_sha,
        "temperature": 0.0,
        "thinking": "disabled",
        "concurrency": args.batch_size,
        "base_url": args.base_url,
        "server_pid": server_pid,
        "server_model_path": (
            str(args.server_model_path.resolve())
            if args.server_model_path is not None
            else None
        ),
        "server_model_files_manifest_sha256": server_model_manifest_sha256,
        "hardware": args.hardware,
        "runtime_environment": {
            "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get(
                "VLLM_USE_FLASHINFER_SAMPLER"
            ),
        },
    }
    _atomic_json(manifests / "campaign_manifest.json", payload)


def _write_server_model_manifest(model_path: Path, output_path: Path) -> None:
    """Hash every non-hidden file in a local standalone model directory."""

    if not (model_path / "config.json").is_file():
        raise ValueError("local server model does not contain config.json")
    files: list[dict[str, object]] = []
    for path in sorted(model_path.rglob("*")):
        relative = path.relative_to(model_path)
        if not path.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not files or not any(
        str(item["path"]).endswith(".safetensors") for item in files
    ):
        raise ValueError("local server model does not contain safetensors weights")
    _atomic_json(
        output_path,
        {
            "schema_version": 1,
            "model_path": str(model_path),
            "file_count": len(files),
            "total_bytes": sum(int(item["bytes"]) for item in files),
            "files": files,
        },
    )


def _write_status(
    output_root: Path, status: str, model_id: str, completed: list[str]
) -> None:
    _atomic_json(
        output_root / "campaign_status.json",
        {"status": status, "model_id": model_id, "completed_benchmarks": completed},
    )


def _write_commands(output_root: Path, commands: list[list[str]]) -> None:
    _atomic_json(output_root / "manifests" / "commands.json", commands)


def _existing_run(
    output_root: Path, benchmark_id: str, model_id: str
) -> tuple[str, Path | None]:
    parent = output_root / benchmark_id / model_id
    if not parent.is_dir():
        return "new", None
    runs = sorted(
        (path for path in parent.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run in runs:
        if (run / "metrics.json").is_file():
            return "complete", run
    for run in runs:
        if (run / "run_manifest.yaml").is_file():
            return "resume", run
    return "new", None


def _validate_manifest_file(root: Path, section: dict[str, object]) -> None:
    path = root / str(section.get("task_ids_file", ""))
    expected = section.get("sha256")
    if not path.is_file() or not isinstance(expected, str):
        raise ValueError(f"invalid cohort manifest entry: {section}")
    if file_sha256(path) != expected:
        raise ValueError(f"cohort artifact hash mismatch: {path}")


def _read_pid(path: Path) -> int:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw.isdigit():
        raise ValueError(f"invalid server PID file: {path}")
    return int(raw)


def _is_commit_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
