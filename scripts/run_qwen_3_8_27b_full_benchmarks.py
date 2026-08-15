#!/usr/bin/env python3
"""Run the frozen Qwen 3.8 27B ISEPDermaBench and DermoBench campaign.

The campaign covers the five held-out ISEP tasks used for cross-model
comparison and every public DermoBench task. It generates open-ended answers
but deliberately leaves LLM judging to the repository's dedicated judge
commands after the raw predictions have been preserved.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from src.benchmark.efficiency import BenchmarkResourceMonitor
from src.benchmark.efficiency_report import build_efficiency_report


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """One benchmark invocation in the frozen campaign."""

    selector: str
    benchmark_id: str
    evaluation_set: str
    source: str


ISEP_INTERNAL_TASKS = (
    BenchmarkSpec(
        "visual_top_k_closed_set",
        "visual_top_k_closed_set",
        "internal_benchmark",
        "local",
    ),
    BenchmarkSpec(
        "clinical_context_ablation",
        "clinical_context_ablation",
        "internal_benchmark",
        "local",
    ),
    BenchmarkSpec(
        "visual_disease_confusion_sets",
        "visual_disease_confusion_sets",
        "internal_benchmark",
        "local",
    ),
    BenchmarkSpec(
        "evidence_grounded_diagnosis",
        "evidence_grounded_diagnosis",
        "internal_benchmark",
        "local",
    ),
    BenchmarkSpec(
        "open_ended_diagnosis",
        "open_ended_diagnosis",
        "internal_benchmark",
        "local",
    ),
)

DERMOBENCH_TASK_IDS = (
    "task_1_1_description_without_morphology",
    "task_1_2_description_with_morphology",
    "task_1_3_derm7pt_morphology_mcq",
    "task_1_4_skincon_morphology_mcq",
    "task_2_1_diagnosis_mcq_25_choices",
    "task_2_1_diagnosis_mcq_4_choices",
    "task_2_1_ddi_diagnosis_mcq",
    "task_2_1_derm1m_edu_diagnosis_mcq",
    "task_2_1_derm7pt_diagnosis_mcq",
    "task_2_1_snu134_diagnosis_mcq",
    "task_3_1_diagnostic_reasoning_without_morphology",
    "task_3_2_diagnostic_reasoning_with_morphology",
    "task_4_ddi_fairness_mcq",
)

DERMOBENCH_TASKS = tuple(
    BenchmarkSpec(
        f"dermobench/{task_id}",
        f"dermobench_{task_id}",
        "filtered",
        "local",
    )
    for task_id in DERMOBENCH_TASK_IDS
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/qwen_3_8_27b_temp0_full_campaign"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--server-pid-file", type=Path)
    parser.add_argument("--resource-sample-interval", type=float, default=0.5)
    parser.add_argument("--idle-baseline-seconds", type=float, default=10.0)
    parser.add_argument(
        "--hardware",
        default="NVIDIA RTX PRO 6000 Blackwell",
        help="Stable hardware label written to thesis tables.",
    )
    parser.add_argument(
        "--suite",
        choices=("all", "isep", "dermobench"),
        default="all",
    )
    return parser.parse_args()


def endpoint_ready(base_url: str) -> bool:
    """Return whether the local endpoint health check succeeds."""

    try:
        with urlopen(base_url.removesuffix("/v1") + "/health", timeout=5) as response:
            return 200 <= response.status < 300
    except (URLError, TimeoutError):
        return False


def find_existing_run(output_root: Path, benchmark_id: str) -> tuple[str, Path | None]:
    """Return the complete, resumable, or new state for one task."""

    parent = output_root / benchmark_id / "qwen_3_8_27b"
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


def run_one(
    *,
    root: Path,
    output_root: Path,
    spec: BenchmarkSpec,
    base_url: str,
    batch_size: int,
    limit: int | None,
    dry_run: bool,
) -> None:
    """Run, resume, skip, or validate one benchmark task."""

    state, existing = find_existing_run(output_root, spec.benchmark_id)
    if state == "complete" and not dry_run:
        print(f"SKIP complete: {spec.benchmark_id} / {existing}", flush=True)
        return

    command = [
        sys.executable,
        "-m",
        "src.benchmark.cli",
        "run",
        "--model",
        "qwen_3_8_27b",
        "--benchmark",
        spec.selector,
        "--evaluation-set",
        spec.evaluation_set,
        "--benchmark-source",
        spec.source,
        "--base-url",
        base_url,
        "--thinking-mode",
        "disabled",
        "--temperature",
        "0",
        "--batch-size",
        str(batch_size),
        "--reasoning-capture",
        "available",
    ]
    if limit is not None:
        command.extend(("--limit", str(limit)))
    if dry_run:
        command.extend(("--dry-run", "--output-root", str(output_root)))
        action = "DRY RUN"
    elif state == "resume" and existing is not None:
        command.extend(("--resume", str(existing)))
        action = "RESUME"
    else:
        command.extend(("--output-root", str(output_root)))
        action = "START"
    print(f"{action}: {spec.benchmark_id}", flush=True)
    subprocess.run(command, cwd=root, check=True)


def write_campaign_status(
    output_root: Path, status: str, completed: list[str], current: str | None
) -> None:
    """Persist a concise machine-readable campaign status."""

    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "temperature": 0.0,
        "thinking": "disabled",
        "status": status,
        "completed_benchmarks": completed,
        "current_benchmark": current,
    }
    temporary = output_root / "campaign_status.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_root / "campaign_status.json")


def main() -> int:
    """Execute the selected frozen benchmark suite."""

    args = parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("batch size and limit must be positive")
    if args.resource_sample_interval <= 0 or args.idle_baseline_seconds < 0:
        raise ValueError("resource interval must be positive and baseline non-negative")
    if not args.dry_run and not endpoint_ready(args.base_url):
        raise RuntimeError(f"Endpoint is not ready: {args.base_url}")

    root = args.project_root.resolve()
    output_root = (root / args.output_root).resolve()
    specs = {
        "all": ISEP_INTERNAL_TASKS + DERMOBENCH_TASKS,
        "isep": ISEP_INTERNAL_TASKS,
        "dermobench": DERMOBENCH_TASKS,
    }[args.suite]
    completed: list[str] = []
    monitor: BenchmarkResourceMonitor | None = None
    if not args.dry_run:
        server_pid: int | None = None
        if args.server_pid_file is not None:
            raw_pid = args.server_pid_file.read_text(encoding="utf-8").strip()
            if not raw_pid.isdigit():
                raise ValueError("server PID file does not contain a valid PID")
            server_pid = int(raw_pid)
        monitor = BenchmarkResourceMonitor(
            output_root=output_root,
            server_pid=server_pid,
            interval_seconds=args.resource_sample_interval,
        )
        monitor.start()
        monitor.set_phase("idle_baseline")
        time.sleep(args.idle_baseline_seconds)
    write_campaign_status(output_root, "running", completed, None)
    try:
        for index, spec in enumerate(specs, start=1):
            print(f"[{index:02d}/{len(specs):02d}]", end=" ", flush=True)
            write_campaign_status(output_root, "running", completed, spec.benchmark_id)
            if monitor is not None:
                monitor.set_phase(spec.benchmark_id)
            run_one(
                root=root,
                output_root=output_root,
                spec=spec,
                base_url=args.base_url,
                batch_size=args.batch_size,
                limit=args.limit,
                dry_run=args.dry_run,
            )
            completed.append(spec.benchmark_id)
            if monitor is not None:
                monitor.set_phase("between_tasks")
    except Exception:
        write_campaign_status(output_root, "failed", completed, spec.benchmark_id)
        raise
    finally:
        if monitor is not None:
            monitor.stop()
    if not args.dry_run:
        build_efficiency_report(
            output_root,
            model_id="qwen_3_8_27b",
            model="Qwen 3.8 27B",
            parameters_billions=27.781427952,
            dtype="BF16",
            hardware=args.hardware,
        )
    write_campaign_status(
        output_root,
        "dry_run" if args.dry_run else "completed",
        completed,
        None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
