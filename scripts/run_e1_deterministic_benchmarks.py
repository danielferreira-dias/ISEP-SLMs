#!/usr/bin/env python3
"""Run all judge-free E1 benchmarks against one OpenAI-compatible endpoint.

The runner executes the deterministic ISEPDermaBench tasks followed by the
deterministic DermoBench MCQ tasks. Completed task/model combinations are
skipped and interrupted runs are resumed through the benchmark CLI so a remote
GPU interruption does not discard already persisted predictions.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """One deterministic benchmark invocation."""

    selector: str
    benchmark_id: str
    evaluation_set: str


ISEP_BENCHMARKS = (
    BenchmarkSpec(
        "visual_top_k_closed_set",
        "visual_top_k_closed_set",
        "internal_benchmark",
    ),
    BenchmarkSpec(
        "clinical_context_ablation",
        "clinical_context_ablation",
        "internal_benchmark",
    ),
    BenchmarkSpec(
        "visual_disease_confusion_sets",
        "visual_disease_confusion_sets",
        "internal_benchmark",
    ),
    BenchmarkSpec(
        "evidence_grounded_diagnosis",
        "evidence_grounded_diagnosis",
        "internal_benchmark",
    ),
    BenchmarkSpec(
        "visual_grounding_no_image",
        "visual_grounding_no_image",
        "validation",
    ),
    BenchmarkSpec(
        "general_visual_hallucination_audit",
        "general_visual_hallucination_audit",
        "validation",
    ),
    BenchmarkSpec(
        "dermatology_counterfactual_hallucination",
        "dermatology_counterfactual_hallucination",
        "validation",
    ),
)

DERMOBENCH_VISUAL_DIAGNOSIS_TASKS = (
    "task_2_1_diagnosis_mcq_25_choices",
    "task_2_1_diagnosis_mcq_4_choices",
    "task_2_1_ddi_diagnosis_mcq",
    "task_2_1_derm1m_edu_diagnosis_mcq",
    "task_2_1_derm7pt_diagnosis_mcq",
    "task_2_1_snu134_diagnosis_mcq",
    "task_4_ddi_fairness_mcq",
)

DERMOBENCH_BENCHMARKS = tuple(
    BenchmarkSpec(
        selector=f"dermobench/{task_id}",
        benchmark_id=f"dermobench_{task_id}",
        evaluation_set="filtered",
    )
    for task_id in DERMOBENCH_VISUAL_DIAGNOSIS_TASKS
)

MODEL_ENVIRONMENT = {
    "qwen_3_5_4b_e1_frozen_vision": "ISEP_FROZEN_VISION_REQUEST_MODEL",
    "qwen_3_5_4b_e1_frozen_vision_t06": "ISEP_FROZEN_VISION_REQUEST_MODEL",
    "qwen_3_5_4b_e1_vision_lora": "ISEP_VISION_LORA_REQUEST_MODEL",
    "qwen_3_5_4b_e1_vision_lora_t06": "ISEP_VISION_LORA_REQUEST_MODEL",
}


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model", required=True, choices=tuple(MODEL_ENVIRONMENT))
    parser.add_argument(
        "--request-model",
        required=True,
        help="Exact model name advertised by the vLLM endpoint.",
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def endpoint_ready(base_url: str) -> bool:
    """Return whether the endpoint health route responds successfully."""

    health_url = base_url.removesuffix("/v1") + "/health"
    try:
        with urlopen(health_url, timeout=5) as response:
            return 200 <= response.status < 300
    except (URLError, TimeoutError):
        return False


def existing_run(
    *, output_root: Path, benchmark_id: str, model_id: str
) -> tuple[str, Path | None]:
    """Find a completed or resumable run for one benchmark/model pair."""

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


def execute_benchmark(
    *,
    root: Path,
    model: str,
    request_model: str,
    base_url: str,
    output_root: Path,
    spec: BenchmarkSpec,
    batch_size: int,
    limit: int | None,
    dry_run: bool,
) -> None:
    """Execute, resume, or skip one deterministic benchmark."""

    state, run = existing_run(
        output_root=output_root,
        benchmark_id=spec.benchmark_id,
        model_id=model,
    )
    if state == "complete" and not dry_run:
        print(f"SKIP complete: {spec.benchmark_id} / {run}", flush=True)
        return

    command = [
        sys.executable,
        "-m",
        "src.benchmark.cli",
        "run",
        "--model",
        model,
        "--benchmark",
        spec.selector,
        "--evaluation-set",
        spec.evaluation_set,
        "--benchmark-source",
        "local",
        "--base-url",
        base_url,
        "--thinking-mode",
        "disabled",
        "--batch-size",
        str(batch_size),
        "--reasoning-capture",
        "available",
    ]
    if limit is not None:
        command.extend(("--limit", str(limit)))
    if dry_run:
        command.append("--dry-run")
        command.extend(("--output-root", str(output_root)))
        print(f"DRY RUN: {spec.benchmark_id}", flush=True)
    elif state == "resume" and run is not None:
        command.extend(("--resume", str(run)))
        print(f"RESUME: {spec.benchmark_id} / {run}", flush=True)
    else:
        command.extend(("--output-root", str(output_root)))
        print(f"START: {spec.benchmark_id}", flush=True)

    environment = os.environ.copy()
    environment[MODEL_ENVIRONMENT[model]] = request_model
    result = subprocess.run(command, cwd=root, env=environment, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed ({result.returncode}): {spec.benchmark_id}"
        )


def main() -> int:
    """Run the complete deterministic suite for one E1 checkpoint."""

    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if not args.dry_run and not endpoint_ready(args.base_url):
        raise RuntimeError(f"Endpoint is not ready: {args.base_url}")

    root = args.project_root.resolve()
    output_root = (root / args.output_root).resolve()
    benchmarks = ISEP_BENCHMARKS + DERMOBENCH_BENCHMARKS
    print(
        f"Running {len(benchmarks)} judge-free benchmarks for {args.model}",
        flush=True,
    )
    for index, spec in enumerate(benchmarks, start=1):
        print(f"[{index:02d}/{len(benchmarks):02d}]", end=" ", flush=True)
        execute_benchmark(
            root=root,
            model=args.model,
            request_model=args.request_model,
            base_url=args.base_url,
            output_root=output_root,
            spec=spec,
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    print("All judge-free benchmarks completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
