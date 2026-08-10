#!/usr/bin/env python3
"""Run the filtered DermoBench suite, then the paired context ablation.

The two local Qwen endpoints are evaluated concurrently, but the clinical
context ablation does not start until both models have completed all thirteen
DermoBench tasks. Completed task/model combinations are skipped, while the
latest interrupted run is resumed after the benchmark CLI validates identity.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from urllib.error import URLError
from urllib.request import urlopen


DERMOBENCH_TASKS = (
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


@dataclass(frozen=True, slots=True)
class ModelEndpoint:
    model_id: str
    config: str
    base_url: str


MODELS = (
    ModelEndpoint(
        model_id="qwen_3_5_4b",
        config="qwen_3_5_4b",
        base_url="http://127.0.0.1:8002/v1",
    ),
    ModelEndpoint(
        model_id="qwen_3_6_27b",
        config="qwen_3_6_27b",
        base_url="http://127.0.0.1:8000/v1",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--skip-dermobench-task",
        action="append",
        default=[],
        choices=DERMOBENCH_TASKS,
        help=(
            "Skip one DermoBench task for every model. Repeat the option to "
            "skip multiple tasks; completed artifacts are preserved."
        ),
    )
    parser.add_argument(
        "--dermobench-output-root",
        type=Path,
        default=Path("outputs/dermobench_full_v1/temp_0_6_thinking_off"),
    )
    parser.add_argument(
        "--context-output-root",
        type=Path,
        default=Path(
            "outputs/clinical_context_ablation_v1/temp_0_6_thinking_off"
        ),
    )
    return parser.parse_args()


def endpoint_ready(model: ModelEndpoint) -> bool:
    try:
        with urlopen(model.base_url.removesuffix("/v1") + "/health", timeout=5) as response:
            return 200 <= response.status < 300
    except (URLError, TimeoutError):
        return False


def existing_run(
    *,
    output_root: Path,
    benchmark_id: str,
    model_id: str,
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


def execute(
    *,
    root: Path,
    model: ModelEndpoint,
    benchmark_selector: str,
    benchmark_id: str,
    evaluation_set: str,
    output_root: Path,
    temperature: float,
    batch_size: int,
    benchmark_source: str,
) -> None:
    state, run = existing_run(
        output_root=output_root,
        benchmark_id=benchmark_id,
        model_id=model.model_id,
    )
    if state == "complete":
        print(f"SKIP complete: {model.model_id} / {benchmark_id} / {run}", flush=True)
        return

    command = [
        sys.executable,
        "-m",
        "src.benchmark.cli",
        "run",
        "--model",
        model.config,
        "--benchmark",
        benchmark_selector,
        "--evaluation-set",
        evaluation_set,
        "--benchmark-source",
        benchmark_source,
        "--base-url",
        model.base_url,
        "--thinking-mode",
        "disabled",
        "--temperature",
        str(temperature),
        "--batch-size",
        str(batch_size),
        "--reasoning-capture",
        "available",
    ]
    if state == "resume" and run is not None:
        command.extend(("--resume", str(run)))
        print(f"RESUME: {model.model_id} / {benchmark_id} / {run}", flush=True)
    else:
        command.extend(("--output-root", str(output_root)))
        print(f"START: {model.model_id} / {benchmark_id}", flush=True)

    result = subprocess.run(command, cwd=root, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed ({result.returncode}): "
            f"{model.model_id} / {benchmark_id}"
        )


def run_dermobench_model(
    model: ModelEndpoint,
    *,
    tasks: tuple[str, ...],
    root: Path,
    output_root: Path,
    temperature: float,
    batch_size: int,
) -> None:
    for task in tasks:
        execute(
            root=root,
            model=model,
            benchmark_selector=f"dermobench/{task}",
            benchmark_id=f"dermobench_{task}",
            evaluation_set="filtered",
            output_root=output_root,
            temperature=temperature,
            batch_size=batch_size,
            benchmark_source="local",
        )


def run_stage(workers: dict[ModelEndpoint, callable]) -> None:
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        pending = {
            executor.submit(worker): model.model_id
            for model, worker in workers.items()
        }
        for future in as_completed(pending):
            model_id = pending[future]
            try:
                future.result()
            except Exception as exc:  # preserve the other model's output
                failures.append(f"{model_id}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError("; ".join(failures))


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    dermobench_output = (root / args.dermobench_output_root).resolve()
    context_output = (root / args.context_output_root).resolve()

    unavailable = [model.model_id for model in MODELS if not endpoint_ready(model)]
    if unavailable:
        raise RuntimeError(f"Endpoints are not ready: {', '.join(unavailable)}")

    print("STAGE 1/2: filtered DermoBench", flush=True)
    skipped_tasks = set(args.skip_dermobench_task)
    selected_tasks = tuple(
        task for task in DERMOBENCH_TASKS if task not in skipped_tasks
    )
    if skipped_tasks:
        print(
            "SKIP requested DermoBench tasks: "
            + ", ".join(task for task in DERMOBENCH_TASKS if task in skipped_tasks),
            flush=True,
        )
    run_stage(
        {
            model: (
                lambda model=model: run_dermobench_model(
                    model,
                    tasks=selected_tasks,
                    root=root,
                    output_root=dermobench_output,
                    temperature=args.temperature,
                    batch_size=args.batch_size,
                )
            )
            for model in MODELS
        }
    )

    print("STAGE 2/2: paired clinical context ablation", flush=True)
    run_stage(
        {
            model: (
                lambda model=model: execute(
                    root=root,
                    model=model,
                    benchmark_selector="clinical_context_ablation",
                    benchmark_id="clinical_context_ablation",
                    evaluation_set="internal_benchmark",
                    output_root=context_output,
                    temperature=args.temperature,
                    batch_size=args.batch_size,
                    benchmark_source="local",
                )
            )
            for model in MODELS
        }
    )
    print("All inference stages completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
