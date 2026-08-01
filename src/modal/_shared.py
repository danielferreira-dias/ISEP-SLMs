"""Shared local orchestration for temporary Modal benchmark servers."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class SmokeRun:
    """One benchmark invocation and its exact number of scored tasks."""

    benchmark: str
    evaluation_set: str
    selection_limit: int
    expected_task_count: int


def smoke_runs(
    *,
    benchmark: str,
    evaluation_set: str,
    limit: int,
    all_benchmarks: bool,
    evidence_and_top_k: bool = False,
    validation_suite: bool = False,
) -> tuple[SmokeRun, ...]:
    """Resolve a single run or one of the fixed benchmark smoke suites."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    selected_suites = sum(
        (all_benchmarks, evidence_and_top_k, validation_suite)
    )
    if selected_suites > 1:
        raise ValueError(
            "all_benchmarks, evidence_and_top_k, and validation_suite are "
            "mutually exclusive"
        )
    if validation_suite:
        if limit % 2:
            raise ValueError(
                "The validation-suite task limit must be even because the "
                "confusion benchmark evaluates complete two-task pairs"
            )
        return (
            SmokeRun(
                benchmark="visual_top_k_closed_set",
                evaluation_set="validation",
                selection_limit=limit,
                expected_task_count=limit,
            ),
            SmokeRun(
                benchmark="visual_disease_confusion_sets",
                evaluation_set="validation",
                selection_limit=limit // 2,
                expected_task_count=limit,
            ),
            SmokeRun(
                benchmark="evidence_grounded_diagnosis",
                evaluation_set="validation",
                selection_limit=limit,
                expected_task_count=limit,
            ),
            SmokeRun(
                benchmark="open_ended_diagnosis",
                evaluation_set="validation",
                selection_limit=limit,
                expected_task_count=limit,
            ),
        )
    if evidence_and_top_k:
        return (
            SmokeRun(
                benchmark="visual_top_k_closed_set",
                evaluation_set="internal_benchmark_1000",
                selection_limit=limit,
                expected_task_count=limit,
            ),
            SmokeRun(
                benchmark="evidence_grounded_diagnosis",
                evaluation_set="external_ddi_evidence",
                selection_limit=limit,
                expected_task_count=limit,
            ),
        )
    if not all_benchmarks:
        multiplier = (
            2
            if benchmark == "visual_disease_confusion_sets"
            else 1
        )
        return (
            SmokeRun(
                benchmark=benchmark,
                evaluation_set=evaluation_set,
                selection_limit=limit,
                expected_task_count=limit * multiplier,
            ),
        )
    if limit % 2:
        raise ValueError(
            "The all-benchmarks task limit must be even because the "
            "confusion benchmark evaluates complete two-task pairs"
        )
    return (
        SmokeRun(
            benchmark="visual_top_k_closed_set",
            evaluation_set="internal_benchmark_1000",
            selection_limit=limit,
            expected_task_count=limit,
        ),
        SmokeRun(
            benchmark="visual_disease_confusion_sets",
            evaluation_set="paired_confusion_tasks",
            selection_limit=limit // 2,
            expected_task_count=limit,
        ),
        SmokeRun(
            benchmark="evidence_grounded_diagnosis",
            evaluation_set="external_ddi_evidence",
            selection_limit=limit,
            expected_task_count=limit,
        ),
    )


def structured_output_modes(value: str) -> tuple[str, ...]:
    """Expand one requested structured-output experiment condition."""

    if value == "both":
        return ("prompt_only", "json_schema")
    if value in {"prompt_only", "json_schema"}:
        return (value,)
    raise ValueError(
        "structured_output must be prompt_only, json_schema, or both"
    )


def run_benchmark(
    *,
    project_root: Path,
    model_config_id: str,
    model_id: str,
    run: SmokeRun,
    seed: int,
    batch_size: int,
    reasoning_capture: str,
    structured_output: str,
    output_root: str | None,
    dry_run: bool,
    server_url: str | None,
) -> None:
    """Invoke the repository CLI for one resolved smoke run."""

    command = [
        sys.executable,
        "-m",
        "src.benchmark.cli",
        "run",
        "--model",
        model_config_id,
        "--benchmark",
        run.benchmark,
        "--evaluation-set",
        run.evaluation_set,
        "--limit",
        str(run.selection_limit),
        "--seed",
        str(seed),
        "--batch-size",
        str(batch_size),
        "--reasoning-capture",
        reasoning_capture,
        "--structured-output",
        structured_output,
    ]
    if output_root:
        command.extend(["--output-root", output_root])
    if dry_run:
        command.append("--dry-run")
    else:
        if not server_url:
            raise ValueError("A server URL is required outside dry-run mode")
        command.extend(
            ["--base-url", f"{server_url.rstrip('/')}/v1"]
        )

    print(
        f"Running {run.benchmark} with {model_id}: "
        f"{run.expected_task_count} scored task(s), seed {seed}, "
        f"structured output {structured_output}.",
        flush=True,
    )
    subprocess.run(command, cwd=project_root, check=True)


def wait_until_healthy(url: str, *, timeout_seconds: int) -> None:
    """Wait through a Modal cold start until vLLM answers its health route."""

    health_url = f"{url.rstrip('/')}/health"
    deadline = time.monotonic() + timeout_seconds
    last_status: str | None = None
    print(f"Waiting for vLLM at {health_url}", flush=True)
    while time.monotonic() < deadline:
        request = Request(health_url, method="GET")
        try:
            with urlopen(request, timeout=60) as response:
                status = getattr(response, "status", 200)
                if 200 <= status < 300:
                    print("vLLM is ready.", flush=True)
                    return
                last_status = f"HTTP {status}"
        except HTTPError as exc:
            last_status = f"HTTP {exc.code}"
        except (URLError, TimeoutError) as exc:
            last_status = type(exc).__name__
        time.sleep(1)

    detail = f" Last response: {last_status}." if last_status else ""
    raise TimeoutError(
        f"vLLM did not become healthy within {timeout_seconds} seconds."
        f"{detail}"
    )
