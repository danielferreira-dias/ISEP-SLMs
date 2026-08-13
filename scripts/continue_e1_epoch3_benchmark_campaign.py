#!/usr/bin/env python3
"""Continue the E1 epoch-3 benchmark campaign after the frozen run starts.

This controller waits for the already-running frozen-vision deterministic
suite, validates its sixteen completed metric files, replaces only that vLLM
server with the standalone vision-LoRA checkpoint, runs real 1/10-case gates,
and finally executes the same deterministic suite for the vision condition.
It deliberately stops before any continued fine-tuning experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

EXPECTED_FROZEN_BENCHMARKS = 16
EXPECTED_VISION_BENCHMARKS = 14


def parse_args() -> argparse.Namespace:
    """Parse campaign paths and process identifiers."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--frozen-runner-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def process_exists(pid: int) -> bool:
    """Return whether a process exists without signalling it."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def completed_metrics(output_root: Path) -> tuple[Path, ...]:
    """Return completed metric files below one condition output root."""

    return tuple(sorted(output_root.glob("**/metrics.json")))


def wait_for_frozen_run(
    *, output_root: Path, runner_pid: int, poll_seconds: int
) -> None:
    """Wait for a complete frozen suite or fail if its runner disappears."""

    while True:
        metrics = completed_metrics(output_root)
        print(
            f"Frozen progress: {len(metrics)}/{EXPECTED_FROZEN_BENCHMARKS} completed",
            flush=True,
        )
        if len(metrics) == EXPECTED_FROZEN_BENCHMARKS:
            return
        if len(metrics) > EXPECTED_FROZEN_BENCHMARKS:
            raise RuntimeError("Frozen output contains unexpected extra metric files")
        if not process_exists(runner_pid):
            raise RuntimeError(
                "Frozen runner exited before all deterministic tasks completed"
            )
        time.sleep(poll_seconds)


def read_server_pid(pid_path: Path) -> int:
    """Read and validate the current vLLM API server PID."""

    raw = pid_path.read_text(encoding="utf-8").strip()
    pid = int(raw)
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not cmdline_path.is_file():
        raise RuntimeError(f"vLLM PID does not exist: {pid}")
    command = cmdline_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    expected = "qwen_3_5_4b_e1_frozen_vision_epoch3"
    if "vllm serve" not in command or expected not in command:
        raise RuntimeError(f"Refusing to stop unexpected PID {pid}: {command}")
    return pid


def stop_server(pid: int) -> None:
    """Gracefully stop the exact validated API server PID."""

    print(f"Stopping frozen vLLM API server PID {pid}", flush=True)
    os.kill(pid, signal.SIGTERM)
    for _ in range(60):
        if not process_exists(pid):
            return
        time.sleep(1)
    raise RuntimeError(f"Frozen vLLM PID {pid} did not stop after SIGTERM")


def wait_for_gpu_release() -> None:
    """Wait until the prior vLLM process releases its GPU allocation."""

    command = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    for _ in range(60):
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        memory_used_mib = int(result.stdout.strip().splitlines()[0])
        if memory_used_mib < 2_000:
            print(f"GPU released: {memory_used_mib} MiB in use", flush=True)
            return
        time.sleep(2)
    raise RuntimeError("GPU memory was not released after stopping frozen vLLM")


def endpoint_ready(base_url: str, model_name: str) -> bool:
    """Return whether health and model-list endpoints identify the target model."""

    try:
        with urlopen(base_url.removesuffix("/v1") + "/health", timeout=5) as response:
            if not 200 <= response.status < 300:
                return False
        with urlopen(base_url + "/models", timeout=5) as response:
            payload = json.load(response)
    except (URLError, TimeoutError, json.JSONDecodeError):
        return False
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return False
    return any(isinstance(item, dict) and item.get("id") == model_name for item in data)


def start_vision_server(root: Path) -> int:
    """Start the standalone vision-LoRA model and wait until it is healthy."""

    model_path = "outputs/merged_models/qwen_3_5_4b_e1_vision_lora_epoch3"
    model_name = "qwen_3_5_4b_e1_vision_lora"
    log_path = root / "runs/vllm/e1-vision-merged-8002.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(root / ".venv/bin/vllm"),
        "serve",
        model_path,
        "--served-model-name",
        model_name,
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
    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (root / "runs/vllm/e1-current-8002.pid").write_text(
        f"{process.pid}\n", encoding="utf-8"
    )
    print(f"Started vision vLLM API server PID {process.pid}", flush=True)
    for _ in range(90):
        if process.poll() is not None:
            raise RuntimeError(
                f"Vision vLLM exited during startup with code {process.returncode}"
            )
        if endpoint_ready("http://127.0.0.1:8002/v1", model_name):
            return process.pid
        time.sleep(10)
    raise RuntimeError("Vision vLLM did not become ready within 15 minutes")


def run_gate(
    *,
    root: Path,
    benchmark: str,
    evaluation_set: str,
    limit: int,
    output_root: Path,
) -> None:
    """Run one real deterministic gate through the benchmark CLI."""

    command = [
        str(root / ".venv/bin/python"),
        "-m",
        "src.benchmark.cli",
        "run",
        "--model",
        "qwen_3_5_4b_e1_vision_lora",
        "--benchmark",
        benchmark,
        "--evaluation-set",
        evaluation_set,
        "--benchmark-source",
        "local",
        "--limit",
        str(limit),
        "--base-url",
        "http://127.0.0.1:8002/v1",
        "--thinking-mode",
        "disabled",
        "--temperature",
        "0",
        "--batch-size",
        "8" if limit > 1 else "1",
        "--output-root",
        str(output_root),
    ]
    environment = os.environ.copy()
    environment["ISEP_VISION_LORA_REQUEST_MODEL"] = "qwen_3_5_4b_e1_vision_lora"
    subprocess.run(command, cwd=root, env=environment, check=True)


def run_vision_gates(root: Path) -> None:
    """Run 1/10-case gates in both benchmark families."""

    gate_root = root / "outputs/e1_epoch3_deterministic_smoke/vision"
    for limit, suffix in ((1, "one"), (10, "ten")):
        run_gate(
            root=root,
            benchmark="visual_top_k_closed_set",
            evaluation_set="internal_benchmark",
            limit=limit,
            output_root=gate_root / f"isep_{suffix}",
        )
        run_gate(
            root=root,
            benchmark="dermobench/task_2_1_diagnosis_mcq_4_choices",
            evaluation_set="filtered",
            limit=limit,
            output_root=gate_root / f"dermobench_{suffix}",
        )


def run_full_vision_suite(root: Path) -> None:
    """Execute the same sixteen deterministic tasks for the vision model."""

    command = [
        str(root / ".venv/bin/python"),
        "scripts/run_e1_deterministic_benchmarks.py",
        "--model",
        "qwen_3_5_4b_e1_vision_lora",
        "--request-model",
        "qwen_3_5_4b_e1_vision_lora",
        "--base-url",
        "http://127.0.0.1:8002/v1",
        "--output-root",
        "outputs/e1_epoch3_deterministic_benchmarks/vision",
        "--batch-size",
        "8",
    ]
    subprocess.run(command, cwd=root, check=True)
    metrics = completed_metrics(
        root / "outputs/e1_epoch3_deterministic_benchmarks/vision"
    )
    if len(metrics) != EXPECTED_VISION_BENCHMARKS:
        raise RuntimeError(
            "Vision suite produced "
            f"{len(metrics)} completed metrics, expected {EXPECTED_VISION_BENCHMARKS}"
        )


def write_status(root: Path, *, status: str, detail: str) -> None:
    """Persist the controller's terminal state without clinical data."""

    path = root / "outputs/e1_epoch3_deterministic_benchmarks/campaign_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "detail": detail,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """Continue the frozen-to-vision benchmark campaign."""

    args = parse_args()
    root = args.project_root.resolve()
    try:
        wait_for_frozen_run(
            output_root=root / "outputs/e1_epoch3_deterministic_benchmarks/frozen",
            runner_pid=args.frozen_runner_pid,
            poll_seconds=args.poll_seconds,
        )
        server_pid = read_server_pid(root / "runs/vllm/e1-current-8002.pid")
        stop_server(server_pid)
        wait_for_gpu_release()
        start_vision_server(root)
        run_vision_gates(root)
        run_full_vision_suite(root)
    except Exception as error:
        write_status(root, status="failed", detail=f"{type(error).__name__}: {error}")
        raise
    write_status(
        root,
        status="completed",
        detail="Frozen and vision epoch-3 deterministic suites completed",
    )
    print("Epoch-3 deterministic campaign completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
