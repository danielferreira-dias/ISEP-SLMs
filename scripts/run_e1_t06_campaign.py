#!/usr/bin/env python3
"""Run the paired E1 epoch-3 benchmark campaign with Qwen's t=0.6 profile."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

EXPECTED_METRICS = 14
BASE_URL = "http://127.0.0.1:8002/v1"


@dataclass(frozen=True, slots=True)
class Condition:
    """One E1 checkpoint and its benchmark-facing identifiers."""

    key: str
    model_id: str
    request_model: str
    model_path: str
    environment_name: str


CONDITIONS = (
    Condition(
        key="frozen",
        model_id="qwen_3_5_4b_e1_frozen_vision_t06",
        request_model="qwen_3_5_4b_e1_frozen_vision",
        model_path="outputs/merged_models/qwen_3_5_4b_e1_frozen_vision_epoch3",
        environment_name="ISEP_FROZEN_VISION_REQUEST_MODEL",
    ),
    Condition(
        key="vision",
        model_id="qwen_3_5_4b_e1_vision_lora_t06",
        request_model="qwen_3_5_4b_e1_vision_lora",
        model_path="outputs/merged_models/qwen_3_5_4b_e1_vision_lora_epoch3",
        environment_name="ISEP_VISION_LORA_REQUEST_MODEL",
    ),
)


def process_exists(pid: int) -> bool:
    """Return whether a process exists without altering it."""

    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.is_file():
        fields = stat_path.read_text(encoding="utf-8").split()
        if len(fields) >= 3 and fields[2] == "Z":
            return False

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_command(pid: int) -> str:
    """Read one Linux process command line."""

    path = Path(f"/proc/{pid}/cmdline")
    if not path.is_file():
        raise RuntimeError(f"Process does not exist: {pid}")
    return path.read_bytes().replace(b"\0", b" ").decode(errors="replace")


def read_validated_server_pid(root: Path) -> int:
    """Return the PID only when it is one of this campaign's vLLM servers."""

    pid = int((root / "runs/vllm/e1-current-8002.pid").read_text().strip())
    command = process_command(pid)
    allowed_paths = tuple(condition.model_path for condition in CONDITIONS)
    if "vllm serve" not in command or not any(
        model_path in command for model_path in allowed_paths
    ):
        raise RuntimeError(f"Refusing to stop unexpected PID {pid}: {command}")
    return pid


def stop_server(root: Path) -> None:
    """Gracefully stop the exact validated current vLLM server."""

    pid = read_validated_server_pid(root)
    print(f"Stopping vLLM PID {pid}", flush=True)
    os.kill(pid, signal.SIGTERM)
    for _ in range(90):
        if not process_exists(pid):
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"vLLM PID {pid} did not stop after SIGTERM")

    for _ in range(90):
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
        if int(result.stdout.strip().splitlines()[0]) < 2_000:
            return
        time.sleep(2)
    raise RuntimeError("GPU allocation was not released")


def endpoint_ready(model_name: str) -> bool:
    """Return whether health and model-list routes identify the target model."""

    try:
        with urlopen(BASE_URL.removesuffix("/v1") + "/health", timeout=5) as response:
            if not 200 <= response.status < 300:
                return False
        with urlopen(BASE_URL + "/models", timeout=5) as response:
            payload = json.load(response)
    except (URLError, TimeoutError, json.JSONDecodeError):
        return False
    data = payload.get("data") if isinstance(payload, dict) else None
    return isinstance(data, list) and any(
        isinstance(item, dict) and item.get("id") == model_name for item in data
    )


def start_server(root: Path, condition: Condition) -> int:
    """Start a condition's local standalone checkpoint and await readiness."""

    log_path = root / f"runs/vllm/e1-t06-{condition.key}-8002.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(root / ".venv/bin/vllm"),
        "serve",
        condition.model_path,
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
    print(f"Started {condition.key} vLLM PID {process.pid}", flush=True)
    for _ in range(90):
        if process.poll() is not None:
            raise RuntimeError(
                f"{condition.key} vLLM exited with code {process.returncode}"
            )
        if endpoint_ready(condition.request_model):
            return process.pid
        time.sleep(10)
    raise RuntimeError(f"{condition.key} vLLM did not become ready")


def run_gate(root: Path, condition: Condition, *, limit: int) -> None:
    """Run one real multimodal Top-K gate with the final decoding profile."""

    output = root / f"outputs/e1_epoch3_historical_t06_smoke/{condition.key}_{limit}"
    command = [
        str(root / ".venv/bin/python"),
        "-m",
        "src.benchmark.cli",
        "run",
        "--model",
        condition.model_id,
        "--benchmark",
        "visual_top_k_closed_set",
        "--evaluation-set",
        "internal_benchmark",
        "--benchmark-source",
        "local",
        "--limit",
        str(limit),
        "--base-url",
        BASE_URL,
        "--thinking-mode",
        "disabled",
        "--batch-size",
        "8" if limit > 1 else "1",
        "--output-root",
        str(output),
    ]
    environment = os.environ.copy()
    environment[condition.environment_name] = condition.request_model
    subprocess.run(command, cwd=root, env=environment, check=True)


def run_suite(root: Path, condition: Condition) -> None:
    """Run and validate all 14 judge-free benchmark tasks for one condition."""

    output = root / f"outputs/e1_epoch3_historical_t06_benchmarks/{condition.key}"
    command = [
        str(root / ".venv/bin/python"),
        "scripts/run_e1_deterministic_benchmarks.py",
        "--model",
        condition.model_id,
        "--request-model",
        condition.request_model,
        "--base-url",
        BASE_URL,
        "--output-root",
        str(output),
        "--batch-size",
        "8",
    ]
    subprocess.run(command, cwd=root, check=True)
    metrics = tuple(output.glob("**/metrics.json"))
    if len(metrics) != EXPECTED_METRICS:
        raise RuntimeError(
            f"{condition.key} produced {len(metrics)} metrics; "
            f"expected {EXPECTED_METRICS}"
        )


def write_status(root: Path, *, status: str, detail: str) -> None:
    """Persist a clinical-data-free campaign status document."""

    output = root / "outputs/e1_epoch3_historical_t06_benchmarks"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "detail": detail,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    temporary = output / ".campaign_status.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "campaign_status.json")


def main() -> int:
    """Execute the frozen and vision conditions sequentially."""

    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/e1_epoch3_historical_t06_benchmarks"
    if any((output / condition.key).exists() for condition in CONDITIONS):
        raise RuntimeError("Refusing to mix with an existing t=0.6 campaign")

    write_status(root, status="running", detail="Preparing frozen condition")
    try:
        stop_server(root)
        for index, condition in enumerate(CONDITIONS):
            write_status(root, status="running", detail=f"Running {condition.key}")
            start_server(root, condition)
            run_gate(root, condition, limit=1)
            run_gate(root, condition, limit=10)
            run_suite(root, condition)
            if index < len(CONDITIONS) - 1:
                stop_server(root)
        write_status(
            root,
            status="completed",
            detail="Frozen and vision t=0.6 suites completed",
        )
    except Exception as exc:
        write_status(root, status="failed", detail=f"{type(exc).__name__}: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
