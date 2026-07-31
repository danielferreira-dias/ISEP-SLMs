"""Run the repository benchmark pipeline against Gemma 4 31B on Modal.

The remote Modal Server exposes vLLM's OpenAI-compatible API. The local
entrypoint waits for that server and invokes the repository benchmark CLI, so
selection, prompts, validation, metrics, and HTML reports are identical to
the API-model runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import modal


APP_NAME = "isep-gemma-4-31b-it-benchmark"
MODEL_CONFIG_ID = "gemma_4_31b_it"
MODEL_ID = "google/gemma-4-31B-it"
MODEL_REVISION = "main"
GPU = "A100-80GB"
VLLM_PORT = 8000

MAX_MODEL_LEN = 16_384
GPU_MEMORY_UTILIZATION = 0.90
STARTUP_TIMEOUT_SECONDS = 30 * 60
SCALEDOWN_WINDOW_SECONDS = 60

HF_CACHE_PATH = "/cache/huggingface"
VLLM_CACHE_PATH = "/cache/vllm"

app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name(
    "isep-huggingface-cache",
    create_if_missing=True,
)
vllm_cache = modal.Volume.from_name(
    "isep-vllm-cache",
    create_if_missing=True,
)

runtime_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.23.0",
        "huggingface-hub[hf_xet]>=0.36.0",
    )
    .env(
        {
            "HF_HOME": HF_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_CACHE_ROOT": VLLM_CACHE_PATH,
        }
    )
)


@app.server(
    image=runtime_image,
    gpu=GPU,
    cpu=8.0,
    memory=98_304,
    port=VLLM_PORT,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    target_concurrency=8,
    max_containers=1,
    unauthenticated=True,
    volumes={
        HF_CACHE_PATH: hf_cache,
        VLLM_CACHE_PATH: vllm_cache,
    },
)
class GemmaVllmServer:
    """OpenAI-compatible Gemma 4 server on one A100 80 GB."""

    @modal.enter()
    def start(self) -> None:
        """Start vLLM with settings aligned to the model YAML."""

        self.process = subprocess.Popen(
            [
                "vllm",
                "serve",
                MODEL_ID,
                "--revision",
                MODEL_REVISION,
                "--served-model-name",
                MODEL_ID,
                "--host",
                "0.0.0.0",
                "--port",
                str(VLLM_PORT),
                "--dtype",
                "bfloat16",
                "--max-model-len",
                str(MAX_MODEL_LEN),
                "--gpu-memory-utilization",
                str(GPU_MEMORY_UTILIZATION),
                "--enforce-eager",
                "--limit-mm-per-prompt",
                json.dumps({"image": 1}),
                "--reasoning-parser",
                "gemma4",
                "--default-chat-template-kwargs",
                json.dumps({"enable_thinking": False}),
            ]
        )

    @modal.exit()
    def stop(self) -> None:
        """Stop the vLLM child process when Modal retires the container."""

        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()


@app.local_entrypoint()
def main(
    benchmark: str = "visual_top_k_closed_set",
    evaluation_set: str = "internal_benchmark_1000",
    limit: int = 10,
    seed: int = 42,
    batch_size: int = 8,
    reasoning_capture: str = "available",
    output_root: str | None = None,
    dry_run: bool = False,
    startup_timeout: int = STARTUP_TIMEOUT_SECONDS,
) -> None:
    """Run a benchmark through the temporary Modal vLLM server."""

    project_root = Path(__file__).resolve().parents[2]
    if limit <= 0:
        raise ValueError("limit must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if startup_timeout <= 0:
        raise ValueError("startup_timeout must be positive")

    command = [
        sys.executable,
        "-m",
        "src.benchmark.cli",
        "run",
        "--model",
        MODEL_CONFIG_ID,
        "--benchmark",
        benchmark,
        "--evaluation-set",
        evaluation_set,
        "--limit",
        str(limit),
        "--seed",
        str(seed),
        "--batch-size",
        str(batch_size),
        "--reasoning-capture",
        reasoning_capture,
    ]
    if output_root:
        command.extend(["--output-root", output_root])

    if dry_run:
        command.append("--dry-run")
        subprocess.run(command, cwd=project_root, check=True)
        return

    server_url = GemmaVllmServer.get_url()
    if not server_url:
        raise RuntimeError("Modal did not return a URL for the vLLM server")
    _wait_until_healthy(server_url, timeout_seconds=startup_timeout)

    command.extend(["--base-url", f"{server_url.rstrip('/')}/v1"])
    print(
        f"Running {benchmark} with {MODEL_ID}: "
        f"{limit} selected case(s), seed {seed}.",
        flush=True,
    )
    subprocess.run(command, cwd=project_root, check=True)


def _wait_until_healthy(url: str, *, timeout_seconds: int) -> None:
    """Wait through Modal cold starts until vLLM answers its health route."""

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
