"""Run MedGemma 1.5 4B through the benchmark pipeline on Modal."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

from src.modal._shared import (
    run_benchmark,
    smoke_runs,
    wait_until_healthy,
)


APP_NAME = "isep-medgemma-1-5-4b-benchmark"
MODEL_CONFIG_ID = "medgemma_1_5_4b"
MODEL_ID = "google/medgemma-1.5-4b-it"
MODEL_REVISION = "main"
GPU = "L40S"
VLLM_PORT = 8000

MAX_MODEL_LEN = 16_384
GPU_MEMORY_UTILIZATION = 0.90
STARTUP_TIMEOUT_SECONDS = 30 * 60
SCALEDOWN_WINDOW_SECONDS = 60

HF_CACHE_PATH = "/cache/huggingface"
VLLM_CACHE_PATH = "/cache/vllm"

app = modal.App(APP_NAME)
hf_secret = modal.Secret.from_name(
    "huggingface-secret",
    required_keys=["HF_TOKEN"],
)
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
    .add_local_python_source("src.modal._shared")
)


@app.server(
    image=runtime_image,
    gpu=GPU,
    cpu=8.0,
    memory=49_152,
    port=VLLM_PORT,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    target_concurrency=8,
    max_containers=1,
    unauthenticated=True,
    secrets=[hf_secret],
    volumes={
        HF_CACHE_PATH: hf_cache,
        VLLM_CACHE_PATH: vllm_cache,
    },
)
class MedGemmaVllmServer:
    """OpenAI-compatible MedGemma server on one L40S."""

    @modal.enter()
    def start(self) -> None:
        """Start vLLM with the parameters in the model YAML."""

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
    structured_output: str = "prompt_only",
    output_root: str | None = None,
    all_benchmarks: bool = False,
    evidence_and_top_k: bool = False,
    dry_run: bool = False,
    startup_timeout: int = STARTUP_TIMEOUT_SECONDS,
) -> None:
    """Run one benchmark or the complete three-benchmark smoke suite."""

    project_root = Path(__file__).resolve().parents[2]
    runs = smoke_runs(
        benchmark=benchmark,
        evaluation_set=evaluation_set,
        limit=limit,
        all_benchmarks=all_benchmarks,
        evidence_and_top_k=evidence_and_top_k,
    )
    if structured_output != "prompt_only":
        raise ValueError(
            "MedGemma supports prompt_only in this pipeline. Its embedded "
            "reasoning is separated from final content client-side."
        )
    modes = ("prompt_only",)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if startup_timeout <= 0:
        raise ValueError("startup_timeout must be positive")

    server_url: str | None = None
    if not dry_run:
        server_url = MedGemmaVllmServer.get_url()
        if not server_url:
            raise RuntimeError("Modal did not return a vLLM server URL")
        wait_until_healthy(
            server_url,
            timeout_seconds=startup_timeout,
        )
    for mode in modes:
        for run in runs:
            run_benchmark(
                project_root=project_root,
                model_config_id=MODEL_CONFIG_ID,
                model_id=MODEL_ID,
                run=run,
                seed=seed,
                batch_size=batch_size,
                reasoning_capture=reasoning_capture,
                structured_output=mode,
                output_root=output_root,
                dry_run=dry_run,
                server_url=server_url,
            )
