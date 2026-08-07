---
name: runpod-vllm-setup
description: Prepare, validate, and troubleshoot a RunPod GPU pod for serving this ISEP repository's Hugging Face multimodal models through local OpenAI-compatible vLLM endpoints. Use when connecting to a new RunPod, installing the pinned Linux runtime, checking GPU/CUDA/Hugging Face access, starting one or more model YAMLs, configuring thinking, reusing model caches, exposing or tunnelling endpoints, or diagnosing vLLM startup, memory, timeout, GDN, and endpoint errors.
---

# RunPod vLLM setup

Set up the pod in this order. Do not start model downloads before the hardware,
disk, authentication, and intended server layout are known.

## 1. Resolve the execution plan

Collect or discover:

- the direct-TCP SSH command (prefer it because it supports `rsync`/`scp`);
- GPU model, GPU count, visible VRAM, volume size, and free disk;
- model YAML paths, thinking condition, ports, and desired concurrency;
- whether benchmarks run inside the pod or call a remotely exposed endpoint.

Prefer running the benchmark code inside the pod against loopback endpoints.
Do not expose an unauthenticated vLLM port publicly. If remote access is needed,
use an SSH tunnel unless the user explicitly requests an authenticated service.

## 2. Inspect before mutating

Run read-only checks first:

```bash
nvidia-smi
df -h /workspace
ps -ef | grep '[v]llm serve'
ss -ltnp
```

Copy `scripts/runpod_bootstrap.sh` to the pod and run it without `--install`.
Read [references/troubleshooting.md](references/troubleshooting.md) if CUDA,
driver, GDN, memory, or authentication checks fail.

## 3. Transfer and install reproducibly

Use the repository's existing checkout when present. Otherwise use `rsync` over
the direct-TCP SSH connection or clone the authorized repository. Preserve
`uv.lock`; do not independently `pip install` a different vLLM/Torch stack.

From the repository root on the pod:

```bash
bash skills/runpod-vllm-setup/scripts/runpod_bootstrap.sh --install
```

The script installs `uv` only when missing, executes `uv sync --frozen`, checks
the pinned imports, and never prints or persists `HF_TOKEN`. Supply Hugging Face
credentials through the process environment or RunPod secrets.

Do not update the NVIDIA driver from inside a running pod. Apply the documented
CUDA forward-compatibility remedy only when the checks demonstrate that it is
needed.

## 4. Allocate servers explicitly

Inspect every selected model YAML. Treat its repository, dtype, context length,
image limit, reasoning parser, and generation settings as authoritative.

For the proven single-H200 layout in this repository:

| Model | Port | GPU memory utilization | Max sequences |
| --- | ---: | ---: | ---: |
| Qwen 3.6 27B | 8000 | 0.65 | 8 |
| Qwen 3.5 4B | 8002 | 0.15 | 8 |

Keep aggregate utilization below physical capacity and retain headroom for
vision embeddings, CUDA graphs/kernels, and transient allocations. Do not copy
this split to a smaller GPU without recalculating it.

Start each server with `scripts/start_vllm_server.py`. It loads the repository's
dataclass config, derives the vLLM command, adds the selected thinking setting,
uses Triton GDN prefill for Qwen 3.5/3.6 by default, writes separate PID/log
files, and waits for both `/health` and `/v1/models`.

Example:

```bash
uv run python skills/runpod-vllm-setup/scripts/start_vllm_server.py \
  configs/models/qwen_3_6_27b.yaml --port 8000 \
  --gpu-memory-utilization 0.65 --max-num-seqs 8 --thinking off
```

Repeat on port `8002` for Qwen 3.5 4B. Start the larger model first, verify its
actual memory consumption, and only then start the smaller model.

## 5. Gate benchmark execution

Require all of the following before running more than a smoke case:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
nvidia-smi
```

Then run one real multimodal case through the same benchmark runner, prompt,
parser, and thinking condition intended for the experiment. Check:

- image ingestion;
- final-answer and reasoning capture;
- JSON/schema behavior where applicable;
- absence of truncation and backend errors;
- log growth, GPU memory, and latency.

Scale from 1 to 10 cases before launching a full cohort. Reuse identical task
IDs for paired model/thinking comparisons.

## 6. Preserve provenance and control cost

Record GPU, driver, CUDA, Torch, vLLM, model revision, port, memory fraction,
context length, concurrency, thinking state, and exact benchmark command.
Keep each server log with the run artifacts.

After completion, stop only the verified server PIDs and remind the user to
stop the RunPod itself; terminating an SSH process does not stop pod billing.

