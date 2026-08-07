# RunPod vLLM troubleshooting

## Contents

- [Known-good ISEP baseline](#known-good-isep-baseline)
- [CUDA and driver mismatch](#cuda-and-driver-mismatch)
- [Hugging Face authentication](#hugging-face-authentication)
- [Slow Qwen GDN startup](#slow-qwen-gdn-startup)
- [Out of memory while co-serving](#out-of-memory-while-co-serving)
- [Endpoint and timeout failures](#endpoint-and-timeout-failures)
- [Thinking and context limits](#thinking-and-context-limits)
- [Operational safeguards](#operational-safeguards)

## Known-good ISEP baseline

The completed local experiments used one H200 with about 139.8 GiB visible
VRAM, vLLM 0.23.0, PyTorch 2.11.0+cu130, BF16, a 32,768-token runtime context,
and eight concurrent sequences per model. Qwen 3.6 27B used port 8000 and GPU
memory utilization 0.65; Qwen 3.5 4B used port 8002 and 0.15.

This is evidence for that exact pod and software stack, not a universal sizing
rule. Re-run preflight checks on every new pod.

## CUDA and driver mismatch

Observed setup: RunPod driver branch 570 with a CUDA 13 PyTorch build required
NVIDIA's `cuda-compat-13-0` forward-compatibility package and the compatibility
libraries on `LD_LIBRARY_PATH`.

Symptoms include `torch.cuda.is_available()` returning false, CUDA driver
version errors, or vLLM failing before weight loading.

Do not replace the host driver inside the container. First capture:

```bash
nvidia-smi
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Only if the installed Torch CUDA runtime is newer than the driver-supported
runtime, install the matching NVIDIA forward-compatibility package in the
container and prepend its `compat` directory to `LD_LIBRARY_PATH`. Re-test
Torch before attempting vLLM again. Prefer a RunPod image whose CUDA runtime is
already compatible when creating a fresh pod.

## Hugging Face authentication

The warning about unauthenticated Hub requests means the vLLM process did not
inherit a usable token, even if an interactive shell was authenticated.

Check presence without printing the secret:

```bash
test -n "${HF_TOKEN:-}" && echo HF_TOKEN-present || echo HF_TOKEN-missing
uv run hf auth whoami
```

Pass `HF_TOKEN` through RunPod secrets or the server process environment. Never
put it in a command-line flag, log, committed `.env`, shell history, or result
JSON. Public models can download without a token, but authenticated downloads
have better limits and reliability.

## Slow Qwen GDN startup

Qwen 3.5/3.6 hybrid models may spend a long time compiling many FlashInfer GDN
prefill variants. The successful ISEP run selected Triton explicitly:

```bash
--gdn-prefill-backend triton
```

Use this as the default for those model families on the pinned vLLM runtime.
If upgrading vLLM, verify the option against that version and rerun a smoke
test; kernel defaults and compatibility change over time.

## Out of memory while co-serving

`gpu_memory_utilization` is per server, not a global scheduler. Two values that
sum close to 1.0 leave insufficient room for model-independent allocations.

Start the largest model first. Wait for readiness, inspect `nvidia-smi`, then
start the next server. On the proven H200 layout, 0.65 + 0.15 was stable. If a
server fails:

1. stop only the failed/verified PID;
2. confirm no orphaned engine workers remain;
3. lower memory utilization or concurrency;
4. shorten runtime context only if the benchmark output/context contract still
   fits;
5. avoid changing multiple variables in the same comparison run.

## Endpoint and timeout failures

`GET /metrics` entries are normal monitoring scrapes and do not represent
extra inference cases.

A listening TCP port is not proof that the model is ready. Require both:

- `GET /health` returns success;
- `GET /v1/models` contains the requested served model ID.

Use streaming for long generations and disable SDK retries for local vLLM.
Otherwise a client timeout can duplicate an expensive request while the first
generation continues server-side. The repository backend already implements
streaming, preflight, and `max_retries=0`.

If startup times out, inspect the server log before increasing the timeout.
Distinguish active weight download/kernel compilation from a crashed engine.

## Thinking and context limits

Thinking can produce very long traces without improving visual accuracy. Keep
the primary benchmark condition explicit (`on` or `off`) and use the same task
IDs for ablations.

The maximum completion must remain below `max_model_len` after accounting for
the system prompt, user prompt, image tokens, and schema. The ISEP runtime used
32,768 context so a 14,336-token thinking-on completion cap still had input
headroom. Do not solve truncation by increasing completion tokens beyond the
runtime context.

## Operational safeguards

- Keep one log and one PID file per model/port.
- Never kill broad process patterns without inspecting exact PIDs.
- Never expose an unauthenticated `0.0.0.0` vLLM endpoint to the internet.
- Pin model revisions for final thesis measurements; `main` can change.
- Keep model caches on persistent `/workspace` storage when available.
- Stop the RunPod in the control plane after work; SSH exit is not shutdown.

