# Modal inference

This directory contains reproducible Modal applications for running the
dermatology benchmark pipeline on cloud GPUs.

## Qwen 3.6 27B

`qwen_3_6_27b.py` exposes `Qwen/Qwen3.6-27B` through vLLM's
OpenAI-compatible API on one NVIDIA A100 80 GB. Its local entrypoint calls
the same benchmark runner used for Luna and Kimi. The application:

- retains Hugging Face weights and vLLM cache data in Modal Volumes;
- accepts one image plus text per benchmark request;
- disables Qwen thinking while retaining its general-task sampling settings;
- uses the generation parameters from `configs/models/qwen_3_6_27b.yaml`;
- allows up to 20 minutes for each long thinking request and disables silent
  SDK retries for vLLM requests;
- streams vLLM deltas back to the local runner so long generations remain
  active through Modal's HTTP proxy;
- writes the standard `predictions.jsonl`, metrics, run manifest, and
  `report.html` artifacts;
- shuts the GPU container down after 60 seconds without another invocation.

Authenticate the local Modal client once:

```bash
modal setup
```

Run the default ten-case `visual_top_k` smoke test:

```bash
modal run src/modal/qwen_3_6_27b.py
```

Choose another case count:

```bash
modal run src/modal/qwen_3_6_27b.py \
  --limit 100
```

Run the evidence-grounded benchmark:

```bash
modal run src/modal/qwen_3_6_27b.py \
  --benchmark evidence_grounded_diagnosis \
  --evaluation-set external_ddi_evidence \
  --limit 10
```

Validate the complete benchmark setup without starting a GPU:

```bash
modal run src/modal/qwen_3_6_27b.py --dry-run
```

The first invocation is slower because Modal must build the runtime image and
download the model. Later invocations reuse the persistent model cache. The
server currently uses vLLM eager execution because TorchInductor compilation
fails for this model/runtime combination on Modal. The Hugging Face revision
follows `main` to match the model YAML; pin both files to a commit before
collecting final thesis results.

The benchmark runner remains local. It reads and normalizes the local images,
sends the resulting image-and-prompt requests to the temporary Modal endpoint,
and stores outputs under `outputs/benchmark_runs/`. The endpoint created by
`modal run` is ephemeral; use `modal deploy` only when a persistent service is
actually required.

## Qwen 3.5 9B

`qwen_small_9b.py` runs `Qwen/Qwen3.5-9B` in BF16 on one NVIDIA L40S.
The 48 GB GPU comfortably accommodates the model and its 16,384-token context
while costing less than the A100 80 GB used by the larger candidates. Thinking
is disabled at both the vLLM server and request levels, while the sampling
parameters come from `configs/models/qwen_small_9b.yaml`.

Run the default ten-case visual Top-K smoke test:

```bash
modal run src/modal/qwen_small_9b.py
```

Run another benchmark or evaluation set:

```bash
modal run src/modal/qwen_small_9b.py \
  --benchmark evidence_grounded_diagnosis \
  --evaluation-set external_ddi_evidence \
  --limit 10
```

Validate the configuration locally without allocating an L40S:

```bash
modal run src/modal/qwen_small_9b.py --dry-run
```

## Gemma 4 31B IT

`gemma_4_31b_it.py` runs `google/gemma-4-31B-it` in BF16 on a dedicated
A100 80 GB. It uses the Gemma 4 reasoning parser, but thinking is disabled in
both the server default and the per-request model configuration. The launcher
uses the same persistent Hugging Face and vLLM cache volumes as Qwen while
running in an independent Modal App, so both models can run simultaneously.

Run the default ten-case visual Top-K smoke test:

```bash
modal run src/modal/gemma_4_31b_it.py
```

Run ten paired confusion-set images, producing twenty low/high tasks:

```bash
modal run src/modal/gemma_4_31b_it.py \
  --benchmark visual_disease_confusion_sets \
  --evaluation-set paired_confusion_tasks \
  --limit 10
```

Run ten evidence-grounded DDI cases:

```bash
modal run src/modal/gemma_4_31b_it.py \
  --benchmark evidence_grounded_diagnosis \
  --evaluation-set external_ddi_evidence \
  --limit 10
```

Validate locally without allocating the A100:

```bash
modal run src/modal/gemma_4_31b_it.py --dry-run
```

## MedGemma 1.5 4B, MiniCPM-V 4.6, Gemma 4 E4B, and Qwen 3.5 4B

The four small-model launchers use independent Modal Apps on NVIDIA L40S
GPUs:

| Launcher | Hugging Face model | Default concurrency |
| --- | --- | ---: |
| `medgemma_1_5_4b.py` | `google/medgemma-1.5-4b-it` | 8 |
| `minicpm_v_4_6.py` | `openbmb/MiniCPM-V-4.6` | 4 |
| `gemma_4_e4b_it.py` | `google/gemma-4-E4B-it` | 8 |
| `qwen_small_4b.py` | `Qwen/Qwen3.5-4B` | 8 |

All four attach the existing `huggingface-secret` Modal secret and share the
persistent Hugging Face and vLLM cache volumes. MedGemma therefore receives
the token required for its gated repository, while public-model downloads
also avoid anonymous Hub rate limits. MiniCPM forwards its configured
`downsample_mode` and `max_slice_nums` processor arguments. Gemma uses the
Gemma 4 reasoning parser with thinking disabled.

MedGemma may emit reasoning inside the content channel between `<unused94>`
and `<unused95>`. The shared inference client separates that block into
`response.reasoning` before validation and keeps only the remaining content
as the final answer. MedGemma is intentionally restricted to `prompt_only`;
vLLM JSON Schema constraints caused prolonged generation and truncation on
the evidence-grounded schema. MiniCPM's ordered string-list JSON is retained
as the strict raw output and additionally projected into a canonical
ranked-object view; reports therefore show both strict and canonical metrics.

The MiniCPM image applies upstream vLLM fix `aa1df36c` on top of vLLM 0.23.0.
The released wheel accessed `image_processor.version`, but the current
Transformers 5.7+ `MiniCPMV4_6ImageProcessor` no longer exposes that
attribute. The pinned one-line fallback is the change merged upstream in
vLLM PR `#44980`; keeping it explicit avoids an unpinned nightly wheel.

Run exactly ten scored tasks from each of the three benchmarks while keeping
one model server alive:

```bash
modal run src/modal/medgemma_1_5_4b.py \
  --all-benchmarks --limit 10

modal run src/modal/minicpm_v_4_6.py \
  --all-benchmarks --limit 10

modal run src/modal/gemma_4_e4b_it.py \
  --all-benchmarks --limit 10
```

For MiniCPM, Gemma E4B, and Qwen 4B, compare prompt-only generation with vLLM
JSON Schema constraints on exactly the visual Top-K and evidence-grounded
benchmarks:

```bash
modal run src/modal/<launcher>.py \
  --evidence-and-top-k \
  --structured-output both \
  --limit 10
```

`both` creates independent benchmark runs for `prompt_only` and
`json_schema`; the condition is stored in each run manifest. Use `--dry-run`
to validate all four combinations without starting the model server.

Run the same two-benchmark suite for MedGemma in prompt-only mode:

```bash
modal run src/modal/medgemma_1_5_4b.py \
  --evidence-and-top-k \
  --structured-output prompt_only \
  --limit 10
```

For the paired confusion benchmark, the suite selects five pairs to produce
ten scored low/high-confusability tasks. Outside suite mode, `--limit` retains
the benchmark CLI's native meaning and therefore selects pairs.

Validate all three benchmark configurations without allocating a GPU:

```bash
modal run src/modal/medgemma_1_5_4b.py \
  --all-benchmarks --limit 10 --dry-run
```
