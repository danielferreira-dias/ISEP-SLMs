# Modal inference

This directory contains reproducible Modal applications for running the
dermatology benchmark pipeline on cloud GPUs.

## Validation teacher screening with thinking

The model launchers accept the same auditable thinking and completion-budget
overrides as the benchmark CLI. The thinking-on phase uses the fixed 100-case
Validation cohorts and a 14,336-token total generation cap:

```bash
modal run src/modal/<launcher>.py \
  --teacher-screening \
  --thinking-mode enabled \
  --max-output-tokens 14336 \
  --output-root outputs/validation_screening_v1/thinking_on
```

The override is passed to every task in the suite and recorded in each run
identity, manifest, and config snapshot. Every controllable model additionally
reads its 10,240-token reasoning budget from the model YAML. Local vLLM sends
it as `thinking_token_budget`; OpenRouter sends it as `reasoning.max_tokens`.
This leaves up to 4,096 tokens for the final answer. The published
ISEPDermaBench YAML files retain their frozen 8,192-token default.

## Qwen 3.6 27B

`qwen_3_6_27b.py` exposes `Qwen/Qwen3.6-27B` through vLLM's
OpenAI-compatible API on one NVIDIA A100 80 GB. Its local entrypoint calls
the same benchmark runner used for API and local models. The application:

- retains Hugging Face weights and vLLM cache data in Modal Volumes;
- accepts one image plus text per benchmark request;
- defaults to Qwen thinking disabled while allowing an explicit audited
  thinking override;
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

## Qwen 3.5 4B official student

`qwen_small_4b.py` runs the official student, `Qwen/Qwen3.5-4B`, in BF16 on
one NVIDIA L40S. It attaches the `huggingface-secret` Modal secret and reuses
the persistent Hugging Face and vLLM cache volumes. Thinking is disabled by
default but can still be overridden in an explicitly audited experiment.

Run exactly ten scored tasks from each of the three benchmarks while keeping
one model server alive:

```bash
modal run src/modal/qwen_small_4b.py \
  --all-benchmarks --limit 10
```

Compare prompt-only generation with vLLM JSON Schema constraints on the visual
Top-K and evidence-grounded benchmarks:

```bash
modal run src/modal/qwen_small_4b.py \
  --evidence-and-top-k \
  --structured-output both \
  --limit 10
```

`both` creates independent benchmark runs for `prompt_only` and
`json_schema`; the condition is stored in each run manifest. Use `--dry-run`
to validate all combinations without starting the model server.

For the paired confusion benchmark, the suite selects five pairs to produce
ten scored low/high-confusability tasks. Outside suite mode, `--limit` retains
the benchmark CLI's native meaning and therefore selects pairs.

Validate all benchmark configurations without allocating a GPU:

```bash
modal run src/modal/qwen_small_4b.py \
  --all-benchmarks --limit 10 --dry-run
```
