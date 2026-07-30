# Multimodal benchmark pipeline

This package runs the project's three dermatology benchmarks through one
reproducible command-line pipeline. Model and benchmark YAML files are loaded
into validated dataclasses, images are resolved from direct files, ZIP files,
or embedded Parquet columns, inference is delegated to the selected backend,
and task-specific adapters validate and score only the model's final JSON.

The pipeline is intended for thesis experiments, not for clinical deployment
or patient-care decisions.

## What can be run

| Benchmark ID | Purpose | Default evaluation set | Output cap |
| --- | --- | --- | ---: |
| `visual_top_k_closed_set` | Rank six diseases from the fixed 21-class taxonomy | `internal_benchmark_1000` | 8,192 |
| `visual_disease_confusion_sets` | Rank three candidates under paired low/high-confusability conditions | `paired_confusion_tasks` | 8,192 |
| `evidence_grounded_diagnosis` | Findings, observation-only description, six diagnoses, confidence, and evidence links | `external_ddi_evidence` | 8,192 |

All three protocols use `prompt_only` structured output. A provider is not
given constrained JSON decoding when another model cannot receive the same
constraint. This keeps the comparison focused on the models rather than on a
provider-specific output feature.

The output caps are intentionally much larger than the expected JSON. They
provide a finite execution boundary without encouraging terse answers. Before
freezing a final experiment, run a pilot and confirm that `truncated_output`
is effectively absent. A cap hit is recorded as an invalid, truncated case;
the pipeline does not repair it, continue it, or give that model another
attempt.

## Image normalization

Every benchmark model receives the same deterministic image representation.
The `image_preprocessing` section in each benchmark YAML corrects EXIF
orientation, converts the image to RGB, limits the longest edge to 768 pixels,
and encodes it as JPEG. Encoding starts at quality 85 and is reduced in
five-point steps; resolution is reduced only when needed to keep the encoded
image at or below 45,000 bytes.

The byte budget is required because some OpenAI-compatible Azure gateways
limit base64 data URLs to approximately 64 KB. A 45,000-byte image becomes a
data URL below that transport limit. Applying the profile at benchmark level,
rather than inside one provider backend, prevents API and local models from
receiving different pixels. The profile and its parameters are retained in
the benchmark config snapshot for every run.

## Runtime architecture

```text
configs/models/*.yaml ──> typed ModelConfig ──> inference backend
                                                     │
configs/benchmarks/*.yaml ─> typed BenchmarkConfig   │
              │                                      │
              ├─ prompt + schema + taxonomies        │
              └─ Parquet manifest ─> stable subset   │
                                      │              │
                                      └─ task adapter
                                             │
                                  final JSON + reasoning
                                             │
                         deterministic validation and metrics
                                             │
                              append-safe run artifacts
```

The subset is selected by the lowest SHA-256 scores derived from the benchmark
release hash, seed, and case ID. It never depends on the model ID. For the
confusion benchmark, `--limit N` selects `N` image pairs and therefore runs
`2N` tasks.

## Backend compatibility

The local path targets vLLM `0.23.0` on Linux x86-64 with NVIDIA CUDA. The
configuration and request-building paths are unit-tested on macOS, but loading
the weights and measuring GPU memory must be validated on the actual compute
machine.

| Model configuration | Default transport | Managed vLLM | Important behavior |
| --- | --- | --- | --- |
| `qwen_3_5_4b` | vLLM | Yes | Qwen reasoning parser and official sampling recipe |
| `qwen_3_5_9b` | vLLM | Yes | Qwen reasoning parser and official sampling recipe |
| `qwen_3_6_27b` | vLLM | Yes | Qwen reasoning parser; requires substantially more GPU memory |
| `gemma_4_e4b_it` | vLLM | Yes | Gemma 4 reasoning parser and sampling recipe |
| `gemma_4_31b_it` | vLLM | Yes | Gemma 4 reasoning parser; large-model GPU requirements |
| `minicpm_v_4_6` | vLLM | Yes | MiniCPM image processor arguments are forwarded |
| `medgemma_1_5_4b` | vLLM | Yes | Gated weights; image-first prompt and no separate system role |
| `kimi_k2_6` | Azure Chat Completions | No | Optional existing `vllm_endpoint` profile |
| `gpt_5_6_luna` | Azure Responses | No | Official reasoning summary only |

“Managed” means the CLI may start and stop `vllm serve`. It does not mean the
repository downloads a model in advance or provisions a GPU. Kimi's optional
vLLM profile points to an already hosted OpenAI-compatible endpoint because
the checked-in source is a provider model, not a local Hugging Face weight
repository. GPT uses the Azure Responses API and cannot use vLLM.

See [the inference package](../inference/README.md) for transport and
multimodal message details.

## Installation

Configuration inspection, dataset validation, dry runs, and tests work on
macOS:

```bash
uv sync
```

On the Linux NVIDIA machine that will host local models:

```bash
uv sync --extra gpu
```

Fine-tuning uses a separate environment:

```bash
uv sync --extra training
```

The `gpu` and `training` extras are intentionally marked as conflicting.
vLLM 0.23 and the current Unsloth release require incompatible
Torch/Transformers combinations. Benchmark inference and fine-tuning should
therefore use separate virtual environments rather than an unreproducible
mixture of both stacks.

Accept the MedGemma terms on Hugging Face before running that model, and make
`HF_TOKEN` available to the vLLM process. Never place tokens in YAML files or
command-line arguments.

## Discover configurations

```bash
uv run python -m src.benchmark.cli list-models
uv run python -m src.benchmark.cli list-benchmarks
```

IDs from the first column can be passed directly to `--model` or
`--benchmark`. YAML filenames and paths are also accepted.

After `uv sync`, `uv run isep-benchmark ...` is an equivalent shorter entry
point for every `python -m src.benchmark.cli ...` command shown below.

## Validate before inference

`--dry-run` validates both typed configs, prompt and schema references,
taxonomy compatibility, the Parquet schema, deterministic selection, and
access to every selected image. It does not start vLLM, load weights, use a
GPU, require provider credentials, or make a network request.

```bash
uv run python -m src.benchmark.cli run \
  --model qwen_3_5_4b \
  --benchmark visual_top_k_closed_set \
  --limit 10 \
  --dry-run
```

The same check for the two newer tasks:

```bash
uv run python -m src.benchmark.cli run \
  --model qwen_3_5_4b \
  --benchmark visual_disease_confusion_sets \
  --limit 10 \
  --dry-run

uv run python -m src.benchmark.cli run \
  --model gpt_5_6_luna \
  --benchmark evidence_grounded_diagnosis \
  --limit 10 \
  --dry-run
```

## Run against an existing vLLM server

Start vLLM separately with the model and arguments represented by its YAML,
then provide its OpenAI-compatible base URL:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1

uv run python -m src.benchmark.cli run \
  --model qwen_3_5_4b \
  --benchmark visual_top_k_closed_set \
  --evaluation-set internal_benchmark_1000 \
  --limit 100 \
  --seed 42
```

An explicit `--base-url` overrides `VLLM_BASE_URL`. The executor sends
concurrent HTTP requests according to the benchmark batch size so vLLM can
perform continuous batching.

## Start a managed vLLM server

On a compatible Linux CUDA machine:

```bash
uv run python -m src.benchmark.cli run \
  --model medgemma_1_5_4b \
  --benchmark evidence_grounded_diagnosis \
  --limit 100 \
  --server-mode managed \
  --startup-timeout 900
```

The exact argument vector is built without a shell. Dtype, tensor parallelism,
context length, GPU-memory utilization, image limits, reasoning parser, and
MiniCPM processor options come from the model YAML. Server stdout and stderr
are written to `vllm_server.log` inside the run directory.

### Qwen3.5-2B pipeline smoke test

`src/test_qwen.py` runs a real one-case `visual_top_k` test through the same
configuration loader, dataset selection, prompt rendering, vLLM backend,
response validation, metrics, and result writer used by full benchmark runs.
The 2B model has a smoke-only config and is not restored to the main teacher
shortlist.

Validate everything except model inference:

```bash
uv run src/test_qwen.py --dry-run
```

Run direct inference on Apple MPS while preserving
`enable_thinking: true`:

```bash
uv run src/test_qwen.py --transformers --limit 10
```

Start and stop vLLM automatically on a compatible Linux CUDA machine:

```bash
uv run --extra gpu src/test_qwen.py
```

Alternatively, use an existing vLLM server:

```bash
vllm serve Qwen/Qwen3.5-2B \
  --max-model-len 16384 \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image": 1}'

uv run src/test_qwen.py --base-url http://127.0.0.1:8000/v1
```

## Provider APIs

Kimi's default Azure Chat Completions profile uses:

```text
KIMI_K2_6_AZURE_ENDPOINT
KIMI_K2_6_AZURE_API_KEY
KIMI_K2_6_AZURE_DEPLOYMENT
KIMI_K2_6_AZURE_API_VERSION
```

Run it with:

```bash
uv run python -m src.benchmark.cli run \
  --model kimi_k2_6 \
  --benchmark visual_top_k_closed_set \
  --limit 100
```

Its optional existing vLLM endpoint uses:

```text
KIMI_K2_6_VLLM_BASE_URL
KIMI_K2_6_VLLM_API_KEY
KIMI_K2_6_VLLM_MODEL
```

```bash
uv run python -m src.benchmark.cli run \
  --model kimi_k2_6 \
  --backend-profile vllm_endpoint \
  --benchmark visual_top_k_closed_set \
  --limit 100
```

GPT's Azure Responses profile uses:

```text
GPT_5_6_LUNA_AZURE_ENDPOINT
GPT_5_6_LUNA_AZURE_API_KEY
GPT_5_6_LUNA_AZURE_DEPLOYMENT
GPT_5_6_LUNA_AZURE_API_VERSION
```

```bash
uv run python -m src.benchmark.cli run \
  --model gpt_5_6_luna \
  --benchmark evidence_grounded_diagnosis \
  --limit 100
```

Provider-managed sampling fields are omitted for Kimi and GPT. Their YAML
files do not invent local-model temperature or penalty values.
Both provider configurations explicitly freeze `reasoning_effort: high`.
The runtime sends it as `reasoning_effort="high"` to Kimi Chat Completions
and as `reasoning.effort="high"` to GPT Responses.

## Reasoning capture

The default `--reasoning-capture available` keeps the richest public channel
that the backend actually exposes:

| Backend | Stored level |
| --- | --- |
| vLLM or Chat Completions with a reasoning field | Full provider-exposed field |
| Azure Responses | Official reasoning summary |
| Provider exposing only usage | Reasoning-token count, when available |
| Provider exposing none | No reasoning |

The explicit alternatives are `full`, `summary`, `tokens_only`, and `none`.
“Full” never means hidden chain of thought: it means only a reasoning field
deliberately returned by the endpoint. Azure Responses is downgraded to its
official summary because raw chain of thought is not part of that API.

Reasoning is stored at:

```text
response.reasoning.capture_mode
response.reasoning.availability
response.reasoning.text
response.reasoning.token_count
response.reasoning.source
```

The answer used by validators is stored separately in
`response.final_text` and `response.parsed_output`. Reasoning is never parsed
as the benchmark schema and never contributes to a metric. Keep the same
capture setting across compared runs, because requesting a provider summary
can change cost or latency.

Use `--reasoning-capture none` if the output may contain sensitive material.
Run outputs are ignored by Git by default.

## Results and resume

Each new run is written under:

```text
outputs/benchmark_runs/<benchmark_id>/<model_id>/<timestamp>_<hash>/
├── run_manifest.yaml
├── config_snapshot.yaml
├── selection.json
├── environment.json
├── rendered_prompts.jsonl
├── predictions.jsonl
├── metrics.json
└── vllm_server.log
```

Every case is appended and flushed as soon as it reaches a terminal status:
`ok`, `invalid_output`, `truncated_output`, `backend_error`, or
`image_error`. A backend or image failure affects that case rather than
silently removing it from the denominator.

Resume an interrupted or failed run with its exact directory:

```bash
uv run python -m src.benchmark.cli run \
  --model qwen_3_5_4b \
  --benchmark visual_top_k_closed_set \
  --limit 100 \
  --seed 42 \
  --resume outputs/benchmark_runs/visual_top_k_closed_set/qwen_3_5_4b/<run>
```

Resume is refused when the model config, benchmark config, prompt, schema,
taxonomy, dataset, selected IDs, output cap, backend profile, endpoint
binding, batch size, reasoning policy, or seed has changed. A truncated final
JSONL line from an interrupted write is ignored; malformed earlier records
remain a hard error.

## Deterministic scoring

The Top-K and confusion tasks calculate ranking accuracy, reciprocal rank,
macro-F1, and structural compliance. Confusion tasks also preserve the paired
low/high-confusability unit and calculate the paired gap with a fixed
bootstrap seed.

The evidence-grounded task calculates:

- controlled morphology overlap;
- morphology recovered from the free clinical description;
- description/findings consistency and forbidden-content rate;
- Top-1/3/6 diagnosis accuracy, reciprocal rank, and covered-class macro-F1;
- evidence-reference validity and visible-evidence grounding;
- confidence calibration;
- JSON, schema, and cross-field semantic compliance.

BERTScore and subgroup analyses are declared as deferred in the benchmark
YAML and are not silently approximated. The implementation uses frozen
concept aliases and deterministic rules; it does not ask another language
model to judge the answer.

## Tests

Run the complete suite:

```bash
uv run python -m unittest discover -s tests
```

The CLI and inference tests use injected clients or `--dry-run`; they do not
call external APIs or download model weights.
