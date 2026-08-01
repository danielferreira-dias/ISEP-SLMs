# Multimodal benchmark pipeline

This package runs the project's four frozen ISEPDermaBench protocols through
one reproducible command-line pipeline. Model YAML files are loaded into
validated dataclasses. Benchmark images, rendered prompts, and response
schemas come directly from the task Parquets; gold references are loaded from
the corresponding isolated `_references` configuration and joined only inside
the scorer.

The pipeline is intended for thesis experiments, not for clinical deployment
or patient-care decisions.

## What can be run

| Benchmark ID | Purpose | Default evaluation set | Output cap |
| --- | --- | --- | ---: |
| `visual_top_k_closed_set` | Rank six diseases from the fixed 21-class taxonomy | `internal_benchmark` | 8,192 |
| `visual_disease_confusion_sets` | Rank three candidates under paired low/high-confusability conditions | `internal_benchmark` | 8,192 |
| `evidence_grounded_diagnosis` | Findings, observation-only description, six diagnoses, confidence, and evidence links | `internal_benchmark` | 8,192 |
| `open_ended_diagnosis` | Natural clinical prose with visible findings and an explicitly ranked Top-3 | `internal_benchmark` | 8,192 |

The three structured protocols use `prompt_only` structured output. A provider is not
given constrained JSON decoding when another model cannot receive the same
constraint. This keeps the comparison focused on the models rather than on a
provider-specific output feature.

`open_ended_diagnosis` deliberately has no response schema and exposes no
candidate taxonomy. The evaluated answer is retained verbatim and is scored
in a second command by one blinded GPT-5.6 Luna judge. The judge sees the
image, correct diagnosis, optional exact-match SKINCON/SkinCAP references, and
only the final user-visible answer. It never receives the evaluated model's
identity or provider reasoning.

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

## Browse generated answers

Every completed CLI run automatically creates a self-contained `report.html`
next to `predictions.jsonl`. Open it in any browser; it does not need a
Jupyter kernel or a running web server. The report provides:

- search and filters for status, reference disease, source dataset, and skin
  tone;
- paginated cases with an embedded image thumbnail;
- the final answer and provider-returned reasoning in separate panes;
- parsed output, validation errors, token usage, prompts, and metadata.

`predictions.jsonl` remains the authoritative machine-readable artifact. The
HTML file is a read-only derived view and can be regenerated at any time:

```bash
uv run python -m src.benchmark.report \
  outputs/benchmark_runs/<benchmark>/<model>/<run>
```

Use `--no-images` for a smaller report or `--output PATH` to choose another
destination.

### Skin-tone performance in visual Top-K

The `visual_top_k_closed_set` scorer reports both exact and prespecified
aggregate skin-tone results. Exact labels are scale-qualified, for example
`fitzpatrick:FST_3` and `monk:MST_3`; values from Fitzpatrick and Monk are
never merged merely because their numbers match. Aggregate reporting uses
`FST_1-2`, `FST_3-4`, `FST_5-6` and `MST_1-3`, `MST_4-6`, `MST_7-10`.

Each row in the `by_skin_tone` and `by_skin_tone_aggregate` report tables
contains the image and independent-group counts, Top-1/3/6 accuracy with
95% Wilson intervals, mean reciprocal rank, and a disease-adjusted Top-1
summary where enough per-disease groups exist. Missing annotations remain in
the global score and appear separately as `unknown`.

`statistically_supported` is true only when the subgroup reaches the
configured minimum number of unique leakage groups. The worst-group accuracy
and best-to-worst gap exclude `unknown` and unsupported exact groups. These
descriptive fairness metrics must always be interpreted with their sample
counts and intervals; the report does not treat the skin-tone annotation as a
measured biological attribute.

## Runtime architecture

```text
configs/models/*.yaml ──> typed ModelConfig ──> inference backend
                                                     │
ISEPDermaBench tasks ──> image + frozen request      │
              │                                      │
              └─ isolated references ─> stable subset│
                                      │              │
                                      └─ task adapter
                                             │
                                  final JSON + reasoning
                                             │
                         deterministic validation and metrics
                                             │
                              append-safe run artifacts
```

For the open-ended protocol, the final free-text answer follows a separate
stage:

```text
final prose + image + isolated reference
                 │
                 └──> single blinded Luna judge ──> judge metrics + HTML
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
`--benchmark`. The benchmark loader uses the local
`data/benchmarks/ISEPDermaBench` mirror by default and falls back to the
private Hugging Face dataset. Use `--benchmark-source local` or
`--benchmark-source hub` to require one source explicitly.

After `uv sync`, `uv run isep-benchmark ...` is an equivalent shorter entry
point for every `python -m src.benchmark.cli ...` command shown below.

## Validate before inference

`--dry-run` validates the model config, ISEPDermaBench task/reference
contract, embedded response schema, deterministic selection, and every
selected image. It does not start vLLM, load weights, use a GPU, require
provider credentials, or make a network request when the local mirror exists.

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

uv run python -m src.benchmark.cli run \
  --model gpt_5_6_luna \
  --benchmark open_ended_diagnosis \
  --evaluation-set validation \
  --limit 10 \
  --dry-run
```

## Open-ended diagnosis and single-judge scoring

First run a model normally. Its response must be concise clinical prose with
exactly three clearly ranked diagnoses; JSON and private chain-of-thought are
not requested.

```bash
uv run python -m src.benchmark.cli run \
  --model qwen_3_5_4b \
  --benchmark open_ended_diagnosis \
  --evaluation-set validation \
  --limit 100
```

Then judge the completed run directory. A dry-run validates all task,
reference, prompt, and judge-schema joins without calling the selected judge:

```bash
uv run python -m src.benchmark.cli judge \
  --run outputs/benchmark_runs/open_ended_diagnosis/qwen_3_5_4b/<run> \
  --dry-run

uv run python -m src.benchmark.cli judge \
  --run outputs/benchmark_runs/open_ended_diagnosis/qwen_3_5_4b/<run>
```

The second command writes `judgments.jsonl`, `judge_metrics.json`,
`judge_manifest.yaml`, and `judge_report.html` beside the original run. The
principal metrics are judge Top-1/Top-3 accuracy and reciprocal rank, plus
0–4 scores for diagnosis correctness, visible findings, evidence grounding,
clinical-rationale quality, and differential quality. Unsupported claims and
the overall-verdict distribution are also reported. These metrics depend on
the fixed judge and prompt; they must be reported as judge-based estimates,
not deterministic ground truth.

Luna remains the default judge. An alternative judge candidate can be tested
without overwriting Luna's artifacts:

```bash
uv run python -m src.benchmark.cli judge \
  --run outputs/benchmark_runs/open_ended_diagnosis/qwen_3_5_4b/<run> \
  --judge-model qwen_3_7_flash_openrouter
```

Alternative outputs are isolated under
`judges/<judge_model_id>/`. Judge comparisons are a protocol-development
experiment: select one judge and freeze its model, prompt, schema, and decoding
settings before reporting final benchmark results.

For the frozen primary-judge protocol, Qwen 3.7 Flash can be enabled only as a
content-policy fallback for Luna:

```bash
uv run python -m src.benchmark.cli judge \
  --run outputs/benchmark_runs/open_ended_diagnosis/qwen_3_5_4b/<run> \
  --judge-model gpt_5_6_luna \
  --fallback-judge-model qwen_3_7_flash_openrouter
```

The fallback is not used for ordinary transport errors, timeouts, invalid JSON,
or a clinical disagreement. Those conditions remain attributable to the
primary judge. Combined artifacts are isolated under
`judges/gpt_5_6_luna__fallback_qwen_3_7_flash_openrouter/`. Each judgment stores
the primary and effective judge plus the fallback reason. The metrics and HTML
report include judge usage, fallback rate, invalid-judgment count, and score
summaries separated by effective judge.

Before the sealed run, use the deterministic 50-case calibration subset for
every candidate model:

```bash
uv run python -m src.benchmark.cli run \
  --model <candidate_model> \
  --benchmark open_ended_diagnosis \
  --evaluation-set validation \
  --limit 50 \
  --seed 42
```

Then run the primary/fallback judge command above on each completed directory.
The selected 50-case subset covers all 21 diseases in the frozen local release.
Do not change the seed between models.

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
asynchronous HTTP requests through `AsyncOpenAI`, limited by the benchmark
batch size, so vLLM can perform continuous batching. Results are consumed
with `asyncio.as_completed` and persisted as each request finishes.

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

Run direct inference on Apple MPS with thinking disabled while preserving the
configured general-task sampling parameters:

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

Kimi K2.6 runs without extended thinking, with temperature `0.6` and top-p
`0.95`. Moonshot's API uses `thinking: {type: disabled}`, but the tested
Direct-from-Azure gateway rejects that field and accepts
`reasoning_effort: none`; the model YAML records this profile-specific
mapping. Its vLLM endpoint maps disabled thinking to
`chat_template_kwargs.thinking: false`. GPT uses `reasoning_effort: high`,
sent as `reasoning.effort` to Responses.

Use `--structured-output json_schema` for a separate constrained-output Luna
run. This does not alter the default prompt-only comparison track. The
provider schema copy omits `uniqueItems`, which Azure Structured Outputs does
not accept; deterministic post-generation validation still applies the full
project contract.

### OpenRouter profiles

`gpt_5_6_luna` and `gemma_4_31b_it` also define optional `openrouter`
profiles. Both use the OpenAI-compatible Chat Completions endpoint and require:

```text
OPENROUTER_API_KEY
```

The profiles pin the provider model slugs rather than relying on environment
defaults:

```text
openai/gpt-5.6-luna-pro
google/gemma-4-31b-it:free
```

OpenRouter receives text before the base64 image, its unified
`reasoning` object is used instead of provider-specific thinking fields, and
unsupported extended sampling fields such as `top_k` are omitted. Gemma uses
temperature `1.0`, top-p `0.95`, and reasoning disabled. Luna Pro maps the
configured high effort to `reasoning: {effort: high}`.

Reproduce an image-level provider refusal with the frozen benchmark task:

```bash
uv run python -m src.test_openrouter_image \
  --model gpt_5_6_luna \
  --reasoning-effort none
```

Run the free Gemma endpoint conservatively because its shared upstream pool
can return HTTP 429 even when the OpenRouter account itself has quota:

```bash
uv run python -m src.benchmark.cli run \
  --model gemma_4_31b_it \
  --backend-profile openrouter \
  --benchmark visual_top_k_closed_set \
  --evaluation-set validation \
  --limit 10 \
  --batch-size 1 \
  --request-interval-seconds 6
```

`--request-interval-seconds` spaces request starts and is included in the
immutable run identity. It controls client-side pacing but cannot eliminate a
429 caused by exhaustion of OpenRouter's shared upstream free-provider pool.

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

Requesting an Azure Responses summary does not guarantee that its
`output[*].summary` array will contain text for every response. When the
provider returns an empty summary array but reports reasoning-token usage,
the run records `availability: tokens_only`; it does not fabricate a summary
or mistake the requested summary mode for generated text.

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

For MedGemma, a configured model-specific parser separates content enclosed
by `<unused94>` and `<unused95>` before benchmark validation. The extracted
text is retained under `response.reasoning`; only text outside that complete
block is eligible to become the final JSON answer. Reasoning is never used to
repair or infer a missing diagnosis.

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
`ok`, `format_invalid`, `schema_invalid`, `semantic_noncompliant`,
`truncated_output`, `safety_refusal`, `backend_error`, or `image_error`.
The precedence is truncation, strict JSON format, schema, semantic contract,
and finally `ok`. `invalid_output` remains readable only for compatibility
with older artifacts; reports map it to the corresponding granular layer. A
backend or image failure affects that case rather than silently removing it
from the denominator.

`json_validity_rate` is the strict prompt-contract metric.
`recoverable_json_validity_rate` additionally recognizes exactly one complete
Markdown JSON fence with no surrounding prose. Recovery is reported for
production diagnostics but does not turn the strict benchmark case into
`ok`. Provider content-policy blocks use `safety_refusal` and retain only
safe category, severity, code, and request-ID metadata when available.

Ranked-list benchmarks also expose a separate canonical view for transparent
post-processing analysis. A JSON array of disease IDs is deterministically
projected to the requested `{rank, disease_id}` objects because array order
fully determines rank. This produces `response.canonical_output`,
`response.canonicalization_rules`, and `canonical_*` metrics while leaving
the strict status and strict metrics unchanged. The canonicalizer never
extracts labels from prose or reasoning.

Evidence-grounded clinical metrics use recoverable, individually valid fields
from the parsed response. They are not zeroed merely because another contract
layer failed. For example, a correct Top-1 diagnosis still counts when
`case_confidence` uses the wrong band, while
`semantic_compliance_rate` records that inconsistency separately.

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

The open-ended task itself reports only response availability and length.
Clinical correctness is calculated by its explicit single-judge stage, never
by extracting disease names from prose with hidden heuristics.

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
