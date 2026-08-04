# Visual hallucination audit results

## Decision

The two development-only hallucination audits were executed on the same fixed
cases for every selected model. The paired thinking ablation does not support
using thinking by default for these tasks:

- thinking produced small and inconsistent changes in the general visual
  status task;
- thinking reduced dermatology counterfactual full success for every model
  that supports an off/on comparison;
- thinking substantially increased generated tokens, wall-clock time, and
  raw JSON-format failures;
- no corrected run produced a backend error, safety refusal, image error, or
  truncated output.

The recommended primary condition for the next benchmark phase is therefore
**thinking off**. Thinking on remains an auxiliary reasoning/latency ablation,
not the default teacher-selection condition.

These are development robustness results. They are not clinical accuracy
estimates and do not replace the sealed Internal Benchmark.

## Paired protocol

The compared conditions used identical task rows, images, references, prompt
versions, schemas, and deterministic ordering:

- `general_visual_hallucination_audit`: 100 fixed HaloQuest cases;
- `dermatology_counterfactual_hallucination`: 50 fixed cases, comprising 25
  pixel-shuffled images and 25 hard-negative disease-image swaps;
- thinking off: model sampling recipe with thinking disabled and an 8,192
  total completion-token cap;
- thinking on: the same sampling recipe with thinking enabled, a 10,240-token
  reasoning budget, and a 14,336 total completion-token cap.

Luna does not have a valid paired off condition in the current provider
configuration and was executed once with its fixed `high` reasoning profile.
The official Qwen 3.8 Max endpoint explicitly rejected `reasoning=off` with
HTTP 400 (`Reasoning is mandatory for this endpoint and cannot be disabled`).
A separate provider-supported `low` versus `high` Qwen 3.8 ablation was
therefore executed after the primary table was frozen.

## Prompt correction before measurement

Prompt version 1.0.0 under-specified the concrete JSON field types. The first
Qwen 3.7 off run consequently produced:

Column legend:

- **Task**: evaluated benchmark task.
- **OK**: outputs that passed JSON parsing, the response schema, and the
  task-specific semantic checks.
- **Format invalid**: outputs whose raw model response was not valid JSON,
  even if a deterministic parser could later extract a JSON object.
- **Schema invalid**: JSON objects that did not satisfy the required fields,
  types, enumerations, or task-specific structural rules.

| Task | OK | Format invalid | Schema invalid |
| --- | ---: | ---: | ---: |
| General visual audit | 8/100 | 1/100 | 91/100 |
| Dermatology counterfactual | 24/50 | 1/50 | 25/50 |

The prompts were corrected to version 1.0.1 by explicitly enumerating the
allowed strings, arrays, nullable fields, and object members. The exact same
Qwen 3.7 cases were rerun and reached 100/100 and 50/50 valid outputs. All
tables below use only the corrected 1.0.1 runs. The failed 1.0.0 runs remain
in `outputs/` as development provenance and are excluded from comparisons.

## General Visual Hallucination Audit

All values are percentages. `Halluc.` and `Overconf.` are error rates, so
lower is better. The other columns are accuracies, so higher is better.

Column legend:

- **Model**: evaluated multimodal model.
- **Condition**: `off` disables model reasoning, `on` enables it, and
  `fixed high` means the provider configuration exposes only the recorded
  high-reasoning condition, without a paired off run.
- **Status**: percentage of all 100 cases where `question_status` matched the
  reference category: `false_premise`, `insufficient_visual_evidence`, or
  `answerable`.
- **Unanswerable**: clean-abstention accuracy over the false-premise and
  insufficient-context cases. The status must be correct and the answer must
  be `null`.
- **False premise**: percentage of false-premise cases correctly rejected.
- **Insufficient context**: percentage of cases correctly identified as
  lacking enough visual evidence.
- **Visual challenge**: percentage of answerable visual-challenge cases
  correctly classified as answerable. This does not judge whether the
  free-text answer itself is correct.
- **Halluc.**: unanswerable-case hallucination rate. A case is counted when
  the model marks it answerable or supplies a non-empty answer despite the
  missing or contradictory visual evidence.
- **Overconf.**: percentage of unanswerable cases where the model failed to
  abstain cleanly and nevertheless reported high confidence.

| Model | Condition | Status | Unanswerable | False premise | Insufficient context | Visual challenge | Halluc. | Overconf. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 4B | off | 59.0 | 54.3 | 48.0 | 70.0 | 70.0 | 8.6 | 17.1 |
| Qwen 3.5 4B | on | 71.0 | 65.7 | 56.0 | 90.0 | 83.3 | 14.3 | 17.1 |
| Qwen 3.6 27B | off | 78.0 | 78.6 | 80.0 | 75.0 | 76.7 | 7.1 | 21.4 |
| Qwen 3.6 27B | on | 81.0 | 81.4 | 86.0 | 70.0 | 80.0 | 8.6 | 12.9 |
| Qwen 3.7 Flash | off | 80.0 | 80.0 | 90.0 | 55.0 | 80.0 | 4.3 | 17.1 |
| Qwen 3.7 Flash | on | 76.0 | 82.9 | 86.0 | 75.0 | 60.0 | 4.3 | 17.1 |
| MiniMax M3 | off | 77.0 | 74.3 | 76.0 | 70.0 | 83.3 | 8.6 | 24.3 |
| MiniMax M3 | on | 79.0 | 77.1 | 78.0 | 75.0 | 83.3 | 10.0 | 18.6 |
| MiMo V2.5 | off | 82.0 | 81.4 | 86.0 | 70.0 | 83.3 | 5.7 | 18.6 |
| MiMo V2.5 | on | 84.0 | 84.3 | 86.0 | 80.0 | 83.3 | 5.7 | 15.7 |
| Qwen 3.8 Max | fixed high | **85.0** | **85.7** | 86.0 | 85.0 | 83.3 | **4.3** | **8.6** |
| GPT-5.6 Luna | fixed high | 80.0 | 78.6 | 78.0 | 80.0 | 83.3 | 12.9 | 17.1 |

This task measures whether the model recognizes answerability and false
premises. Free-text answer correctness for answerable HaloQuest questions is
not currently scored; the benchmark metadata explicitly reports
`not_scored_requires_semantic_judge`. Status accuracy must therefore not be
presented as complete HaloQuest answer accuracy.

## Dermatology counterfactual audit

All values are percentages. Pixel finding/diagnosis hallucination and source
label persistence are error rates; lower is better. The other columns are
success rates or accuracies; higher is better.

Column legend:

- **Model** and **Condition**: model identity and reasoning setting, using the
  same meanings as in the general visual audit.
- **Full success**: success over all 50 counterfactual cases. A shuffled image
  must trigger a complete low-confidence abstention; a hard-negative image
  swap must be declared evaluable and place the diagnosis belonging to the
  actually displayed replacement image at rank 1.
- **Grounding**: percentage of all cases where the response followed the
  supplied pixels at the basic task level: correct abstention for shuffled
  images or an evaluable-image decision for hard-negative swaps. It does not
  by itself require the swapped-image diagnosis to be correct.
- **Pixel abstention**: percentage of the 25 pixel-shuffled cases returning
  `not_evaluable`, no visual findings, no diagnoses, and low confidence.
- **Pixel finding halluc.**: percentage of pixel-shuffled cases containing at
  least one claimed visual finding.
- **Pixel diagnosis halluc.**: percentage of pixel-shuffled cases containing
  at least one diagnosis.
- **Hard-neg Top-1**: percentage of the 25 hard-negative swaps where the
  diagnosis associated with the displayed replacement image ranked first.
- **Hard-neg Top-3**: percentage where that diagnosis appeared within the
  first three positions.
- **Source-label persistence**: percentage of hard-negative swaps where rank
  1 repeated the original case label associated with the prompt provenance,
  rather than following the replacement image. This is a shortcut/error
  rate.

| Model | Condition | Full success | Grounding | Pixel abstention | Pixel finding halluc. | Pixel diagnosis halluc. | Hard-neg Top-1 | Hard-neg Top-3 | Source-label persistence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 4B | off | 70.0 | 98.0 | 100.0 | 0.0 | 0.0 | 40.0 | 56.0 | 4.0 |
| Qwen 3.5 4B | on | 60.0 | 96.0 | 100.0 | 0.0 | 0.0 | 20.0 | 40.0 | 8.0 |
| Qwen 3.6 27B | off | 74.0 | 100.0 | 100.0 | 0.0 | 0.0 | 48.0 | 64.0 | 4.0 |
| Qwen 3.6 27B | on | 68.0 | 98.0 | 100.0 | 0.0 | 0.0 | 36.0 | 64.0 | 12.0 |
| Qwen 3.7 Flash | off | **80.0** | 100.0 | 100.0 | 0.0 | 0.0 | **60.0** | **76.0** | 8.0 |
| Qwen 3.7 Flash | on | 68.0 | 94.0 | 100.0 | 0.0 | 0.0 | 36.0 | 56.0 | 4.0 |
| MiniMax M3 | off | 68.0 | 96.0 | 100.0 | 0.0 | 0.0 | 36.0 | 60.0 | 8.0 |
| MiniMax M3 | on | 64.0 | 98.0 | 100.0 | 0.0 | 0.0 | 28.0 | 52.0 | 16.0 |
| MiMo V2.5 | off | 64.0 | 98.0 | 100.0 | 0.0 | 0.0 | 28.0 | 44.0 | 4.0 |
| MiMo V2.5 | on | 58.0 | 94.0 | 96.0 | 4.0 | 4.0 | 20.0 | 44.0 | 8.0 |
| Qwen 3.8 Max | fixed high | 66.0 | 96.0 | 100.0 | 0.0 | 0.0 | 32.0 | 64.0 | 4.0 |
| GPT-5.6 Luna | fixed high | 62.0 | 100.0 | 100.0 | 0.0 | 0.0 | 24.0 | 56.0 | 12.0 |

The pixel-shuffle condition was nearly saturated. Most discrimination comes
from the 25 hard-negative swaps, so the Top-1/3 confidence intervals are wide.
This small audit is suited to detecting large grounding failures, not to
ranking clinically close models by a few percentage points.

## Paired thinking effect

Positive accuracy deltas favor thinking; negative error-rate deltas favor
thinking. Token multipliers compare mean provider-reported output tokens.

Column legend:

- Every **delta** is `thinking on - thinking off`, expressed in percentage
  points.
- **General status delta**: change in general question-status accuracy;
  positive values favor thinking.
- **Unanswerable halluc. delta**: change in the hallucination error rate on
  unanswerable cases; negative values favor thinking and positive values are
  worse.
- **Derm full-success delta**: change in complete dermatology
  counterfactual success; positive values favor thinking.
- **Hard-neg Top-1 delta**: change in replacement-image Top-1 accuracy;
  positive values favor thinking.
- **General/Derm token multiplier**: mean output tokens with thinking on
  divided by mean output tokens with thinking off. For example, `10.6x`
  means that the thinking run generated about 10.6 times as many tokens.

| Model | General status delta | Unanswerable halluc. delta | Derm full-success delta | Hard-neg Top-1 delta | General token multiplier | Derm token multiplier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 4B | +12.0 | +5.7 | -10.0 | -20.0 | 41.7x | 10.6x |
| Qwen 3.6 27B | +3.0 | +1.4 | -6.0 | -12.0 | 24.0x | 18.6x |
| Qwen 3.7 Flash | -4.0 | 0.0 | -12.0 | -24.0 | 47.0x | 13.9x |
| MiniMax M3 | +2.0 | +1.4 | -4.0 | -8.0 | 7.8x | 4.0x |
| MiMo V2.5 | +2.0 | 0.0 | -6.0 | -8.0 | 21.4x | 14.6x |

The general task contains mixed changes and no uniform improvement. The
dermatology task shows the same direction for all five paired models:
thinking reduced both full counterfactual success and hard-negative Top-1.

## Qwen 3.8 Max: low versus high reasoning

The exact same 100 general and 50 dermatology cases were rerun through the
pinned official Alibaba route with reasoning effort `low`. All 150 low-effort
responses were valid and schema compliant. Disabling reasoning is not a
supported condition for this endpoint.

| Metric | Low | High | Low - high |
| --- | ---: | ---: | ---: |
| General question-status accuracy | 86.0% | 85.0% | +1.0 pp |
| General unanswerable detection | 87.1% | 85.7% | +1.4 pp |
| General hallucination on unanswerable cases | 4.3% | 4.3% | 0.0 pp |
| General overconfidence on unanswerable cases | 12.9% | 8.6% | +4.3 pp |
| Dermatology full counterfactual success | 70.0% | 66.0% | +4.0 pp |
| Dermatology hard-negative Top-1 | 40.0% | 32.0% | +8.0 pp |
| Dermatology hard-negative Top-3 | 76.0% | 64.0% | +12.0 pp |
| General mean reasoning tokens | 170.2 | 662.2 | -74.3% |
| Dermatology mean reasoning tokens | 262.6 | 705.6 | -62.8% |

Low reasoning was at least competitive on these small audits while using
substantially fewer reasoning tokens. The apparent dermatology improvement is
based on only 25 hard-negative cases and must not be treated as a precise
ranking result. For Qwen 3.8 Max, `low` is the more economical condition to
carry into a larger paired evaluation; `off` is technically unavailable.

## Output-contract disclosure

`S/R/C` means strict JSON validity / recoverable JSON validity / schema
compliance. Recovery removes code fences or extracts a valid JSON object; it
does not change diagnoses or answers. Invalid or non-compliant cases remain in
the metric denominator and therefore lower task accuracy.

Column legend:

- **General S/R/C** and **Derm S/R/C**: three percentages for the respective
  task. `S` is valid JSON exactly as returned, `R` is JSON obtainable through
  deterministic non-semantic cleanup, and `C` is compliance with the complete
  response schema after parsing.
- **Reasoning text, general/derm**: percentage of cases in each task for which
  the provider exposed reasoning text that the runner could store separately
  from the scored answer. `0` does not prove that the model performed no
  internal computation; it means no reasoning text was returned under that
  condition.

| Model | Condition | General S/R/C | Derm S/R/C | Reasoning text, general/derm |
| --- | --- | --- | --- | --- |
| Qwen 3.5 4B | off | 100/100/98 | 100/100/100 | 0/0 |
| Qwen 3.5 4B | on | 100/100/99 | 86/100/100 | 100/100 |
| Qwen 3.6 27B | off | 100/100/98 | 100/100/100 | 0/0 |
| Qwen 3.6 27B | on | 67/99/99 | 90/100/100 | 100/100 |
| Qwen 3.7 Flash | off | 100/100/100 | 100/100/100 | 0/0 |
| Qwen 3.7 Flash | on | 48/100/100 | 96/100/100 | 100/100 |
| MiniMax M3 | off | 100/100/100 | 100/100/100 | 0/0 |
| MiniMax M3 | on | 100/100/100 | 100/100/100 | 100/100 |
| MiMo V2.5 | off | 100/100/100 | 100/100/100 | 0/0 |
| MiMo V2.5 | on | 72/100/100 | 98/100/100 | 100/100 |
| Qwen 3.8 Max | fixed high | 100/100/100 | 98/98/98 | 100/100 |
| GPT-5.6 Luna | fixed high | 100/100/100 | 100/100/100 | 70/44 |

Across the 1,800 corrected predictions there were 129 raw format-invalid
outputs and five schema-invalid outputs, but zero backend errors, image
errors, safety refusals, semantic refusals, or truncations. Overall strict
JSON validity was 92.8%, recoverable JSON validity was 99.9%, and schema
compliance was 99.6%.

The local vLLM Qwen runs expose full reasoning text through streaming deltas,
but do not report a separate reasoning-token count. Their total output-token
usage includes the reasoning and final answer. OpenRouter and Azure token
accounting are provider-specific, so token totals are useful for within-model
off/on comparisons but not exact cross-provider billing comparisons.

## Generation cost and elapsed time

`Tokens` is the mean provider-reported output-token count; `max` is the
largest case. Elapsed time is wall-clock time for the whole task. Local models
were co-served on one H200 and API jobs were also run concurrently, so elapsed
time documents this execution but is not an isolated throughput benchmark.

Column legend:

- **General/Derm tokens (max)**: mean output tokens per case, followed in
  parentheses by the largest single-case output. Where reasoning is exposed,
  provider accounting may include reasoning and final-answer tokens.
- **General/Derm elapsed**: wall-clock duration for the complete 100-case
  general task or 50-case dermatology task, not latency per image.
- Token definitions and elapsed times are reliable for paired comparisons
  within the same model/provider, but not as exact cross-provider cost or
  throughput comparisons.

| Model | Condition | General tokens (max) | General elapsed | Derm tokens (max) | Derm elapsed |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen 3.5 4B | off | 31.6 (52) | 24.9 s | 97.4 (187) | 22.5 s |
| Qwen 3.5 4B | on | 1,315.9 (9,133) | 686.5 s | 1,036.9 (5,007) | 191.8 s |
| Qwen 3.6 27B | off | 30.9 (35) | 36.2 s | 106.2 (193) | 39.3 s |
| Qwen 3.6 27B | on | 742.2 (10,272) | 976.6 s | 1,979.8 (10,426) | 602.5 s |
| Qwen 3.7 Flash | off | 31.0 (54) | 86.8 s | 111.2 (207) | 41.8 s |
| Qwen 3.7 Flash | on | 1,459.4 (7,529) | 1,248.4 s | 1,549.1 (6,148) | 547.3 s |
| MiniMax M3 | off | 21.0 (43) | 44.9 s | 61.1 (149) | 26.9 s |
| MiniMax M3 | on | 164.6 (924) | 62.7 s | 246.5 (609) | 51.0 s |
| MiMo V2.5 | off | 26.5 (31) | 43.6 s | 73.6 (148) | 41.3 s |
| MiMo V2.5 | on | 566.7 (7,010) | 380.0 s | 1,070.8 (3,083) | 212.1 s |
| Qwen 3.8 Max | fixed high | 694.6 (4,356) | 721.9 s | 826.0 (4,891) | 364.6 s |
| GPT-5.6 Luna | fixed high | 161.1 (1,056) | 134.7 s | 205.9 (595) | 50.7 s |

## Local execution provenance

The local models ran simultaneously on one NVIDIA H200 with 139.8 GiB of
visible VRAM:

- vLLM 0.23.0 and PyTorch 2.11.0+cu130;
- RunPod driver branch 570, bridged to CUDA 13 with NVIDIA's official
  `cuda-compat-13-0` forward-compatibility package;
- Qwen 3.6 27B at GPU-memory utilization 0.65 on port 8000;
- Qwen 3.5 4B at GPU-memory utilization 0.15 on port 8002;
- BF16 weights, 32,768-token runtime context, batch size 8, and the Qwen3
  reasoning parser;
- Triton GDN prefill backend, selected to avoid compiling 65 FlashInfer GDN
  kernel variants during startup.

The model YAML runtime context was increased from 16,384 to 32,768 so the
14,336-token thinking-on completion cap leaves adequate space for image and
prompt tokens. This does not change the model weights or generation recipe.

## Limitations and interpretation

- The audits use Validation/development cases and were inspected while
  developing prompts and parsers. They are not sealed final estimates.
- General status accuracy does not include semantic grading of free-text
  answers.
- The dermatology hard-negative result is based on only 25 swapped images.
- Pixel shuffling is an intentionally strong corruption and is close to
  saturated for most models.
- Thinking implementations differ by provider. A shared on/off label does not
  imply identical hidden algorithms.
- Parser recovery measures output-contract robustness separately from task
  intelligence. Recovered JSON is scored without altering its semantic
  content.
- Co-serving and provider concurrency make elapsed time operational rather
  than controlled latency evidence.

The evidence supports retaining the concise, grounded prompt and using
thinking off for the primary full benchmark. If reasoning traces are needed
for synthetic-data research, they should be generated in a separate pipeline
with explicit quality filters, token limits, and rejection of traces that do
not improve the final grounded answer.
