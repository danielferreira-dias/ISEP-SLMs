# Internal Benchmark: complete pre-training model comparison

## Status and objective

This note records the complete pre-training Internal Benchmark baseline for the
official student, Qwen 3.5 4B, and the larger local comparison model, Qwen 3.6
27B. Both models completed the same 2,262 benchmark cases with thinking
disabled.

It was subsequently extended with the same 2,262 cases for Qwen 3.8 Max
through the official Alibaba OpenRouter provider. Qwen 3.8 uses mandatory
reasoning, so it was evaluated with the frozen `reasoning_effort=low` setting
selected by the preceding hallucination audit. Its results are reported in a
separate section because this is not a controlled thinking-off comparison.

MiniMax M3 and MiMo V2.5 subsequently completed the same benchmark through
their official OpenRouter providers with thinking requested as disabled. The
provider still returned residual reasoning for some responses, so this label
describes the request configuration rather than a guarantee of zero internal
reasoning. Gemini 3.5 Flash-Lite subsequently completed the same benchmark
through Google AI Studio with provider-default decoding and its mandatory
`minimal` thinking level. The six completed models are compared below using the frozen
prompts, parsers, metrics, and blinded-judge protocol.

The purpose of this run was to measure the student's starting point and to
determine how much performance is gained by scaling the same model family
before any dermatology-specific fine-tuning. These results do not yet complete
teacher selection: Qwen 3.7 Flash has not run the full Internal Benchmark. All
open-ended responses documented here have been scored with the frozen
blinded-judge protocol.

Because the Internal Benchmark has now been inspected, these results must be
treated as a frozen pre-training baseline. Future prompt, parser, and inference
adjustments should be developed on Validation, not repeatedly optimized on
these Internal Benchmark cases.

## Execution protocol

- Hardware: one RunPod H200 with 141 GB VRAM.
- Backend: two local OpenAI-compatible vLLM servers.
- Qwen 3.6 27B: port 8000, BF16, maximum context 32,768, GPU memory
  utilization 0.65, maximum 8 concurrent sequences.
- Qwen 3.5 4B: port 8002, BF16, maximum context 32,768, GPU memory
  utilization 0.15, maximum 8 concurrent sequences.
- Thinking: disabled for both models.
- Sampling: each model used the frozen parameters in its model configuration.
- Benchmark output limit: 8,192 tokens.
- Structured tasks: prompt-only JSON generation followed by the frozen parser
  and schema validator.
- Open-ended task: unrestricted clinical prose using the frozen prompt.
- Evaluated-model retry policy: no answer repair or selective rerun was used
  for scoring.
- Pairing: both models received exactly the same task IDs, images, prompts,
  candidate classes, response schemas, and references.

Invalid outputs remain in the denominator. A recoverable answer can therefore
be scored clinically while still being disclosed as invalid in the raw-format
metrics.

## Cases executed

| Task | Cases per model | Notes |
|---|---:|---|
| Visual Top-K | 1,000 | Closed set of 21 diseases; six ranked predictions |
| Visual Confusion Sets | 828 | 414 paired low/high-confusability cases |
| Evidence-Grounded Diagnosis | 134 | Diagnosis, visible findings, and evidence links |
| Open-Ended Diagnosis | 300 | Free-text top-three differential and rationale |
| **Total** | **2,262** | **Same cases for all six completed models** |

## Headline results

Structured-task percentages below use all assigned cases as the denominator,
including invalid outputs where applicable. Open-ended judge percentages use
the successfully judged cases; conservative all-assigned-case rates and judge
coverage are disclosed separately.

| Task and metric | Qwen 3.5 4B | Qwen 3.6 27B | Absolute difference |
|---|---:|---:|---:|
| Visual Top-K — Top-1 | 33.00% | **42.90%** | +9.90 pp |
| Visual Top-K — Top-3 | 62.50% | **74.30%** | +11.80 pp |
| Visual Top-K — Top-6 | 76.60% | **88.60%** | +12.00 pp |
| Visual Top-K — MRR | 49.03% | **60.08%** | +11.06 pp |
| Visual Top-K — macro F1, Top-1 | 28.52% | **38.73%** | +10.22 pp |
| Confusion Sets — Top-1 | 68.84% | **76.21%** | +7.37 pp |
| Confusion Sets — Top-2 | 86.96% | **92.27%** | +5.31 pp |
| Confusion Sets — MRR | 81.80% | **86.33%** | +4.53 pp |
| Confusion Sets — macro F1 | 68.21% | **75.67%** | +7.46 pp |
| Evidence — Top-1 | 41.79% | **46.27%** | +4.48 pp |
| Evidence — Top-3 | 61.19% | **69.40%** | +8.21 pp |
| Evidence — Top-6 | 71.64% | **81.34%** | +9.70 pp |
| Evidence — MRR | 52.24% | **59.29%** | +7.05 pp |
| Evidence — semantic compliance | 38.81% | **49.25%** | +10.45 pp |
| Evidence — grounded Top-1 success | **15.67%** | 7.46% | -8.21 pp |
| Open-ended — judge Top-1 | 16.78% | **24.21%** | +7.43 pp |
| Open-ended — judge Top-3 | 38.11% | **51.23%** | +13.12 pp |
| Open-ended — judge MRR | 26.34% | **35.85%** | +9.51 pp |

### Consolidated six-model results

The following table combines the completed local and API runs. Structured
metrics use every assigned case as the denominator. Open-ended metrics use the
scored outcomes reported by the judge pipeline, including explicit model
failures where present; judge coverage is therefore shown alongside them.

| Task and metric | Qwen 3.5 4B | Qwen 3.6 27B | Qwen 3.8 Max | MiniMax M3 | MiMo V2.5 | Gemini 3.5 Flash-Lite |
|---|---:|---:|---:|---:|---:|---:|
| Top-K — Top-1 | 33.00% | 42.90% | **52.50%** | 29.70% | 32.20% | 50.70% |
| Top-K — Top-3 | 62.50% | 74.30% | 79.70% | 60.70% | 59.50% | **80.00%** |
| Top-K — Top-6 | 76.60% | 88.60% | 87.70% | 78.20% | 77.20% | **90.90%** |
| Top-K — MRR | 49.03% | 60.08% | **66.31%** | 46.77% | 47.94% | 65.79% |
| Confusion — raw Top-1 | 68.84% | 76.21% | 80.43% | 56.04% | 49.40% | **81.04%** |
| Confusion — canonical Top-1 | 69.32% | 77.29% | 80.43% | 69.93% | 68.36% | **82.25%** |
| Confusion — canonical Top-2 | 87.80% | 93.72% | 91.43% | 89.01% | 88.65% | **95.05%** |
| Confusion — raw MRR | 81.80% | 86.33% | 87.50% | 66.38% | 60.29% | **88.97%** |
| Evidence — Top-1 | 41.79% | 46.27% | **58.96%** | 38.81% | 41.79% | 52.99% |
| Evidence — Top-3 | 61.19% | 69.40% | **76.12%** | 61.19% | 62.69% | 72.39% |
| Evidence — Top-6 | 71.64% | 81.34% | 83.58% | 73.13% | 76.87% | **87.31%** |
| Evidence — MRR | 52.24% | 59.29% | **68.48%** | 50.83% | 54.47% | 64.22% |
| Evidence — semantic compliance | 38.81% | 49.25% | 59.70% | 64.18% | 52.99% | **82.84%** |
| Evidence — grounded Top-1 success | **15.67%** | 7.46% | 5.22% | 5.97% | 5.97% | 11.19% |
| Open-ended — judge coverage | **95.33%** | 95.00% | 87.00% | 84.67% | 94.00% | 90.00% |
| Open-ended — Top-1 | 16.78% | 24.21% | **42.15%** | 23.62% | 24.11% | 41.48% |
| Open-ended — Top-3 | 38.11% | 51.23% | **65.52%** | 43.70% | 47.16% | 61.48% |
| Open-ended — MRR | 26.34% | 35.85% | **52.49%** | 32.22% | 33.69% | 50.12% |
| Clinical rationale, 0–4 | 1.49 | 1.95 | **2.55** | 1.87 | 2.00 | 2.41 |
| Evidence grounding, 0–4 | 1.74 | 2.17 | **2.62** | 1.98 | 2.29 | 2.58 |
| Unsupported-claim rate | 94.76% | 88.77% | **70.50%** | 95.67% | 81.56% | 77.04% |

`Raw` Confusion metrics preserve strict format failures in the denominator.
`Canonical` metrics apply only the frozen deterministic parser and therefore
show the clinical ranking recoverable from the response; they do not erase the
strict-format failure disclosed below. Qwen 3.8 leads most diagnostic and
open-ended measures, while Gemini leads Top-K Top-3/Top-6, the paired
Confusion metrics, Evidence Top-6, and Evidence semantic compliance. MiniMax
no longer has the highest semantic-compliance score once Gemini is included.

Judge recovery was not perfectly symmetric: the two local Qwen runs had three
historical `--retry-invalid` passes after an initial concurrent run encountered
Azure rate limits, whereas Qwen 3.8, MiniMax, and MiMo each received one
controlled retry after their initial sequential pass. Gemini used one judge
pass without an invalid-judgment retry. No evaluated-model
answer was regenerated, but open-ended coverage differs across models.
Judge-dependent scores must therefore always be read with the coverage and
conservative all-300-case rates, rather than compared as if every response had
an independent human rating.

Qwen 3.6 27B is consistently stronger on disease ranking. The most important
exception is `grounded_top_1_success`, a strict conjunction requiring a correct
Top-1 diagnosis and compliant evidence grounding. Its reduction must be
investigated at case level; it does not, by itself, demonstrate that Qwen 3.5
has better clinical reasoning.

## Visual Top-K

Both models produced valid JSON and schema-compliant predictions for all 1,000
cases. Neither produced duplicate predictions or invalid disease IDs.

| Metric | Qwen 3.5 4B | Qwen 3.6 27B |
|---|---:|---:|
| JSON validity | 100.00% | 100.00% |
| Recoverable JSON validity | 100.00% | 100.00% |
| Schema compliance | 100.00% | 100.00% |
| Duplicate prediction rate | 0.00% | 0.00% |
| Invalid disease-ID rate | 0.00% | 0.00% |

### Skin-tone slices

Skin-tone metadata was available for 955 of the 1,000 cases. The following are
raw Top-1 accuracies and are not disease-adjusted.

| Skin-tone aggregation | Cases | Qwen 3.5 4B | Qwen 3.6 27B |
|---|---:|---:|---:|
| Fitzpatrick 1–2 | 312 | 37.82% | **44.23%** |
| Fitzpatrick 3–4 | 214 | 33.64% | **44.86%** |
| Fitzpatrick 5–6 | 116 | 31.90% | **38.79%** |
| Monk 1–3 | 205 | 25.37% | **38.54%** |
| Monk 4–6 | 90 | 26.67% | **42.22%** |
| Monk 7–10 | 18 | 5.56% | 33.33% |
| Unknown | 45 | 57.78% | 60.00% |

Across the supported fine-grained groups, the raw Top-1 gap between the best
and worst group was 15.19 percentage points for Qwen 3.5 and 9.65 points for
Qwen 3.6. The Monk 7–10 slice has only 18 cases and is below the support needed
for a stable conclusion. These slices are descriptive diagnostics, not proof
of demographic fairness: class composition, source dataset, image modality,
and sample size are potential confounders.

## Visual Confusion Sets

The benchmark contains matched low- and high-confusability cases. Qwen 3.6
improved accuracy in both conditions and reduced, but did not eliminate, the
confusability gap.

| Metric | Qwen 3.5 4B | Qwen 3.6 27B |
|---|---:|---:|
| Low-confusability Top-1 | 83.82% | **87.44%** |
| High-confusability Top-1 | 53.86% | **64.98%** |
| Confusability gap | 29.95 pp | **22.46 pp** |
| Gap 95% CI | 24.40–35.51 pp | 17.39–27.54 pp |
| Raw JSON validity | **99.03%** | 98.55% |
| Recoverable JSON validity | 100.00% | 100.00% |
| Raw schema compliance | **98.67%** | 98.55% |
| Canonical Top-1 | 69.32% | **77.29%** |
| Canonical Top-2 | 87.80% | **93.72%** |

Qwen 3.5 produced eight format-invalid and three schema-invalid outputs. Qwen
3.6 produced twelve format-invalid outputs. All were recoverable by the frozen
parser, but their raw invalidity is still reported.

## Evidence-Grounded Diagnosis

This remains the most difficult structured task. Qwen 3.6 improves disease
ranking, morphology recall, response format, and semantic compliance, but both
models frequently introduce findings or concepts that are not supported by the
benchmark evidence.

| Metric | Qwen 3.5 4B | Qwen 3.6 27B |
|---|---:|---:|
| Top-1 macro F1 | 33.29% | **39.88%** |
| Finding F1 | 52.97% | **58.53%** |
| Finding precision | **54.30%** | 53.67% |
| Finding recall | 56.55% | **69.07%** |
| Description concept F1 | 53.84% | **54.66%** |
| Description consistency | 87.16% | **91.24%** |
| Supported-concept macro F1 | 63.99% | **67.25%** |
| Visible-evidence precision | 59.88% | **60.65%** |
| Unsupported finding rate | 49.45% | **47.79%** |
| Unsupported description-concept rate | **50.23%** | 51.99% |
| Correct diagnosis with unsupported evidence | 8.93% | **3.23%** |
| Forbidden description-content rate | **2.99%** | 4.48% |
| Invalid concept-ID rate | 5.97% | **2.99%** |
| Invalid disease-ID rate | 1.49% | **0.00%** |
| Valid evidence-link rate | 100.00% | 100.00% |
| Broken evidence-reference rate | 0.00% | 0.00% |
| Top-1 Brier score, lower is better | 35.65% | **33.94%** |
| Top-1 expected calibration error, lower is better | 37.09% | **32.96%** |

The raw JSON-validity difference is substantial: 56.72% for Qwen 3.5 versus
97.01% for Qwen 3.6. The parser recovered 100% of responses from both models,
and final schema compliance was 92.54% and 97.01%, respectively. Recovery does
not erase the raw-format failure from the disclosure metrics.

The final status distribution was:

| Status | Qwen 3.5 4B | Qwen 3.6 27B |
|---|---:|---:|
| OK | 31 | 64 |
| Format invalid | 58 | 4 |
| Schema invalid | 6 | 4 |
| Semantic noncompliance | 39 | 62 |

The low `grounded_top_1_success` score should be reviewed together with its
strict definition. Qwen 3.6 obtains more correct diagnoses but also accumulates
more cases marked semantically noncompliant after parsing, which can cause the
conjunctive metric to fail. Case-level error analysis is required before using
this metric to make a teacher decision.

## Open-Ended Diagnosis

Both models completed all 300 cases without backend errors. The mean response
length was 1,817.68 characters for Qwen 3.5 and 1,835.27 characters for Qwen
3.6.

The final responses were evaluated with the frozen model-identity-blinded
protocol: GPT-5.6 Luna at high reasoning was the primary judge, and Qwen 3.7
Flash was invoked only after a Luna `content_policy_violation`. The judge saw
the image, reference diagnosis, isolated auxiliary references, and the final
model response, but not the evaluated model identity or its private reasoning.

| Judge metric | Qwen 3.5 4B | Qwen 3.6 27B |
|---|---:|---:|
| Evaluated judgments | 286 / 300 | 285 / 300 |
| Judge coverage | **95.33%** | 95.00% |
| Top-1 among evaluated | 16.78% | **24.21%** |
| Top-3 among evaluated | 38.11% | **51.23%** |
| MRR among evaluated | 26.34% | **35.85%** |
| Conservative Top-1 over all 300 | 16.00% | **23.00%** |
| Conservative Top-3 over all 300 | 36.33% | **48.67%** |
| Diagnosis correctness, 0–4 | 1.26 | **1.67** |
| Visible-findings correctness, 0–4 | 2.15 | **2.41** |
| Evidence grounding, 0–4 | 1.74 | **2.17** |
| Clinical-rationale quality, 0–4 | 1.49 | **1.95** |
| Differential quality, 0–4 | 1.49 | **2.18** |
| Unsupported-claim rate | 94.76% | **88.77%** |
| Mean unsupported-claim count | 4.32 | **3.01** |

The verdict distributions over successfully evaluated cases were:

| Verdict | Qwen 3.5 4B | Qwen 3.6 27B |
|---|---:|---:|
| Correct | 8 | **23** |
| Mostly correct | 19 | **39** |
| Partially correct | 83 | 84 |
| Incorrect | 176 | **139** |

Qwen 3.6 is materially stronger on every scored open-ended clinical dimension
and produces fewer unsupported claims. Nevertheless, an unsupported-claim
rate of 88.77% remains too high for evidence-grounded clinical use. These are
judge-dependent results and should not be presented as independent human
expert assessment.

### Judge integrity and retries

The first attempt ran the two 300-case judge jobs concurrently and exceeded
the Azure rate limit. Those transport failures were not accepted as scores.
Both runs were resumed sequentially, retaining completed judgments, and the
same frozen judge prompt and schema were used throughout. Three
`--retry-invalid` passes were applied symmetrically to each model; each judge
request also retained the benchmark's three built-in corrective attempts.

| Final judge outcome | Qwen 3.5 4B | Qwen 3.6 27B |
|---|---:|---:|
| Valid judgment | 286 | 285 |
| Persistent judge-invalid | 13 | **10** |
| Uncovered safety refusal | **1** | 5 |
| Evaluated by Luna | 285 | 285 |
| Evaluated by Qwen fallback | 1 | 0 |
| Model response failure | 0 | 0 |

The official Qwen 3.7 OpenRouter route did not accept provider-native JSON
Schema decoding for fallback requests. It was therefore called in prompt-only
mode, while the same frozen local parser and JSON Schema validator remained
mandatory. A fallback transport failure after a Luna content-policy refusal is
stored as an uncovered `judge_safety_refusal`; it is never converted into a
clinical score. The fallback behavior is covered by the focused regression
suite.

## Complete output-integrity disclosure

| Integrity outcome across 2,262 cases | Qwen 3.5 4B | Qwen 3.6 27B |
|---|---:|---:|
| OK | 2,148 | 2,180 |
| Format invalid | 66 | 16 |
| Schema invalid | 9 | 4 |
| Semantic noncompliance | 39 | 62 |
| Backend error | 0 | 0 |
| Safety refusal | 0 | 0 |
| Image error | 0 | 0 |
| Truncated | 0 | 0 |

`OK` is a pipeline status, not a measure of diagnostic correctness. Conversely,
a recoverable format-invalid answer can still contribute a clinical prediction
while remaining disclosed as a raw-output failure.

## Operational timing

| Task | Qwen 3.5 4B | Qwen 3.6 27B |
|---|---:|---:|
| Visual Top-K | 6 min 14 s | 15 min 04 s |
| Visual Confusion Sets | 2 min 56 s | 4 min 23 s |
| Evidence-Grounded Diagnosis | 2 min 21 s | 3 min 46 s |
| Open-Ended Diagnosis | 3 min 58 s | 5 min 33 s |
| Approximate cumulative wall time | 15 min 51 s | 29 min 07 s |

These are operational measurements, not a controlled throughput benchmark.
The two servers shared one GPU and overlapped during part of the execution, so
the timings should not be used as isolated model-speed estimates.

## Qwen 3.8 Max API comparator

Qwen 3.8 Max received exactly the same task IDs, images, prompts, candidate
classes, response schemas, and references as the two local Qwen models. The
request was restricted to OpenRouter's official Alibaba provider with provider
fallbacks disabled. The model used mandatory reasoning at `low` effort; its
reasoning was captured separately and excluded from the scored final answer.
The benchmark output limit remained 8,192 tokens and no successful response
was marked as truncated or as a safety refusal.

### Diagnostic and grounding results

All structured-task percentages use every assigned case as the denominator.
Consequently, the provider failures described below count as unsuccessful
predictions and reduce accuracy.

| Task and metric | Qwen 3.8 Max |
|---|---:|
| Visual Top-K — Top-1 | 52.50% |
| Visual Top-K — Top-3 | 79.70% |
| Visual Top-K — Top-6 | 87.70% |
| Visual Top-K — MRR | 66.31% |
| Visual Top-K — macro F1, Top-1 | 50.95% |
| Confusion Sets — Top-1 | 80.43% |
| Confusion Sets — Top-2 | 91.43% |
| Confusion Sets — MRR | 87.50% |
| Confusion Sets — macro F1 | 81.95% |
| Confusion Sets — low-confusability Top-1 | 88.89% |
| Confusion Sets — high-confusability Top-1 | 71.98% |
| Confusion Sets — confusability gap | 16.91 pp |
| Evidence — Top-1 | 58.96% |
| Evidence — Top-3 | 76.12% |
| Evidence — Top-6 | 83.58% |
| Evidence — MRR | 68.48% |
| Evidence — macro F1, Top-1 | 55.57% |
| Evidence — semantic compliance | 59.70% |
| Evidence — grounded Top-1 success | 5.22% |

Qwen 3.8 is substantially stronger than both local baselines in primary
diagnostic ranking. It also has the smallest confusion-set gap. Evidence
grounding remains a weakness: the model obtained 50.21% finding F1, 53.65%
visible-evidence precision, and a 57.65% unsupported-finding rate. Its
description concept F1 was 48.85%, while 59.80% of description concepts were
classified as unsupported. A strong diagnosis score therefore must not be
interpreted as equally strong visual grounding.

### Skin-tone slices

Skin-tone metadata was available for 955 cases. These are raw, descriptive
Top-1 results and remain confounded by disease and source composition.

| Skin-tone aggregation | Cases | Top-1 |
|---|---:|---:|
| Fitzpatrick 1–2 | 312 | 52.56% |
| Fitzpatrick 3–4 | 214 | 56.54% |
| Fitzpatrick 5–6 | 116 | 53.45% |
| Monk 1–3 | 205 | 48.78% |
| Monk 4–6 | 90 | 46.67% |
| Monk 7–10 | 18 | 44.44% |
| Unknown | 45 | 62.22% |

The 18-case Monk 7–10 group is below the predefined support threshold and must
not be used to claim demographic fairness. The supported fine-grained-group
Top-1 gap was 13.38 percentage points.

### Open-ended blinded-judge results

The same frozen judge protocol was used: GPT-5.6 Luna was the primary judge,
with Qwen 3.7 Flash invoked only after a Luna content-policy violation. A
single controlled `--retry-invalid` pass retried judge failures without
changing or selectively regenerating any Qwen 3.8 model response.

| Judge metric | Qwen 3.8 Max |
|---|---:|
| Valid clinical judgments | 243 / 300 |
| Model-response failures retained as failures | 18 / 300 |
| Persistent judge-invalid cases | 39 / 300 |
| Scored-outcome coverage, including model failures | 261 / 300 (87.00%) |
| Top-1 over 261 scored outcomes | 42.15% |
| Top-3 over 261 scored outcomes | 65.52% |
| MRR over 261 scored outcomes | 52.49% |
| Conservative Top-1 over all 300 | 36.67% |
| Conservative Top-3 over all 300 | 57.00% |
| Diagnosis correctness, 0–4 | 2.31 |
| Visible-findings correctness, 0–4 | 2.74 |
| Evidence grounding, 0–4 | 2.62 |
| Clinical-rationale quality, 0–4 | 2.55 |
| Differential quality, 0–4 | 2.76 |
| Unsupported-claim rate | 70.50% |
| Mean unsupported-claim count | 1.78 |

The aggregate judge metrics count the 18 failed model responses as failed
scored outcomes. Of the 243 valid judge objects, 240 came from Luna and three
from the Qwen fallback. Luna triggered the content-policy fallback 11 times;
eight fallback attempts remained judge-invalid. The verdict distribution over
the 261 scored outcomes was 61 `correct`, 44 `mostly_correct`, 66
`partially_correct`, and 90 `incorrect`.

The persistent 39 judge-invalid cases are missing judge measurements, not
correct model responses. They remain explicitly disclosed rather than being
silently removed or assigned a favourable score. The conservative 300-case
rates are included to show the lower bound when every uncovered case is
treated as unsuccessful.

### Complete output-integrity disclosure

| Integrity outcome across 2,262 cases | Qwen 3.8 Max |
|---|---:|
| OK | 2,113 |
| Semantic noncompliance | 47 |
| Backend error | 102 |
| Format invalid | 0 |
| Schema invalid | 0 |
| Safety refusal | 0 |
| Image error | 0 |
| Truncated | 0 |

The 102 backend errors were HTTP 400 provider failures distributed across
Visual Top-K (45), Confusion Sets (32), Evidence (7), and Open-ended (18).
They account for 4.51% of all assigned cases. They were not selectively rerun,
because doing so after inspecting the benchmark would create a different retry
policy from the local baselines. For successful structured responses, raw JSON
validity was 95.50% in Top-K, 96.14% in Confusion Sets, and 94.78% in Evidence;
these apparent invalidity rates are entirely explained by the corresponding
backend failures rather than malformed returned JSON.

## MiniMax M3 and MiMo V2.5 API comparators

Both models were routed only to their official OpenRouter providers, with
provider fallbacks disabled. Thinking was requested as disabled according to
the frozen configuration chosen after the screening experiment. OpenRouter
still returned residual reasoning on some requests, so these runs are denoted
`thinking off requested`, not proven reasoning-free inference. Neither model
produced a backend error, safety refusal, image error, or truncation during the
2,262 evaluated-model requests.

### Structured clinical metrics

| Metric | MiniMax M3 | MiMo V2.5 |
|---|---:|---:|
| Top-K Top-1 / Top-3 / Top-6 | 29.70 / 60.70 / 78.20% | **32.20** / 59.50 / 77.20% |
| Top-K MRR | 46.77% | **47.94%** |
| Top-K strict JSON | 100.00% | 100.00% |
| Confusion raw Top-1 / Top-2 | **56.04 / 71.50%** | 49.40 / 65.46% |
| Confusion canonical Top-1 / Top-2 | **69.93 / 89.01%** | 68.36 / 88.65% |
| Confusion low-confusability Top-1 | **67.39%** | 61.35% |
| Confusion high-confusability Top-1 | **44.69%** | 37.44% |
| Confusion gap | **22.71 pp** | 23.91 pp |
| Confusion strict JSON | **79.35%** | 74.03% |
| Confusion recoverable JSON | 100.00% | 100.00% |
| Evidence Top-1 / Top-3 / Top-6 | 38.81 / 61.19 / 73.13% | **41.79 / 62.69 / 76.87%** |
| Evidence MRR | 50.83% | **54.47%** |
| Evidence Top-1 macro F1 | 31.88% | **35.91%** |
| Evidence semantic compliance | **64.18%** | 52.99% |
| Evidence grounded Top-1 success | 5.97% | 5.97% |
| Finding F1 | 44.29% | **53.09%** |
| Visible-evidence precision | 46.49% | **54.56%** |
| Unsupported-finding rate | 63.33% | **53.64%** |
| Description concept F1 | 44.88% | **53.40%** |
| Unsupported description-concept rate | 62.32% | **54.35%** |
| Evidence strict JSON | 47.01% | **68.66%** |
| Evidence recoverable JSON | 100.00% | 100.00% |
| Evidence schema compliance | **95.52%** | 94.78% |

MiniMax is stronger on the paired confusion task and semantic compliance.
MiMo is stronger on Top-K Top-1, Evidence diagnosis, morphology, visible
evidence, and strict output format. The large raw-to-canonical Confusion gap
for both models shows why diagnostic ability and interface reliability must be
reported separately.

### Skin-tone slices

The same 955 Top-K cases had skin-tone metadata for both models. The values are
unadjusted Top-1 accuracy and remain descriptive rather than causal fairness
measurements.

| Skin-tone aggregation | Cases | MiniMax M3 | MiMo V2.5 |
|---|---:|---:|---:|
| Fitzpatrick 1–2 | 312 | 30.13% | **35.58%** |
| Fitzpatrick 3–4 | 214 | 32.24% | **33.18%** |
| Fitzpatrick 5–6 | 116 | **34.48%** | 31.03% |
| Monk 1–3 | 205 | 26.34% | **26.83%** |
| Monk 4–6 | 90 | 20.00% | **28.89%** |
| Monk 7–10 | 18 | **16.67%** | 11.11% |
| Unknown | 45 | 42.22% | **46.67%** |

The Monk 7–10 and unknown rows are below the predefined support threshold.
Differences between rows may reflect class and source composition rather than
skin tone itself.

### Open-ended blinded-judge results

Each model received one initial judge pass and one controlled
`--retry-invalid` pass. No evaluated-model response was regenerated. Luna
remained the primary judge, and Qwen 3.7 Flash was invoked only after a Luna
content-policy violation.

| Judge metric | MiniMax M3 | MiMo V2.5 |
|---|---:|---:|
| Scored outcomes | 254 / 300 | **282 / 300** |
| Judge coverage | 84.67% | **94.00%** |
| Persistent judge-invalid | 38 | **10** |
| Uncovered safety refusals | 8 | 8 |
| Model-response failures | 0 | 0 |
| Top-1 | 23.62% | **24.11%** |
| Top-3 | 43.70% | **47.16%** |
| MRR | 32.22% | **33.69%** |
| Diagnosis correctness, 0–4 | 1.49 | **1.54** |
| Visible-findings correctness, 0–4 | 2.26 | **2.44** |
| Evidence grounding, 0–4 | 1.98 | **2.29** |
| Clinical-rationale quality, 0–4 | 1.87 | **2.00** |
| Differential quality, 0–4 | 1.95 | **2.14** |
| Unsupported-claim rate | 95.67% | **81.56%** |
| Mean unsupported-claim count | 3.77 | **2.29** |

MiniMax produced 14 `correct`, 37 `mostly_correct`, 61
`partially_correct`, and 142 `incorrect` scored outcomes. MiMo produced 34,
25, 74, and 149, respectively. The different judge coverage means small score
differences between the models should not be over-interpreted. Conservative
Top-1 over all 300 cases is 20.00% for MiniMax and 22.67% for MiMo;
conservative Top-3 is 37.00% and 44.33%, respectively.

### Output-integrity disclosure

| Integrity outcome across 2,262 requests | MiniMax M3 | MiMo V2.5 |
|---|---:|---:|
| OK | 1,995 | 1,963 |
| Format invalid | 242 | 257 |
| Schema invalid | 3 | 5 |
| Semantic noncompliance | 22 | 37 |
| Backend error | 0 | 0 |
| Safety refusal | 0 | 0 |
| Image error | 0 | 0 |
| Truncated | 0 | 0 |

The parser recovered every format-invalid response. Recovery allows canonical
clinical scoring, but the original format failure remains counted in JSON and
schema disclosure metrics. This is especially important if either model is
considered as a synthetic-data teacher: a downstream generation pipeline
would need deterministic parsing and validation before accepting annotations.

## Gemini 3.5 Flash-Lite API comparator

Gemini 3.5 Flash-Lite received the same 2,262 Internal Benchmark requests via
OpenRouter's official Google AI Studio provider, with fallbacks disabled. The
request deliberately omitted temperature, top-p, and seed so that decoding
used the provider defaults. The only explicit reasoning control was
`reasoning.effort=minimal`, which is also Google's documented default for this
model. OpenRouter reported zero reasoning tokens for every successful case and
returned no reasoning text; in this run, `minimal` therefore behaved like an
operational no-thinking condition.

### Structured clinical metrics

| Metric | Gemini 3.5 Flash-Lite |
|---|---:|
| Top-K Top-1 / Top-3 / Top-6 | 50.70 / 80.00 / 90.90% |
| Top-K MRR / macro F1 Top-1 | 65.79 / 48.11% |
| Top-K strict JSON / schema compliance | 99.90 / 99.80% |
| Confusion raw Top-1 / Top-2 | 81.04 / 93.84% |
| Confusion canonical Top-1 / Top-2 | 82.25 / 95.05% |
| Confusion raw MRR / macro F1 Top-1 | 88.97 / 81.44% |
| Confusion low / high confusability Top-1 | 92.03 / 70.05% |
| Confusion gap | 21.98 pp |
| Confusion strict / recoverable JSON | 98.79 / 100.00% |
| Evidence Top-1 / Top-3 / Top-6 | 52.99 / 72.39 / 87.31% |
| Evidence MRR / macro F1 Top-1 | 64.22 / 48.45% |
| Evidence semantic compliance | 82.84% |
| Evidence grounded Top-1 success | 11.19% |
| Finding F1 / visible-evidence precision | 60.69 / 62.85% |
| Unsupported-finding rate | 45.20% |
| Description concept F1 | 60.60% |
| Unsupported description-concept rate | 45.63% |
| Evidence strict / recoverable JSON | 87.31 / 100.00% |
| Evidence schema compliance | 98.51% |
| Top-1 Brier score / expected calibration error | 25.97 / 24.09% |

The model is close to Qwen 3.8 Max on diagnostic ranking, leads all six models
on Top-K Top-3/Top-6, canonical Confusion Top-1/Top-2, Evidence Top-6, and
Evidence semantic compliance, and has materially better output reliability.
Qwen 3.8 remains stronger on Top-K Top-1/MRR, Evidence Top-1/Top-3/MRR, and
most blinded open-ended measures.

### Skin-tone slices

These are unadjusted Top-1 results on the same fixed Top-K cases and should be
read with the existing source/class-composition caveat.

| Skin-tone aggregation | Cases | Top-1 |
|---|---:|---:|
| Fitzpatrick 1–2 | 312 | 52.56% |
| Fitzpatrick 3–4 | 214 | 56.07% |
| Fitzpatrick 5–6 | 116 | 53.45% |
| Monk 1–3 | 205 | 43.90% |
| Monk 4–6 | 90 | 51.11% |
| Monk 7–10 | 18 | 38.89% |
| Unknown | 45 | 40.00% |

The Monk 7–10 and unknown rows are below the predefined support threshold.
The supported fine-grained-group Top-1 gap was 23.09 percentage points; this
descriptive gap is confounded by diagnosis and dataset-source composition and
is not evidence of a causal skin-tone effect.

### Open-ended blinded-judge results

The frozen Luna-primary/Qwen-fallback protocol produced 270 scored outcomes,
including the single evaluated-model failure, for 90.00% coverage.

| Judge metric | Gemini 3.5 Flash-Lite |
|---|---:|
| Scored outcomes | 270 / 300 |
| Valid judge objects | 269 / 300 |
| Model-response failures retained as failures | 1 / 300 |
| Persistent judge-invalid | 23 / 300 |
| Uncovered safety refusals | 7 / 300 |
| Top-1 / Top-3 | 41.48 / 61.48% |
| MRR | 50.12% |
| Conservative Top-1 / Top-3 over all 300 | 37.33 / 55.33% |
| Diagnosis correctness, 0–4 | 2.21 |
| Visible-findings correctness, 0–4 | 2.74 |
| Evidence grounding, 0–4 | 2.58 |
| Clinical-rationale quality, 0–4 | 2.41 |
| Differential quality, 0–4 | 2.50 |
| Unsupported-claim rate | 77.04% |
| Mean unsupported-claim count | 1.90 |

Luna triggered 19 content-policy fallbacks. The final valid judgments comprise
267 Luna outputs and two Qwen 3.7 fallback outputs; uncovered fallback
failures remain invalid rather than being imputed. The verdict distribution
over 270 scored outcomes was 51 `correct`, 44 `mostly_correct`, 72
`partially_correct`, and 103 `incorrect`.

### Output integrity, token use, and cost

| Integrity outcome across 2,262 requests | Gemini 3.5 Flash-Lite |
|---|---:|
| OK | 2,208 |
| Format invalid | 27 |
| Schema invalid | 6 |
| Semantic noncompliance | 19 |
| Backend error | 2 |
| Safety refusal | 0 |
| Image error | 0 |
| Truncated | 0 |

The two backend errors correspond to the same Fitzpatrick image in Top-K and
Open-ended: OpenRouter returned a response without a `choices` item. No scored
answer was manufactured or selectively regenerated. Across evaluated-model
inference, the provider reported 3,596,693 input tokens, 242,017 output tokens,
and zero reasoning tokens. At the published OpenRouter list prices of $0.30
per million input tokens and $2.50 per million output tokens, this is an
estimated $1.68 for Gemini inference before caching and excluding the separate
Luna/Qwen judge calls.

## Interpretation and decision gate

1. Qwen 3.6 27B is the stronger zero-shot local model for disease ranking on
   all three scored structured tasks.
2. Qwen 3.5 4B remains the official student. Its lower baseline establishes the
   improvement target for dermatology-specific fine-tuning and distillation.
3. Evidence grounding remains a principal weakness across all six models.
   Better Top-K diagnosis alone does not solve unsupported visual claims or
   semantic noncompliance.
4. Qwen 3.8 Max remains narrowly strongest in primary diagnostic and judged
   open-ended quality, while Gemini 3.5 Flash-Lite is the strongest operational
   challenger: it is much cheaper, has fewer provider failures, and leads
   several coverage, confusion, semantic-compliance, and Top-6 measures.
5. MiMo is a more credible teacher candidate than MiniMax in this comparison:
   it has better Evidence morphology, open-ended grounding, output format, and
   fewer unsupported claims. MiniMax's strengths are semantic compliance and
   paired confusion cases.
6. The final teacher decision should compare Qwen 3.8's small quality lead
   against Gemini's much stronger reliability/cost profile, and should still
   disclose that the Qwen 3.7 full-benchmark comparison is incomplete.
7. Any further prompt or parser development should use the unused Validation
   cases. The Internal Benchmark should now remain fixed for the post-training
   comparison.

## Reproducibility artifacts

The complete predictions, manifests, deterministic metrics, judge metrics,
and HTML reports are stored under:

`outputs/internal_benchmark_full_v1/thinking_off/` and
`outputs/internal_benchmark_full_v1/qwen_3_8_max_low/`, with Gemini under
`outputs/internal_benchmark_full_v1/gemini_3_5_flash_lite_minimal/final/`.

The output directory is intentionally excluded from Git because it contains
large, reproducible run artifacts. This annotation preserves the protocol and
reported results in version control.
