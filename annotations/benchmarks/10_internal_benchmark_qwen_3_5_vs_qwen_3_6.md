# Internal Benchmark: Qwen 3.5 4B versus Qwen 3.6 27B

## Status and objective

This note records the complete pre-training Internal Benchmark baseline for the
official student, Qwen 3.5 4B, and the larger local comparison model, Qwen 3.6
27B. Both models completed the same 2,262 benchmark cases with thinking
disabled.

The purpose of this run was to measure the student's starting point and to
determine how much performance is gained by scaling the same model family
before any dermatology-specific fine-tuning. This result does not yet select a
teacher: the API candidates still require the same full Internal Benchmark.
The open-ended responses in this local-model comparison have now been scored
with the frozen blinded-judge protocol.

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
| **Total** | **2,262** | **Same cases for both models** |

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

## Interpretation and decision gate

1. Qwen 3.6 27B is the stronger zero-shot local model for disease ranking on
   all three scored structured tasks.
2. Qwen 3.5 4B remains the official student. Its lower baseline establishes the
   improvement target for dermatology-specific fine-tuning and distillation.
3. Evidence grounding is the principal weakness for both models. Better Top-K
   diagnosis alone does not solve unsupported visual claims or semantic
   noncompliance.
4. Qwen 3.6 is also materially stronger in the judged open-ended task, but
   teacher selection must still wait for the complete API-model comparison.
5. Any further prompt or parser development should use the unused Validation
   cases. The Internal Benchmark should now remain fixed for the post-training
   comparison.

## Reproducibility artifacts

The complete predictions, manifests, deterministic metrics, judge metrics,
and HTML reports are stored under:

`outputs/internal_benchmark_full_v1/thinking_off/`

The output directory is intentionally excluded from Git because it contains
large, reproducible run artifacts. This annotation preserves the protocol and
reported results in version control.
