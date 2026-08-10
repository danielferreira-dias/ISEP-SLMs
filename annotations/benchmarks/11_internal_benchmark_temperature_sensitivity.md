# Internal Benchmark temperature sensitivity: 1.0 versus 0.6

## Status and interpretation

This note records a complete post-hoc temperature sensitivity run for Qwen
3.5 4B and Qwen 3.6 27B. Both models completed the same four Internal
Benchmark tasks with thinking disabled and `temperature=0.6`. The results are
compared with the previously frozen `temperature=1.0` pre-training baseline.

The lower temperature produced better observed diagnostic ranking on most
metrics for both models. It should nevertheless be described as a
**sensitivity analysis**, not as a second sealed test or a clean causal
temperature experiment:

- the Internal Benchmark had already been inspected before this run;
- the original run used an NVIDIA H200, whereas this run used an NVIDIA RTX
  PRO 6000 Blackwell Workstation Edition;
- both runs used vLLM 0.23.0, but the Blackwell run had to disable the
  FlashInfer top-k/top-p sampler and use vLLM's native sampler;
- decoding is stochastic and the experiment contains one generation per case
  and temperature, rather than repeated seeds.

Consequently, the results support testing `temperature=0.6` on Validation and
freezing it for a future protocol. They do not justify silently replacing the
official `temperature=1.0` Internal Benchmark baseline.

## Controlled inputs and changed parameter

The run used the following fixed settings:

| Setting | Qwen 3.5 4B | Qwen 3.6 27B |
|---|---:|---:|
| Thinking | disabled | disabled |
| Temperature | **0.6** | **0.6** |
| Top-p | 0.95 | 0.95 |
| Top-k | 20 | 20 |
| Min-p | 0.0 | 0.0 |
| Presence penalty | 1.5 | 0.0 |
| Repetition penalty | 1.0 | 1.0 |
| Maximum benchmark output | 8,192 tokens | 8,192 tokens |
| Concurrent requests | 8 | 8 |

Presence penalty differs between the two model YAMLs, as it did in the
original baseline. Within each model, only temperature was overridden by the
runner. The source YAMLs remain at their published `temperature=1.0` recipe.

The following checks passed for every model-task pair:

- the sets of task IDs were exactly equal between the 1.0 and 0.6 runs;
- the rendered system prompt, user prompt, and image URI were identical for
  every task ID;
- model-config, benchmark-config, prompt, schema, taxonomy, and benchmark
  version hashes matched;
- all 2,262 responses per model reported no captured reasoning, confirming
  the thinking-off request.

The release and selection hashes changed after the benchmark repository was
resynchronized, and asynchronous completion changed row order. Direct
task-ID joins confirmed that these packaging differences did not change the
cases or their rendered inputs.

## Cases executed

| Task | Cases per model |
|---|---:|
| Visual Top-K | 1,000 |
| Visual Confusion Sets | 828 |
| Evidence-Grounded Diagnosis | 134 |
| Open-Ended Diagnosis | 300 |
| **Total** | **2,262** |

## Structured diagnostic results

All percentages use every assigned case as the denominator. `Delta` is
`temperature=0.6 - temperature=1.0` in percentage points, except for MRR and
0-4 judge scores where it is the corresponding scale difference.

### Visual Top-K

| Metric | 4B, t=1.0 | 4B, t=0.6 | Delta | 27B, t=1.0 | 27B, t=0.6 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Top-1 | 33.00% | **36.00%** | +3.00 pp | 42.90% | **45.30%** | +2.40 pp |
| Top-3 | 62.50% | **67.30%** | +4.80 pp | 74.30% | **76.60%** | +2.30 pp |
| Top-6 | 76.60% | **80.90%** | +4.30 pp | 88.60% | **89.00%** | +0.40 pp |
| MRR | 49.03% | **52.56%** | +3.53 pp | 60.08% | **61.82%** | +1.74 pp |
| Macro F1, Top-1 | 28.52% | **30.81%** | +2.29 pp | 38.73% | **41.07%** | +2.34 pp |
| JSON / schema validity | 100 / 100% | 100 / 100% | 0 / 0 pp | 100 / 100% | 100 / 100% | 0 / 0 pp |

The strongest and most consistent lower-temperature gain occurred on this
1,000-case task. An exact paired comparison of Top-1 correctness found 47
cases improved and 17 regressed for the 4B model, and 46 improved and 22
regressed for the 27B model. This case-paired result still does not account
for repeated-seed or backend variability.

### Visual Confusion Sets

| Metric | 4B, t=1.0 | 4B, t=0.6 | Delta | 27B, t=1.0 | 27B, t=0.6 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Raw Top-1 | 68.84% | **71.14%** | +2.29 pp | 76.21% | **78.99%** | +2.78 pp |
| Raw Top-2 | 86.96% | **89.25%** | +2.29 pp | 92.27% | **93.24%** | +0.97 pp |
| Raw MRR | 81.80% | **83.70%** | +1.89 pp | 86.33% | **88.37%** | +2.03 pp |
| Macro F1, Top-1 | 68.21% | **70.52%** | +2.31 pp | 75.67% | **78.21%** | +2.54 pp |
| Low-confusability Top-1 | 83.82% | **86.47%** | +2.66 pp | 87.44% | **89.61%** | +2.17 pp |
| High-confusability Top-1 | 53.86% | **55.80%** | +1.93 pp | 64.98% | **68.36%** | +3.38 pp |
| Confusability gap | **29.95 pp** | 30.68 pp | +0.72 pp | 22.46 pp | **21.26 pp** | -1.21 pp |
| Canonical Top-1 | 69.32% | **71.26%** | +1.93 pp | 77.29% | **78.99%** | +1.69 pp |
| Canonical Top-2 | 87.80% | **89.49%** | +1.69 pp | **93.72%** | 93.24% | -0.48 pp |

The lower temperature improved primary ranking for both models. It narrowed
the 27B confusability gap, but the 4B gap widened slightly because its easier
cases improved more than its harder cases.

### Evidence-Grounded Diagnosis

| Metric | 4B, t=1.0 | 4B, t=0.6 | Delta | 27B, t=1.0 | 27B, t=0.6 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Top-1 | 41.79% | **44.78%** | +2.99 pp | 46.27% | **50.00%** | +3.73 pp |
| Top-3 | 61.19% | **61.94%** | +0.75 pp | 69.40% | **71.64%** | +2.24 pp |
| Top-6 | **71.64%** | 70.15% | -1.49 pp | 81.34% | **83.58%** | +2.24 pp |
| MRR | 52.24% | **54.05%** | +1.82 pp | 59.29% | **61.54%** | +2.25 pp |
| Macro F1, Top-1 | 33.29% | **37.92%** | +4.63 pp | 39.88% | **41.30%** | +1.42 pp |
| Semantic compliance | **38.81%** | 38.06% | -0.75 pp | 49.25% | **61.19%** | +11.94 pp |
| Grounded Top-1 success | 15.67% | **16.42%** | +0.75 pp | 7.46% | **9.70%** | +2.24 pp |
| Finding F1 | 52.97% | **53.97%** | +1.00 pp | 58.53% | **58.55%** | +0.02 pp |
| Visible-evidence precision | 59.88% | **63.44%** | +3.56 pp | 60.65% | **60.99%** | +0.34 pp |
| Unsupported-finding rate | 49.45% | **48.39%** | -1.06 pp | **47.79%** | 48.09% | +0.31 pp |

The 27B improvement in semantic compliance is the largest evidence-task
change. The 4B result is mixed: its Top-1 and format reliability improved,
while Top-6 and semantic compliance decreased slightly.

## Open-ended blinded-judge results

The same frozen protocol was used for both temperatures: GPT-5.6 Luna with
high reasoning as the primary judge and Qwen 3.7 Flash only after a Luna
content-policy violation. Three `--retry-invalid` passes were applied to each
new run, matching the original local-model baseline. Evaluated-model answers
were never regenerated.

| Judge metric | 4B, t=1.0 | 4B, t=0.6 | Delta | 27B, t=1.0 | 27B, t=0.6 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Coverage | 95.33% | **99.00%** | +3.67 pp | 95.00% | **99.33%** | +4.33 pp |
| Top-1 among judged | 16.78% | **20.88%** | +4.09 pp | 24.21% | **25.84%** | +1.63 pp |
| Top-3 among judged | 38.11% | **43.77%** | +5.66 pp | 51.23% | **53.36%** | +2.13 pp |
| MRR among judged | 26.34% | **30.70%** | +4.36 pp | 35.85% | **37.53%** | +1.68 pp |
| Diagnosis correctness, 0-4 | 1.26 | **1.44** | +0.18 | 1.67 | **1.73** | +0.06 |
| Visible findings, 0-4 | 2.15 | **2.33** | +0.18 | 2.41 | **2.46** | +0.05 |
| Evidence grounding, 0-4 | 1.74 | **1.97** | +0.23 | 2.17 | **2.27** | +0.09 |
| Clinical rationale, 0-4 | 1.49 | **1.73** | +0.24 | 1.95 | **2.08** | +0.13 |
| Differential quality, 0-4 | 1.49 | **1.85** | +0.36 | 2.18 | **2.40** | +0.23 |
| Unsupported-claim rate | 94.76% | **89.56%** | -5.19 pp | 88.77% | **87.58%** | -1.19 pp |
| Mean unsupported claims | 4.32 | **3.54** | -0.79 | 3.01 | **2.78** | -0.24 |

Conservative rates over all 300 assigned cases also improved:

| Metric | 4B, t=1.0 | 4B, t=0.6 | 27B, t=1.0 | 27B, t=0.6 |
|---|---:|---:|---:|---:|
| Top-1 over all 300 | 16.00% | **20.67%** | 23.00% | **25.67%** |
| Top-3 over all 300 | 36.33% | **43.33%** | 48.67% | **53.00%** |

The final new judge outcomes were 297 valid and three persistent invalid for
the 4B model, and 298 valid and two persistent invalid for the 27B model. No
new run triggered a content-policy fallback or judge safety refusal.

The 4B judgment file contains 952 records rather than 300. Before the valid
pass, two launches omitted `.env` and each recorded 300
`InferenceConfigurationError` outcomes without producing a clinical score.
The later valid pass and controlled retries supersede those records by task
ID. The 27B judgment file contains 340 records: 300 initial judgments plus
24, 12, and four invalid-only retries. The metric aggregator always uses the
latest judgment for each task ID.

## Output integrity

| Integrity outcome across 2,262 cases | 4B, t=1.0 | 4B, t=0.6 | 27B, t=1.0 | 27B, t=0.6 |
|---|---:|---:|---:|---:|
| OK | 2,148 | **2,162** | 2,180 | **2,210** |
| Format invalid | 66 | **58** | 16 | **0** |
| Schema invalid | 9 | **1** | 4 | **0** |
| Semantic noncompliance | **39** | 41 | 62 | **52** |
| Backend error | 0 | 0 | 0 | 0 |
| Safety refusal | 0 | 0 | 0 | 0 |
| Image error | 0 | 0 | 0 | 0 |
| Truncated | 0 | 0 | 0 | 0 |

Task-specific format changes are important:

- Visual Top-K remained 100% JSON-valid and schema-valid for both models.
- Confusion JSON validity improved from 99.03% to 99.76% for 4B and from
  98.55% to 100% for 27B; recoverable validity remained 100%.
- Evidence JSON validity moved from 56.72% to 58.21% for 4B and from 97.01%
  to 100% for 27B.
- Evidence schema compliance improved from 92.54% to 99.25% for 4B and from
  97.01% to 100% for 27B.

The lower temperature also reduced mean open-ended response length from
1,817.68 to 1,711.72 characters for 4B and from 1,835.27 to 1,803.63 for 27B.

## Skin-tone slices

Skin-tone metadata was available for the same 955 Visual Top-K cases. These
are unadjusted descriptive Top-1 rates; source and disease composition remain
potential confounders.

| Aggregate | Cases | 4B, t=1.0 | 4B, t=0.6 | 27B, t=1.0 | 27B, t=0.6 |
|---|---:|---:|---:|---:|---:|
| Fitzpatrick 1-2 | 312 | 37.82% | **41.99%** | 44.23% | **46.79%** |
| Fitzpatrick 3-4 | 214 | 33.64% | **34.58%** | 44.86% | **46.26%** |
| Fitzpatrick 5-6 | 116 | 31.90% | **35.34%** | 38.79% | **43.10%** |
| Monk 1-3 | 205 | 25.37% | **30.24%** | 38.54% | **42.44%** |
| Monk 4-6 | 90 | 26.67% | **27.78%** | 42.22% | **44.44%** |
| Monk 7-10 | 18 | **5.56%** | 0.00% | 33.33% | 33.33% |
| Unknown | 45 | 57.78% | **60.00%** | **60.00%** | 55.56% |

All statistically supported groups improved for both models. The Monk 7-10
and unknown rows remain below the support threshold and must not be used for
fairness claims. The supported-group raw Top-1 gap increased from 15.19 to
16.81 percentage points for 4B and from 9.65 to 11.81 points for 27B, because
the gains were not uniform across groups.

## Runtime and operational disclosure

The new run used:

- NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB VRAM;
- driver 580.178.04;
- Python 3.12.11;
- vLLM 0.23.0 with BF16 weights and a 32,768-token context;
- one local endpoint per model on loopback only;
- GPU-memory utilization 0.65 for 27B and 0.15 for 4B;
- eight maximum concurrent sequences per endpoint.

Initial warm-up failed because vLLM selected FlashInfer sampling on the SM 12
Blackwell GPU and then reported an incompatible capability check. Both
servers were restarted with `VLLM_USE_FLASHINFER_SAMPLER=0`. This changed only
the top-k/top-p sampler to vLLM's native implementation; BF16 weights,
FlashAttention, GDN Triton execution, prompts, and generation parameters were
retained. The failed startup and successful restart are both preserved in
`outputs/parameter_ablation_temp_0_6/runpod_provenance/`.

The models shared the GPU for the first part of the run. Observed cumulative
task time was approximately 22 min 23 s for 4B and 53 min 56 s for 27B, but
these values are not throughput benchmarks. The 4B endpoint was stopped after
its outputs were synchronized, after which the 27B endpoint accelerated.

## Decision and next step

The observed evidence favours `temperature=0.6` for both local Qwen models:

1. both improved Top-K Top-1, Top-3, MRR, macro F1, and primary confusion-set
   ranking;
2. both improved Evidence Top-1 and open-ended judge Top-1/Top-3;
3. the 27B model became fully JSON/schema-valid on every structured task;
4. the 4B model substantially reduced unsupported open-ended claims and
   improved clinical-rationale scores;
5. no backend, safety, image, or truncation failure was introduced.

The improvement is not uniform. The 4B Evidence Top-6 and semantic compliance
decreased slightly, 27B canonical Confusion Top-2 decreased by 0.48 points,
and supported-group accuracy gaps widened. The hardware and sampler change
also prevents attributing every difference solely to temperature.

The defensible next action is therefore:

1. run the same paired 1.0-versus-0.6 comparison on Validation using the same
   GPU, vLLM runtime, native sampler, and at least two or three decoding seeds;
2. select temperature using a predeclared primary metric hierarchy, led by
   Top-K Top-1 and Evidence Top-1, while treating validity and unsupported
   claims as safety/quality constraints;
3. freeze the selected temperature before training and before any future
   untouched evaluation cohort;
4. retain the original Internal Benchmark table as the official sealed
   baseline and label this run explicitly as post-hoc sensitivity analysis.

## Reproduction and artifacts

The four tasks were launched by:

```bash
uv run python scripts/run_internal_parameter_ablation.py \
  --model qwen_3_5_4b \
  --base-url http://127.0.0.1:8002/v1 \
  --temperature 0.6 \
  --output-root outputs/parameter_ablation_temp_0_6/full

uv run python scripts/run_internal_parameter_ablation.py \
  --model qwen_3_6_27b \
  --base-url http://127.0.0.1:8000/v1 \
  --temperature 0.6 \
  --output-root outputs/parameter_ablation_temp_0_6/full
```

The ignored output directory contains predictions, metrics, HTML reports,
manifests, environment snapshots, judge artifacts, and vLLM logs. The source
runner is versioned so the exact task sequence and CLI overrides can be
reproduced without a notebook.
