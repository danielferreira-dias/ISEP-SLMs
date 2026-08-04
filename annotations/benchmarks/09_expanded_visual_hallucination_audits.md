# Expanded visual hallucination audits (ISEPDermaBench 1.8.0)

## Decision and provenance

Release 1.8.0 expands the two development-only hallucination audits while
preserving the original cohorts as strict subsets:

| Audit | Parent | Added | Full cohort |
| --- | ---: | ---: | ---: |
| General Visual Hallucination | 100 | 200 | 300 |
| Dermatology Counterfactual | 50 | 150 | 200 |

The exact parent task-ID sets were compared with the prior prediction files;
both comparisons were exact (`100/100` and `50/50`). Provider inference was
therefore executed only for the added cases. The full metrics and HTML reports
were regenerated from the union of the disjoint parent and expansion outputs.
The aggregation utility also verifies that this union equals every task ID in
the expanded Parquet release.

## General Visual Hallucination

The source remains the official HaloQuest Evaluation CSV. The expanded cohort
contains exactly 100 tasks for each condition:

- false premise;
- insufficient visual evidence;
- answerable visual challenge.

It contains 174 generated and 126 real cases. The selector maximizes image
diversity while retaining the 100 original tasks. Five selected Flickr URLs
returned HTTP 404/410 on 4 August 2026. Deterministic replacement rows were
used, yielding 300 tasks over 295 unique source images. Shared images always
retain the same `leakage_group_id` and use different official questions.
The unavailable URLs and errors are disclosed in
`metadata/general_visual_hallucination_v2/manifest.json`.

This audit deterministically scores answerability and premise grounding. It
does **not** score free-text answer correctness for visual-challenge cases;
that would require a separately frozen semantic judge.

## Dermatology Counterfactual

All 200 source cases come only from Visual Top-K Validation. No Internal
Benchmark image was used. The selected sources are group-unique and cover all
21 diseases:

- 50 deterministic RGB pixel shuffles;
- 150 disease-mismatched hard-negative swaps.

The original 25 pixel-shuffle assignments, 25 hard-negative assignments, and
hard-negative donor mappings are unchanged. The 150 added sources contribute
25 additional pixel shuffles and 125 additional hard-negative swaps. New
hard-negative donors are unique, use a different disease, and prefer the same
clinical confusion set when possible.

## Inference protocol

All structured outputs used prompt-only JSON. No provider-specific constrained
decoding was enabled. Reasoning was excluded from the scored final answer.

| Model | Effective reasoning for this audit |
| --- | --- |
| GPT-5.6 Luna | high, provider summary/tokens captured when available |
| Qwen 3.8 Max | low for the primary table; high rerun completed for General and stopped at 146/150 added Dermatology cases |
| Qwen 3.7 Flash | disabled for the primary table; thinking rerun completed for General and stopped at 21/150 added Dermatology cases |
| MiniMax M3 | disabled |
| MiMo V2.5 | disabled |
| Qwen 3.5 4B | disabled, local vLLM on H200 |
| Qwen 3.6 27B | disabled, local vLLM on H200 |

OpenRouter models were pinned to their official provider and provider fallback
was disabled. The two local models received the same embedded benchmark JPEGs,
prompts, schemas, and task IDs as the API models.

## Full-cohort results

The following table combines the parent and added outputs. Percentages use all
300 general cases or all 200 dermatology cases, as applicable.

| Model | General status | False premise | Insufficient context | Visual challenge | Unanswerable hallucination | Derm full success | Pixel abstention | Hard-neg Top-1 | Hard-neg Top-3 | Source persistence | JSON / schema (G) | JSON / schema (D) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.8 Max (low) | 80.7% | 87.0% | 72.0% | 83.0% | 6.0% | 62.5% | 100.0% | 50.0% | 73.3% | 4.7% | 100.0% / 100.0% | 97.5% / 97.5% |
| GPT-5.6 Luna (high) | 77.7% | 75.0% | 72.0% | 86.0% | 12.0% | 53.5% | 100.0% | 38.0% | 62.0% | 7.3% | 99.7% / 99.7% | 96.5% / 96.5% |
| Qwen 3.6 27B (off) | 74.3% | 79.0% | 73.0% | 71.0% | 5.0% | 57.0% | 100.0% | 42.7% | 69.3% | 6.0% | 100.0% / 97.0% | 100.0% / 100.0% |
| MiMo V2.5 (off) | 73.7% | 84.0% | 61.0% | 76.0% | 5.0% | 47.5% | 100.0% | 30.0% | 52.0% | 6.7% | 100.0% / 100.0% | 100.0% / 100.0% |
| MiniMax M3 (off) | 72.7% | 72.0% | 67.0% | 79.0% | 8.5% | 48.5% | 100.0% | 31.3% | 56.7% | 5.3% | 100.0% / 100.0% | 100.0% / 100.0% |
| Qwen 3.7 Flash (off) | 69.0% | 84.0% | 43.0% | 80.0% | 5.0% | 58.0% | 100.0% | 44.0% | 62.0% | 8.7% | 95.7% / 95.7% | 97.0% / 97.0% |
| Qwen 3.5 4B (off) | 68.7% | 46.0% | 83.0% | 77.0% | 8.5% | 51.5% | 100.0% | 35.3% | 59.3% | 6.7% | 99.7% / 99.0% | 100.0% / 99.5% |

`Unanswerable hallucination` is lower-is-better. `Source persistence` is also
lower-is-better: it measures incorrectly retaining the hidden source-task
label after the image has been replaced. `Derm full success` requires correct
low-confidence abstention for a pixel shuffle or both an evaluable decision
and correct replacement-image Top-1 for a hard negative.

These remain development robustness audits rather than final clinical
accuracy. The General audit includes non-dermatological images, and the
Dermatology audit is derived from Validation. Model selection should consider
these results jointly with the sealed Internal Benchmark and not optimize one
headline percentage in isolation.

## Reasoning escalation comparison and stop decision

After the primary low/off results, the complete 200-case General expansion was
also run with Qwen 3.8 at high reasoning and Qwen 3.7 with thinking enabled.
Each result was combined with its matching 100-case parent, producing two
complete, paired 300-case General cohorts. The models received exactly the same
task IDs, images, prompts, and schemas as their primary low/off runs.

| Model and mode | General accuracy | False premise | Insufficient context | Visual challenge | Unanswerable hallucination | Overconfidence | Raw JSON | Recoverable JSON / schema | Mean reasoning tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.8 Max — low | **80.7%** | **87.0%** | 72.0% | 83.0% | 6.0% | 18.0% | 100.0% | 100.0% / 100.0% | 176.4 |
| Qwen 3.8 Max — high | 80.0% | 80.0% | **76.0%** | **84.0%** | 6.0% | **13.0%** | 100.0% | 100.0% / 100.0% | 669.0 |
| Qwen 3.7 Flash — off | 69.0% | 84.0% | 43.0% | **80.0%** | 5.0% | 29.5% | **95.7%** | 95.7% / 95.7% | 0.0 |
| Qwen 3.7 Flash — thinking | **76.0%** | **88.0%** | **73.0%** | 67.0% | **4.0%** | **19.0%** | 54.0% | **99.7% / 99.7%** | 1,508.9 |

`Unanswerable hallucination` and `overconfidence` are lower-is-better. Qwen
3.8 high reduced overall accuracy by 0.7 percentage points while using about
3.8 times as many reported reasoning tokens. Its condition-level changes were
mixed: false-premise rejection fell by 7 points, insufficient-context
recognition rose by 4 points, and visual-challenge answerability rose by 1
point. This provides no practical justification for replacing low reasoning
with high reasoning in the final screening configuration.

Qwen 3.7 thinking did improve General status accuracy by 7 points, particularly
on insufficient context. It therefore would be incorrect to claim that
thinking reduced every accuracy metric. However, visual-challenge
answerability fell by 13 points, raw JSON validity collapsed from 95.7% to
54.0%, and the model used approximately 1,509 reasoning tokens per case. The
frozen parser recovered nearly all fenced or otherwise wrapped JSON, which is
why recoverable schema compliance remained 99.7%. This is a parser-assisted
semantic gain accompanied by a major deterioration in native output-contract
reliability and substantially greater cost and latency.

The remaining Dermatology expansion requests were stopped once this decision
signal was sufficient:

| Stopped run | Completed added cases | Final partial statuses |
| --- | ---: | --- |
| Qwen 3.8 Max — high | 146 / 150 | 135 ok; 5 backend errors; 3 format invalid; 3 schema invalid |
| Qwen 3.7 Flash — thinking | 21 / 150 | 21 ok |

These partial Dermatology runs are retained for audit only. They are not
combined with the parent cohort, not compared as 200-case results, and not used
for model ranking. The primary full-cohort table therefore continues to use
Qwen 3.8 low and Qwen 3.7 thinking disabled.

The decision is operational rather than universal: explicit reasoning is not
claimed to always reduce visual intelligence. For these models and this
protocol, escalating reasoning did not provide a sufficiently consistent gain
to offset format reliability, token cost, latency, and provider-failure risk.
Thinking remains disabled by default for the final visual benchmarks; targeted
reasoning ablations can remain separate research experiments.

### Output-status disclosure

All assigned cases remain in the relevant metric denominator. Backend errors,
image errors, refusals, and unrecoverable outputs have no usable prediction and
therefore lower accuracy. A raw format-invalid answer may still be scored from
its canonical output when the frozen parser recovers and validates it; its raw
JSON failure remains visible in the format metrics. No primary full-cohort case
was selectively retried or silently repaired.

| Model | General output status (n=300) | Dermatology output status (n=200) |
| --- | --- | --- |
| Qwen 3.8 Max | 300 ok | 195 ok; 5 backend errors |
| GPT-5.6 Luna | 299 ok; 1 format invalid | 193 ok; 7 safety refusals |
| Qwen 3.6 27B | 291 ok; 9 schema invalid | 200 ok |
| MiMo V2.5 | 300 ok | 200 ok |
| MiniMax M3 | 300 ok | 200 ok |
| Qwen 3.7 Flash | 287 ok; 13 backend errors | 194 ok; 5 backend errors; 1 format invalid |
| Qwen 3.5 4B | 297 ok; 2 schema invalid; 1 format invalid | 199 ok; 1 schema invalid |

There were no truncated outputs in these full cohorts. The seven Luna safety
refusals occurred only in the dermatology task. Qwen 3.7 and Qwen 3.8 backend
errors are disclosed separately from model-format errors because they do not
show a completed model answer, but the primary analysis still treats them as
failures under the no-retry protocol.

## Artifacts

- Full reports: `outputs/hallucination_audits_v2/full_combined/`
- Incremental provider runs:
  `outputs/hallucination_audits_v2/expanded_additions/`
- Complete General reasoning comparisons:
  `outputs/hallucination_audits_v2/full_combined_reasoning/`
- Aggregator: `src/benchmark/aggregate_hallucination_runs.py`
- Dataset release: `data/benchmarks/ISEPDermaBench/` (version 1.8.0)

Release 1.8.0 was synchronized to both the private Hugging Face dataset
repository and `hf://buckets/danielfdias98/ISEPDermaBench` after the release
validator and regression tests passed.
