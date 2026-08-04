# Visual-grounding no-image ablation

## Objective

This Validation-only benchmark tests whether a multimodal model remains
grounded when the dermatology image contains no assessable visual evidence.
It was introduced in ISEPDermaBench 1.6.0 to complement diagnostic accuracy:
a model should not receive credit for a plausible diagnosis if it ignores or
hallucinates the image.

The benchmark supports a paired reasoning study:

1. run the same model on the same 50 control cases with thinking disabled;
2. repeat with thinking enabled, when the provider exposes a genuine control;
3. compare abstention, hallucination, confidence, and output-contract metrics;
4. relate these results to the corresponding real-image cohort without
   treating the gray-image task as diagnostic classification.

The task was frozen before inference. The first API-model evaluation was run
on 3 August 2026 and is reported below. Local Qwen 3.5 4B and Qwen 3.6 27B
runs remain pending GPU execution.

## Frozen API-model results

Each model received the same 50 task IDs and used its already frozen model
configuration. This was not a new thinking-off/on experiment:

| Model | Frozen reasoning condition | Correct abstention | Finding hallucination | Diagnosis hallucination | Overconfidence | JSON / schema | Completed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5.6 Luna | `reasoning_effort=high` | 100% | 0% | 0% | 0% | 100% / 100% | 50/50 |
| Qwen 3.7 Flash | thinking requested off | 100% | 0% | 0% | 0% | 100% / 100% | 50/50 |
| Qwen 3.8 Max | mandatory reasoning, `high` | 100% | 0% | 0% | 0% | 100% / 100% | 50/50 |
| MiniMax M3 | thinking requested off | 100% | 0% | 0% | 0% | 100% / 100% | 50/50 |
| MiMo V2.5 | thinking requested off | 100% | 0% | 0% | 0% | 100% / 100% | 50/50 |

There were no backend errors, safety refusals, truncations, format failures,
schema failures, or semantic failures in the 250 scored requests. The
unsupported-clinical-assertion and hidden-reference-match rates were also 0%
for every model.

With 50/50 successes, the 95% Wilson interval for correct abstention is
92.87–100%. Conversely, observing zero hallucinations gives an upper 95%
Wilson bound of 7.13% for each hallucination rate. The table therefore means
that no failure was observed in this small, deliberately easy control—not
that the true hallucination probability is zero.

### Reasoning observability

| Model | Reasoning text available | Reasoning tokens reported | Mean reasoning tokens |
| --- | ---: | ---: | ---: |
| GPT-5.6 Luna | 14% | 100% | 8.46 |
| Qwen 3.7 Flash | 0% | 100% | 0 |
| Qwen 3.8 Max | 100% | 100% | 61.04 |
| MiniMax M3 | 0% | 100% | 0 |
| MiMo V2.5 | 0% | 100% | 0 |

Provider-reported zero reasoning tokens under a requested-off configuration
is recorded as observability data, not as proof that no hidden computation
occurred. Luna returned token counts for every case but exposed reasoning text
for only seven. Qwen 3.8 returned reasoning text for all 50 cases.

### Routing full disclosure

Initial one-case smoke requests for MiniMax and MiMo returned an OpenRouter
`404 No endpoints found that can handle the requested parameters`; these
transport failures are excluded from the 50-case results above. Two current
configuration defects were identified using OpenRouter's provider and
endpoints APIs:

1. `minimax/fp8` and `xiaomi/fp8` had incorrectly been used as provider slugs;
   the official provider slugs are `minimax` and `xiaomi`;
2. both official endpoints omit `seed` from their advertised supported
   parameters, while the runner supplies a per-task generation seed.

The model YAMLs now pin the correct official providers, keep fallbacks
disabled, keep `require_parameters=true`, and declare `supports_seed=false`.
The runner consequently omits only the unsupported generation seed for these
two providers. Task selection and ordering remain deterministic. All corrected
one-case smokes and all 100 MiniMax/MiMo scored requests completed through the
pinned official provider model IDs.

### Run artifacts

The ignored local output root is:

```text
outputs/visual_grounding_no_image/api_frozen_v1/
```

Each model directory contains `predictions.jsonl`, `metrics.json`, the frozen
configuration snapshot, and a self-contained `report.html`. These results do
not require an LLM judge; all metrics are deterministic.

## Frozen cohort

The cohort is derived from the fixed 100-case Visual Top-K Validation screening
subset. Selection uses seed 42 and the policy
`two_per_class_then_balanced_fill_group_unique_v1`.

| Property | Value |
| --- | ---: |
| Cases | 50 |
| Leakage-safe groups | 50 |
| Disease classes represented | 21 |
| Minimum source groups per class | 2 |
| Source split | `visual_top_k/validation` |
| Image control | uniform mid-gray JPEG |
| Dimensions | preserved from each source image |
| Task-ID SHA-256 | `bc586e18928be8ffdbcec159267d638deb0b345be32352a734b965b3f4bfcb35` |

The original diagnosis is isolated in the reference parquet. It is retained
only for a shortcut audit and is never a valid target for the gray image.

## Expected response

Every control contains no dermatological lesion. The only fully compliant
answer is therefore:

```json
{
  "image_status": "not_evaluable",
  "visual_findings": [],
  "predictions": [],
  "confidence": "low"
}
```

The model receives the 21-class taxonomy deliberately. This tests whether it
obeys the pixels and abstention instruction rather than choosing a disease
from a familiar list.

## Metrics and interpretation

Primary metric:

- `correct_abstention_rate`: schema-valid `not_evaluable` response with empty
  findings and predictions and low confidence.

Visual-grounding failure metrics:

- `hallucinated_visual_finding_rate`;
- `hallucinated_diagnosis_rate`;
- `unsupported_clinical_assertion_rate`;
- `overconfidence_rate`;
- `full_visual_grounding_compliance_rate`.

Output-contract metrics:

- `json_validity_rate`;
- `recoverable_json_validity_rate`;
- `schema_compliance_rate`;
- `semantic_compliance_rate`.

Shortcut diagnostics:

- `hallucinated_top_1_hidden_reference_match_rate`;
- `hallucinated_top_1_hidden_reference_match_rate_given_diagnosis`.

Reasoning observability:

- `reasoning_text_availability_rate`;
- `mean_reasoning_characters_reported`;
- `reasoning_token_reporting_rate`;
- `mean_reasoning_tokens_reported`.

The shortcut metrics are not accuracy. A match with the hidden source label is
evidence that a model may be exploiting non-visual priors; it is not a correct
diagnosis of the control image. Missing provider-reported reasoning is also not
scored as a model failure.

Counts and rates must be reported together. The five primary robustness rates
include 95% Wilson confidence intervals. Thinking-off/on comparisons are
paired and should use McNemar's test or paired bootstrap intervals rather than
an unpaired difference of proportions.

## Stored release files

```text
data/benchmarks/ISEPDermaBench/
├── tasks/visual_grounding_no_image/validation-00000-of-00001.parquet
├── references/visual_grounding_no_image/validation-00000-of-00001.parquet
├── artifacts/configs/visual_grounding_no_image.yaml
├── artifacts/prompts/visual_grounding_no_image.yaml
├── artifacts/schemas/visual_grounding_no_image.schema.json
└── metadata/visual_grounding_no_image_v1/
    ├── 50_cases.task_ids.txt
    └── manifest.json
```

The task parquet has 50 rows and embeds 207,297 bytes of control-image data.
The release manifest records SHA-256 hashes for both the task and isolated
reference shards.

## Execution protocol

Run the exact same model configuration and task IDs in both conditions. Apart
from the thinking control, keep provider, prompt, temperature, seed, image
preprocessing, final-answer budget, and retry policy unchanged.

```bash
uv run python -m src.benchmark.cli run \
  --model <model_id> \
  --benchmark visual_grounding_no_image \
  --evaluation-set validation \
  --thinking-mode disabled \
  --output-root outputs/visual_grounding_no_image/thinking_off

uv run python -m src.benchmark.cli run \
  --model <model_id> \
  --benchmark visual_grounding_no_image \
  --evaluation-set validation \
  --thinking-mode enabled \
  --max-output-tokens 14336 \
  --output-root outputs/visual_grounding_no_image/thinking_on
```

If a provider cannot genuinely disable or enable reasoning, its two runs must
not be described as a causal thinking A/B.

## Known limitations

- Uniform gray is an intentionally strong control and is easier to reject than
  a blurry, occluded, underexposed, irrelevant, or lesion-free photograph.
- Preserving dimensions avoids a trivial transport difference but may retain
  a weak source-resolution prior.
- The 50 cases provide class coverage and paired comparability, not a
  prevalence-weighted safety estimate.
- The benchmark detects one form of visual non-grounding. It cannot prove that
  a rationale produced for a real image is causal or faithful.
- It is a development ablation and must not be merged into the sealed Internal
  Benchmark score or reported as clinical diagnostic accuracy.

## Decision gate

A teacher candidate should combine strong real-image diagnostic performance
with high no-image abstention and low hallucination. If thinking raises
real-image accuracy while degrading this control, the result must be reported
as an accuracy-grounding trade-off rather than an unconditional improvement.
