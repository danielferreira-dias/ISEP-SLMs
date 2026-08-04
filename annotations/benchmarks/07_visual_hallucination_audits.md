# Small visual hallucination audits

## Decision

Two independent, development-only evaluations were added in release 1.7.0
and expanded as nested cohorts in release 1.8.0:

1. `general_visual_hallucination_audit`: 300 fixed HaloQuest cases;
2. `dermatology_counterfactual_hallucination`: 200 fixed dermatology cases.

They complement the 50-case uniform-gray control. They do not replace the
sealed Internal Benchmark and must not be reported as clinical accuracy.

## General Visual Hallucination Audit

The source is the official HaloQuest Evaluation protocol. The original 100
cases remain byte-for-byte identifiable as the parent cohort. A deterministic
SHA-256 expansion with seed 42 produces 300 unique source images:

| HaloQuest condition | Generated | Real | Total |
| --- | ---: | ---: | ---: |
| False premises | — | — | 100 |
| Visual challenge | — | — | 100 |
| Insufficient context | — | — | 100 |
| Total | — | — | 300 |

The selector maximizes unique source images. Exact 100/100/100 balance is not
feasible while simultaneously retaining all original cases, rejecting dead
upstream URLs, and forbidding all image reuse. A small number of official rows
therefore share an image but use different questions; their shared
`leakage_group_id` makes this explicit. The evaluated model returns one of
`answerable`, `false_premise`, or `insufficient_visual_evidence`, an optional
answer, and confidence. The primary metric is
`question_status_accuracy`. Secondary metrics disclose false-premise
rejection, insufficient-context recognition, hallucination on unanswerable
questions, overconfidence, JSON validity, and schema compliance.

The official free-text reference answer is kept in the isolated reference
view. It is not scored with string similarity: lexical overlap is too brittle
to represent answer correctness. The current deterministic evaluation
therefore measures answerability and grounding, not complete HaloQuest answer
accuracy. A future answer-correctness score would require a separately frozen
semantic judge.

Source: [Google HaloQuest](https://github.com/google/haloquest).

The materialized v2 cohort contains 174 generated and 126 real cases, with
295 unique images. Five selected Flickr URLs from the official CSV returned
HTTP 404/410 in August 2026. Their identities and errors are frozen in
`metadata/general_visual_hallucination_v2/manifest.json`; deterministic
replacement rows preserve all 300 tasks and exact condition balance.

## Dermatology counterfactual audit

The original 50 source cases are the group-unique, 21-class cohort used by the
no-image audit. The expansion selects 150 additional group-unique cases from
Visual Top-K Validation only. Exactly 50 cases receive a deterministic RGB
pixel permutation and exactly 150 receive a unique image from another disease.
The original 25/25 condition assignments and donor mappings are preserved;
new donors are matched without replacement, with clinically confusable disease
sets preferred when possible.

Expected behavior differs by condition:

- pixel shuffle: `not_evaluable`, no findings, no diagnoses, low confidence;
- hard-negative swap: `evaluable`, findings and diagnoses must follow the
  replacement image rather than the hidden source task label.

The primary `full_counterfactual_success_rate` requires correct abstention for
pixel-shuffled cases and the correct Top-1 replacement-image diagnosis for
hard-negative cases. The report also separates grounding accuracy, Top-1/3,
hallucinated findings/diagnoses, output validity, and
`hard_negative_source_label_persistence_rate`.

Both the original label and replacement-image label are isolated in the
reference Parquet. Neither appears in the prompt or task columns.

## Materialized layout

```text
data/benchmarks/ISEPDermaBench/
├── tasks/general_visual_hallucination_audit/validation-00000-of-00001.parquet
├── references/general_visual_hallucination_audit/validation-00000-of-00001.parquet
├── tasks/dermatology_counterfactual_hallucination/validation-00000-of-00001.parquet
├── references/dermatology_counterfactual_hallucination/validation-00000-of-00001.parquet
├── metadata/general_visual_hallucination_v1/  # frozen parent
├── metadata/general_visual_hallucination_v2/  # parent + added IDs
├── metadata/dermatology_counterfactual_v1/    # frozen parent
└── metadata/dermatology_counterfactual_v2/    # parent + added IDs
```

The complete release validator checks image hashes, task/reference joins,
reference isolation, shard checksums, and declared counts.
