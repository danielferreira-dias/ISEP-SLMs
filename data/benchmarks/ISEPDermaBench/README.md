---
pretty_name: ISEPDermaBench
language:
- en
license: other
task_categories:
- visual-question-answering
- image-classification
tags:
- dermatology
- medical
- multimodal
- benchmark
- private-research-dataset
configs:
- config_name: visual_top_k
  data_files:
  - split: validation
    path: tasks/visual_top_k/validation-*.parquet
  - split: internal_benchmark
    path: tasks/visual_top_k/internal_benchmark-*.parquet
  - split: external_ddi
    path: tasks/visual_top_k/external_ddi-*.parquet
  - split: external_skindisnet
    path: tasks/visual_top_k/external_skindisnet-*.parquet
- config_name: visual_top_k_references
  data_files:
  - split: validation
    path: references/visual_top_k/validation-*.parquet
  - split: internal_benchmark
    path: references/visual_top_k/internal_benchmark-*.parquet
  - split: external_ddi
    path: references/visual_top_k/external_ddi-*.parquet
  - split: external_skindisnet
    path: references/visual_top_k/external_skindisnet-*.parquet
- config_name: clinical_context_ablation
  data_files:
  - split: validation
    path: tasks/clinical_context_ablation/validation-*.parquet
  - split: internal_benchmark
    path: tasks/clinical_context_ablation/internal_benchmark-*.parquet
- config_name: clinical_context_ablation_references
  data_files:
  - split: validation
    path: references/clinical_context_ablation/validation-*.parquet
  - split: internal_benchmark
    path: references/clinical_context_ablation/internal_benchmark-*.parquet
- config_name: visual_confusion_sets
  data_files:
  - split: validation
    path: tasks/visual_confusion_sets/validation-*.parquet
  - split: internal_benchmark
    path: tasks/visual_confusion_sets/internal_benchmark-*.parquet
- config_name: visual_confusion_sets_references
  data_files:
  - split: validation
    path: references/visual_confusion_sets/validation-*.parquet
  - split: internal_benchmark
    path: references/visual_confusion_sets/internal_benchmark-*.parquet
- config_name: evidence_grounded_diagnosis
  data_files:
  - split: validation
    path: tasks/evidence_grounded_diagnosis/validation-*.parquet
  - split: internal_benchmark
    path: tasks/evidence_grounded_diagnosis/internal_benchmark-*.parquet
  - split: external_ddi
    path: tasks/evidence_grounded_diagnosis/external_ddi-*.parquet
- config_name: evidence_grounded_diagnosis_references
  data_files:
  - split: validation
    path: references/evidence_grounded_diagnosis/validation-*.parquet
  - split: internal_benchmark
    path: references/evidence_grounded_diagnosis/internal_benchmark-*.parquet
  - split: external_ddi
    path: references/evidence_grounded_diagnosis/external_ddi-*.parquet
- config_name: open_ended_diagnosis
  data_files:
  - split: validation
    path: tasks/open_ended_diagnosis/validation-*.parquet
  - split: internal_benchmark
    path: tasks/open_ended_diagnosis/internal_benchmark-*.parquet
- config_name: open_ended_diagnosis_references
  data_files:
  - split: validation
    path: references/open_ended_diagnosis/validation-*.parquet
  - split: internal_benchmark
    path: references/open_ended_diagnosis/internal_benchmark-*.parquet
- config_name: visual_grounding_no_image
  data_files:
  - split: validation
    path: tasks/visual_grounding_no_image/validation-*.parquet
- config_name: visual_grounding_no_image_references
  data_files:
  - split: validation
    path: references/visual_grounding_no_image/validation-*.parquet
- config_name: general_visual_hallucination_audit
  data_files:
  - split: validation
    path: tasks/general_visual_hallucination_audit/validation-*.parquet
- config_name: general_visual_hallucination_audit_references
  data_files:
  - split: validation
    path: references/general_visual_hallucination_audit/validation-*.parquet
- config_name: dermatology_counterfactual_hallucination
  data_files:
  - split: validation
    path: tasks/dermatology_counterfactual_hallucination/validation-*.parquet
- config_name: dermatology_counterfactual_hallucination_references
  data_files:
  - split: validation
    path: references/dermatology_counterfactual_hallucination/validation-*.parquet
---

# ISEPDermaBench

ISEPDermaBench is the private, versioned evaluation dataset for the ISEP
small multimodal language model thesis. It packages the exact image and
rendered request seen by a model while keeping scoring references in separate
Hugging Face configurations.

## Configurations

Load task inputs and references independently:

```python
from datasets import load_dataset

tasks = load_dataset(
    "danielfdias98/ISEPDermaBench",
    "visual_top_k",
    split="validation",
)
references = load_dataset(
    "danielfdias98/ISEPDermaBench",
    "visual_top_k_references",
    split="validation",
)
```

Join the two views only inside the scorer by `task_id`. Never include fields
from a `_references` configuration in a model request.

| Benchmark | Split | Tasks | Unique images | Leakage groups |
| --- | --- | ---: | ---: | ---: |
| visual_top_k | validation | 1,000 | 1,000 | 646 |
| visual_top_k | internal_benchmark | 1,000 | 1,000 | 1,000 |
| visual_top_k | external_ddi | 300 | 300 | 299 |
| visual_top_k | external_skindisnet | 1,365 | 1,365 | 333 |
| clinical_context_ablation | validation | 494 (247 pairs) | 247 | 109 |
| clinical_context_ablation | internal_benchmark | 522 (261 pairs) | 261 | 261 |
| visual_confusion_sets | validation | 834 | 417 | 417 |
| visual_confusion_sets | internal_benchmark | 828 | 414 | 414 |
| evidence_grounded_diagnosis | validation | 137 | 137 | 137 |
| evidence_grounded_diagnosis | internal_benchmark | 134 | 134 | 134 |
| evidence_grounded_diagnosis | external_ddi | 636 | 636 | 632 |
| open_ended_diagnosis | validation | 100 | 100 | 100 |
| open_ended_diagnosis | internal_benchmark | 300 | 300 | 300 |
| visual_grounding_no_image | validation | 50 | 50 | 50 |
| general_visual_hallucination_audit | validation | 300 | 295 | 295 |
| dermatology_counterfactual_hallucination | validation | 200 | 200 | 200 |

### `visual_top_k`

Closed-set ranking of exactly six diseases from the frozen 21-class taxonomy.
It includes Validation, the 1,000-case internal paired benchmark, DDI external
evaluation, and SkinDisNet external evaluation.

### `clinical_context_ablation`

A paired SCIN ablation measuring whether explicitly reported clinical context
changes closed-set diagnostic ranking. Every case appears twice with identical
image bytes, taxonomy, schema, model settings, and base prompt: once without
patient context and once with participant-reported duration, body location,
lesion texture, lesion symptoms, and other symptoms.

Validation contains 247 pairs and Internal Benchmark contains 261 pairs. Only
cases with an explicit response in at least one SCIN `condition_symptoms_*`
field are included. `related_category`, dermatologist labels/confidence, and
demographic attributes are excluded from the model prompt. Missing fields mean
“not reported,” never “clinically absent.”

Primary analysis is within-model and paired: Top-1/3/6, MRR, macro F1, JSON
validity, and schema compliance are reported per condition. Context-minus-
image-only deltas use a paired bootstrap interval and an exact McNemar test.
SCIN labels are retrospective dermatologist differentials informed by
self-report, not uniformly pathology-confirmed diagnoses.

### `visual_confusion_sets`

Paired three-way ranking under low- and high-confusability candidate sets. The
same image is evaluated once per condition.

### `evidence_grounded_diagnosis`

Morphology grounding, observation-only clinical description, six-disease
differential diagnosis, and explicit evidence links. It includes Validation,
the newly materialized sealed internal evidence cohort, and external DDI.


### `open_ended_diagnosis`

Free-text, image-only clinical assessment with an explicitly ranked Top-3
differential and concise visible-evidence rationale. The evaluated model sees
no disease taxonomy, candidate IDs, gold label, SKINCON concepts, SkinCAP
description, or JSON schema. A separate stage uses GPT-5.6 Luna as the primary
judge and may use Qwen 3.7 Flash only when Luna returns a content-policy
violation. Each response still receives one final judgment; there is no voting.

### `visual_grounding_no_image`

Validation-only visual-grounding ablation derived from 50 group-unique cases
in the fixed 100-case Visual Top-K screening cohort. Each real image is
replaced by a uniform mid-gray JPEG with the same width and height. The correct
response is explicit `not_evaluable` abstention with no visual findings, no
diagnosis, and low confidence.

The original disease remains isolated in the reference view only to measure
accidental text-prior matches. Matching it is not clinical success and this
control must not be reported as diagnostic accuracy. The cohort covers all 21
diseases with at least two unique groups per class and is intended for paired
normal-image/no-image and thinking-off/thinking-on development comparisons.

### `general_visual_hallucination_audit`

A fixed 300-case subset of the official HaloQuest Evaluation
split. It retains the original 100-case audit unchanged and adds 200 cases.
The final condition distribution is exactly 100 false-premise questions, 100
insufficient-context questions, and 100 answerable visual challenges. The
selection maximizes source-image uniqueness, but permits a small number of
different questions to share an image because exact balance, the frozen parent
cohort, live upstream URLs, and 300 unique source images are not jointly
feasible. The primary deterministic metric is the
model's three-way question-status decision. False-premise rejection,
insufficient-context recognition, unanswerable hallucination, overconfidence,
and output validity are also reported.

The original HaloQuest answers are isolated in the reference configuration.
Free-text answer correctness for the 30 visual challenges is intentionally not
scored by lexical similarity because that would be brittle; it requires a
separately frozen semantic judge if it is later added. This task is a general
visual-grounding audit and must not be reported as dermatology accuracy.

### `dermatology_counterfactual_hallucination`

A fixed 200-case dermatology grounding audit derived only from Visual Top-K
Validation. It retains the original 50 cases unchanged and adds 150 new,
group-unique cases. It contains 50 deterministic RGB pixel shuffles, where
correct behavior is explicit low-confidence abstention, and 150 unique
hard-negative image swaps, where the diagnosis must follow the replacement
image rather than the hidden disease associated with the source task.

Metrics separate corrupted-image hallucination from hard-negative diagnostic
accuracy and include the rate at which a model incorrectly persists with the
source label. The source and counterfactual diagnoses are present only in the
reference view. This is a development robustness audit, not the sealed
clinical benchmark.

## Open-ended prompt protocol

Release 1.5.0 freezes model prompt 1.1.0 after a paired 50-case A/B test against the more prescriptive prompt 1.2.1. The selected prompt retains natural clinical prose, explicit Top-3 ordering, visible-evidence grounding, and no prose example. Judge prompt 1.2.0 and its four-verdict rubric remain unchanged.

Release 1.6.0 leaves that protocol unchanged and adds only the Validation-only
50-case no-image visual-grounding ablation.

Release 1.7.0 adds the two small hallucination audits described above without
changing any existing task, prompt, split, or reference.

Release 1.8.0 expands the general audit from 100 to 300 cases and the
dermatology counterfactual audit from 50 to 200 cases. The v1 cohorts are
strict subsets of the expanded cohorts, allowing prior outputs to be reused.
The expanded HaloQuest cohort contains 174 generated and 126 real cases. Five
official Flickr URLs returned HTTP 404/410, so deterministic replacements were
used; the resulting 300 tasks contain 295 unique source images.

Release 1.9.0 adds the paired SCIN clinical-context ablation without changing
existing task rows, prompts, schemas, or references. Every new pair is checked
against ISEPDermData Train by source-image SHA-256 and leakage group.

## Input schema

Task configurations begin with the multimodal input columns:

```text
image
task_id
sample_id
system_prompt
user_prompt
response_schema_json
candidate_disease_ids
```

Images are deterministic RGB JPEG representations produced with the frozen
`dermatology_api_safe_rgb_jpeg_v1` preprocessing profile. They are the exact
bytes intended for every API and local-model backend, not an additional
training representation.

## Reference isolation

Reference configurations contain the correct disease, morphology concepts,
reference description, scoring flags, and evaluation-only subgroup metadata.
The task Parquets contain no `reference_disease_id`, morphology gold labels,
or reference clinical descriptions.

## Validation rebalance

Release 1.2.0 reduces visual Top-K Validation from 1,683 to 1,000 image tasks using whole leakage groups. All groups required by the other Validation tasks remain protected. The 683 released images are promoted to ISEPDermData Train under the auditable `group_safe_validation_to_train_v1` policy.

## Split policy

- Validation may be used for dry runs, prompt/parser development, teacher
  selection, checkpoint selection, and threshold calibration.
- Internal Benchmark is sealed and supports the paired before/after result.
- External sets measure generalization and must not select the teacher or tune
  the system.
- No split from this repository may be used for student fine-tuning while it
  retains an evaluation role.

## DDI restrictions

DDI rows and images are included because this repository is private and used
for the approved research workflow. They remain governed by the upstream DDI
Research Use Agreement. This repository does not grant permission to make DDI
public, redistribute it, or use it outside the upstream terms.

## Reproducibility

Canonical benchmark YAMLs, prompt templates, JSON schemas, and taxonomies are
copied under `artifacts/`. Every row stores hashes for its rendered prompt,
response schema, benchmark config, taxonomy, source image, and final benchmark
image. Exact shard checksums and counts are recorded in `release.json`.

Build and validate locally with:

```bash
python -m src.data_pipeline.open_ended_benchmark \
  --source data/benchmarks/ISEPDermaBench-v1.0.0 \
  --output data/benchmarks/ISEPDermaBench-v1.1.0
python -m src.data_pipeline.open_ended_benchmark \
  --output data/benchmarks/ISEPDermaBench-v1.1.0 \
  --validate-only
python -m src.data_pipeline.visual_grounding_no_image --validate-only
python -m src.data_pipeline.clinical_context_ablation --validate-only
```
