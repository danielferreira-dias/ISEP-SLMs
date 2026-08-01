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
| visual_top_k | validation | 1,683 | 1,683 | 1,063 |
| visual_top_k | internal_benchmark | 1,000 | 1,000 | 1,000 |
| visual_top_k | external_ddi | 300 | 300 | 299 |
| visual_top_k | external_skindisnet | 1,365 | 1,365 | 333 |
| visual_confusion_sets | validation | 834 | 417 | 417 |
| visual_confusion_sets | internal_benchmark | 828 | 414 | 414 |
| evidence_grounded_diagnosis | validation | 137 | 137 | 137 |
| evidence_grounded_diagnosis | internal_benchmark | 134 | 134 | 134 |
| evidence_grounded_diagnosis | external_ddi | 636 | 636 | 632 |

### `visual_top_k`

Closed-set ranking of exactly six diseases from the frozen 21-class taxonomy.
It includes Validation, the 1,000-case internal paired benchmark, DDI external
evaluation, and SkinDisNet external evaluation.

### `visual_confusion_sets`

Paired three-way ranking under low- and high-confusability candidate sets. The
same image is evaluated once per condition.

### `evidence_grounded_diagnosis`

Morphology grounding, observation-only clinical description, six-disease
differential diagnosis, and explicit evidence links. It includes Validation,
the newly materialized sealed internal evidence cohort, and external DDI.

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
python -m src.data_pipeline.huggingface_benchmark_export
python -m src.data_pipeline.huggingface_benchmark_export --validate-only
```
