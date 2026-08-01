# ISEPDermaBench Hugging Face release

## Objective

Create a private, reproducible Hugging Face benchmark dataset for the four
DermaISEP multimodal tasks while preventing scoring references from being sent
to a model as input.

## Published destinations

- Dataset: `danielfdias98/ISEPDermaBench`
- Bucket: `danielfdias98/ISEPDermaBench`
- Frozen dataset tags: `v1.0.0` and `v1.1.0`
- Initial dataset commit: `9d12adfcc3308b4629e97b3a6bfa0627fb094158`
- Version 1.1.0 dataset commit: `7cb7adcf21d95e9a45b600329e010f40fbb7dace`

Both destinations are private. The dataset repository is the interface for
Hugging Face Dataset Viewer and `load_dataset`; the bucket is a complete
artifact backup and direct object store.

## Reference isolation

Every benchmark is represented by two Hugging Face configurations:

```text
visual_top_k
visual_top_k_references

visual_confusion_sets
visual_confusion_sets_references

evidence_grounded_diagnosis
evidence_grounded_diagnosis_references

open_ended_diagnosis
open_ended_diagnosis_references
```

Task configurations contain the exact model request:

```text
image
system_prompt
user_prompt
response_schema_json
candidate_disease_ids
```

Reference configurations contain the gold diagnosis, morphology concepts,
reference descriptions, scoring flags, and evaluation-only subgroup metadata.
The two views join by `task_id`; `_references` fields must never be included in
the inference request.

## Included splits

| Benchmark | Split | Tasks | Unique images | Groups |
| --- | --- | ---: | ---: | ---: |
| Visual Top-K | Validation | 1,683 | 1,683 | 1,063 |
| Visual Top-K | Internal Benchmark | 1,000 | 1,000 | 1,000 |
| Visual Top-K | External DDI | 300 | 300 | 299 |
| Visual Top-K | External SkinDisNet | 1,365 | 1,365 | 333 |
| Visual Confusion Sets | Validation | 834 | 417 | 417 |
| Visual Confusion Sets | Internal Benchmark | 828 | 414 | 414 |
| Evidence-Grounded Diagnosis | Validation | 137 | 137 | 137 |
| Evidence-Grounded Diagnosis | Internal Benchmark | 134 | 134 | 134 |
| Evidence-Grounded Diagnosis | External DDI | 636 | 636 | 632 |
| Open-Ended Diagnosis | Validation | 100 | 100 | 100 |
| Open-Ended Diagnosis | Internal Benchmark | 300 | 300 | 300 |

Version 1.1.0 adds 400 open-ended tasks across two splits. Its model inputs
contain no candidate list or scoring references. The corresponding isolated
reference views provide the correct label and optional SKINCON/SkinCAP aids
only to the single blinded judge.

## Missing Internal Evidence cohort

The Evidence-Grounded benchmark previously had only internal Validation and
external DDI manifests. Version 1.0.0 materializes the missing sealed internal
cohort by intersecting `internal_benchmark_1000.parquet` with independent
Fitzpatrick17k SKINCON and SkinCAP references:

| Component | Cases |
| --- | ---: |
| Morphology | 134 |
| Clinical description | 119 |
| Diagnosis and grounding | 134 |
| Independent groups | 134 |
| Covered diseases | 19 of 21 |

This cohort must not select the teacher, prompt, parser, checkpoint, generation
settings, or thresholds.

## Image reproducibility

The Hub rows embed deterministic RGB JPEG bytes generated with
`dermatology_api_safe_rgb_jpeg_v1`:

- maximum edge: 768 pixels;
- maximum encoded size: 45,000 bytes;
- initial JPEG quality: 85;
- minimum JPEG quality: 40;
- minimum edge: 224 pixels.

These are the exact bytes intended for all API and local-model backends. Each
row records hashes for the original source image, benchmark image, rendered
prompt, response schema, benchmark YAML, and disease taxonomy.

## DDI policy

DDI images and references are included in both private destinations at the
user's explicit request. They remain governed by the upstream DDI Research Use
Agreement. Neither this release nor its combined dataset card grants public
redistribution rights. The dataset and bucket must remain private unless the
upstream terms are reviewed separately.

## Validation evidence

The release validator confirmed:

- all 50 files and recorded SHA-256 checksums;
- all embedded images decode and match their benchmark image hash;
- every task has a rendered system prompt, user prompt, and valid JSON schema;
- task and reference IDs match exactly within every split;
- task Parquets do not contain scoring-reference columns;
- zero Validation/Internal Benchmark overlap by `leakage_group_id`;
- all copied configs, prompts, schemas, and taxonomies match their hashes.

The v1.1.0 open-ended integration tests validate cohort size, 21-class
coverage, group isolation, free-text preservation, judge-schema scoring, and
the absence of model identity and provider reasoning from judge requests.
The exact bucket sync was verified after publication: all 58 files were
identical, with zero pending uploads, downloads, or deletions.

## Rebuild

```bash
python -m src.data_pipeline.open_ended_benchmark \
  --source data/benchmarks/ISEPDermaBench-v1.0.0 \
  --output data/benchmarks/ISEPDermaBench-v1.1.0
python -m src.data_pipeline.open_ended_benchmark \
  --output data/benchmarks/ISEPDermaBench-v1.1.0 \
  --validate-only
```

The first command requires a local copy of the frozen `v1.0.0` release. The
export is generated under `data/benchmarks/ISEPDermaBench-v1.1.0/`. Generated
Parquet shards and release manifests must not be edited manually.
