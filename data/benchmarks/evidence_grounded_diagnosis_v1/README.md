# Evidence-Grounded Diagnosis Dataset Release

This directory contains the frozen manifest used by the evidence-grounded
dermatology diagnosis benchmark. The benchmark combines DDI images and
diagnoses with SKINCON morphology concepts and SkinCAP descriptions.

## Structure

```text
evidence_grounded_diagnosis_v1/
├── datasets/
│   └── external/
│       └── evidence_grounded_ddi.parquet
└── release/
    └── benchmark_release_v1.yaml
```

## Why it is under `external`

This benchmark does not have an internal cohort with morphology references.
The available SKINCON annotations and SkinCAP descriptions are associated
with DDI images, which are reserved for external evaluation in this project.
Moving or copying these cases into `datasets/internal` would not make them
internal data and would result in methodologically incorrect provenance.

This file must not be used for training, prompt selection, checkpoint
selection, or threshold definition.

## Evaluation cohorts

The manifest contains 636 images and 632 leakage groups. The scoring flags
indicate which references are available for each row:

| Flag | Images | Evaluation |
| --- | ---: | --- |
| `score_morphology` | 636 | SKINCON visual concepts |
| `score_description` | 635 | Clinical description with a SkinCAP reference |
| `score_diagnosis` | 294 | Diagnosis, grounding, and calibration |

The 20 images marked by SKINCON as `Do not consider this image` are excluded.
Of the 300 DDI images eligible for the diagnosis taxonomy, six have this
annotation, leaving 294 images for the diagnosis metrics. The remaining cases
are still useful for morphology and description evaluation, even when their
diagnosis is outside the closed taxonomy of 21 diseases.

## Main columns

| Column | Contents |
| --- | --- |
| `sample_id` | Stable DDI case identifier |
| `image_uri` | Image path |
| `leakage_group_id` | Group used to control duplicates |
| `disease_id` | Normalized diagnosis, when eligible |
| `morphology_concept_ids` | List of positive SKINCON concepts |
| `reference_clinical_description` | SkinCAP description, when available |
| `score_morphology` | Whether morphology can be scored |
| `score_description` | Whether the description can be scored |
| `score_diagnosis` | Whether diagnosis and grounding can be scored |

`reference_clinical_description` is a secondary reference: SkinCAP captions
may contain diagnoses, tests, or recommendations and are not always strictly
observation-only descriptions.

## Experiment configuration

The zero-shot model comparison is declared in:

```text
configs/experiments/zero_shot_evidence_grounded_v1.yaml
```

It evaluates the same nine model candidates used by the other zero-shot
benchmarks. Because DDI is held out for external evaluation, its results must
not be used for teacher, prompt, checkpoint, or threshold selection.

## Rebuild and validate

Deterministically rebuild the manifest and release:

```bash
python -m src.data_pipeline.evidence_grounded
```

Validate the existing file without rebuilding it:

```bash
python -m src.data_pipeline.evidence_grounded --validate-only
```

The generated Parquet and YAML files must not be edited manually. Changes
should be made to the source data, taxonomy, or benchmark configuration,
followed by rebuilding the release.
