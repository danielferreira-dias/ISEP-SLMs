# Normalization implementation

## Objective

This stage implemented reproducible source adapters and generated one
schema-stable Parquet manifest per classification dataset.

## Implementation

The pipeline is implemented under `src/data_pipeline/`:

| Module | Responsibility |
| --- | --- |
| `common.py` | Arrow schema, YAML loading, disease mapping, row construction, and Parquet writing |
| `adapters.py` | Fitzpatrick17k-C, PAD-UFES-20, SCIN, and DDI source adapters |
| `reporting.py` | Cross-dataset concatenation, label inventory, and coverage reports |
| `pipeline.py` | Command-line orchestration and manifest validation |

The pipeline is executed from the repository root:

```bash
.venv/bin/python -m src.data_pipeline.pipeline
```

Existing manifests can be validated without rebuilding:

```bash
.venv/bin/python -m src.data_pipeline.pipeline --validate-only
```

## Dataset-specific transformations

### Fitzpatrick17k-C

- The corrected 11,394-row metadata file is used instead of the original
  release.
- `label` is the source disease target.
- `md5hash` provides the image and fallback group identifier.
- Every referenced archive member is checked before the row is emitted.
- Source partitions are retained only as provenance and are not adopted as the
  final project split.
- Fitzpatrick values outside 1 through 6 are treated as unavailable.

No patient identifier is available. Image-level grouping therefore cannot
prevent leakage between visually related images; URL and perceptual duplicate
analysis remains required.

### PAD-UFES-20

- All 2,298 metadata rows are normalized.
- `patient_id` is used as `group_id`, while `lesion_id` remains case metadata.
- The six diagnostic codes are mapped through reviewed dataset-specific rules.
- `biopsed` distinguishes pathology from clinical-consensus labels.
- Every image filename is resolved against exactly one of the three archives.

### SCIN

- All 26 Parquet shards are read using column projection so embedded image
  bytes are not loaded.
- One case row is expanded to one row for each non-null image path.
- `weighted_skin_condition_label` is parsed as an ordered differential.
- Candidates are sorted by descending weight; source order breaks equal-weight
  ties.
- The highest-weight diagnosis is the primary label used by the current
  single-reference benchmark policy.
- `group_id` is based on `case_id`.
- The US Monk Skin Tone annotation is preferred when available, followed by
  the India annotation. Both raw annotations remain in source metadata.

The local SCIN revision contains 5,033 cases and 10,406 non-null image paths:
5,032 first images, 3,085 second images, and 2,289 third images. The commonly
reported total is 10,408 images. The local release has one null first-image
entry, and the remaining difference must be treated as a source-release
discrepancy rather than silently corrected.

### DDI

- All 656 metadata rows resolve to local PNG files.
- The released disease label is treated as pathology-grounded.
- The grouped Fitzpatrick value is retained.
- The public metadata does not provide a patient identifier, although the
  dataset documentation describes 570 patients. Image-level fallback groups
  are therefore used in the manifest.

DDI is not concatenated into the development pool.

## Generated manifests

| Manifest | Images | Unique groups |
| --- | ---: | ---: |
| `data/manifests/fitzpatrick17k_c_v1.parquet` | 11,394 | 11,394 |
| `data/manifests/pad_ufes_20_v1.parquet` | 2,298 | 1,373 |
| `data/manifests/scin_v1.parquet` | 10,406 | 5,033 |
| `data/manifests/ddi_v1.parquet` | 656 | 656 fallback groups |

The three contributors are concatenated into
`data/combined/visual_top_k_development_pool_v1.parquet`, containing 24,098
rows. The pool retains unmapped and excluded rows with explicit status fields;
it is not yet a final training dataset.
