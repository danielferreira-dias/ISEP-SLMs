# Dermatology multimodal training corpus v1

This directory is the training layer. It combines the frozen 6,417-image
internal training split with eligible clinical photographs from Derm1M and
HIBA. It never rewrites validation, internal test, internal benchmark, or
external evaluation data.

The Derm1M selection uses a conservative metadata filter followed by a manual
source-stratified image audit. Derm1M `validation_data` and
`public_medical_dataset` are not selected automatically because the audit
found ambiguous non-clinical modalities.

## Artifacts

| File | Meaning |
| --- | --- |
| `train_images.parquet` | Final image-level training manifest after leakage checks |
| `teacher_annotation_queue.parquet` | Same eligible images, ready for teacher-generated structured targets |
| `all_candidates.parquet` | Included and excluded candidates with audit reasons |
| `release/training_release_v1.json` | Counts, protected sets, and release rules |
| `reports/source_summary.csv` | Counts by source, role, and inclusion status |
| `reports/class_distribution.csv` | Directly mapped in-domain label counts |
| `reports/excluded_candidates.csv` | Decode failures and duplicate exclusions |
| `reports/derm1m_filter_summary.csv` | Derm1M modality-filter counts |
| `reports/derm1m_filter_decisions.parquet` | Per-row Derm1M filter decision |

The queue is not yet a completed SFT dataset. It becomes one only after a
teacher model produces and validates the target JSON/text for the chosen
tasks.

`training_role` distinguishes what each image can safely teach:

- `in_domain_diagnosis`: directly mapped to one of the 21 thesis diseases;
- `out_of_domain`: diagnosed dermatology image outside that closed taxonomy;
- `description_only`: clinical image/caption without a definitive diagnosis.

## Build

```bash
uv run python -m src.data_pipeline.training_corpus
```

For a quick pipeline smoke test:

```bash
uv run python -m src.data_pipeline.training_corpus --limit-derm1m 10
```

Validate the materialized release without rebuilding or decoding the images:

```bash
uv run python -m src.data_pipeline.training_corpus --validate-only
```

## Private Hugging Face export

The 21-class diagnosis subset can be exported as `ISEPDermData`. The canonical
Hugging Face release excludes Derm1M after a pilot label-quality audit found
source-derived entity-linking errors and images whose diagnosis depended on
article context rather than visible evidence. Derm1M remains in the local
training layer for provenance and further research, but is not exported. The
export embeds the remaining original image bytes in sharded Parquet files and
removes captions, raw source metadata, local image URIs, and demographic
fields:

```bash
uv run python -m src.data_pipeline.huggingface_dataset_export
uv run python -m src.data_pipeline.huggingface_dataset_export --validate-only
```

The generated release is stored in `data/training/ISEPDermData/`. Its first
three columns are `image`, `source`, and `label`; stable disease IDs,
`leakage_group_id`, source labels, checksums, and licence IDs are retained for
auditability. The release contains only `in_domain_diagnosis` rows from sources
other than Derm1M and does not include `description_only` or `out_of_domain`
records.

## Leakage policy

Every new image receives an encoded-byte SHA-256 and a 64-bit perceptual hash.
Exact and near-duplicate matches against validation, internal test, DDI, or
SkinDisNet are excluded. Exact duplicates already present in training are also
excluded. Near duplicates wholly inside training are retained but marked for
review.
