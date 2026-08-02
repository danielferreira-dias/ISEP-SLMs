---
pretty_name: ISEPDermData
language:
- en
license: other
task_categories:
- image-classification
tags:
- dermatology
- medical
- image
- multimodal
- private-research-dataset
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# ISEPDermData

ISEPDermData is a private, research-only collection of 7,541
clinical dermatology images mapped to the 21-class ISEP thesis taxonomy.
The release is intended as the source pool for teacher annotation and later
multimodal student fine-tuning.

## Dataset summary

| Statistic | Value |
| --- | ---: |
| Images | 7,541 |
| Leakage-safe groups | 5,671 |
| Active disease classes | 21 |
| Split | `train` (unsplit source pool) |

This release is deliberately an unsplit pool. A later version will
materialize `sft_train` and `sft_dev` by `leakage_group_id`; related images
must never cross those splits.

## Schema

The first three columns are the primary training view:

```text
image | source | label
```

Audit columns preserve stable IDs, original source labels, grouping,
diagnostic provenance, checksums, and per-row source licences:

```text
image
source
label
disease_id
sample_id
source_image_id
source_label
leakage_group_id
diagnosis_basis
image_sha256
license_id
```

Free-text captions, raw source metadata, local paths, and demographic fields
are intentionally excluded from this release.

## Source distribution

| Source | Images |
| --- | ---: |
| Fitzpatrick17k-C | 3,226 |
| HIBA | 318 |
| PAD-UFES-20 | 1,629 |
| SCIN | 2,368 |

## Class distribution

| Disease ID | Label | Images | Groups |
| --- | --- | ---: | ---: |
| D001 | melanoma | 298 | 281 |
| D002 | melanocytic_nevus | 215 | 169 |
| D003 | psoriasis | 697 | 611 |
| D004 | basal_cell_carcinoma | 1,006 | 753 |
| D005 | squamous_cell_carcinoma | 451 | 405 |
| D006 | seborrheic_keratosis | 222 | 179 |
| D007 | actinic_keratosis | 693 | 541 |
| D009 | contact_dermatitis | 682 | 458 |
| D011 | acne_vulgaris | 403 | 361 |
| D012 | rosacea | 95 | 75 |
| D013 | vitiligo | 93 | 91 |
| D014 | eczema | 892 | 487 |
| D015 | seborrheic_dermatitis | 105 | 90 |
| D016 | folliculitis | 351 | 227 |
| D017 | urticaria | 403 | 245 |
| D018 | scabies | 193 | 173 |
| D019 | pityriasis_rosea | 166 | 131 |
| D022 | prurigo_nodularis | 106 | 82 |
| D023 | granuloma_annulare | 150 | 125 |
| D024 | erythema_multiforme | 120 | 113 |
| D025 | drug_eruption | 200 | 154 |

## Provenance and licences

This is a mixed-source research dataset. Every row retains `source` and
`license_id`; no single licence should be interpreted as replacing the terms
of the original source. The current sources include Fitzpatrick17k-C,
PAD-UFES-20, SCIN, and HIBA. Derm1M was removed in release 1.1.0 after a
label-quality audit identified source-derived entity-linking errors and
context-dependent images unsuitable as direct image-classification targets.
Release 1.2.0 promotes 123 images from 63 previously unrepresented internal
reserve groups into Train. Release 1.3.0 additionally promotes 683 images from
417 whole groups released when visual Top-K Validation was reduced to 1,000
tasks. Images from groups still represented by Validation or the sealed
Internal Benchmark remain excluded from training.
See `metadata/source_licenses.json` and the upstream dataset documentation
before any redistribution, commercial use, or publication of derived
artifacts.

## Intended use

- dermatology image classification within the fixed 21-class taxonomy;
- teacher-generated visual findings, differential diagnoses, and short
  evidence-grounded rationales;
- research on small multimodal language models;
- group-safe supervised fine-tuning after a separate split is released.

## Limitations

- source labels have heterogeneous diagnostic certainty;
- class and source distributions are imbalanced;
- some groups contain multiple related images;
- the dataset is not a medical device and must not be used as a substitute
  for clinical diagnosis;
- this release does not include out-of-domain or description-only records.

## Reproducibility

The release is generated from the local thesis repository with:

```bash
python -m src.data_pipeline.huggingface_dataset_export
```

Checksums and exact counts are recorded in `release.json`.
