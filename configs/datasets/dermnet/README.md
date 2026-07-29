# Dermnet (Kaggle mirror)

## Local status

The original Kaggle archive is stored without extraction at
`data/raw/dermnet.zip`. It contains 19,559 JPEG images:

| Supplied split | Images |
| --- | ---: |
| `train/` | 15,557 |
| `test/` | 4,002 |
| **Total** | **19,559** |

The ZIP passed an archive-integrity check. Its size is 1,847,748,564 bytes and
its SHA-256 is
`7ac4d9233a2799c40fc5bdb938be3a85fc50a683ccd0c8851d02d43a705be1fe`.

Source: <https://www.kaggle.com/datasets/shubhamgoel27/dermnet>.

## Dataset

This is a third-party Kaggle mirror of clinical photographs collected from the
skin-disease atlas formerly available at `dermnet.com`. It is not DermNet NZ's
separate, paid dataset designed and licensed for AI.

Images are RGB JPEG files of varying dimensions. There is no accompanying CSV
or patient-level metadata. The directory structure supplies both the split and
the target:

```text
train/<category>/<image>.jpg
test/<category>/<image>.jpg
```

The folders are broad educational categories rather than consistently
fine-grained diagnoses. Some folders combine several diseases, and the
`Melanoma Skin Cancer Nevi and Moles` folder combines malignant and benign
entities. The labels should therefore not be interpreted as
pathology-confirmed diagnoses.

## Labels

There are 23 directory labels:

| # | Folder label | Train | Test | Total |
| ---: | --- | ---: | ---: | ---: |
| 1 | Acne and Rosacea Photos | 840 | 312 | 1,152 |
| 2 | Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions | 1,149 | 288 | 1,437 |
| 3 | Atopic Dermatitis Photos | 489 | 123 | 612 |
| 4 | Bullous Disease Photos | 448 | 113 | 561 |
| 5 | Cellulitis Impetigo and other Bacterial Infections | 288 | 73 | 361 |
| 6 | Eczema Photos | 1,235 | 309 | 1,544 |
| 7 | Exanthems and Drug Eruptions | 404 | 101 | 505 |
| 8 | Hair Loss Photos Alopecia and other Hair Diseases | 239 | 60 | 299 |
| 9 | Herpes HPV and other STDs Photos | 405 | 102 | 507 |
| 10 | Light Diseases and Disorders of Pigmentation | 568 | 143 | 711 |
| 11 | Lupus and other Connective Tissue diseases | 420 | 105 | 525 |
| 12 | Melanoma Skin Cancer Nevi and Moles | 463 | 116 | 579 |
| 13 | Nail Fungus and other Nail Disease | 1,040 | 261 | 1,301 |
| 14 | Poison Ivy Photos and other Contact Dermatitis | 260 | 65 | 325 |
| 15 | Psoriasis pictures Lichen Planus and related diseases | 1,405 | 352 | 1,757 |
| 16 | Scabies Lyme Disease and other Infestations and Bites | 431 | 108 | 539 |
| 17 | Seborrheic Keratoses and other Benign Tumors | 1,371 | 343 | 1,714 |
| 18 | Systemic Disease | 606 | 152 | 758 |
| 19 | Tinea Ringworm Candidiasis and other Fungal Infections | 1,300 | 325 | 1,625 |
| 20 | Urticaria Hives | 212 | 53 | 265 |
| 21 | Vascular Tumors | 482 | 121 | 603 |
| 22 | Vasculitis Photos | 416 | 105 | 521 |
| 23 | Warts Molluscum and other Viral Infections | 1,086 | 272 | 1,358 |

## Missing metadata and limitations

The archive does not provide:

- patient, case, lesion, or encounter identifiers;
- age, sex, anatomical site, country, or skin-tone annotations;
- a fine-grained diagnosis table;
- diagnostic confirmation or annotator information;
- an explanation of how the supplied train/test split was constructed.

Consequently, the image path is only a fallback `group_id`; it cannot prevent
different photographs of the same patient from crossing splits. Exact and
perceptual duplicate analysis is required before using the supplied split.

This is an image-classification dataset. A text-only language model cannot
learn from its pixels without a vision encoder or an independently validated
image-to-text transformation.

## Licence and provenance warning

At download time, Kaggle reported the dataset licence as CC BY-NC-ND 4.0.
That is a non-commercial, no-derivatives licence. The Kaggle page says that
the images were taken from `dermnet.com`, but it does not document the
copyright and patient-consent chain for each image or provide an upstream
licence expressly authorising AI training.

Keep this archive local and do not redistribute it, derived images, or trained
weights without confirming the applicable rights in writing. This project
records the dataset as `excluded_from_disease_coverage` until provenance,
licensing, grouping, and label quality are resolved.
