# PAD-UFES-20

## Local status

`data/raw/` contains the official Mendeley version 1 release:

- `metadata.csv`
- `imgs_part_1.zip`
- `imgs_part_2.zip`
- `imgs_part_3.zip`

The three archives contain the smartphone images and are kept compressed to
avoid duplicating approximately 3.59 GB of data. Source:
<https://doi.org/10.17632/zr7vgbcyr2.1>.

## Dataset

PAD-UFES-20 contains 2,298 smartphone images representing 1,641 lesions from
1,373 patients collected by the Dermatological and Surgical Assistance
Program at the Federal University of Espírito Santo, Brazil.

`metadata.csv` provides patient, lesion, image, demographic, exposure, and
clinical attributes. Important fields include:

- `patient_id`, `lesion_id`, `img_id`
- `age`, `gender`, `fitspatrick`
- `region`, `diameter_1`, `diameter_2`, `elevation`
- `itch`, `grew`, `hurt`, `changed`, `bleed`
- `smoke`, `drink`, `pesticide`
- `skin_cancer_history`, `cancer_history`
- `biopsed`

Always split by `patient_id` or at least `lesion_id`, never independently by
`img_id`.

## Labels

The target column is `diagnostic`:

| Code | Diagnosis | Local rows |
| --- | --- | ---: |
| `ACK` | Actinic keratosis | 730 |
| `BCC` | Basal cell carcinoma | 845 |
| `MEL` | Melanoma | 52 |
| `NEV` | Nevus | 244 |
| `SCC` | Squamous cell carcinoma, including Bowen disease/SCC in situ | 192 |
| `SEK` | Seborrheic keratosis | 235 |

All BCC, melanoma, and SCC cases are biopsy-proven. Some benign cases use
clinical consensus; `biopsed` records this distinction. The severe class
imbalance, especially the 52 melanoma images, must be considered in sampling
and evaluation.

## Integrity and licence

Official SHA-256 values:

```text
metadata.csv    14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527
imgs_part_1.zip 0ab44f60938bf57445e12f518a8878954cc734e6b0aec6d01194e2d26b4b2dca
imgs_part_2.zip e2d9a3cbd58e823f5ae33163c48643e7d1b54ae3f9e145f01f8e9f16a363a60b
imgs_part_3.zip ecc4ef10143a43e1d01cb736773148607a78b530417eb76f2c38ad24bf5d0d2c
```

PAD-UFES-20 version 1 is CC BY 4.0.

