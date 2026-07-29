# Fitzpatrick17k

## Local status

The local copy contains:

- `fitzpatrick17k.csv`: original 16,577-row release.
- `Fitzpatrick17k-C.csv`: corrected 11,394-row release.
- `Fitzpatrick17k_DiagnosisMapping.xlsx`: diagnosis mapping released with the
  corrected dataset.
- `data/raw/fitzpatrick17k.zip`: the complete original archive of 16,577 JPEG
  images, stored without extraction to avoid duplicating approximately 1.36 GB.

The ZIP passed an archive-integrity check and has SHA-256
`9e3ea84ee8b349dad514b0acf1f553cf39726bccfbe888d4faec4b3045d13d88`.
Its members use the path `data/finalfitz17k/<md5hash>.jpg`. All 11,394 rows in
Fitzpatrick17k-C match an image in the archive both by `orig_img_name` and by
`md5hash + ".jpg"`; no corrected-set image is missing.

The archive was obtained through the authors' access process. Keep it local
and do not redistribute it independently of the upstream licence and terms.

Sources:

- Original dataset: <https://github.com/mattgroh/fitzpatrick17k>
- Corrected dataset: <https://github.com/sfu-mial/Corrected-Skin-Image-Datasets>
- Corrected release: <https://doi.org/10.5281/zenodo.11101337>

## Dataset

Fitzpatrick17k contains 16,577 clinical photographs collected from the
DermaAmin and Atlas Dermatologico atlases. It covers 114 skin conditions and
contains two skin-tone annotations:

- `fitzpatrick_scale`: expert/crowd Fitzpatrick type (`1` through `6`);
  `-1` means unavailable.
- `fitzpatrick_centaur`: the Centaur Labs Fitzpatrick estimate.

The corrected Fitzpatrick17k-C release removes duplicates, erroneous images,
and visual outliers, retaining 11,394 records and adding a standardized
`partition` field. Disease labels are still inherited from web atlases and are
not uniformly pathology-confirmed.

## Labels

The main target is `label`, a fine-grained disease name. Examples include
`psoriasis`, `basal cell carcinoma`, `squamous cell carcinoma`, `melanoma`,
`eczema`, `acne vulgaris`, `lichen planus`, `scabies`, and `vitiligo`.

Two coarser targets are also supplied:

- `three_partition_label`: `benign`, `malignant`, or `non-neoplastic`.
- `nine_partition_label`: `inflammatory`, `genodermatoses`,
  `benign epidermal`, `benign dermal`, `benign melanocyte`,
  `malignant epidermal`, `malignant dermal`, `malignant melanoma`, or
  `malignant cutaneous lymphoma`.

Important original columns:

| Column | Meaning |
| --- | --- |
| `md5hash` | Image identifier/checksum |
| `label` | One of 114 disease labels |
| `nine_partition_label` | Nine-class diagnostic grouping |
| `three_partition_label` | Benign/malignant/non-neoplastic grouping |
| `fitzpatrick_scale` | Fitzpatrick type 1-6 or -1 |
| `qc` | Original quality-control field |
| `url` | Upstream atlas image URL |

Fitzpatrick17k-C adds `filename`, `diag`, `fst`, `partition`,
`orig_img_name`, and `new_img_name`.

## Licence and use

The original project identifies the images as
CC BY-NC-SA 3.0. The corrected repository's Apache-2.0 licence applies to its
code; it does not replace the upstream image licence.

Use lesion/source-aware deduplication before training. For this project,
Fitzpatrick17k-C is preferred over the original metadata, but it should still
be treated as noisy diagnostic supervision.
