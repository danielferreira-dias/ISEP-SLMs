# DDI — Diverse Dermatology Images

## Local status

The local copy contains:

- `data/ddi_metadata.csv`, the verified Redivis metadata export;
- `data/images/000001.png` through `data/images/000656.png`, the complete
  official image release.

Validation results:

- 656 PNG files and 238,522,560 image bytes, exactly matching the official
  Redivis file manifest;
- 656 rows with no missing values or duplicate image identifiers;
- 656 unique filenames, covering `000001.png` through `000656.png`;
- every metadata filename resolves to exactly one local image;
- no missing, extra, zero-byte, size-mismatched, or invalid-signature PNGs;
- 78 fine-grained disease labels;
- 171 malignant and 485 non-malignant records.

DDI requires individual Stanford AIMI registration and acceptance of its
Research Use Agreement:

<https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images>

This local copy was downloaded through the researcher's authenticated Redivis
data access. Keep the files unmodified and do not redistribute them.

## Dataset

DDI contains 656 clinical photographs from 570 unique patients across 78
fine-grained diagnoses. Diagnoses are biopsy/pathology-confirmed and reviewed
by a board-certified dermatologist and dermatopathologist.

The dataset was designed for evaluation across skin tones:

| Fitzpatrick group | Images |
| --- | ---: |
| I-II | 208 |
| III-IV | 241 |
| V-VI | 207 |

Canonical documentation: <https://ddi-dataset.github.io/>.

## Labels

The released metadata associates an image with:

- `DDI_ID`, a unique numeric image identifier;
- `DDI_file`, the corresponding PNG filename;
- a fine-grained `disease` diagnosis;
- a binary `malignant`/benign target;
- a grouped `skin_tone` value: `12`, `34`, or `56`.

The `_unnamed_var` column in the Redivis export is a redundant zero-based row
index and may be dropped during preprocessing.

Disease examples include melanoma, melanoma in situ, nodular melanoma,
acral-lentiginous melanoma, basal cell carcinoma, and benign mimics.

DDI is most valuable as a pathology-grounded, skin-tone-stratified external
evaluation set. Its small size makes it unsuitable as the sole training
corpus.

## Usage restriction

The published agreement limits use to approved non-commercial research and
restricts redistribution, modification, derivative works, and clinical use.
Obtain written clarification before using DDI to create or release trained
model weights. Do not copy the DDI subset from another derived dataset as a
way around the agreement.
