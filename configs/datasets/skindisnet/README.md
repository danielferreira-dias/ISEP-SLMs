# SkinDisNet

## Local status

`data/raw/SkinDisNet_2.zip` is the official Mendeley Data version 2 archive:

- DOI: <https://doi.org/10.17632/yj3md44hxg.2>
- Published: 26 June 2025
- Size: 1,382,276,708 bytes
- SHA-256:
  `7367d2d63d3df9dfafd8f09529c078cfb758ad314760865982f632fe356f7428`

The local archive matches the size and SHA-256 published by Mendeley's public
file manifest and passed a complete ZIP integrity test. It is intentionally
stored without extraction.

Version 2 supersedes the initial image-only release by adding
`SkinDisNet_Metadata.csv`. The archive contains:

- 1,710 preprocessed clinical images;
- 11,970 augmented derivatives;
- metadata linking the 1,710 source images to 416 patients.

Only the preprocessed images are indexed by this project's normalized manifest.
The augmented directory is retained as part of the official archive but must not
be used as independent benchmark observations.

Sources:

- Dataset: <https://data.mendeley.com/datasets/yj3md44hxg/2>
- Data article: <https://doi.org/10.1016/j.dib.2025.112239>

## Dataset

SkinDisNet contains clinical photographs collected between April 2023 and June
2024 at Rangpur Medical College and Shaheed Syed Nazrul Islam Medical College
Hospital in Bangladesh. Researchers captured the images with three smartphone
cameras, in natural light, during consultations with dermatology specialists.

The released images are not untouched camera originals. The dataset pipeline
labels, crops, removes backgrounds, and resizes them to 512 by 512 pixels before
creating the `Preprocessed/` directory. This makes SkinDisNet useful for
geographically distinct smartphone-image evaluation, but it does not reproduce
all acquisition noise found in unprocessed patient submissions.

The metadata columns are:

| Column | Meaning |
| --- | --- |
| `Folder_name` | Disease directory in the ZIP |
| `Patient_id` | Patient grouping identifier |
| `Image_id` | Image identifier and filename stem |
| `Age` | Patient age in years, including fractional values for young children |
| `Sex` | Source-provided sex value |
| `Leision_location` | Anatomical location; spelling is from the release |
| `Diagnosis` | Six-class reference diagnosis |

Always split by `Patient_id`. A patient can contribute multiple photographs,
but no patient in the release has images assigned to more than one diagnosis.

## Labels and verified counts

The local metadata and preprocessed image directories match exactly: every
metadata row resolves to one image, with no missing or extra preprocessed
images.

| Diagnosis | Code | Images | Patients |
| --- | --- | ---: | ---: |
| Atopic dermatitis | `AD` | 70 | 15 |
| Contact dermatitis | `CD` | 477 | 110 |
| Eczema | `EC` | 466 | 115 |
| Scabies | `SC` | 343 | 90 |
| Seborrheic dermatitis | `SD` | 79 | 18 |
| Tinea corporis | `TC` | 275 | 68 |
| **Total** |  | **1,710** | **416** |

The diagnoses are clinically reviewed labels associated with dermatologist
consultations; they are not described as pathology-confirmed diagnoses.

## Benchmark use

SkinDisNet is configured as `external_evaluation_only` so its geographically
distinct cases do not influence taxonomy selection.

Four source diagnoses currently map directly to active benchmark IDs:

- contact dermatitis;
- eczema;
- scabies;
- seborrheic dermatitis.

Atopic dermatitis and tinea corporis remain outside the active taxonomy because
their former IDs are retired. Do not silently merge atopic dermatitis into
generic eczema or tinea corporis into another fungal label. Reintroducing those
classes requires an explicit taxonomy review.

Do not mix files from `Augmented/` into training, validation, or test splits.
Each source image has seven augmented derivatives, so image-level random
splitting would create severe leakage and inflated results.

## Licence and privacy

Mendeley Data releases SkinDisNet under CC BY-NC 4.0. Use is therefore limited
to non-commercial purposes and requires attribution.

The data article reports patient consent, anonymization, institutional
permission, and ethical approval. Nevertheless, these remain sensitive medical
images. Keep the archive local, preserve patient grouping, and do not
redistribute it outside the licence terms.
