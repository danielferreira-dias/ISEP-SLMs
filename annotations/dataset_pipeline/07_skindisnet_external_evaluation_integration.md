# SkinDisNet external evaluation integration

> Historical stage note: this document records the state before global
> perceptual grouping. Stage 8 completes hashing and reduces SkinDisNet from
> 416 source patient groups to 414 leakage-safe groups.

## Objective

This stage integrates SkinDisNet into the normalized data inventory as a
second external-evaluation dataset. The dataset is not added to the taxonomy
contributors and therefore cannot change disease eligibility or the
twenty-one-class closed-set taxonomy.

The integration has four goals:

1. preserve the six source diagnoses and patient groupings;
2. identify the subset compatible with the current candidate taxonomy;
3. make age and sex metadata available for evaluation-only subgroup analysis;
4. exclude augmented derivatives from all case and image counts.

No final external-test split is created at this stage. That remains blocked on
cross-dataset exact and perceptual duplicate analysis.

## Source inspection

The local input is the official Mendeley Data version 2 archive:

- DOI: `10.17632/yj3md44hxg.2`;
- release date: 26 June 2025;
- licence: CC BY-NC 4.0;
- 1,710 preprocessed images;
- 11,970 augmented derivatives;
- 416 patients;
- one metadata row for every preprocessed image.

Source references:

- Mendeley Data: <https://data.mendeley.com/datasets/yj3md44hxg/2>
- Data article: <https://doi.org/10.1016/j.dib.2025.112239>

The metadata contains `Folder_name`, `Patient_id`, `Image_id`, `Age`, `Sex`,
`Leision_location`, and `Diagnosis`. Every metadata row resolves to exactly one
file under `Preprocessed/`, and no additional preprocessed image is left
without metadata.

The augmented files are excluded completely. They are transformations of the
source observations and cannot be treated as independent images or patient
cases.

## Normalized manifest

The source adapter creates:

`data/manifests/skindisnet_v3.parquet`

Each row represents one preprocessed image. `Patient_id` becomes the
leakage-safe `group_id`, ensuring that all images from a patient remain
together in a future external-test manifest.

The normalized fields include:

- source image and patient identifiers;
- source diagnosis and normalized disease ID when compatible;
- exact integer age when available;
- a standardized age band for every patient;
- source sex with `source_gender` as the measurement system;
- anatomical site and original fractional age in `source_metadata`;
- a record that cropping, background removal, and resizing were applied.

The source contains fractional ages for young children. Because `age_years` is
an integer field, fractional values are preserved in source metadata and used
to derive the correct `under_18` band rather than being presented as an exact
integer age.

SkinDisNet does not provide race, ethnicity, Fitzpatrick type, or Monk Skin
Tone annotations. None are inferred.

## Taxonomy compatibility

Four diagnoses map directly to active benchmark classes:

| Source diagnosis | Disease ID | Patients | Images |
| --- | --- | ---: | ---: |
| Contact Dermatitis | D009 | 110 | 477 |
| Eczema | D014 | 115 | 466 |
| Seborrheic Dermatitis | D015 | 18 | 79 |
| Scabies | D018 | 90 | 343 |
| **Closed-set total** |  | **333** | **1,365** |

Two diagnoses remain outside the active taxonomy:

| Source diagnosis | Patients | Images | Reason |
| --- | ---: | ---: | --- |
| Atopic Dermatitis | 15 | 70 | The retired candidate is not silently merged into broad eczema. |
| Tinea Corporis | 68 | 275 | The retired candidate is not reintroduced by an external dataset. |

These 345 out-of-taxonomy images remain auditable in the manifest. They are
not scored in the primary closed-set external benchmark. A later exploratory
open-set analysis may use them without changing the main taxonomy.

## Why the dataset remains external

SkinDisNet provides a geographically and technically distinct smartphone
domain. Its images were collected in Bangladesh and processed differently from
the three taxonomy-contributor datasets. Keeping all SkinDisNet patients out
of training preserves an independent test of dataset shift.

The role is enforced in both:

- `configs/datasets/catalog.yaml`;
- `configs/datasets/disease_inclusion.yaml`.

An automated contract test verifies that SkinDisNet is present under
`external_evaluation_only` and absent from `taxonomy_contributors`.

The development pool therefore remains limited to Fitzpatrick17k-C,
PAD-UFES-20, and SCIN. SkinDisNet is included in source accounting and
demographic support reports but cannot affect `disease_coverage_v3.csv` or
`included_diseases_v3.yaml`.

## Evaluation limitations

SkinDisNet cannot serve as a complete external evaluation of all twenty-one
classes because only four active classes are represented.

Seborrheic dermatitis has only 18 patients, below the current overall subgroup
reporting threshold of 30 unique groups. Its class metric may be shown as
exploratory with an explicit sample count and confidence interval, but it
should not be treated as a stable primary estimate.

The preprocessed images have been cropped, had their backgrounds removed, and
been resized to 512 by 512 pixels. Results therefore measure robustness to this
specific processed smartphone domain rather than untouched clinical
photography.

An exact-hash inspection found no duplicate byte sequences among the 1,710
preprocessed images. Cross-dataset and perceptual duplicates remain unchecked
and are the next pipeline gate.

## Generated report results

After rebuilding the version 3 reports:

- SkinDisNet contributes six dataset/source-label inventory rows;
- all 1,710 images and 416 patient groups appear in source accounting;
- 1,365 images and 333 patient groups appear as benchmark-mapped rows;
- age-band and sex availability are reported for all 416 patient groups;
- exact integer age is available for 391 patient groups;
- no skin-tone or race/ethnicity coverage is reported;
- contributor-only disease counts and the 21-class included list remain
  unchanged.

The complete report inventory now contains 597 dataset/source-label pairs and
519 consolidated canonical source labels. The contributor development pool
remains unchanged at 24,098 rows. `drug_eruption` also remains unchanged at
215 contributor groups and 286 images, confirming that the external dataset
did not alter taxonomy coverage.

All six normalized manifests pass validation and contain 46,023 globally
unique sample IDs. Sixteen unit tests pass.

## Next decision gate

The next stage is cross-dataset deduplication:

1. calculate image SHA-256 values from decoded source bytes;
2. calculate perceptual hashes;
3. identify exact, near-duplicate, and source-lineage groups;
4. prevent duplicate groups from crossing any future split;
5. rebuild support reports before generating internal and external test
   manifests.
