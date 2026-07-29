# Disease mapping and coverage analysis

> Historical stage note: this document records the initial 13-class v1
> analysis. The current 20-class taxonomy and complete canonical-label
> coverage are documented in
> `05_twenty_class_taxonomy_expansion.md`.

## Objective

This stage inventories all source disease labels, maps clinically compatible
labels to the candidate benchmark taxonomy, and measures preliminary support
using unique case or patient groups rather than raw image counts.

## Mapping process

The candidate taxonomy is defined in
`configs/taxonomies/diseases.yaml`. Additional reviewed source mappings are
stored in `configs/taxonomies/source_disease_mappings.yaml`.

Labels are processed in this order:

1. Normalize Unicode, case, punctuation, separators, and whitespace.
2. Match a dataset-specific reviewed mapping.
3. Match a global reviewed mapping.
4. Match a taxonomy canonical name, display name, or alias.
5. Retain the original label and mark the primary diagnosis as unmapped.

Normalization is lexical, not clinical. It does not automatically collapse
broad and specific diagnoses.

Two conservative decisions are important:

- `Eczema` is not automatically mapped to `atopic_dermatitis`.
- `Tinea` is not automatically mapped to `tinea_corporis`.

Those source labels are broader than the candidate taxonomy classes. Mapping
them without more evidence would create artificial ground truth.

The current mapping groups melanoma in situ under the candidate melanoma class
and squamous cell carcinoma in situ under the candidate squamous cell
carcinoma class. These grouping decisions are explicitly preliminary and
require clinical review before the taxonomy is frozen.

## Source-label inventory

`data/reports/source_disease_inventory_v1.csv` contains one row per
dataset/source-label pair and records:

- the original source label;
- the lexically normalized label;
- the mapped disease ID, if available;
- image and unique-group counts when the label appears anywhere in a
  differential;
- image and unique-group counts when it is the primary diagnosis.

The inventory contains 568 dataset/source-label pairs:

| Dataset | Source labels | Mapped to candidate taxonomy | Unmapped |
| --- | ---: | ---: | ---: |
| DDI | 78 | 9 | 69 |
| Fitzpatrick17k-C | 114 | 12 | 102 |
| PAD-UFES-20 | 6 | 6 | 0 |
| SCIN | 370 | 11 | 359 |

The high number of unmapped labels is expected because the candidate taxonomy
contains only 13 diseases. Unmapped records are retained for audit and future
taxonomy revision.

## Counting policy

Coverage uses only:

- gradable rows;
- normalized primary diagnoses;
- Fitzpatrick17k-C, PAD-UFES-20, and SCIN;
- one count per unique `group_id` for each disease.

SCIN secondary differentials appear in the source inventory but do not
increase primary-disease coverage. DDI does not influence inclusion.

The preliminary support thresholds are:

- at least 100 unique groups across contributors;
- presence in at least two independent contributor datasets.

Train, validation, and test minimums cannot be evaluated until group-safe
splits exist.

## Preliminary result

Eleven candidate diseases pass total-support and source-diversity thresholds:

| ID | Disease | Unique groups | Images | Contributing datasets |
| --- | --- | ---: | ---: | ---: |
| D001 | Melanoma | 303 | 321 | 3 |
| D002 | Melanocytic nevus | 190 | 257 | 2 |
| D003 | Psoriasis | 643 | 768 | 2 |
| D004 | Basal cell carcinoma | 931 | 1,270 | 3 |
| D005 | Squamous cell carcinoma | 513 | 580 | 3 |
| D006 | Seborrheic keratosis | 229 | 288 | 2 |
| D007 | Actinic keratosis | 679 | 893 | 3 |
| D009 | Contact dermatitis | 623 | 976 | 2 |
| D011 | Acne vulgaris | 481 | 543 | 2 |
| D012 | Rosacea | 108 | 136 | 2 |
| D013 | Vitiligo | 115 | 117 | 2 |

`atopic_dermatitis` and `tinea_corporis` currently have zero safely mapped
primary groups and are classified as long-tail candidates. This does not mean
that no eczema or fungal-condition images exist; it means that the available
labels are not specific enough to support these two exact classes under the
current conservative mapping.

The generated outputs are:

- `data/reports/source_disease_inventory_v1.csv`
- `data/reports/disease_coverage_v1.csv`
- `data/reports/disease_coverage_v1.parquet`
- `data/reports/included_diseases_v1.yaml`
- `data/reports/long_tail_diseases_v1.csv`
- `data/reports/unmapped_disease_labels_v1.csv`

The included-disease file is marked
`provisional_pending_split_validation`; it is not a frozen benchmark taxonomy.
