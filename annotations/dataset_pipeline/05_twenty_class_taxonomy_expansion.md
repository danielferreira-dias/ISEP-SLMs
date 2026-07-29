# Twenty-class taxonomy expansion

## Objective

The benchmark taxonomy was expanded from 13 to 20 active classes. The
expansion had two goals:

1. include visually difficult inflammatory and infectious differentials such
   as eczema and seborrheic dermatitis;
2. ensure that every source diagnosis remains countable even when it is not an
   active benchmark class.

## Two-level mapping policy

The implementation separates source-label accounting from benchmark-label
selection.

Every non-empty source diagnosis receives:

- `normalized_source_label`: lexical normalization of case and punctuation;
- `canonical_source_label`: a stable countable label, using a reviewed
  benchmark canonical name when available and otherwise the normalized source
  concept;
- `mapping_status`: either `benchmark_mapped` or
  `out_of_benchmark_scope`.

Only clinically compatible labels receive a `benchmark_disease_id`. This
prevents hundreds of distinct diseases from being forced into 20 unrelated
classes while still providing total counts for every observed label.

The normalized manifest schema was updated to version 1.1.0. Both the primary
row and every item in `reference_diagnoses` now store
`canonical_source_label` and `mapping_status`, making the complete mapping
auditable at image level as well as in aggregate reports.

The full count table is
`data/reports/all_source_disease_coverage_v2.csv`. It contains per-dataset and
all-source counts for primary and any-reference occurrences.

## Clinical-boundary decisions

The broad `eczema` class was introduced because the largest relevant source
label is simply `Eczema`; it cannot be interpreted specifically as atopic
dermatitis. Explicit contact dermatitis remains a separate class.

Seborrheic dermatitis is also retained separately. It is a specific chronic
or relapsing form of dermatitis with characteristic anatomical distribution
and is an important visual differential for eczema, psoriasis, rosacea, and
tinea.

Generic `Tinea` remains outside the active taxonomy. Tinea names are
site-specific, and tinea corporis excludes several sites represented by other
tinea terms. Generic tinea and tinea versicolor therefore cannot be treated as
tinea corporis.

Clinical terminology references:

- DermNet, Dermatitis: <https://dermnetnz.org/topics/dermatitis>
- DermNet, Seborrheic dermatitis:
  <https://dermnetnz.org/topics/seborrhoeic-dermatitis>
- DermNet, Tinea: <https://dermnetnz.org/topics/tinea>
- DermNet, Tinea corporis:
  <https://dermnetnz.org/topics/tinea-corporis>

These references support terminology boundaries; they do not replace
dermatologist review of dataset-specific mappings.

## Identifier stability

Disease IDs are never reassigned to a different clinical concept. Unsupported
or rejected candidates remain under `retired_diseases`:

| ID | Retired candidate | Reason |
| --- | --- | --- |
| D008 | Atopic dermatitis | Available labels support broad eczema, not enough explicitly atopic cases |
| D010 | Tinea corporis | Available tinea labels are broad or refer to other anatomical sites |
| D020 | Pyogenic granuloma | Only 97 contributor groups; DDI cannot satisfy selection thresholds |
| D021 | Dermatomyositis | Contributor support is concentrated in one dataset |

The active taxonomy consequently contains 20 labels with non-contiguous IDs.

## Active taxonomy and preliminary coverage

Coverage is computed only from Fitzpatrick17k-C, PAD-UFES-20, and SCIN. DDI
does not influence class selection.

| ID | Disease | Unique contributor groups | Images | Contributor datasets |
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
| D014 | Eczema | 664 | 1,255 | 2 |
| D015 | Seborrheic dermatitis | 127 | 144 | 2 |
| D016 | Folliculitis | 324 | 488 | 2 |
| D017 | Urticaria | 348 | 576 | 2 |
| D018 | Scabies | 224 | 247 | 2 |
| D019 | Pityriasis rosea | 167 | 216 | 2 |
| D022 | Prurigo nodularis | 103 | 135 | 2 |
| D023 | Granuloma annulare | 157 | 186 | 2 |
| D024 | Erythema multiforme | 161 | 170 | 2 |

All 20 pass the preliminary requirement of at least 100 unique contributor
groups and representation in at least two contributor datasets.

## Complete label accounting

The v2 inventory contains:

- 568 dataset/source-label pairs;
- 495 consolidated canonical source labels;
- 20 benchmark-mapped canonical labels;
- 475 canonical labels outside benchmark scope.

An out-of-scope label is not missing or discarded. Its original label,
canonical label, primary counts, differential counts, source dataset, and
group counts remain in the reports.

Relevant outputs:

- `data/reports/source_disease_inventory_v2.csv`
- `data/reports/all_source_disease_coverage_v2.csv`
- `data/reports/disease_coverage_v2.csv`
- `data/reports/included_diseases_v2.yaml`
- `data/reports/out_of_scope_source_labels_v2.csv`

## Remaining qualification

The 20-class result is still marked `pending_split_validation`. Exact and
perceptual duplicate grouping may reduce support, and final train,
validation, and test minimums have not yet been evaluated.
