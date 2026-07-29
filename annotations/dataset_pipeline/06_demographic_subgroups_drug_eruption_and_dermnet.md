# Demographic subgroups, drug eruption, and Dermnet audit integration

> Historical stage note: this document records the five-manifest version 3
> state. SkinDisNet is added as a sixth, external-only manifest in stage 7.

## Objective

This stage extended the normalized data layer in three related ways:

1. preserve available age, sex or gender, race or ethnicity, and skin-tone
   metadata for later fairness and robustness analysis;
2. add `drug_eruption` as the twenty-first active benchmark class;
3. index the newly added Dermnet Kaggle mirror without allowing it to affect
   taxonomy selection, training, or benchmark metrics.

The changes produce an auditable inventory and eligibility analysis. They do
not create final train, validation, internal-test, or external-test splits.

## Manifest schema revision

The shared manifest schema was updated to version 1.2.0. It now includes:

- exact age when explicitly supplied by the source;
- the original age band and a standardized project age band;
- normalized sex or gender plus a field identifying whether the source
  represents sex at birth or an unspecified source gender;
- source-provided combined race or ethnicity and its provenance;
- the existing skin-tone value, measurement system, and provenance.

No missing demographic value is inferred. An exact age is never reconstructed
from a source age band. Metadata is configured as evaluation-only and must not
be shown to a model in the visual disease-ranking prompt.

The standardized age bands are:

- `under_18`;
- `18_to_29`;
- `30_to_39`;
- `40_to_49`;
- `50_to_59`;
- `60_to_69`;
- `70_and_over`;
- `unknown`.

`unknown`, `other_or_unspecified`, and `prefer_not_to_answer` remain preserved
where supplied, but they are not treated as metric-eligible subgroups.

## Dataset-specific demographic mappings

### PAD-UFES-20

PAD-UFES-20 supplies exact age, a source gender field, and Fitzpatrick values.
The patient identifier remains the grouping unit, so availability and subgroup
support count a patient once even when several images exist.

Among the 1,373 benchmark-mapped patient groups:

- exact and standardized age are available for all 1,373 groups;
- source gender is available for 794 groups;
- Fitzpatrick values are available for 794 groups.

### SCIN

SCIN supplies age bands, sex at birth, combined self-reported race or
ethnicity, and several skin-tone annotations. The manifest uses the preferred
Monk Skin Tone annotation already selected by the adapter while retaining
other source annotations in metadata.

The boolean helper indicating that a participant selected more than one race
is not exposed as a race category. The source `combined_race` value
`TWO_OR_MORE_AFTER_MITIGATION` is normalized to `two_or_more_races`.

Among the 1,579 benchmark-mapped SCIN case groups:

- an informative standardized age band is available for 730 groups;
- informative sex-at-birth values are available for 833 groups;
- informative combined race or ethnicity is available for 895 groups;
- a Monk Skin Tone value is available for 1,578 groups.

### Fitzpatrick17k-C and DDI

Fitzpatrick17k-C supplies Fitzpatrick values for 4,047 of its 4,158
benchmark-mapped image groups. The available local metadata does not supply
age, sex or gender, or race or ethnicity.

DDI is held out from taxonomy selection and supplies grouped Fitzpatrick values
for all 300 benchmark-mapped external-evaluation images. Its grouped values are
not merged with six-level Fitzpatrick values.

## Subgroup reporting policy

Two version 3 reports were introduced:

- `data/reports/demographic_availability_v3.csv` reports group-level metadata
  coverage for all source rows and benchmark-mapped rows;
- `data/reports/subgroup_coverage_v3.csv` reports unique groups and images for
  each usable subgroup, both overall and per disease.

A subgroup requires at least 30 unique groups for an overall metric. A
disease-specific subgroup requires at least 10 unique groups. These thresholds
control whether a metric is reported; they do not change training membership
or disease inclusion.

Fitzpatrick, grouped Fitzpatrick, and Monk Skin Tone are kept separate. Race or
ethnicity is also not treated as a substitute for measured skin tone. Metrics
should include confidence intervals, missingness, and counts, and should be
reported by source dataset because metadata collection procedures differ.

The current contributor pool has enough overall support for all six adult age
bands, but only 23 unique groups under age 18, below the threshold of 30.
Several low-frequency race or ethnicity categories and the darkest Monk
categories are also below threshold. Per-disease support is more restrictive
and is recorded explicitly in the subgroup report.

## Drug eruption class

Taxonomy version 2.2.0 adds:

- ID: `D025`;
- canonical name: `drug_eruption`;
- display name: `Drug eruption`;
- reviewed aliases: `drug rash`, `cutaneous drug eruption`, and
  `medication eruption`.

This class represents a cutaneous eruption attributed to a medication. It is
not a generic label for every allergic reaction. Urticaria and contact
dermatitis remain separate active classes, and vague unsupported labels must
not be forced into `D025`.

Preliminary primary-reference coverage is:

| Dataset | Unique groups | Images |
| --- | ---: | ---: |
| Fitzpatrick17k-C | 157 | 157 |
| SCIN | 58 | 129 |
| All contributors | 215 | 286 |

The class passes the preliminary thresholds of 100 unique contributor groups
and two independent contributor datasets. Like every active class, its status
remains `pending_split_validation`.

## Dermnet Kaggle integration

The local Dermnet Kaggle archive contains 19,559 JPEG images arranged into 23
broad directory categories. A fifth normalized manifest,
`data/manifests/dermnet_kaggle_v3.parquet`, indexes every archive member and
retains the directory category and supplied train or test folder.

All 19,559 rows are assigned:

- `include: false`;
- `exclusion_reason: dataset_excluded_from_benchmark`;
- image-level fallback groups because no patient, lesion, or case identifiers
  are supplied.

Dermnet is audit-only for two reasons. First, categories such as
`Exanthems and Drug Eruptions` combine multiple clinical concepts and cannot be
mapped safely to `drug_eruption`. Second, the Kaggle mirror's displayed licence
does not resolve provenance and upstream rights for every atlas image.
Therefore Dermnet contributes zero cases to disease eligibility, the combined
development pool, or either benchmark test set.

## Generated version 3 artifacts

- `data/manifests/fitzpatrick17k_c_v3.parquet`
- `data/manifests/pad_ufes_20_v3.parquet`
- `data/manifests/scin_v3.parquet`
- `data/manifests/ddi_v3.parquet`
- `data/manifests/dermnet_kaggle_v3.parquet`
- `data/combined/visual_top_k_development_pool_v3.parquet`
- `data/reports/source_disease_inventory_v3.csv`
- `data/reports/all_source_disease_coverage_v3.csv`
- `data/reports/disease_coverage_v3.csv`
- `data/reports/included_diseases_v3.yaml`
- `data/reports/out_of_scope_source_labels_v3.csv`
- `data/reports/demographic_availability_v3.csv`
- `data/reports/subgroup_coverage_v3.csv`

The complete source inventory now contains 591 dataset/source-label pairs and
517 consolidated canonical source labels. Of those canonical labels, 21 map
to the active benchmark and 496 remain outside benchmark scope.

## Validation and limitations

All five manifests pass schema and cross-manifest uniqueness validation,
covering 44,313 globally unique image samples. Fourteen unit tests pass,
including taxonomy/schema synchronization, age-band normalization, and the
SCIN multi-race normalization rule.

These subgroup figures describe data availability, not model fairness. A
fairness conclusion requires final leakage-safe splits, prediction results,
uncertainty estimates, and careful interpretation of selection bias.
Fitzpatrick type was designed around sun response and is not equivalent to
constitutive skin colour; source annotations may also be subjective. Final
reporting must preserve the named measurement system and acknowledge these
limitations.

## Next decision gate

The next pipeline stage should:

1. calculate exact and perceptual image hashes;
2. form cross-dataset duplicate groups;
3. create group-safe internal and external evaluation splits;
4. recompute disease and subgroup support after deduplication;
5. freeze the approved taxonomy and split manifests;
6. implement prediction-level subgroup metrics with confidence intervals.
