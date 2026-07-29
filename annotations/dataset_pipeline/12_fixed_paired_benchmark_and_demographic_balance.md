# Fixed paired benchmark and demographic balance

## Objective

This stage created the fixed 1,000-case internal benchmark used for the paired
comparison between the pre-training and post-training model states.

The same `sample_id` values, images, taxonomy, prompt, and schema must be used
for both stages. This supports per-sample deltas, paired bootstrap confidence
intervals, and direct accounting of corrected and regressed cases.

## Source and selection

The source is the sealed `internal_test` split:

- 1,063 independent `leakage_group_id` values;
- 1,722 eligible images;
- all 21 active diseases.

The selection creates:

- `internal_benchmark_1000.parquet`: 1,000 images from 1,000 distinct leakage
  groups;
- `internal_test_reserve.parquet`: one representative image from each of the
  remaining 63 groups.

The source internal test is retained unchanged for audit. The benchmark and
reserve reconstruct all 1,063 internal-test groups without overlap.

## Representative image policy

Each selected group contributes exactly one image. When a group contains
multiple eligible images, selection is deterministic:

1. prefer a disease with lower group-level support;
2. prefer stronger source diagnostic evidence;
3. break remaining ties with a seeded SHA-256 digest of `sample_id`.

This avoids manual image-quality selection and preserves minority diseases.
The benchmark contains all 21 diseases, with 14 to 129 cases per disease.

## Stratification dimensions

Group selection balances:

- disease ID;
- source dataset;
- disease and dataset combination;
- standardized age group;
- skin-tone system and value;
- sex-or-gender system and value;
- race or ethnicity;
- demographic missingness.

Demographic data is used only for stratification and reporting. It is never
included in the model prompt.

## Age management

Exact ages and source age bands are normalized into:

- under 18;
- 18 to 29;
- 30 to 39;
- 40 to 49;
- 50 to 59;
- 60 to 69;
- 70 and over;
- missing or unknown.

No missing age is imputed and no age is inferred from an image.

| Age category | Benchmark cases |
| --- | ---: |
| Under 18 | 8 |
| 18 to 29 | 75 |
| 30 to 39 | 44 |
| 40 to 49 | 44 |
| 50 to 59 | 55 |
| 60 to 69 | 40 |
| 70 and over | 46 |
| Missing or unknown | 688 |

The under-18 group is retained descriptively but is below the configured
minimum support for standalone benchmark claims.

## Sex and gender management

Source systems are not silently merged. Values remain namespaced:

- `sex_at_birth:female`;
- `sex_at_birth:male`;
- `source_gender:female`;
- `source_gender:male`;
- `missing_or_unknown`.

| Sex/gender category | Benchmark cases |
| --- | ---: |
| `sex_at_birth:female` | 132 |
| `sex_at_birth:male` | 70 |
| `source_gender:female` | 57 |
| `source_gender:male` | 50 |
| Missing or unknown | 691 |

No sex or gender value is inferred from image appearance. Because 69.1% of
cases lack informative sex/gender metadata, subgroup conclusions must report
missingness and avoid claims about the full benchmark population.

## Distribution audit

`benchmark_1000_balance_v1.csv` compares the one-case-per-group internal test
with the selected benchmark. Maximum absolute distribution drift is:

- dataset: 0.31 percentage points;
- disease: 0.58 percentage points;
- age: 0.40 percentage points;
- sex/gender system and value: 0.43 percentage points.

These values show that removing 63 groups did not materially change the
recorded distributions.

## Integrity

The release validator confirms:

- exactly 1,000 images;
- exactly 1,000 leakage groups;
- one image per group;
- all 21 internal-test diseases retained;
- benchmark is a strict subset of internal test;
- zero benchmark-to-reserve group overlap;
- benchmark plus reserve reconstructs all internal-test groups.

## Evaluation rule

The 1,000 cases must not be used to choose prompts, checkpoints, post-training
hyperparameters, or teacher candidates. Those decisions use validation. The
paired benchmark is opened for the frozen pre/post comparison.

External DDI and SkinDisNet evaluations remain separate and must not be pooled
into the 1,000-case internal metric.
