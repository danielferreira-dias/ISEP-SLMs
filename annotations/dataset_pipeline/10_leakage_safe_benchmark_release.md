# Leakage-safe benchmark release

## Objective

This stage generated the first reproducible dataset release for the visual
top-k benchmark. It separates model-development data, an internal held-out
test, and two external evaluation datasets while keeping patients, cases, and
duplicate candidates intact.

## Configuration

`configs/datasets/visual_top_k_split.yaml` defines:

- release: `visual_top_k_dataset_v1`, version `1.0.0`;
- seed: `42`;
- algorithm: `greedy_multilabel_group_stratification_v1`;
- internal ratios: 70% train, 15% validation, 15% internal test;
- grouping key: `leakage_group_id`.

The internal pool contains Fitzpatrick17k-C, PAD-UFES-20, and SCIN. DDI and
SkinDisNet are external-only and do not influence internal splitting or
taxonomy selection.

## Split algorithm

The deterministic greedy algorithm assigns complete leakage groups and balances
group-level incidence for disease, dataset, disease-dataset combinations, age,
skin tone, sex or gender, and race or ethnicity. Demographic metadata is used
only for split balance and reporting, never in the model prompt.

Rare disease-dataset combinations are assigned first. Candidate costs measure
deviation from feature and total-group targets. A seeded SHA-256 tie-breaker
makes assignment independent of input row order.

## Generated evaluation sets

| Evaluation set | Images | Leakage groups | Diseases |
| --- | ---: | ---: | ---: |
| Train | 6,417 | 4,962 | 21 |
| Validation | 1,683 | 1,063 | 21 |
| Internal test | 1,722 | 1,063 | 21 |
| Internal paired benchmark | 1,000 | 1,000 | 21 |
| External DDI | 300 | 299 | 8 |
| External SkinDisNet | 1,365 | 333 | 4 |

Every internal split contains all 21 diseases. Per-disease group counts range
from 71 to 600 in train, 16 to 162 in validation, and 16 to 164 in internal
test. The paired benchmark contains 14 to 129 cases per disease. External
metrics use the same fixed taxonomy and schema but only the
classes present in each dataset.

## Integrity

The release passed:

- zero leakage-group overlap between internal splits;
- zero sample-ID overlap between internal splits;
- zero eligible group overlap between the internal pool and either external
  dataset;
- all 21 diseases present in every internal split.

## Frozen release

`data/benchmarks/visual_top_k_v1/benchmark_release_v1.yaml` stores SHA-256
checksums for the source manifests, taxonomy, benchmark, prompt, schema,
policies, review decisions, generated manifests, and reports. Validation fails
if any referenced file changes without rebuilding the release.

## Outputs

- `train.parquet`, `validation.parquet`, and `internal_test.parquet`
- `internal_benchmark_1000.parquet` and `internal_test_reserve.parquet`
- `external_ddi.parquet` and `external_skindisnet.parquet`
- `split_summary_v1.csv` and `subgroup_summary_v1.csv`
- `benchmark_1000_balance_v1.csv`
- `integrity_report_v1.yaml`
- `benchmark_release_v1.yaml`

All outputs are under `data/benchmarks/visual_top_k_v1/`.

## Reproduction

```bash
.venv/bin/python -m src.data_pipeline.pipeline
.venv/bin/python -m src.data_pipeline.splitting --validate-only
```

## Limitation

External macro metrics cover eight DDI classes or four SkinDisNet classes and
are not numerically equivalent to 21-class internal macro metrics. Pending
pHash candidates are conservatively kept together; reviewed false positives
can be separated in a future release.
