# Validation findings and current limitations

## Automated validation

The implementation includes unit tests for:

- lexical disease-label normalization;
- dataset-specific diagnostic-code mapping;
- conservative handling of ambiguous eczema labels;
- SCIN differential ordering;
- Fitzpatrick-value validation;
- preliminary coverage thresholds.

The test command is:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Ten tests pass, including a contract test that keeps the 20 active taxonomy
IDs, retired IDs, benchmark configuration, and JSON output schema synchronized.

The manifest validator additionally checks:

- uniqueness of `sample_id` within and across manifests;
- non-null `group_id`;
- absence of included rows without a normalized disease ID;
- agreement between `disease_id` and the first ranked reference;
- consecutive reference ranks;
- exact alignment between the YAML manifest field list and Parquet columns.

All four manifests pass validation. They contain 24,754 globally unique image
samples, including the separately held-out DDI manifest.

All configuration YAML files parse successfully, source archive members and
direct image paths are checked during normalization, and the dependency
lockfile is synchronized.

## Current limitations

### No image-content deduplication

The first-stage manifests do not calculate SHA-256 or perceptual hashes.
Fitzpatrick17k-C is already a corrected release, but cross-dataset duplicates,
near duplicates, and known SCIN duplicates have not yet been removed.

The current coverage result is therefore preliminary. Final splits must not be
created until exact, perceptual, and source-lineage duplicate analysis is
complete.

### Incomplete patient grouping

PAD-UFES-20 and SCIN have suitable patient or case groups. Fitzpatrick17k-C has
no patient identifier, and the local DDI metadata does not expose the
documented patient grouping. Image-level fallback groups are used for those
datasets.

This may overestimate the number of independent cases. DDI remains a single
external evaluation set, but repeated-patient effects should still be reported
if a patient mapping becomes available.

### Heterogeneous label evidence

The sources do not provide equivalent ground truth:

- DDI is pathology-grounded.
- PAD-UFES-20 mixes pathology and clinical consensus.
- SCIN provides retrospective dermatologist differentials.
- Fitzpatrick17k-C inherits noisy web-atlas labels.

Metrics must therefore be reported by source dataset and diagnosis basis in
addition to any combined result.

### Candidate taxonomy only

The current taxonomy contains 20 active classes selected from the complete
source-label inventory. All pass preliminary total-support and source-diversity
thresholds, but the taxonomy still requires clinical approval and duplicate-
adjusted recounting before it can be frozen.

The JSON output schema enumerates the 20 active candidate IDs and is currently
synchronized with taxonomy version 2.1.0. It must be regenerated after any
future taxonomy revision.

### No final splits or benchmark dataset

The following artifacts do not yet exist:

- final train and validation manifests;
- `internal_test.parquet`;
- `ddi_external_test.parquet`;
- decoded-image resolver used by model inference.

These are intentionally deferred until mapping review and duplicate analysis
are complete.

## Next decision gate

The next stage should:

1. review the out-of-benchmark-scope label inventory;
2. approve or revise clinical grouping mappings;
3. decide whether the two unsupported candidate classes remain;
4. compute exact and perceptual duplicate groups;
5. recalculate unique-case coverage;
6. freeze the benchmark taxonomy;
7. create group-safe train, validation, internal-test, and DDI external-test
   manifests.
