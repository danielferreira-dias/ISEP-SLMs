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

Nineteen tests pass, including a contract test that keeps the 21 active taxonomy
IDs, retired IDs, benchmark configuration, and JSON output schema synchronized.
The tests also enforce the SkinDisNet mapping boundaries and its external-only
dataset role. New tests cover encoding-stable perceptual hashing, exact
canonical selection, and exact-label-conflict exclusion.

The manifest validator additionally checks:

- uniqueness of `sample_id` within and across manifests;
- non-null `group_id`;
- absence of included rows without a normalized disease ID;
- agreement between `disease_id` and the first ranked reference;
- consecutive reference ranks;
- exact alignment between the YAML manifest field list and Parquet columns.

All six manifests pass validation. They contain 46,023 globally unique image
samples, including the separately held-out DDI and SkinDisNet manifests and
the audit-only Dermnet Kaggle manifest.

All configuration YAML files parse successfully, source archive members and
direct image paths are checked during normalization, and the dependency
lockfile is synchronized.

## Current limitations

### Perceptual candidates require review

All six manifests now contain encoded-byte SHA-256 values, 64-bit DCT
perceptual hashes, duplicate decisions, and leakage-safe group IDs. Exact
redundancies are removed automatically, and exact label conflicts are
excluded.

Perceptual hashing is intentionally conservative and produces candidates, not
clinical proof that two photographs are the same observation. The 1,020
duplicate components marked `requires_review` must be inspected before final
splits are frozen. Candidates already share a leakage group, preventing them
from crossing splits if review is incomplete.

### Incomplete patient grouping

PAD-UFES-20, SCIN, and SkinDisNet have suitable patient or case groups.
Fitzpatrick17k-C has no patient identifier, and the local DDI metadata does not
expose the documented patient grouping. Image-level fallback groups are used
for those datasets.

This may overestimate the number of independent cases. DDI remains a single
external evaluation set, but repeated-patient effects should still be reported
if a patient mapping becomes available.

### Heterogeneous label evidence

The sources do not provide equivalent ground truth:

- DDI is pathology-grounded.
- PAD-UFES-20 mixes pathology and clinical consensus.
- SCIN provides retrospective dermatologist differentials.
- Fitzpatrick17k-C inherits noisy web-atlas labels.
- SkinDisNet provides clinically reviewed labels but not pathology-confirmed
  diagnoses.

Metrics must therefore be reported by source dataset and diagnosis basis in
addition to any combined result.

### Candidate taxonomy only

The current taxonomy contains 21 active classes selected from the complete
source-label inventory. All pass preliminary total-support and source-diversity
thresholds, but the taxonomy still requires clinical approval and duplicate-
adjusted recounting before it can be frozen.

The JSON output schema enumerates the 21 active candidate IDs and is currently
synchronized with taxonomy version 2.2.0. It must be regenerated after any
future taxonomy revision.

### No final splits or benchmark dataset

The following artifacts do not yet exist:

- final train and validation manifests;
- `internal_test.parquet`;
- `ddi_external_test.parquet`;
- `skindisnet_external_test.parquet`;
- decoded-image resolver used by model inference.

These are intentionally deferred until perceptual-candidate review and final
taxonomy approval are complete.

## Next decision gate

The next stage should:

1. review perceptual duplicate candidates, prioritizing cross-dataset and
   label-conflicting groups;
2. approve or reject candidate edges and rebuild leakage groups;
3. complete clinical approval of the candidate taxonomy;
4. freeze the benchmark taxonomy;
5. create group-safe train, validation, internal-test, DDI external-test, and
   SkinDisNet external-test manifests.
