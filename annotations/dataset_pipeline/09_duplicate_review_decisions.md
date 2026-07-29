# Duplicate review decisions

## Objective

This stage converted duplicate-candidate inspection into versioned,
machine-readable decisions. It aimed to recover supported image-to-label
associations without treating visual inspection as a substitute for clinical
or histopathological diagnosis.

## Inputs

- `data/reports/duplicate_pairs_v3.csv`
- `data/reports/duplicate_groups_v3.csv`
- PAD-UFES-20 source metadata and additional same-lesion images
- full-resolution images for high-priority perceptual candidates
- `configs/datasets/duplicate_review.yaml`

## Exact-label conflict review

Twelve PAD-UFES-20 groups associated identical encoded image bytes with
different lesion IDs and disease labels. Four groups had an independent image
assigned to one candidate lesion:

| Patient | Retained sample | Disease | Supporting sample |
| --- | --- | --- | --- |
| `PAT_56` | `PAD_UFES_20_PAT_56_86_479` | `D005` | `PAD_UFES_20_PAT_56_86_802` |
| `PAT_202` | `PAD_UFES_20_PAT_202_307_424` | `D004` | `PAD_UFES_20_PAT_202_307_927` |
| `PAT_691` | `PAD_UFES_20_PAT_691_1311_2` | `D004` | `PAD_UFES_20_PAT_691_1311_686` |
| `PAT_691` | `PAD_UFES_20_PAT_691_1311_890` | `D004` | `PAD_UFES_20_PAT_691_1311_686` |

Those independent views supported the image-to-lesion association. One
canonical record was retained in each group and the conflicting copy was
excluded with `exact_duplicate_rejected_label_association`. The remaining
eight groups lacked independent evidence and retained `exclude_all`.

The review resolved metadata linkage only. It did not create or change a
disease label based on image appearance.

## High-priority perceptual review

Four perceptual groups were visually reviewed:

- a PAD-UFES-20 mapped-label conflict;
- a SkinDisNet mapped-label conflict;
- a Fitzpatrick17k-C to SkinDisNet overlap candidate;
- a SCIN to SkinDisNet overlap candidate.

All were false positives. The full-resolution sources were different
photographs, lesions, body regions, or compositions. Similar low-frequency
colour and SkinDisNet background removal caused the pHash collisions. Their
perceptual edges are removed before duplicate components are constructed.

## Final results

- source images fingerprinted: 46,023;
- candidate pairs: 2,672;
- duplicate components: 2,095;
- rows newly excluded by deduplication: 30;
- unresolved exact-label-conflict rows: 16;
- pending perceptual-review pairs: 1,245;
- pending perceptual-review components: 1,004.

| Priority | Reason | Pair count |
| --- | --- | ---: |
| 3 | Cross-dataset overlap | 805 |
| 4 | pHash distance 0 | 122 |
| 5 | pHash distance 2 | 129 |
| 6 | pHash distance 4 | 189 |

No priority-1 mapped-label conflict or priority-2 internal-to-external
candidate remains. Pending candidates remain conservatively grouped by
`leakage_group_id`.

## Implementation and validation

`src/data_pipeline/deduplication.py` now loads and validates the review file,
applies reviewed exact decisions, removes rejected perceptual edges, and writes
a prioritized review queue. Unlisted exact conflicts remain excluded; unlisted
perceptual candidates remain included but leakage-grouped.

All six manifests passed validation. Twenty-five unit tests passed after the
review, splitting, and benchmark-runner changes.

## Limitation

Pending components are not confirmed duplicates. Most cross-dataset candidates
involve the excluded Dermnet mirror. Future review decisions must create a new
review-file version and benchmark release checksum.
