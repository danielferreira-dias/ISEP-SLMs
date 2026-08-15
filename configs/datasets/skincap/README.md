# SkinCAP

## Local snapshot

`data/` contains the complete gated Hugging Face snapshot of
[`joshuachou/SkinCAP`](https://huggingface.co/datasets/joshuachou/SkinCAP),
pinned to revision
`4119044b3e14085d7439f88016d93376d433da5f`.

The snapshot was verified against the Hub on 15 August 2026 with no missing
remote files. Raw data remain ignored by Git and must not be redistributed.

| Asset | Count |
| --- | ---: |
| Metadata rows | 4,000 |
| Main PNG images | 4,000 |
| Additional PNGs in `not_include/` | 346 |
| Fitzpatrick17k rows | 3,345 |
| DDI rows | 655 |
| Distinct raw disease strings | 187 |
| Non-empty English captions | 4,000 |

The current `v240715` workbook contains Google Translate English captions.
The older `v240623` CSV contains GPT-4 English translations. Their IDs,
source paths, disease fields, usability flags, and source counts agree; only
eight English captions are byte-for-byte identical.

## Eligibility for ISEP training

The eligibility audit applies the source-provided usability flag before the
frozen thesis leakage exclusions:

| Audit stage | Remaining rows | Removed at stage |
| --- | ---: | ---: |
| Downloaded SkinCAP rows | 4,000 | — |
| `Do not consider this image = 0` | 3,561 | 439 |
| Exclude frozen ISEPDermaBench Validation overlap | 3,437 | 124 |
| Exclude frozen ISEPDermaBench Internal overlap | **3,318** | 119 |

The 3,318 technical candidates comprise 2,683 Fitzpatrick17k rows and 635 DDI
rows. They represent 3,317 distinct `leakage_group_id` values because two
Fitzpatrick17k images belong to one duplicate/leakage group. The two images
must remain in the same train/dev split.

This is a *technical* candidate count, not automatic approval for training.
Written permission to create the private thesis derivatives was attested by the
project owner on 15 August 2026; the permission document itself is deliberately
not stored in the repository. Fitzpatrick17k and DDI upstream terms still
apply, and the resulting dataset remains private.

## Target-quality caution

SkinCAP is free text, but it is not a clean observation-only caption corpus.
A conservative lexical screen flags diagnostic/differential language in
3,304 of the 3,318 eligible captions, and flags some form of diagnosis,
testing, or management language in 3,306. This screen is only an audit aid,
not a clinical quality label.

For E2, do not copy the complete caption blindly as a morphology target.
A versioned high-precision prefix transform was subsequently implemented as
`skincap_observation_prefix_v1`. It accepts 3,250 of the 3,318 technical
candidates: 2,649 Fitzpatrick17k and 601 DDI rows. The remaining 68 are rejected
because the safe prefix is empty/short or retains gold-diagnosis leakage. The
accepted targets have a median length of 19 words (range 5–65; p95 34).

The aggregate audit is stored in `observation_transform_audit.json`. It does
not contain captions, diagnoses, image IDs, or derived text. After the written
permission attestation, the 3,250 accepted targets were materialized as the
private `caption` configuration of `isep_distill_dataset_v0.4.0`: 2,770 train
and 480 dev rows. That release was withdrawn after a cross-task split audit.
The corrected `isep_distill_dataset_v0.4.1` contains 2,767 train and 483 dev
rows. Its captions inherit the already frozen E1 or morphology assignment,
giving zero cross-task train/dev group overlap. Trainer-visible shards exclude
the raw caption, diagnosis,
and removed suffix; they retain only the filtered target and provenance hashes.

## Integrity

| File | SHA-256 |
| --- | --- |
| `README.md` | `f4e7c7fb263e225ff0349620003ee8b0cf294ea23a83f3b451e6eaf28b17d2eb` |
| `skincap_v240623.csv` | `2591fbc7492cb3048a456d6b2bea4d4749f85f17dcb90d833be3a07c2c874f06` |
| `skincap_v240715.xlsx` | `0286ec8ba295eda97b89db04969e5436862b8c8f68aa03b3104705e2a9b39df3` |

The local payload occupies approximately 575 MiB including the Hugging Face
download cache. Main images contain 547,605,606 encoded bytes; the 346
`not_include/` images contain 22,689,729 encoded bytes.

## Sources

- Dataset: <https://huggingface.co/datasets/joshuachou/SkinCAP>
- Paper: <https://arxiv.org/abs/2405.18004>
- Upstream Fitzpatrick17k: <https://github.com/mattgroh/fitzpatrick17k>
- Upstream DDI: <https://ddi-dataset.github.io/>
- Upstream SKINCON: <https://skincon-dataset.github.io/>
