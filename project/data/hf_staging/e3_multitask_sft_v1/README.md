---
pretty_name: ISEPDistillDataset
language:
  - en
  - pt
license: other
task_categories:
  - image-classification
  - visual-question-answering
  - image-to-text
tags:
  - dermatology
  - multimodal
  - knowledge-distillation
  - clinical-reasoning
  - private-research-release
size_categories:
  - 10K<n<100K
configs:
  - config_name: diagnosis
    default: true
    data_files:
      - split: sft_train
        path: data/diagnosis/sft_train-*.parquet
      - split: sft_dev
        path: data/diagnosis/sft_dev-*.parquet
  - config_name: morphology
    data_files:
      - split: sft_train
        path: data/morphology/sft_train-*.parquet
      - split: sft_dev
        path: data/morphology/sft_dev-*.parquet
  - config_name: caption
    data_files:
      - split: sft_train
        path: data/caption_v0_4_1/sft_train-*.parquet
      - split: sft_dev
        path: data/caption_v0_4_1/sft_dev-*.parquet
  - config_name: e3_multitask_sft_v1
    data_files:
      - split: sft_train
        path: data/e3_multitask_sft_v1/sft_train-*.parquet
---

# ISEPDistillDataset

Private research repository for the multimodal training corpus used in the
ISEP thesis on dermatology-specialized small multimodal language models.

> **Status:** corrected additive source release `isep_distill_dataset_v0.4.1`
> and the derived trainer-ready release `e3_multitask_sft_v1` are materialized
> and validated. The E3 release adds a new configuration without changing the
> frozen `diagnosis`, `morphology`, or `caption` shards. v0.4.0 remains
> withdrawn because its task-local split check missed 125 cross-task train/dev
> group conflicts.

## Released configurations

| Configuration | sft_train | sft_dev | Total | Target |
|---|---:|---:|---:|---|
| `diagnosis` | 6,312 | 1,229 | 7,541 | One canonical label from the frozen 21-class taxonomy |
| `morphology` | 3,068 | 527 | 3,595 | All positive concepts from the 48-concept SKINCON ontology |
| `caption` | 2,767 | 483 | 3,250 | Short observation-only prefix filtered from authorized SkinCAP text |
| `e3_multitask_sft_v1` | 25,084 | — | 25,084 | Task-isolated diagnosis, morphology, caption, grounded differential, or image-replacement conversation |

`diagnosis` reproduces the frozen E1 split exactly. `morphology` excludes 271
SKINCON rows that overlap ISEPDermaBench Validation/Internal groups. Its 606
rows already present in E1 inherit their frozen split; remaining groups use a
deterministic 85/15 split with seed 42.

`caption` excludes 439 author-rejected cases and 243 frozen benchmark overlaps
before transformation. A versioned high-precision transform accepts 3,250 of
3,318 technical candidates. The source captions were produced with knowledge
of the diagnosis, so these rows are explicitly labelled
`human_caption_gold_conditioned_filtered`; they are not presented as
answer-blind descriptions. Written derivative permission was attested by the
project owner on 15 August 2026 and the permission document itself is not
stored in this repository.

Every caption group now inherits the already frozen E1 or morphology split.
The combined `diagnosis + morphology + caption` release has zero train/dev
overlap by `leakage_group_id`; the manifest records the source of each caption
split. This cross-task invariant is checked by the runtime before training.

`e3_multitask_sft_v1` expands the 6,312 frozen `sft_train` diagnosis images
into one independently prompted conversation per task. It contains 6,312
diagnosis, 6,312 structured morphology, 6,312 answer-blind caption, 6,127
grounded differential, and 21 request-new-image rows. Stage B accepted 6,148
images and clinically rejected 164; rejected Stage B reasoning is not a
trainer-visible target. No distilled `sft_dev` is published, preserving the
frozen development protocol for comparable checkpoint selection.

## Objective

The dataset will support two controlled thesis phases after the existing
label-only baseline:

- `E2_structured`: supervision for visual concepts, short captions, grounded
  differentials, evidence links, uncertainty, and next action.
- `E3_hard_kd`: the best E2 condition plus filtered response targets generated
  by a frozen open-weight teacher.

The teacher protocol is two-stage with a strict information boundary. Stage A
is answer-blind and describes only what is observable in the image. Stage B
receives the same image, the frozen Stage A output, and the private gold
diagnosis as a supervised anchor; it produces an auditable differential and a
concise student-facing clinical justification without inventing findings.

## Configuration roadmap

| Configuration | Row unit | Student target | Primary source |
|---|---|---|---|
| `diagnosis` **released** | One diagnostic task per image | Canonical label | Normalized gold label |
| `morphology` **released** | One perception task per image | Visible SKINCON concepts | All eligible SKINCON |
| `caption` **released** | One description task per image | Short visible observation | Authorized, filtered SkinCAP prefix |
| `e3_multitask_sft_v1` **released** | One task-isolated conversation per image/task | Diagnosis, morphology JSON, caption, grounded differential, or request for a better image | Human gold plus accepted frozen Stage A/B targets |
| `structured` | One complete clinical task per image | Observations, differential, evidence, uncertainty, action | Accepted Stage A and Stage B |
| `open_response` | One open clinical task per image | Short natural-language response consistent with the canonical JSON | Rendering of the accepted structured target |

The optional `preferences`, feature-distillation, logit-distillation, and
on-policy artifacts are not part of the first core dataset release.

## Dataset Viewer

The release uses explicit dataset-card configs and image-aware Parquet metadata.
The Viewer offers `diagnosis`, `morphology`, `caption`, and
`e3_multitask_sft_v1`, and renders the embedded clinical image plus the exact
trainer-visible prompt/target messages. Only the three frozen source configs
include `sft_dev`; the derived E3 config contains `sft_train` only.

| config | sample_id | image | target_text | quality_status | messages |
|---|---|---|---|---|---:|
| `diagnosis` | source-stable ID | embedded source image | canonical label | `accepted` | 2 |
| `morphology` | SKINCON-stable ID | embedded source image | deterministic concept JSON | `accepted` | 2 |
| `caption` | SkinCAP-stable ID | embedded source image | filtered visible observation | `accepted` | 2 |
| `e3_multitask_sft_v1` | source-stable task-row ID | embedded source image | task-specific accepted target | `accepted` | 2 |

The synthetic examples under `examples/` document future schemas only and are
excluded by the explicit `configs` manifest above.

## Common public columns

| Group | Fields |
|---|---|
| Identity | `sample_id`, `case_id`, `task_id`, `image_asset_id`, `view_type` |
| Leakage control | `leakage_group_id`, `split`, `split_inherited_from_e1`, `split_source` where applicable |
| Provenance | `source_dataset`, `source_sample_id`, `license_id`, `image_sha256` |
| Gold | `disease_id`, `gold_diagnosis`, `gold_provenance`, `taxonomy_version` |
| Human morphology | `skincon.ontology_version`, `skincon.source_subset`, `skincon.positive_concepts`, `skincon.all_concepts_annotated` |
| Target | `target_variant`, `target_source`, config-specific structured fields, `messages` |
| Prompt/target | `prompt`, `prompt_sha256`, `target_text`, `messages` |
| Quality | `schema_version`, `quality_status` |

## Uneven target coverage

Not every source image will appear in every configuration. Missing supervision
is represented by the absence of a task row, never by an invented caption or
an all-null placeholder target.

- SKINCON provides 48 binary visual concepts, not prose descriptions. Positive
  concepts are stored in `skincon.positive_concepts`; when
  `skincon.all_concepts_annotated=true`, every omitted ontology concept is an
  annotated negative rather than a missing value.
- The reproduced source audit contains 3,866 usable annotations with upstream
  diagnostic labels: 3,230 Fitzpatrick17k and 636 DDI.
- The training policy admits 3,595 morphology candidates after removing 271
  Fitzpatrick17k cases that overlap the frozen internal benchmark. The 480 rows
  marked `Do not consider this image` by SKINCON remain excluded.
- In the released morphology config, 1,198 rows map through the versioned
  taxonomy mapping to the 21 classes. The remaining 2,397 retain valid upstream
  disease labels but are morphology-only examples; they do not expand the
  diagnostic output vocabulary silently.
- DDI is now admitted to morphology training. Consequently, DDI results after
  this release are in-domain/contaminated diagnostics and cannot be presented
  as independent external generalization evidence.
- `caption` contains 3,250 filtered SkinCAP observations. Raw captions,
  diagnoses, and removed suffixes are excluded from trainer-visible shards.
- `structured` requires accepted Stage A and Stage B components.
- `open_response` is created only when an accepted canonical structured target
  can be rendered consistently.
- `e3_multitask_sft_v1` keeps diagnosis, morphology, and caption for all 6,312
  training images. Only the fourth clinical-response row depends on Stage B
  acceptance; 164 clinically rejected Stage B samples therefore contribute no
  grounded-differential or request-new-image target.

The release manifest reports exact shards, hashes, coverage and split counts in
`metadata/release.json` for v0.3 and
`releases/isep_distill_dataset_v0.4.1/release.json` for v0.4.1; the matching
quality summaries record the validation gates.

The frozen ontology order and the reproduced coverage audit are stored in
`metadata/skincon_ontology.json` and `metadata/skincon_coverage.json`. A
documentation-only morphology record is available at
`examples/morphology.example.json`.

Only an explicit per-task allowlist determines what is visible to the student.
Audit fields must never leak gold diagnosis or private metadata into an
answer-blind teacher call.

## Private audit material

The following do not belong in the distributable core configs:

- raw private reasoning or chain-of-thought from the teacher;
- long unfiltered teacher outputs;
- free-text reviewer notes or reviewer identity;
- potentially re-identifiable source identifiers;
- sealed benchmark responses and references;
- full-vocabulary logits, hidden states, or feature tensors.

When needed for an experiment, model-side KD artifacts are versioned separately
and linked through `sample_id`, `task_id`, and `generation_id`.

## Physical layout

```text
ISEPDistillDataset/
  README.md
  data/
    diagnosis/
    morphology/
    caption/                 # withdrawn v0.4.0, retained for auditability
    caption_v0_4_1/          # current Viewer/training shards
    e3_multitask_sft_v1/     # trainer-ready E3 task rows, sft_train only
  examples/
  metadata/
  releases/
    isep_distill_dataset_v0.4.0/
    isep_distill_dataset_v0.4.1/
    e3_multitask_sft_v1/
  schemas/
```

The Parquet shards were written by versioned builders after split, license,
leakage, schema, image-decode, and checksum gates passed. `diagnosis`,
`morphology`, `caption`, and `e3_multitask_sft_v1` are materialized;
`structured` and `open_response` remain optional planned configs rather than
prerequisites for E3 training.

## Versioning policy

- Schema versions follow semantic versioning.
- Every generated release freezes source revisions, taxonomy, prompts, teacher
  revision, generation parameters, split assignments, and file hashes.
- `sample_id + task_id + target_variant` is unique inside a release.
- Every image belonging to the same patient, lesion, case, or equivalent image
  group remains in one `leakage_group_id`.
- A release tag is created only after automatic audit and manual approval.

## License and intended use

This repository is private and research-only. Each row retains its source-level
`license_id`; mixed-source data are not assigned a blanket permissive license.
DDI image bytes are included under the user's accepted research-use terms and
must not be redistributed publicly. This corpus is not a medical device and is
not intended for clinical diagnosis or patient care.
