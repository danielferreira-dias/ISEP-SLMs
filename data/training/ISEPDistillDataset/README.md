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
---

# ISEPDistillDataset

Private research repository for the multimodal training corpus used in the
ISEP thesis on dermatology-specialized small multimodal language models.

> **Status:** release `isep_distill_dataset_v0.3.0` is materialized and
> validated. The Dataset Viewer exposes only the real `diagnosis` and
> `morphology` Parquet configs; documentation examples and JSON schemas are not
> interpreted as training rows.

## Released configurations

| Configuration | sft_train | sft_dev | Total | Target |
|---|---:|---:|---:|---|
| `diagnosis` | 6,312 | 1,229 | 7,541 | One canonical label from the frozen 21-class taxonomy |
| `morphology` | 3,068 | 527 | 3,595 | All positive concepts from the 48-concept SKINCON ontology |

`diagnosis` reproduces the frozen E1 split exactly. `morphology` excludes 271
SKINCON rows that overlap ISEPDermaBench Validation/Internal groups. Its 606
rows already present in E1 inherit their frozen split; remaining groups use a
deterministic 85/15 split with seed 42.

## Objective

The dataset will support two controlled thesis phases after the existing
label-only baseline:

- `E2_structured`: supervision for visual concepts, short captions, grounded
  differentials, evidence links, uncertainty, and next action.
- `E3_hard_kd`: the best E2 condition plus filtered response targets generated
  by a frozen open-weight teacher.

The core teacher protocol is answer-blind and two-stage. Stage A describes only
what is observable in the image. Stage B receives the image and the frozen
Stage A output, then produces a differential and evidence links. The gold
diagnosis is used only after generation for filtering and partial acceptance.

## Configuration roadmap

| Configuration | Row unit | Student target | Primary source |
|---|---|---|---|
| `diagnosis` **released** | One diagnostic task per image | Canonical label | Normalized gold label |
| `morphology` **released** | One perception task per image | Visible SKINCON concepts | All eligible SKINCON |
| `caption` | One description task per image | Short clinical caption | Eligible human text or accepted Stage A rendering |
| `structured` | One complete clinical task per image | Observations, differential, evidence, uncertainty, action | Accepted Stage A and Stage B |
| `open_response` | One open clinical task per image | Short natural-language response consistent with the canonical JSON | Rendering of the accepted structured target |

The optional `preferences`, feature-distillation, logit-distillation, and
on-policy artifacts are not part of the first core dataset release.

## Dataset Viewer

The release uses explicit dataset-card configs and image-aware Parquet metadata.
The Viewer therefore offers `diagnosis` and `morphology`, each with
`sft_train`/`sft_dev`, and renders the embedded clinical image plus nested
`messages` and `skincon` fields.

| config | sample_id | image | target_text | quality_status | messages |
|---|---|---|---|---|---:|
| `diagnosis` | source-stable ID | embedded source image | canonical label | `accepted` | 2 |
| `morphology` | SKINCON-stable ID | embedded source image | deterministic concept JSON | `accepted` | 2 |

The synthetic examples under `examples/` document future schemas only and are
excluded by the explicit `configs` manifest above.

## Common public columns

| Group | Fields |
|---|---|
| Identity | `sample_id`, `case_id`, `task_id`, `image_asset_id`, `view_type` |
| Leakage control | `leakage_group_id`, `split`, `split_inherited_from_e1` where applicable |
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
- `caption` requires eligible human text or an accepted Stage A rendering.
- `structured` requires accepted Stage A and Stage B components.
- `open_response` is created only when an accepted canonical structured target
  can be rendered consistently.

The release manifest reports exact shards, hashes, coverage and split counts in
`metadata/release.json`; `metadata/quality_summary.json` records the validation
gates.

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
  examples/
  metadata/
  schemas/
```

The Parquet shards were written by the versioned builder after split, license,
leakage, schema, image-decode, and checksum gates passed. `caption`,
`structured`, and `open_response` remain planned configs and will be exposed
only after real accepted targets exist.

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
