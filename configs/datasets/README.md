# Dermatology datasets

This directory contains local research copies and access documentation for the
dermatology datasets considered for this project.

Raw payloads live under each dataset's `data/` directory and are intentionally
ignored by Git. The READMEs remain trackable and record the source, schema,
labels, licence, and completeness of the local copy.

Each dataset also has a `config.yaml` that describes how its native fields map
to the shared normalized manifest. The configuration declares the output; it
does not mean that the Parquet manifest has already been generated.

The normalization and selection flow is:

1. Read the source-specific `config.yaml`.
2. Produce one normalized manifest row per image using
   `manifest_schema.yaml`.
3. Map source disease names to stable IDs in
   `configs/taxonomies/diseases.yaml`.
4. Build the cross-dataset coverage report using
   `disease_inclusion.yaml`.
5. Split eligible cases by `group_id` and produce the generated benchmark
   manifest consumed by `configs/benchmarks/visual_top_k.yaml`.

`catalog.yaml` is the registry of all dataset configurations and records
whether a dataset contributes to disease selection, is reserved for external
evaluation, or is excluded because it duplicates another source or has an
incompatible target.

## Build normalized manifests and coverage reports

Run the complete first-stage pipeline from the repository root:

```bash
.venv/bin/python -m src.data_pipeline.pipeline
```

Validate existing outputs without rebuilding them:

```bash
.venv/bin/python -m src.data_pipeline.pipeline --validate-only
```

The pipeline builds normalized manifests for Fitzpatrick17k-C, PAD-UFES-20,
SCIN, SkinDisNet, DDI, and the Dermnet Kaggle mirror. It then concatenates the
three taxonomy contributors into
`data/combined/visual_top_k_development_pool_v3.parquet` and creates the
source-label inventory, demographic-availability report, subgroup-support
report, and preliminary disease-coverage reports under `data/reports/`.

The combined development pool retains out-of-scope rows with their `include`
and `exclusion_reason` fields. Every non-empty source diagnosis receives a
countable `canonical_source_label`, but only clinically reviewed labels map to
one of the 21 active benchmark disease IDs. The pool is not a final training
dataset.
Final train, validation, and test splits must only be created after mapping
review, duplicate analysis, and taxonomy approval.

Image bytes are not duplicated during normalization. Direct files use normal
relative paths, images inside ZIP archives use `zip://` locators, and embedded
SCIN images use `parquet://` locators containing the shard, source row, and
image column.

Demographic fields are source-dependent and are used only for evaluation
stratification. They are never added to the model prompt. Age, sex or gender,
race or ethnicity, and skin-tone values retain their provenance. Fitzpatrick,
grouped Fitzpatrick, and Monk Skin Tone values are reported as separate
measurement systems rather than converted into a single scale.

| Directory | Dataset | Local payload |
| --- | --- | --- |
| `fitzpatrick17k/` | Fitzpatrick17k and Fitzpatrick17k-C | Complete original image archive plus original and corrected metadata |
| `scin/` | SCIN | Complete official Hugging Face snapshot |
| `pad-ufes-20/` | PAD-UFES-20 | Metadata and three official image archives |
| `ddi/` | Diverse Dermatology Images | Complete metadata and all 656 official images |
| `skindisnet/` | SkinDisNet | Complete official version 2 archive with 1,710 preprocessed smartphone images and patient metadata |
| `skincon/` | SKINCON | Complete concept annotations; images remain subject to Fitzpatrick17k/DDI access |
| `skincap/` | SkinCAP | Complete gated Hugging Face snapshot for the locally authenticated account |
| `skincare/` | SkinCaRe | Complete gated Hugging Face snapshot for the locally authenticated account |
| `dermavqa/` | DermaVQA | Public OSF files, including the IIYI image archive and Reddit answer annotations |
| `dermnet/` | Dermnet Kaggle mirror | Complete 19,559-image archive; excluded pending upstream-rights and label-quality review |

## Important constraints

- These datasets contain sensitive or graphic medical images.
- They are research data, not a substitute for clinical validation.
- Keep patient, case, lesion, or encounter groups intact when creating splits.
- Treat SkinDisNet's augmented images as derivatives, not independent cases;
  benchmark only the preprocessed images and split them by patient.
- Count unique groups when measuring disease support. Do not use raw image
  counts to decide whether a disease has enough data.
- Keep excluded low-frequency diseases in the long-tail report so selection
  decisions remain auditable.
- Do not commit or redistribute raw payloads without checking the applicable
  licence and access agreement.
- DDI, SkinCAP, SkinCaRe, and the DDI portion of SKINCON have additional access
  restrictions. Access to one derived dataset does not automatically grant
  rights to all upstream images.
- Dermnet is a third-party mirror whose displayed Kaggle licence does not
  resolve the upstream rights for every atlas image. Keep it excluded unless
  those rights are confirmed in writing.
- A text-only language model cannot learn directly from image pixels. Image-only
  datasets need a visual encoder or a carefully audited structured-to-text
  transformation.
