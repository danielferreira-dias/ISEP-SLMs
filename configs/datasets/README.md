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
4. Resolve every image, calculate exact and perceptual hashes, remove exact
   redundancies, apply `duplicate_review.yaml`, and create leakage-safe
   duplicate groups.
5. Build the cross-dataset coverage report using
   `disease_inclusion.yaml`.
6. Split eligible cases by `leakage_group_id` and produce the generated
   benchmark manifest consumed by
   `configs/benchmarks/derma_isep/visual_top_k.yaml`.

`catalog.yaml` is the registry of all dataset configurations and records
whether a dataset contributes to disease selection, is reserved for external
evaluation, is training-only, or is excluded because it duplicates another
source or has an incompatible target.

## Build the augmented training corpus

HIBA is handled by a separate training-only pipeline. It does not change the
frozen benchmark splits. Derm1M was removed after its label-quality audit:

```bash
uv run python -m src.data_pipeline.training_corpus
```

The output is documented under
`data/training/dermatology_multimodal_v1/README.md`. It starts from the frozen
6,417-image training split, adds only eligible clinical photographs, and
screens every new image against validation, internal test, DDI, and
SkinDisNet using exact and perceptual hashes.

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
SCIN, SkinDisNet, and DDI. It then concatenates the three taxonomy contributors into
`data/combined/visual_top_k_development_pool_v3.parquet` and creates the
source-label inventory, demographic-availability report, subgroup-support
report, and preliminary disease-coverage reports under `data/reports/`.

The hashing stage calculates SHA-256 over encoded image bytes and a
64-bit DCT perceptual hash over orientation-normalized grayscale pixels.
Exact redundant rows are excluded automatically. Perceptual matches are
conservative candidates: they remain included but share a
`leakage_group_id` until reviewed.

The same command creates the frozen `visual_top_k_dataset_v1` release under
`data/benchmarks/derma_isep/visual_top_k_v1/`. Model-ready Parquet files are separated
into `datasets/internal/` and `datasets/external/`; human-readable summaries
are stored under `reports/`, and integrity metadata under `release/`. Its
release manifest records checksums for the source manifests, configurations,
review decisions, and generated artifacts.

After the visual Top-K release is available, the pipeline also creates the
provisional paired confusion-set release under
`data/benchmarks/derma_isep/visual_confusion_sets_v1/`. It selects a balanced subset of
the sealed 1,000-case benchmark and creates one low-confusability and one
high-confusability three-way ranking task for every selected image.

The primary paired pre-training/post-training evaluation uses
`datasets/internal/internal_benchmark_1000.parquet`: exactly 1,000 images from
1,000 distinct internal-test leakage groups. Age, sex/gender, skin tone,
race/ethnicity, disease, dataset, and missingness distributions are audited in
`reports/benchmark_1000_balance_v1.csv`. Demographic values are never included
in the model prompt.

The combined development pool retains out-of-scope rows with their `include`
and `exclusion_reason` fields. Every non-empty source diagnosis receives a
countable `canonical_source_label`, but only clinically reviewed labels map to
one of the 21 active benchmark disease IDs. The pool is not a final training
dataset.

Validate the frozen release without rebuilding it:

```bash
.venv/bin/python -m src.data_pipeline.splitting --validate-only
```

Build or validate only the confusion-set release:

```bash
.venv/bin/python -m src.data_pipeline.confusion_sets
.venv/bin/python -m src.data_pipeline.confusion_sets --validate-only
```

Run the benchmark execution smoke test:

```bash
.venv/bin/python -m src.benchmark.smoke_test
.venv/bin/python -m src.benchmark.confusion_smoke_test
```

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
| `skincare/` | SkinCaRe | Complete gated Hugging Face snapshot, including its SkinCAP and SkinCoT components |
| `skincap/` | SkinCAP | Complete pinned standalone snapshot; 3,250 authorized filtered captions materialized in corrected private ISEPDistillDataset v0.4.1 |
| `dermobench/` | DermoBench | Complete gated Hub release: official task annotations and bundled image archive |
| `hiba/` | HIBA Skin Lesions | Complete official archive; only 355 clinical overview/close-up images are training-only |

## Important constraints

- These datasets contain sensitive or graphic medical images.
- They are research data, not a substitute for clinical validation.
- Keep patient, case, lesion, or encounter groups intact when creating splits.
- Use `leakage_group_id`, which combines source grouping and duplicate
  relationships, for every final split.
- Treat SkinDisNet's augmented images as derivatives, not independent cases;
  benchmark only the preprocessed images and split them by patient.
- Count unique groups when measuring disease support. Do not use raw image
  counts to decide whether a disease has enough data.
- Keep excluded low-frequency diseases in the long-tail report so selection
  decisions remain auditable.
- Do not commit or redistribute raw payloads without checking the applicable
  licence and access agreement.
- DDI, SkinCaRe (including its SkinCAP component), and the DDI portion of SKINCON have additional access
  restrictions. Access to one derived dataset does not automatically grant
  rights to all upstream images.
- A text-only language model cannot learn directly from image pixels. Image-only
  datasets need a visual encoder or a carefully audited structured-to-text
  transformation.
