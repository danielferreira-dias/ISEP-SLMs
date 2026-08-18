# ISEP training pipeline

This package implements the first two controlled fine-tuning phases of the
thesis. `E1_label` compares two otherwise identical BF16 LoRA runs of
`Qwen/Qwen3.5-4B`:

- `E1_frozen_vision`: language-side LoRA with the visual encoder frozen;
- `E1_unsloth_all`: Unsloth's multimodal LoRA recipe with visual layers
  trainable through adapters.

The comparison is an ablation, not two unrelated training recipes. The CLI
checks the frozen dataset release, configuration hashes and trainable-module
manifest so that the vision policy is the only intended scientific change.

`E2_skincon` starts again from the same official Qwen base with a fresh
Vision-LoRA adapter. The baseline mixes the frozen E1 diagnosis labels with
all eligible human SKINCON morphology targets. A separate
`E2_skincon_skincap` ablation adds authorized, filtered SkinCAP captions; it
does not replace the two-task baseline. E2 contains no teacher output;
distillation starts only in E3.

## Environment

Training requires Linux, one NVIDIA GPU with BF16 support and a dedicated
environment. The vLLM benchmark and Unsloth environments deliberately remain
separate.

```bash
uv sync --extra training
```

Unsloth, TRL and CUDA are imported only by GPU commands. Configuration
validation, dataset preparation, reports and comparison remain usable on a
CPU-only development machine.

## Workflow

```bash
# Validate both immutable experiment recipes.
isep-train validate-config \
  --config configs/training/e1_label_frozen_vision.yaml
isep-train validate-config \
  --config configs/training/e1_label_unsloth_all.yaml
isep-train validate-config \
  --config configs/training/e2_skincon_unsloth_all.yaml
isep-train validate-config \
  --config configs/training/e2_skincon_skincap_unsloth_all.yaml

# Materialize the group-safe split release once.
isep-train prepare-data \
  --config configs/training/e1_label_frozen_vision.yaml
isep-train inspect-data \
  --release data/training/ISEPDermData/releases/e1_label_v1

# Verify the full GPU path before a real experiment.
isep-train smoke-test \
  --config configs/training/e1_label_frozen_vision.yaml
isep-train smoke-test \
  --config configs/training/e2_skincon_unsloth_all.yaml
isep-train smoke-test \
  --config configs/training/e2_skincon_skincap_unsloth_all.yaml

# Run or resume a scientific experiment.
isep-train run \
  --config configs/training/e1_label_frozen_vision.yaml \
  --seed 42
isep-train run \
  --config configs/training/e1_label_frozen_vision.yaml \
  --resume-from outputs/training/e1_label_frozen_vision/<run>/checkpoints/checkpoint-<step>

# Rebuild derived outputs or compare paired runs.
isep-train evaluate --run-dir <run-directory> --checkpoints all
isep-train report --run-dir <run-directory>
isep-train compare --runs \
  <frozen-42> <frozen-3407> <frozen-2026> \
  <vision-42> <vision-3407> <vision-2026>
```

Every command validates YAML before loading a model. `run` accepts only a
prepared release and never creates a new split implicitly.

## Data contract

The source is `ISEPDermData` 1.3.0. `prepare-data` assigns complete
`leakage_group_id` values to `sft_train` or `sft_dev` with a deterministic
85/15 split and creates a 210-case development panel containing ten unique
groups per disease. The release stores identifiers and hashes rather than
copying private image bytes.

Each E1 target contains one image and one canonical diagnosis. It contains no
teacher response, metadata or rationale. Only assistant response tokens
contribute to the loss and Qwen thinking is disabled.

The E2 baseline consumes `ISEPDistillDataset` release
`isep_distill_dataset_v0.3.0`: 6,312 diagnosis + 3,068 morphology train rows,
and 1,229 diagnosis + 527 morphology dev rows. The additive SkinCAP ablation
uses corrected v0.4.1 and adds 2,767 caption train rows and 483 caption dev
rows. The withdrawn v0.4.0 is rejected because 125 shared groups crossed task
splits. Every
task row appears once per epoch in a deterministic interleave. The pipeline
verifies the pinned Hub revision, release manifest, 48-concept ontology, every
Parquet shard and embedded image hash before using a row. Morphology rows do
not expose diagnostic labels; caption rows expose only the filtered target and
not the raw caption, diagnosis, or removed suffix.

## Checkpoint evaluation

The base checkpoint is evaluated on the fixed development panel before the
first update. Every immutable epoch checkpoint is replayed deterministically
on that panel after training; this avoids loading a second 4B model inside a
Trainer callback while preserving the exact epoch states. After training, the
base and all candidate checkpoints are evaluated on the complete `sft_dev`
split. The selected checkpoint maximizes macro-F1, with balanced accuracy,
evaluation loss and the earliest epoch as ordered tie-breakers.

For E2, the same diagnosis metric selects the checkpoint so comparison with E1
remains direct. Base and epoch checkpoints are also evaluated on the 527 human
SKINCON dev rows using exact match, micro/macro F1, per-concept precision,
recall and F1, Hamming loss, and invalid-JSON rate. These results produce PNG,
SVG and source CSV figures alongside the standard training artefacts.

The SkinCAP ablation also evaluates every state on its 483 caption dev rows.
It reports clinical-format compliance, prohibited-content rate, concept
precision/recall/F1, unsupported-concept rate, ROUGE-L, token F1 and their
declared deterministic reference-similarity average. Caption quality is not
called accuracy. The E2 report therefore contains both the task-specific
metrics and a transparent `global_multitask_score`: the unweighted macro mean
of diagnosis macro-F1, morphology macro-F1 and (when present) the SkinCAP
caption task score. This composite is comparable only between runs containing
the same task set and is never used to hide the disaggregated results.

Optimizer-time monitoring records train/eval loss, learning rate, seconds per
step, examples per second and tokens per second. Full generative diagnosis,
morphology and caption metrics are computed for the base model and every epoch
checkpoint after the optimizer run. This preserves every checkpoint state
without contaminating step-time, energy or throughput measurements with
generation callbacks.

The smoke profile records its identity separately from a full run, validates
finite/improving loss, verifies the assistant-only token mask, validates the
resume state hashes and reloads the saved adapter twice to confirm stable
prediction. A smoke checkpoint cannot be resumed accidentally as a full run.

ISEPDermaBench, DermoBench, DDI and SkinDisNet are not checkpoint-selection
sets. They remain reserved for later frozen evaluation.

## Private checkpoint repository

Every full run mirrors each completed epoch to the private Hugging Face model
repository `danielfdias98/ISEP-training-checkpoints`. The remote hierarchy is:

```text
e1_label_frozen_vision/
└── seed-3407/
    └── <run-id>/
        ├── checkpoint-epoch-01/
        ├── checkpoint-epoch-02/
        └── checkpoint-epoch-03/
```

The local directories retain Trainer's native `checkpoint-<global-step>`
names because those paths are required for reliable resume. The mirror maps
their integer epoch to `checkpoint-epoch-NN` without changing local state.
Each epoch is uploaded as one Hub commit and recorded in
`manifests/checkpoint_uploads.json` with the remote path, commit URL, file list
and a SHA-256 tree hash.

Only an explicit allowlist of adapter, optimizer, scheduler, RNG, tokenizer
and provenance files may be uploaded. Clinical images, predictions and report
artefacts are rejected. The pipeline verifies that the destination remains
private and accessible before allocating the GPU, never uploads smoke tests
and treats a later upload failure as a failed, resumable run. Resuming retries
the last checkpoint upload before performing new optimizer updates.

## Thesis artefacts

Runs are written below `outputs/training/<experiment>/<run-id>/` and include:

- immutable configuration, model, dataset, prompt and environment manifests;
- canonical JSONL metrics plus validated CSV/Parquet history snapshots;
- per-sample predictions in CSV and Parquet;
- TensorBoard event files;
- resumable LoRA checkpoints;
- PNG and SVG figures with their source CSV files;
- CSV and LaTeX tables;
- a Markdown thesis summary and self-contained HTML report.

The production collator also writes `logs/sample_costs.jsonl` and materializes
`metrics/sample_costs.csv` plus `metrics/sample_costs.parquet`. Each unique
task row contains `sample_id`, split, `leakage_group_id`, original and resized
image geometry, original pixel count, exact visual/prompt/target token counts,
available annotations, phase and task. Token counts come from the actual
post-collation `input_ids`, attention mask and assistant-only loss labels, not
from an offline estimate.

`tables/resource_summary.csv` and its LaTeX counterpart report duration,
GPU-hours, peak/average VRAM, peak/average process RAM, steps/examples/tokens
per second, mean step time, GPU utilization, peak/mean power, integrated Wh,
peak/mean temperature, best-checkpoint size and trainable-parameter count.
NVML fields remain null on unsupported hardware rather than being inferred.

Reports do not embed clinical images. Resource measurements are explicitly
marked unavailable when NVML is not present; missing values are never
fabricated.

## Reproducibility rules

- Model and dataset revisions must be immutable commit hashes.
- Existing releases and run directories are never overwritten.
- Resume validates model, data and configuration identity.
- Epoch uploads require an authenticated Hugging Face token with write access.
- OOMs, NaNs and missing CUDA/BF16 support fail explicitly.
- The pipeline never retries with a smaller batch, quantized model or changed
  preprocessing.
- Confirmation runs use `--seed 42`, `--seed 3407` and `--seed 2026` for each
  vision condition; single-seed runs are pilots. Confirmatory `compare` refuses
  any set other than the paired six runs.

`E1_label` and human-only `E2_skincon` are executable training phases. E3 now
has an executable, gated teacher-generation pilot for GPT-5.6 Sol at medium
reasoning effort, a source-traceable 66-concept Stage-A terminology contract,
strict Stage-A/Stage-B review contracts, durable progress, and five CPU-tested
task renderings. It remains absent from the student-training
phase registry until a versioned teacher release passes the offline preflight,
one-case external smoke, 100-case generation pilot, and materialization audit.
Reinforcement learning, QLoRA, crop experiments and multi-GPU training require
later, separately reviewed phases.
