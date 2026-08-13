# ISEP training pipeline

This package implements the first controlled fine-tuning phase of the thesis:
`E1_label`. It compares two otherwise identical BF16 LoRA runs of
`Qwen/Qwen3.5-4B`:

- `E1_frozen_vision`: language-side LoRA with the visual encoder frozen;
- `E1_unsloth_all`: Unsloth's multimodal LoRA recipe with visual layers
  trainable through adapters.

The comparison is an ablation, not two unrelated training recipes. The CLI
checks the frozen dataset release, configuration hashes and trainable-module
manifest so that the vision policy is the only intended scientific change.

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

# Materialize the group-safe split release once.
isep-train prepare-data \
  --config configs/training/e1_label_frozen_vision.yaml
isep-train inspect-data \
  --release data/training/ISEPDermData/releases/e1_label_v1

# Verify the full GPU path before a real experiment.
isep-train smoke-test \
  --config configs/training/e1_label_frozen_vision.yaml

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

## Checkpoint evaluation

The base checkpoint is evaluated on the fixed development panel before the
first update. Every immutable epoch checkpoint is replayed deterministically
on that panel after training; this avoids loading a second 4B model inside a
Trainer callback while preserving the exact epoch states. After training, the
base and all candidate checkpoints are evaluated on the complete `sft_dev`
split. The selected checkpoint maximizes macro-F1, with balanced accuracy,
evaluation loss and the earliest epoch as ordered tie-breakers.

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

Only `E1_label` is executable in this version. Structured supervision,
distillation, reinforcement learning, QLoRA, crop experiments and multi-GPU
training require later, separately reviewed phases.
