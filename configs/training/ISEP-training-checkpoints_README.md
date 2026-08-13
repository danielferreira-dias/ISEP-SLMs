---
base_model: Qwen/Qwen3.5-4B
library_name: peft
pipeline_tag: image-text-to-text
tags:
  - unsloth
  - lora
  - multimodal
  - dermatology
  - thesis
---

# ISEP training checkpoints

Private, resumable LoRA checkpoints produced by the controlled training
experiments of the ISEP master's thesis.

This repository is an experiment store, not a single final model and not a
clinical product. The checkpoints must not be used for medical diagnosis.

## Layout

```text
<experiment-id>/
└── seed-<seed>/
    └── <run-id>/
        ├── checkpoint-epoch-01/
        ├── checkpoint-epoch-02/
        └── checkpoint-epoch-03/
```

For example:

```text
e1_label_frozen_vision/
└── seed-3407/
    └── <run-id>/
        └── checkpoint-epoch-01/
```

Each epoch directory contains the PEFT adapter and the trainer state required
to resume training, including optimizer, scheduler and random-number-generator
state. Every directory also contains `isep_checkpoint.json`, which binds the
checkpoint to immutable model, dataset, configuration and run identities.

## Privacy and provenance

- The repository must remain private.
- No clinical image, per-case prediction or Hugging Face token is uploaded.
- The training pipeline uses an explicit filename allowlist.
- One epoch is written in one Hub commit.
- The local run records the commit identifier, URL and SHA-256 tree hash in
  `manifests/checkpoint_uploads.json`.
- Smoke-test checkpoints are intentionally excluded.

The source dataset and model revisions, exact prompt, trainable parameter
manifest and software environment remain recorded in the corresponding local
training run.

## Current experiments

- `e1_label_frozen_vision`: BF16 LoRA with the visual component frozen.
- `e1_label_unsloth_all`: BF16 LoRA following the Unsloth multimodal recipe,
  including visual LoRA.

The two conditions use the same group-safe data split, seeds, update budget,
image preprocessing and optimization settings. They differ only in whether
visual LoRA is enabled.
