# Training-step annotations

This directory records the execution, validation, interpretation, and decision
gates of each model-training phase. Raw predictions, checkpoints, telemetry,
TensorBoard events, and automatically generated reports remain under
`outputs/training/`; the annotations here provide the thesis-facing account and
point to the preserved evidence.

## Reports

1. [E1 label-only: frozen vision versus Vision LoRA](01_e1_label_only_vision_lora_ablation.md)
2. [E1 continued fine-tuning: epochs 4 and 5](02_e1_continued_fine_tuning_epochs_4_5.md)
3. [E2 learning-rate and visual-LoRA smoke pilots](03_e2_learning_rate_and_vision_smoke_pilots.md)
4. [E2 full multitask campaign and E1 comparison](04_e2_full_multitask_campaign_and_e1_comparison.md)

## Figure policy

Figures selected for the dissertation are stored under `figures/` as both PNG
and SVG. Every curated figure must have an adjacent CSV containing its source
values. The full automatically generated figure set remains inside each run's
`outputs/training/<experiment>/<run-id>/figures/` directory.
