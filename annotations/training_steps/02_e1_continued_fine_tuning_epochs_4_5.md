# E1 continued fine-tuning: epochs 4 and 5

Date of completion: 14 August 2026  
Experiment status: complete, exploratory single-seed continuation  
Decision: **retain epoch 3 for both conditions**

## Material Passport

- **Research question:** does continuing the label-only E1 training from epoch
  3 for two additional epochs improve validation performance?
- **Conditions:** frozen vision and Vision LoRA, both resumed from their own
  epoch-3 checkpoint.
- **Intervention:** two additional epochs without image augmentation.
- **Data:** unchanged group-safe `sft_train`/`sft_dev` release from
  ISEPDermData 1.3.0.
- **Evaluation:** deterministic generation on all 1,229 `sft_dev` cases.
- **Selection metric:** macro-F1; Top-1, balanced accuracy, evaluation loss, and
  invalid-output rate are secondary diagnostics.
- **Scope:** one seed (3407). The result is sufficient to reject these
  particular continuation checkpoints, but it does not replace the planned
  multi-seed confirmation of the epoch-3 comparison.
- **Integrity:** both complete runs were copied from RunPod to the Mac and
  verified file by file with SHA-256.

## 1. Protocol

Each condition resumed from optimizer step 2,367, the end of epoch 3, and ran
to steps 3,156 and 3,945, corresponding to epochs 4 and 5. Dataset, prompt,
seed, model revision, LoRA topology, image processing, batch size, optimizer,
and absence of augmentation were unchanged. The continuation therefore tests
additional training exposure rather than a new data or architecture method.

The two conditions remained:

1. **Frozen vision:** 32,464,896 trainable parameters; visual layers frozen.
2. **Vision LoRA:** 38,756,352 trainable parameters, including 6,291,456
   visual-layer LoRA parameters.

The continuation ran on one NVIDIA L40S using BF16 LoRA with Unsloth. No
QLoRA, full vision unfreezing, synthetic data, image augmentation, distillation,
or benchmark feedback was introduced.

## 2. Validation results

| Condition | Epoch | Top-1 | Macro-F1 | Balanced accuracy | Eval loss | Invalid |
|---|---:|---:|---:|---:|---:|---:|
| Frozen vision | 3 | **60.13%** | **57.93%** | **56.63%** | **0.2619** | 0/1,229 |
| Frozen vision | 4 | 55.17% | 53.72% | 52.97% | 0.3781 | 0/1,229 |
| Frozen vision | 5 | 57.61% | 55.24% | 53.57% | 0.6045 | 3/1,229 |
| Vision LoRA | 3 | **61.68%** | **62.32%** | **62.16%** | **0.2157** | 0/1,229 |
| Vision LoRA | 4 | 57.04% | 59.28% | 58.44% | 0.3009 | 0/1,229 |
| Vision LoRA | 5 | 58.91% | 58.79% | 58.29% | 0.3780 | 0/1,229 |

For frozen vision, epoch 5 recovered part of the Top-1 loss observed at epoch
4 but remained below epoch 3 on every selection metric. Its macro-F1 was 2.69
percentage points below epoch 3 and its evaluation loss more than doubled.
Three invalid labels also appeared for the first time.

For Vision LoRA, neither additional checkpoint recovered the epoch-3 result.
Epoch 5 was 2.77 points lower in Top-1, 3.53 points lower in macro-F1, and 3.87
points lower in balanced accuracy. Evaluation loss rose from 0.2157 to 0.3780.

The joint pattern—worsening generalization metrics with increasing evaluation
loss—supports an **overfitting interpretation** for this continuation recipe.
Training for longer without changing the data or regularization is therefore
not justified by these results.

## 3. Scientific decision

The selected checkpoint remains **epoch 3 for both conditions**. Epochs 4 and
5 are preserved as negative experimental evidence and must not replace the
selected adapters.

No internal or external benchmark is needed to select among epochs 3–5:
checkpoint selection was pre-defined on `sft_dev`, and using benchmark results
to rescue a rejected checkpoint would leak test information into model
selection. The deterministic internal benchmark results already obtained for
epoch 3 remain the appropriate downstream evidence.

The next data intervention should be evaluated as a separate controlled arm,
not mixed with this continuation result. In particular, an augmentation study
should compare equal update budgets from the same parent checkpoint, with and
without a conservative, clinically label-preserving policy. The failed
continuation alone does not prove that augmentation will help.

## 4. Computational cost

| Continuation metric | Frozen vision | Vision LoRA |
|---|---:|---:|
| Runtime | 4,551 s | 5,395 s |
| GPU-hours | 1.264 | 1.499 |
| Peak VRAM | 13.24 GiB | 13.89 GiB |
| Throughput | 6.94 samples/s | 5.85 samples/s |
| Maximum recorded temperature | 57 °C | 56 °C |
| Maximum recorded power | 271.98 W | 264.63 W |

The two extra epochs consumed about 2.76 GPU-hours in total and produced no
validation improvement. This is a useful efficiency finding for the thesis:
additional compute was not a substitute for better supervision or data.

## 5. Preserved evidence and thesis figures

The complete frozen-vision run is at:

`outputs/training/e1_label_frozen_vision_continued/continued-l40s-frozen-seed3407-epoch5-20260814/`

The complete Vision-LoRA run is at:

`outputs/training/e1_label_unsloth_all_continued/continued-l40s-vision-seed3407-epoch5-20260814/`

Each directory contains resolved configuration, environment and package
versions, checkpoint manifests, adapters, optimizer/scheduler/RNG state,
TensorBoard events, JSONL logs, GPU telemetry, predictions, JSON/CSV/Parquet
metrics, PNG/SVG figures with CSV sources, LaTeX tables, `report.html`, and
`thesis_summary.md`.

The most useful existing dissertation figures are:

- `figures/checkpoint_quality.{png,svg}` — Top-1, macro-F1, and balanced
  accuracy by checkpoint;
- `figures/checkpoint_eval_loss.{png,svg}` — evaluation-loss trajectory;
- `figures/loss_curves.{png,svg}` and `figures/learning_rate.{png,svg}` —
  optimization dynamics;
- `figures/per_class_f1.{png,svg}` and `figures/confusion_matrix.{png,svg}` —
  class-level behavior at the selected checkpoint;
- `figures/resource_*.{png,svg}` — VRAM, throughput, utilization, power, and
  temperature.

Their adjacent CSV files are the authoritative source values. A consolidated
campaign report and the file-level checksum manifests are stored at:

`outputs/training/reports/e1_continued_epoch3_to_5_seed3407/`

## 6. Reproducibility caveat

Re-evaluating the staged epoch-3 adapters during the continuation produced 18
changed predictions for frozen vision and 10 for Vision LoRA relative to the
original archived evaluation, despite matching sample IDs and software-version
manifests. The resulting metric differences were small and did not change the
ranking: epoch 3 remained best in both runs. This is recorded as generation or
GPU replay variability and motivates paired multi-seed confirmation rather
than reporting excessive decimal precision.

## 7. Conclusion

Continued fine-tuning to five epochs without augmentation did not improve the
E1 label-only models. It increased evaluation loss and reduced Top-1,
macro-F1, and balanced accuracy in both conditions. The thesis should retain
the epoch-3 Vision-LoRA checkpoint as the strongest E1 result, preserve the
epoch-4/5 checkpoints as an overfitting ablation, and move to a new controlled
data or supervision intervention rather than adding more identical epochs.
