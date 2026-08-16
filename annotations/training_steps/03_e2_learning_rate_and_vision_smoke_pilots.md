# E2 structured supervision: learning-rate and visual-LoRA smoke pilots

Date of completion: 15 August 2026  
Experiment status: complete, exploratory smoke comparison  
Decision: **advance learning rate `1e-4`; retain both frozen-vision and
Vision-LoRA conditions for the full E2 experiment**

## Material Passport

- **Material type:** E2 training-pipeline smoke report and provisional
  hyperparameter decision.
- **Research object:** Qwen 3.5 4B trained jointly on canonical diagnosis,
  SKINCON morphology, and SkinCAP-derived clinical caption tasks.
- **Base checkpoint:**
  `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- **Human-supervision release:** `ISEPDistillDataset` 0.4.1 at Hub revision
  `b215f0474e4931b5951da768e79a0d579d26919d`.
- **Execution scope:** one NVIDIA L40S; seed 42; 64 training examples; 30
  optimizer steps; 32 evaluation cases per task in smoke mode.
- **Compared interventions:** Vision LoRA at `2e-4`, Vision LoRA at `1e-4`,
  and frozen vision at `1e-4`.
- **Controlled factors:** data release, task mixture, sample budget, seed,
  preprocessing, LoRA rank/alpha/dropout, language-side targets, optimizer,
  scheduler, batch size, prompts, and deterministic decoding.
- **Checkpoint selection:** diagnosis macro-F1, with balanced accuracy,
  evaluation loss, and earlier checkpoint as tie-breakers.
- **Verification status:** **ANALYZED**. All three runs passed the smoke
  integrity gates, but the 32-case metrics have not been confirmed on the full
  `sft_dev` split or across multiple seeds.
- **Preservation scope:** the `2e-4` Vision run is preserved locally; at the
  user's request, the two `1e-4` runs were inspected on the RunPod and were not
  copied to the Mac.

## 1. Objective

E2 extends label-only supervision with two additional targets:

1. a deterministic SKINCON morphology representation;
2. a constrained one-sentence clinical description derived from SkinCAP.

The smoke experiments addressed two questions before committing to the full
training cost:

1. Does reducing the E2 learning rate from the Unsloth-style `2e-4` starting
   point to `1e-4` better preserve the three-task behavior?
2. At `1e-4`, does applying LoRA to visual layers improve the E2 trade-off
   relative to keeping the visual component frozen?

These are pipeline and recipe-selection pilots, not final E2 experiments.

## 2. Protocol

All conditions used BF16 LoRA with rank 16, alpha 16, dropout 0, bias disabled,
and `all-linear` targets. The effective batch size was eight (microbatch two,
gradient accumulation four), with AdamW 8-bit, weight decay `0.001`, 5% warmup,
linear decay, assistant-only loss, no packing, no QLoRA, no augmentation, no
teacher output, and thinking disabled.

Smoke mode limited training to 64 examples and 30 updates. With an effective
batch size of eight, the saved states at steps 8, 16, and 24 correspond to
approximately one, two, and three passes over the smoke sample; step 30 is the
final 3.75-pass state. Evaluation was deterministic on 32 cases for each of
diagnosis, morphology, and caption generation.

The two `1e-4` configurations were validated as a controlled pair: they differ
only in `vision_profile` and `finetune_vision_layers`. The model base predictions
and evaluation sample IDs were also common across the learning-rate pilot.

## 3. Checkpoint trajectories

`Global` below is the macro task score
`mean(diagnosis_macro_f1, morphology_macro_f1, caption_task_score)`. It is **not
an accuracy metric** and is comparable only when the task set is identical.

| Condition | Step | Diagnosis macro-F1 | SKINCON macro-F1 | SkinCAP score | Global |
|---|---:|---:|---:|---:|---:|
| Vision LoRA, `2e-4` | 8 | 23.29% | 12.15% | **39.71%** | 25.05% |
| Vision LoRA, `2e-4` | 16 | **27.76%** | 11.84% | 24.83% | 21.48% |
| Vision LoRA, `2e-4` | 24 | 24.78% | **17.22%** | 11.48% | 17.82% |
| Vision LoRA, `2e-4` | 30 | 25.45% | 15.38% | 11.80% | 17.54% |
| Vision LoRA, `1e-4` | 8 | **30.17%** | **13.65%** | 36.79% | **26.87%** |
| Vision LoRA, `1e-4` | 16 | 23.55% | 9.79% | **41.14%** | 24.82% |
| Vision LoRA, `1e-4` | 24 | 28.06% | 11.80% | 36.26% | 25.37% |
| Vision LoRA, `1e-4` | 30 | 28.21% | 11.66% | 36.74% | 25.54% |
| Frozen vision, `1e-4` | 8 | **31.69%** | **13.63%** | **38.46%** | **27.92%** |
| Frozen vision, `1e-4` | 16 | 23.55% | 9.30% | 39.49% | 24.11% |
| Frozen vision, `1e-4` | 24 | 25.23% | 9.39% | 36.61% | 23.74% |
| Frozen vision, `1e-4` | 30 | 25.23% | 11.31% | 36.93% | 24.49% |

The `2e-4` condition reduced optimization loss more aggressively, but its
SkinCAP score collapsed after the first checkpoint and its global score fell
throughout training. The lower training loss therefore did not indicate better
multitask generalization in this pilot.

## 4. Selected-checkpoint comparison

The fixed diagnosis-macro-F1 rule selected step 16 for Vision LoRA at `2e-4`
and step 8 for both `1e-4` conditions.

| Metric | Vision `2e-4`, step 16 | Vision `1e-4`, step 8 | Frozen `1e-4`, step 8 |
|---|---:|---:|---:|
| Diagnosis Top-1 | **40.63%** | 34.38% | **40.63%** |
| Diagnosis macro-F1 | 27.76% | 30.17% | **31.69%** |
| Diagnosis balanced accuracy | 34.29% | 32.38% | **35.24%** |
| SKINCON macro-F1 | 11.84% | **13.65%** | 13.63% |
| SKINCON micro-F1 | **56.14%** | 51.72% | 51.16% |
| SKINCON exact match | **18.75%** | 9.38% | 9.38% |
| SkinCAP task score | 24.83% | 36.79% | **38.46%** |
| SkinCAP concept-F1 | 3.39% | 15.65% | **16.81%** |
| SkinCAP reference similarity | **14.85%** | 13.48% | 14.19% |
| SkinCAP clinical compliance | 56.25% | 81.25% | **84.38%** |
| Global macro task score | 21.48% | 26.87% | **27.92%** |

The unadapted base scored 40.63% diagnosis Top-1, 30.07% diagnosis macro-F1,
0% SKINCON macro-F1, 46.52% SkinCAP task score, and 25.53% global macro task
score. Step 8 Frozen therefore preserved base Top-1, slightly improved diagnosis
macro-F1, learned the structured morphology format, and remained below the base
on caption quality. Its global score was 2.39 percentage points above the base.

Every selected trained checkpoint produced valid diagnosis and caption outputs.
The base model had a 100% invalid-format rate on the structured SKINCON task;
all trained checkpoints reduced this to 0% in the 32-case smoke evaluation.

## 5. Frozen versus Vision LoRA at `1e-4`

At the commonly selected step 8, Frozen minus Vision LoRA was:

- diagnosis Top-1: **+6.25 percentage points**;
- diagnosis macro-F1: **+1.52 points**;
- diagnosis balanced accuracy: **+2.86 points**;
- SKINCON macro-F1: **-0.02 points**;
- SKINCON micro-F1: **-0.56 points**;
- SkinCAP task score: **+1.66 points**;
- SkinCAP concept-F1: **+1.15 points**;
- SkinCAP clinical compliance: **+3.13 points**;
- global macro task score: **+1.05 points**.

This provisional result differs from E1, where Vision LoRA produced the best
full-validation macro-F1. It is a useful hypothesis for the full E2 ablation:
structured human morphology and caption supervision may initially be absorbed
by language-side adapters without requiring additional visual adaptation. The
smoke sample is too small to establish this mechanism or to discard the Vision
LoRA arm.

## 6. Optimization and resource observations

| Metric | Vision `2e-4` | Vision `1e-4` | Frozen `1e-4` |
|---|---:|---:|---:|
| Early loss mean | 1.2807 | 1.4213 | 1.4227 |
| Late loss mean | **0.6470** | 0.8131 | 0.8263 |
| Training duration | 209.1 s | 133.5 s | **116.5 s** |
| Training energy | 10.29 Wh | 7.74 Wh | **6.58 Wh** |
| Peak VRAM | 13.48 GiB | **12.29 GiB** | 13.18 GiB |
| Trainable parameters | 38,756,352 | 38,756,352 | **32,464,896** |
| Checkpoint size | 0.237 GiB | 0.237 GiB | **0.202 GiB** |

Frozen vision used 6,291,456 fewer trainable parameters than Vision LoRA. In
this run it was approximately 12.8% faster and consumed about 15.0% less energy
than the `1e-4` Vision smoke. Peak VRAM did not follow the same direction and
was 0.90 GiB higher in the Frozen run, illustrating that a single short run is
too noisy for a memory-efficiency conclusion.

The timing and energy values are operational evidence only. The runs were
sequential and could differ in cache warmth and one-time compilation overhead;
learning rate must not be credited with the runtime difference. A full
same-hardware efficiency comparison requires repeated, warmed measurements.

## 7. Smoke integrity gates

All three runs passed:

- finite and decreasing observed loss;
- assistant-only mask audit present;
- adapter save/reload prediction stability;
- checkpoint resume-manifest validation;
- no OOM or NaN;
- no silent QLoRA, CPU, batch-size, or split fallback.

The two `1e-4` smokes completed remotely. They were intentionally not copied to
the Mac and were not uploaded to the private checkpoint repository because
`upload_smoke=false`.

## 8. Decision and next experiment

The learning-rate pilot supports replacing `2e-4` with **`1e-4` for E2**. The
reason is not lower training loss; it is better retention of diagnosis and
caption behavior alongside successful morphology learning.

The architecture question remains open. Frozen vision currently has the better
32-case trade-off, but selecting it as the sole E2 topology would overinterpret
a smoke test. The confirmatory E2 design remains:

1. start both conditions independently from the same pinned Qwen 3.5 4B base;
2. train Frozen Vision and Vision LoRA with the same `1e-4` recipe, seed 42,
   full task mixture, and matched update budget;
3. select checkpoints only on the complete group-safe `sft_dev` metrics;
4. compare diagnosis, SKINCON, SkinCAP, global macro task score, and cost;
5. use the winning topology as the candidate for E3 hard knowledge
   distillation, subject to later seed confirmation.

No full E2 run was launched after these smokes. The RunPod had approximately
9 GB free at completion, which is insufficient for the two complete E2 runs;
storage must be expanded or completed smoke caches must be removed under an
explicit preservation/deletion decision.

## 9. Remote provenance

The inspected run directories were:

- `outputs/training/e2_skincon_skincap_unsloth_all/`
  `smoke-l40s-e2-skincap-vision-seed42-v2-20260815/` (`2e-4`, locally
  preserved);
- `outputs/training/e2_skincon_skincap_unsloth_all_lr1e4_pilot/`
  `smoke-l40s-e2-skincap-vision-lr1e4-seed42-20260815/`;
- `outputs/training/e2_skincon_skincap_frozen_vision_lr1e4_pilot/`
  `smoke-l40s-e2-skincap-frozen-lr1e4-seed42-20260815/`.

The frozen pilot configuration is
`configs/training/e2_skincon_skincap_frozen_vision_lr1e4_pilot.yaml`; the
Vision pilot configuration is
`configs/training/e2_skincon_skincap_unsloth_all_lr1e4_pilot.yaml`.

## 10. Interpretation boundary

These findings support pipeline readiness and a provisional E2 recipe. They do
not establish a statistically reliable Frozen-versus-Vision difference,
external generalization, clinical validity, or superiority over larger models.
The sample contains only 32 evaluation cases per task and one training seed.
The full `sft_dev` evaluation and, ideally, later seed replication are required
before a dissertation claim about the E2 architecture is made.
