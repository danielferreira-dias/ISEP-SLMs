# SkinFlow and the visual-encoder strategy

## Executive decision

SkinFlow supports the thesis hypothesis that a small, domain-specialized
multimodal model can outperform much larger general models. It is also relevant
because it treats fine-grained visual encoding—not language-model scale—as a
potential dermatology bottleneck.

However, the exact SkinFlow architecture should **not** be added to the
official Qwen 3.5 4B student before establishing a reproducible fine-tuning
baseline. The defensible order is:

1. train and evaluate the unchanged student architecture;
2. add morphology/description supervision;
3. test low-risk visual adaptation such as projector and late vision-block
   LoRA or selective unfreezing;
4. test higher-resolution or lesion-crop inputs under a matched compute budget;
5. only then run FDLinear as an isolated experimental branch.

The most immediately useful SkinFlow idea for this thesis is therefore its
**describe-then-diagnose training curriculum**, not its unreleased custom
operator.

## What SkinFlow actually changes

[SkinFlow](https://arxiv.org/abs/2601.09136) is a January 2026 preprint built
from Qwen2.5-VL-Instruct-7B. It combines two interventions:

1. a Dynamic Vision Encoder (DVE), implemented with Frequency Dynamic Linear
   (`FDLinear`) operators;
2. two stages of reinforcement learning: dermatological caption learning,
   followed by ranked Top-K diagnosis.

In the Qwen vision transformer, the authors replace static linear layers in
the MLPs at layers 8, 16, 24, and 32. A static projection uses one matrix for
all samples. FDLinear instead builds a sample-conditioned matrix from
frequency-disjoint bases:

```text
global image descriptor -> mixing coefficients alpha
frequency bases + alpha  -> sample-specific W(image)
visual tokens            -> W(image) × tokens
```

The intended effect is to adapt the projection to the image and separate
subtle lesion texture from healthy skin and background without physically
widening the encoder. The paper reports less than 5% parameter overhead
relative to the vision encoder.

Stage I uses about 5,000 labelled dermatology images. Approximately 4,000
captions are machine-generated and accepted only when an LLM can recover the
gold diagnosis from the description; failed captions are regenerated up to
five times and then sent for expert revision. Captions use fields such as
colour, location, shape, lesion type, size, border, surface, and distribution.
The generated caption is scored field by field to form an RL reward.

Stage II continues with RL and teaches a ranked Top-K differential. The reward
depends on the first position at which an LLM evaluator accepts a diagnosis as
clinically equivalent to the reference. Training uses VERL, with a reported
Stage I learning rate of `1e-6` and Stage II learning rate of `5e-7`.

## Why they had this idea

The authors describe a “powerful brain, weak retina” asymmetry: Qwen2.5-VL has
a language backbone of roughly 7B parameters but a vision encoder of roughly
0.6B. They hypothesize that subtle morphology can be irreversibly lost before
the language model starts reasoning. In their framing, diagnosis is an
image-compression and semantic-decoding process:

- explicit, describable evidence is learned through medical captioning;
- implicit texture is retained by a more adaptive visual representation;
- diagnosis is decoded only after this alignment.

There is also a direct methodological lineage. SkinFlow coauthor Linwei Chen
is the first author of [Frequency Dynamic Convolution for Dense Image
Prediction](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Frequency_Dynamic_Convolution_for_Dense_Image_Prediction_CVPR_2025_paper.html),
the CVPR 2025 paper cited as the source of the frequency-dynamic operator.
FDConv partitions a fixed Fourier-domain parameter budget into disjoint
frequency groups and dynamically modulates the resulting filters. The
SkinFlow authors adapt that idea from convolutional kernels to linear
projections in a ViT. This origin is an inference from shared authorship,
citation, and mechanism; the SkinFlow paper does not narrate the ideation
process explicitly.

## What the ablation really shows

SkinFlow evaluates 1,000 randomly selected Fitzpatrick17k images and about 200
internally curated images. Its main ablation is:

| Variant | Fitzpatrick17k Top-1 | Top-6 | Internal Top-1 | Top-6 |
| --- | ---: | ---: | ---: | ---: |
| No Stage I, no DVE | 15.22% | 45.36% | 27.46% | 66.84% |
| Stage I, no DVE | 24.45% | 57.69% | 35.64% | 74.75% |
| Stage I + DVE | 29.19% | 71.16% | 36.63% | 79.21% |

Consequently:

- Stage I adds 9.23 percentage points Top-1 and 12.33 points Top-6 on
  Fitzpatrick17k;
- DVE then adds 4.74 points Top-1 and 13.47 points Top-6;
- on the internal set, DVE adds only 0.99 points Top-1 and 4.46 points Top-6.

This is evidence that both interventions may help, but the largest and most
consistent Top-1 gain comes from caption alignment. There is no `DVE-only`
row, so the independent effect of DVE and its interaction with Stage I cannot
be fully identified.

## Evidence that supports the general direction

The exact FDLinear-in-Qwen method is not independently replicated, but several
peer-reviewed studies support its broader premises:

- [FDConv, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Frequency_Dynamic_Convolution_for_Dense_Image_Prediction_CVPR_2025_paper.html)
  validates frequency-diverse dynamic weights in object detection,
  segmentation, and classification. Its official implementation reports a
  modest parameter increase, but it implements convolution rather than the
  unreleased FDLinear adaptation.
- [PanDerm, Nature Medicine 2025](https://www.nature.com/articles/s41591-025-03747-y)
  pretrains a dermatology-specific ViT on more than two million images from 11
  institutions and four imaging modalities. It reports state-of-the-art
  results across 28 datasets, often with only 10% of labelled downstream data.
  This strongly supports investing in domain-specific visual representations,
  although it does not validate dynamic weights.
- [MONET, Nature Medicine 2024](https://www.nature.com/articles/s41591-024-02887-x)
  aligns 105,550 dermatology images with natural-language concepts from
  medical literature and demonstrates dermatologist-validated concept
  recognition. This supports explicit morphology supervision and auditable
  concept alignment.
- [SkinGPT-4, Nature Communications 2024](https://www.nature.com/articles/s41467-024-50043-3)
  aligns a vision transformer and LLM using 52,929 skin images paired with
  clinical concepts and doctors' notes in a two-step training design. This is
  a close precedent for description/knowledge alignment before downstream
  interaction.
- [Anatomy-VLM, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Gu_Anatomy-VLM_A_Fine-grained_Vision-Language_Model_for_Medical_Interpretation_WACV_2026_paper.html)
  uses localization and multi-scale regions to prevent a medical VLM from
  treating the image only as a holistic object. It is radiology rather than
  dermatology, but supports explicit fine-grained visual encoding.
- [HuatuoGPT-Vision, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.418/)
  shows that high-quality, cleaned medical image-text supervision improves
  multimodal medical capabilities. This supports improving visual-language
  data before assuming that architecture is the main bottleneck.

These studies support specialized visual pretraining, fine-grained regions,
and image-concept alignment. They do **not** constitute independent evidence
that SkinFlow's exact Fourier-linear replacement is superior to simpler vision
fine-tuning under matched data and compute.

## Reproducibility and validity concerns

SkinFlow is promising but currently insufficient as an implementation recipe:

- It is a preprint and no official SkinFlow code, model weights, or linked
  dataset were found on arXiv or its Hugging Face paper page.
- The public FDConv repository exposes `FDConv`, not the FDLinear module used
  in SkinFlow.
- The paper uses `K=64` in its virtual-width example but says `K=d/2` in the
  implementation section. For `d=1280`, those values are 64 and 640. The text
  also claims less than 5% vision-encoder parameter overhead without enough
  implementation detail to reproduce the storage and factorization exactly.
- It does not report a `DVE-only` ablation, a matched vision-LoRA/unfreezing
  baseline, a high-resolution/crop baseline, or a larger-vision-encoder
  baseline.
- It claims comparable inference complexity but does not provide a complete
  latency, throughput, memory, or FLOP comparison for the final Qwen model.
- Localization evidence consists of final-token attention heatmaps and
  attention-weight distributions over 500 images. No lesion masks or
  localization IoU are used; concentrated attention is not causal proof that
  the diagnosis relies on the correct lesion.
- Predictions are evaluated by Gemini 2.5 Pro three times using a permissive
  clinical hierarchy. This is useful for synonyms and near-misses but is not a
  dermatologist reader study and makes direct comparison with exact-match
  benchmarks difficult.
- Fitzpatrick17k uses 1,000 randomly sampled images, but the paper does not
  provide the exact IDs. Dataset correction status and reproducibility of the
  sample therefore remain unclear.
- The second evaluation set is private, small, and from the same development
  organization. It is not equivalent to external validation.
- The authors acknowledge simple image backgrounds and incomplete systematic
  interpretability evaluation.

## Recommended experiment for this thesis

Do not modify the official student yet. First freeze a baseline with the same
data split, image resolution, prompts, sampling, and compute budget. Then use a
small architecture study:

| Arm | Change | Question |
| --- | --- | --- |
| A | unchanged Qwen 3.5 4B | reference baseline |
| B | morphology/description SFT | does explicit visual supervision help? |
| C | B + projector/late-vision LoRA | is low-risk visual adaptation enough? |
| D | B + multi-scale or lesion crop | is resolution/localization the bottleneck? |
| E | B + FDLinear pilot | does the custom operator add value beyond B–D? |

Evaluate all arms on the same Internal Benchmark only after development is
finished. During development, use Validation and report:

- Visual Top-K Top-1, Top-3, and Top-6;
- Evidence-Grounded Diagnosis accuracy and finding/evidence metrics;
- open-ended judge diagnosis and clinical-rationale scores;
- the no-image grounding ablation;
- subgroup performance by skin tone where available;
- external DDI and SkinDisNet generalization;
- trainable parameters, peak memory, throughput, and latency.

If masks are available or can be curated for a small subset, add lesion
localization IoU or pointing-game accuracy. Attention concentration alone
should remain exploratory.

For FDLinear, use a separate branch and require, before a full run:

1. an unambiguous definition of the basis count and parameterization;
2. initialization that preserves the pretrained static projection;
3. a 10–50 case forward-pass and memory smoke test;
4. a small Stage-I-only ablation;
5. comparison against Arm C with matched trainable parameters and compute.

## Final assessment

SkinFlow is highly relevant as a **research hypothesis** and as support for a
describe-then-diagnose curriculum. It is not yet strong enough to justify
making FDLinear part of the main student architecture. The best thesis path is
to obtain a clean Qwen 3.5 4B baseline, test morphology-aligned training, and
only implement FDLinear if the baseline analysis shows a persistent visual
bottleneck that simpler visual adaptation does not resolve.
