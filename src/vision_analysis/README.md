# Student visual attribution

This module produces a qualitative, reproducible view of which image regions
affect a fixed dermatology diagnosis score. It is separate from benchmark
accuracy and must not be described as access to a model's private reasoning.

The first supported method is blurred patch occlusion. For every grid region,
the runner replaces that region with a strongly blurred version of the same
image and recomputes the teacher-forced log-probability of a disease ID. A
positive score drop means that the region supported the target; a negative
drop means that the region suppressed it.

Validate the frozen pilot without loading the model:

```bash
uv run python -m src.vision_analysis.cli --validate-only
```

Run one case on Apple MPS:

```bash
uv run python -m src.vision_analysis.cli \
  --device mps \
  --dtype float16 \
  --max-cases 1
```

Run all pilot cases on a CUDA GPU:

```bash
uv run python -m src.vision_analysis.cli \
  --device cuda \
  --dtype bfloat16
```

The output directory contains the original image, raw score-drop arrays,
overlays, per-tile metadata, a complete manifest, and a self-contained
`report.html`. The red regions support the selected target, while blue regions
suppress it under this specific perturbation.

The final before/after study must reuse the same frozen images, prompt, target
definitions, occlusion configuration, and model revision across `E0_base` and
all trained checkpoints. Gradient-based attribution and patch occlusion should
then be reported together as complementary views.
