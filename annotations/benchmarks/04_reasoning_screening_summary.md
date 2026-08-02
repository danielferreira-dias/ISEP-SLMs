# Reasoning screening summary

Date: 2026-08-02

Qwen 3.7 Flash, MiMo V2.5, and MiniMax M3 were tested on the same fixed
validation subsets with thinking requested both off and on. Across the
benchmarks, thinking did not produce a large or consistent improvement:
individual metrics sometimes increased and sometimes decreased, while the
models' overall relative capability remained broadly similar.

The OpenRouter endpoints also did not always honor the thinking-off request,
so this experiment should not be treated as a strict causal reasoning
ablation. The practical conclusion is that performance at this stage appears
to depend mainly on the model's underlying capability, rather than on the
thinking switch alone. This evidence is sufficient to continue to the next
teacher-selection stage without further reasoning-mode screening for now.
