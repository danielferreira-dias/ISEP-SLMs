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

## Expanded 300-case visual follow-up

On 4 August 2026, Qwen 3.8 Max and Qwen 3.7 Flash were compared again on the
same complete 300-case General Visual Hallucination cohort. The detailed
protocol, condition-level metrics, stopped Dermatology runs, and artifacts are
recorded in
[`09_expanded_visual_hallucination_audits.md`](09_expanded_visual_hallucination_audits.md).

| Comparison | Primary mode | Escalated thinking mode |
| --- | ---: | ---: |
| Qwen 3.8 General accuracy | **80.7% low** | 80.0% high |
| Qwen 3.8 mean reasoning tokens | **176.4** | 669.0 |
| Qwen 3.7 General accuracy | 69.0% off | **76.0% thinking** |
| Qwen 3.7 raw JSON validity | **95.7% off** | 54.0% thinking |
| Qwen 3.7 mean reasoning tokens | **0.0** | 1,508.9 |

This follow-up strengthens the operational decision to avoid escalating
thinking for the final visual benchmarks. Qwen 3.8 high was slightly less
accurate and substantially more expensive. Qwen 3.7 thinking improved the
semantic status metric, so the result is not evidence that thinking always
reduces visual accuracy; however, it caused a 41.7-point reduction in native
JSON validity, lower visual-challenge answerability, and much greater token
use. The parser recovered most responses, but parser-assisted recovery is not
equivalent to reliable native adherence to the model contract.

The correct conclusion is therefore metric-specific: thinking did not offer a
consistent accuracy/reliability/cost advantage sufficient to make it the
default. The remaining expanded Dermatology requests were stopped and are not
used as complete-cohort results.
