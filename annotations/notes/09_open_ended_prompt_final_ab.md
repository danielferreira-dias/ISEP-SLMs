# Final 50-case open-ended prompt A/B

Date: 2026-08-02

## Controlled design

- model and judge: `gpt_5_6_luna`;
- judge prompt: 1.2.0;
- Validation cases: 50 identical task IDs;
- seed: 42;
- reasoning effort: high;
- A: example-free natural ranked prose prompt 1.1.0;
- B: prescriptive evidence-constrained prompt 1.2.1;
- no fallback judge.

Prompt 1.2.1 included the corrected rule that `Image limitations` may appear
only for a case-specific image-quality or framing problem, never merely
because photographs cannot support palpation.

## Results

| Metric | A: 1.1.0 | B: 1.2.1 | B − A |
| --- | ---: | ---: | ---: |
| Valid responses | 47/50 | 47/50 | 0 |
| Safety refusals | 3 | 3 | 0 |
| Top-1 accuracy | 32% | 26% | -6 pp |
| Top-3 accuracy | 44% | 44% | 0 pp |
| Mean reciprocal rank | 0.370 | 0.337 | -0.033 |
| Diagnosis correctness, 0–4 | 1.58 | 1.50 | -0.08 |
| Visual findings, 0–4 | 3.02 | 3.08 | +0.06 |
| Evidence grounding, 0–4 | 3.42 | 3.46 | +0.04 |
| Clinical rationale, 0–4 | 2.68 | 2.48 | -0.20 |
| Differential quality, 0–4 | 2.66 | 2.50 | -0.16 |
| Unsupported-claim rate | 16% | 16% | 0 pp |

The paired Top-1 discordances were four A-only correct cases and one B-only
correct case; twelve cases were correct under both prompts. Top-3 discordances
were balanced at two cases in each direction. The 6-point Top-1 difference is
not statistically conclusive at this sample size, but B provides no aggregate
advantage that offsets its lower rationale and differential scores.

The corrected limitation instruction worked: B used `Image limitations` in
4/47 valid responses, compared with 19/20 under prompt 1.2.0 in the earlier
acceptance run. All 47 valid B responses contained the requested visible-
findings and three explicit rank labels.

## Final decision

Freeze model prompt 1.1.0 with judge prompt 1.2.0 in ISEPDermaBench release
1.5.0. It preserves Top-3 performance, has higher Top-1, clinical-rationale,
and differential scores, and has the same unsupported-claim rate as the more
prescriptive alternative. Prompt 1.2.1 remains stored as an audited rejected
variant and must not be silently substituted during model comparisons.
