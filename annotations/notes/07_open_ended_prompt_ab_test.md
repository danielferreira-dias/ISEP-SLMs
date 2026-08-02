# Open-ended diagnosis prompt A/B test

Date: 2026-08-01

## Objective

Test whether removing the prose example from the evaluated-model prompt reduces
style anchoring without harming the open-ended dermatology task.

## Controlled design

- evaluated model: `gpt_5_6_luna`;
- judge: `gpt_5_6_luna`, judge prompt 1.2.0;
- split: ISEPDermaBench Validation;
- sample: 10 identical task IDs;
- seed: 42;
- A: model prompt 1.0.0 with the prose example;
- B: model prompt 1.1.0 without an example;
- no fallback judge;
- both arms were scored with the same updated judge and schema.

An earlier B run selected different cases because deterministic subset selection
includes the benchmark release hash. It was excluded. The final comparison used
`--task-ids-file` to force an exactly paired cohort.

## Results

| Metric | A: example | B: no example | Change |
| --- | ---: | ---: | ---: |
| Valid model responses | 10/10 | 10/10 | — |
| Judge coverage | 100% | 100% | — |
| Top-1 accuracy | 40% | 20% | -20 pp |
| Top-3 accuracy | 60% | 60% | 0 pp |
| Mean reciprocal rank | 0.500 | 0.383 | -0.117 |
| Diagnosis correctness, 0–4 | 2.2 | 1.9 | -0.3 |
| Visual findings, 0–4 | 3.2 | 3.3 | +0.1 |
| Evidence grounding, 0–4 | 3.2 | 3.5 | +0.3 |
| Clinical rationale, 0–4 | 2.9 | 2.5 | -0.4 |
| Differential quality, 0–4 | 2.9 | 2.9 | 0.0 |
| Unsupported claim rate | 60% | 30% | -30 pp |
| Mean unsupported claims | 0.7 | 0.4 | -0.3 |

## Style anchoring

| Phrase from the example | A | B |
| --- | ---: | ---: |
| `The image shows` | 9/10 | 0/10 |
| `ranks second` | 3/10 | 0/10 |
| `ranks third` | 3/10 | 0/10 |
| `most likely diagnosis because` | 1/10 | 0/10 |

Mean response length was 762.5 characters for A and 825.2 for B. Removing the
example therefore changed style rather than merely shortening responses.

## Decision

Keep prompt 1.1.0 as the current development candidate because it removes
strong template imitation, preserves Top-3 accuracy, improves evidence
grounding, and halves unsupported claims. Do not claim an overall diagnostic
improvement: Top-1 accuracy and clinical rationale were lower in this
ten-case run, and the sample is too small to distinguish a prompt effect from
stochastic model variation.

Before freezing the protocol, confirm A versus B on a larger fixed Validation
cohort, recommended at 50 cases. Use the exact same task-ID file and judge
version for both arms.
