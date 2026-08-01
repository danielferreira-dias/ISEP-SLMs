# Open-ended diagnosis with a single blinded judge

## Objective

Add a fourth ISEPDermaBench task that measures whether a multimodal model can
write a useful clinical assessment without seeing a closed list of diseases
or being forced into JSON. This complements, rather than replaces, the three
deterministic benchmarks.

## Cohorts

The v1.1.0 release contains two leakage-safe, 21-class cohorts derived from
the frozen v1.0.0 internal data:

| Split | Cases | Groups | Classes | SKINCON references | SkinCAP references |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 100 | 100 | 21 | 84 | 82 |
| Internal Benchmark | 300 | 300 | 21 | 134 | 119 |

Validation is for prompt, parser, model-setting, and judge-pipeline dry runs.
Internal Benchmark remains sealed for the principal before/after comparison.
The two cohorts have no shared `leakage_group_id`.

## Evaluated-model contract

The model receives only the image and a frozen prompt. It must answer in
natural clinical prose, describe relevant visible findings, and make its
first, second, and third diagnoses explicit. The prompt contains a neutral
placeholder example showing the desired style. It does not expose the
21-class taxonomy, disease IDs, gold diagnosis, morphology references,
SkinCAP description, or a JSON schema.

The answer should contain a concise visible-evidence rationale, not hidden
chain-of-thought. Provider-returned reasoning may still be stored by the
normal benchmark transport for audit purposes, but it is never sent to the
judge or scored.

## Judge protocol

There is exactly one judge: `gpt_5_6_luna`, configured with high reasoning
effort. Each model response is judged independently. The judge receives:

- the same benchmark image;
- the correct diagnosis and ID;
- exact-match SKINCON morphology concepts when available;
- a secondary SkinCAP description when available;
- the final user-visible model response.

It never receives the model name, backend, provider reasoning, or another
model's response. SKINCON concepts are treated as incomplete positive
references; SkinCAP text is secondary because it may be diagnosis-conditioned
or contain non-visual claims. The judge must inspect the image rather than
matching only against those annotations.

## Metrics

The strict judge schema records the position of the reference diagnosis
(first, second, third, or absent), 0–4 scores for diagnosis correctness,
visible-findings correctness, evidence grounding, clinical-rationale quality,
and differential quality, plus unsupported claims and an overall verdict.

Aggregates include Top-1, Top-3, mean reciprocal rank, mean dimension scores,
unsupported-claim rate/count, model failures, and verdict distribution.
These are judge-dependent measurements. Comparisons are valid only when the
judge model revision, prompt, schema, reasoning setting, and benchmark release
remain frozen.

## Artifacts and execution

The benchmark task and its isolated references live in:

```text
data/benchmarks/ISEPDermaBench/
├── tasks/open_ended_diagnosis/
├── references/open_ended_diagnosis/
└── artifacts/
    ├── configs/open_ended_diagnosis.yaml
    ├── prompts/open_ended_diagnosis.yaml
    ├── judges/open_ended_diagnosis_judge.yaml
    └── schemas/open_ended_diagnosis_judge.schema.json
```

The editable source resources under
`src/benchmark/resources/open_ended_diagnosis/` are inputs to the release
builder and runtime implementation. They are copied and hashed into the
published release; they are not a separate benchmark dataset.

The model run writes the normal `predictions.jsonl` and `report.html`. The
separate `judge` command writes `judgments.jsonl`, `judge_metrics.json`,
`judge_manifest.yaml`, and `judge_report.html` into that run directory.
Predictions and judgments are experimental results and are not uploaded as
part of ISEPDermaBench.

## Limitations

One judge avoids voting ambiguity and cost multiplication but leaves results
sensitive to that judge's errors and preferences. No second judge or human
adjudication is used by design. The benchmark therefore reports a reproducible
single-judge protocol, not an independent clinical gold standard for prose
quality.
