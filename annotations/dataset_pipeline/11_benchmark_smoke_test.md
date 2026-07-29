# Benchmark execution smoke test

## Objective

This stage validated the executable path from a frozen benchmark sample to
ranked metrics without requiring model weights or API credentials.

## Implementation

`src/benchmark/runner.py` renders prompts, loads normalized image bytes, calls a
shared backend protocol, parses JSON, and validates prediction count, ranks,
taxonomy IDs, and uniqueness.

`src/benchmark/metrics.py` calculates top-1, top-3, top-6, mean reciprocal
rank, macro top-1 F1, JSON validity, schema compliance, invalid-ID rate, and
duplicate-prediction rate.

`src/benchmark/smoke_test.py` uses a deterministic backend. It does not measure
model quality. It verifies that a real internal-test image can be resolved, the
prompt rendered, a six-item response validated, and all metrics calculated.

## Result

The smoke test passed. Image loading, JSON parsing, schema validation, taxonomy
validation, rank validation, and metric computation all completed.

The deterministic response intentionally ranks the known label first. Its
one-sample accuracy and MRR are therefore 1.0. Macro F1 is `1/21` because it
averages all active classes. These are implementation checks, not model
results.

## Outputs

- `outputs/smoke/visual_top_k_v1/smoke_predictions.jsonl`
- `outputs/smoke/visual_top_k_v1/smoke_metrics.json`
- `outputs/smoke/visual_top_k_v1/smoke_report.yaml`

## Reproduction

```bash
.venv/bin/python -m src.benchmark.smoke_test
```

## Next gate

Run a small real-model pilot on the validation split. Start with one local
multimodal model, record latency and raw responses, and verify its image
preprocessing and chat template before full internal and external evaluation.
