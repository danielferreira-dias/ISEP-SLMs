# Efficiency and agentic benchmark strategy

## Material Passport

- Origin Skill: `experiment-agent`
- Origin Mode: `plan`
- Origin Date: 2026-08-14
- Verification Status: `PLANNED`
- Version Label: `efficiency_agentic_benchmark_strategy_v1`
- Overall Confidence: `CAUTION`

## 1. Objective

The central thesis comparison cannot be reduced to diagnostic accuracy. A
specialized 4B model may be scientifically preferable to a larger model when it
achieves comparable or better clinical-task quality while using less memory,
time, energy, and money. Future evaluations must therefore treat quality and
efficiency as joint outcomes.

This note records two sequential stages:

1. add inference-efficiency measurements to the next controlled
   ISEPDermaBench and DermoBench campaign;
2. after fine-tuning and distillation are frozen, evaluate the selected model
   as an agent, including action-space scaling and efficiency per successful
   task.

Agentic experiments must not begin while model-level fine-tuning choices are
still changing. Otherwise model adaptation, retrieval, tool policy, and agent
orchestration become inseparable confounders.

## 2. Model-level efficiency metrics

For every local model that can be served on the same GPU, inference engine,
precision, image preprocessing, prompt set, and concurrency profile, collect:

| Metric | Definition and thesis relevance |
|---|---|
| Time to first token (TTFT) | Time from request submission to the first generated token; report p50, p95, and p99. |
| End-to-end latency | Time until the complete response; report p50, p95, and p99. |
| Time per output token (TPOT) | Decode latency after the first token; separates prefill from decoding. |
| Output tokens/second | Decode throughput per completed request. |
| Requests/second | System throughput at a frozen concurrency and batch policy. |
| GPU-seconds/request | GPU occupancy attributable to one request under the declared load profile. |
| Peak GPU memory | Maximum allocated or device memory during the measured window. |
| Input/output tokens | Necessary for interpreting latency, energy, and cost differences. |
| Monetary cost/request | Local GPU rental cost apportioned by runtime, or provider-billed API cost. |

Latency distributions must be computed from per-request observations, not from
one aggregate wall-clock duration. Warm-up requests must be separated from the
measured cohort. Cache state, batch/concurrency, retries, failures, prompt
length, image resolution, output cap, and stop conditions must be frozen and
reported.

## 3. Energy measurement

For local NVIDIA GPUs, sample board power through NVML throughout the complete
measurement window. Subtracting an idle baseline may be reported as a
sensitivity analysis, but raw board energy remains the primary reproducible
measurement.

For samples at timestamps \(t_i\), power \(P_i\), and elapsed intervals
\(\Delta t_i\), approximate energy as:

\[
E_{Wh} = \frac{\sum_i P_i \Delta t_i}{3600}.
\]

Report at least:

\[
EnergyPerQuery = \frac{E_{Wh}}{N_{requests}},
\qquad
EnergyPerCorrect = \frac{E_{Wh}}{N_{correct}}.
\]

`EnergyPerCorrect` is meaningful only when models receive the same task IDs,
task mixture, decoding budget, and failure policy. It must be reported beside
both total energy and accuracy so that a small denominator cannot be hidden.
For agentic experiments, use:

\[
EnergyPerSuccess = \frac{E_{Wh}}{N_{successful\ tasks}}.
\]

Provider-side energy for API models is unobservable unless the provider exposes
a verifiable measurement. API rows must therefore show energy as unavailable;
local NVML measurements must never be used as a proxy for a remote model.

## 4. Joint quality-efficiency analysis

The primary visualization is not a single accuracy ranking. Generate Pareto
frontiers for:

- diagnostic quality versus p50 and p95 latency;
- diagnostic quality versus peak VRAM;
- diagnostic quality versus Wh/query and Wh/correct answer;
- diagnostic quality versus monetary cost/query;
- diagnostic quality versus model parameters.

A model is Pareto-dominated when another model is at least as good on both
axes and strictly better on one. Report the complete frontier instead of
choosing an arbitrary scalar weighting. A larger model gaining a few accuracy
points at substantially higher latency, energy, or cost is a different result
from an unqualified accuracy victory.

The quality axis must match the benchmark task: Top-1/macro-F1 for
classification, grounded diagnostic success for structured evidence, or an
appropriately frozen judged score for open-ended responses. Do not combine
incompatible tasks into one frontier without a predeclared aggregation rule.

## 5. Minimum controlled protocol

The next local ISEPDermaBench or DermoBench evaluation should freeze:

- identical task IDs and order, with patient/leakage-group pairing preserved;
- one GPU model, driver, CUDA stack, inference engine, and server revision;
- model dtype and quantization condition, reported explicitly;
- image preprocessing and resolution;
- prompt, chat template, thinking mode, decoding profile, output limit, and
  retry policy;
- concurrency levels, recommended at minimum `1` for interactive latency and a
  fixed higher level for throughput;
- a declared warm-up count and at least three repeated measured passes when
  feasible;
- failure inclusion in both quality and efficiency denominators.

The comparison must include the Qwen 3.5 4B base model, the selected 4B
specialized checkpoint, and relevant larger local baselines. API models remain
contextual comparisons unless the transport and provider conditions are
equivalent.

## 6. Required artifacts

Each measured benchmark run should preserve, without clinical images in logs:

```text
metrics/per_request_efficiency.parquet
metrics/efficiency_summary.json
logs/nvml_samples.jsonl
tables/quality_efficiency.csv
tables/quality_efficiency.tex
figures/quality_latency_pareto.png
figures/quality_latency_pareto.svg
figures/quality_energy_pareto.png
figures/quality_energy_pareto.svg
figures/quality_cost_pareto.png
figures/quality_cost_pareto.svg
```

Per-request records should contain task/sample ID, request start, first-token
time, completion time, input/output token counts, status, retry count, and
correctness outcome. NVML samples should contain only timestamp, power,
utilization, memory, temperature, and device identity.

Old benchmark runs must not be assigned retroactive TTFT or energy values when
the necessary measurements were not captured. Existing durations and GPU logs
may be reported only under the semantics actually recorded.

## 7. Later agentic evaluation

After E2 structured training, E3 hard knowledge distillation, checkpoint
selection, and the final model-level comparison are frozen, introduce an
agentic research question:

> How does a specialized 4B model compare with larger models in task success,
> tool-use reliability, and efficiency as the available action space grows?

Required agent metrics:

| Metric | What it measures |
|---|---|
| Task success rate | Whether the task was actually completed. |
| Tool-selection accuracy | Whether the correct tool was selected. |
| Argument accuracy | Whether tool arguments were complete and correct. |
| Executable call rate | Whether emitted calls can be parsed and executed. |
| Invalid tool-call rate | Invented tools, invalid schemas, or impossible calls. |
| Steps/task | Planning efficiency, including failed steps. |
| Tool calls/task | Operational efficiency. |
| Loop rate | Repeated states or calls without progress. |
| Recovery rate | Successful recovery after an injected or natural error. |
| Tokens/successful task | Generation efficiency conditioned on success. |
| Latency/successful task | End-to-end operational efficiency. |
| Cost/successful task | Monetary efficiency. |
| Energy/successful task | Local energy efficiency. |

Task success remains the primary outcome; tool-call validity alone is not
success. Denominators, timeouts, maximum steps, retry policy, and partial-credit
rules must be frozen before comparing models.

## 8. Tool-space scaling experiment

Evaluate the same underlying tasks with progressively larger advertised tool
sets, initially:

```text
5, 10, 25, 50, and 100 tools
```

Only the number of plausible distractor tools should change. The correct tool,
task difficulty, prompt budget, schemas, timeout, and success evaluator must
remain fixed. Sample multiple distractor sets with frozen seeds so that one
fortunate tool inventory does not determine the result.

Compare at least:

- specialized 4B without retrieval;
- specialized 4B with a frozen tool-retrieval/router component;
- one relevant larger model under the same tool protocol.

Plot task success, invalid-call rate, latency/success, tokens/success, and
energy/success against tool count. This measures the practical action-space
limit of the 4B agent and whether retrieval shifts that limit.

## 9. Decision gates

1. Finish and freeze model-level E2/E3 experiments.
2. Add efficiency instrumentation before the next benchmark rerun; do not
   reconstruct missing measurements from old runs.
3. Run one small same-hardware pilot to validate TTFT, NVML integration, and
   per-request accounting.
4. Execute the paired local model comparison and generate Pareto plots.
5. Only then implement the agent harness and tool-space scaling experiment.
6. Keep model-level and agent-level claims separate in the dissertation.

## 10. Known limitations

- Board power is not full datacenter energy and excludes cooling and host
  infrastructure unless separately measured.
- Batched inference complicates attribution of GPU-seconds and energy to one
  request; allocation rules must be declared.
- Latency depends on load, engine, prompt length, and transport, not only model
  size.
- API latency includes network and provider scheduling; API energy is usually
  unavailable.
- Energy per correct answer can look artificially favorable on easier or
  imbalanced task mixtures.
- Agent benchmarks are sensitive to tool descriptions, schemas, timeout,
  evaluator design, and distractor sampling.
- A Pareto advantage on the evaluated workload does not establish universal
  deployment superiority.
