# Title

This repository is about the development of ISEP's thesis about Small Language Models.

## Folder Structure

- @doc/ is the folder that contains the dissertation which includes different chapters
- @src/ is the source folder that consists of coding

## Permanent Thesis Comparison Principle

The central thesis hypothesis is that a smaller multimodal model, after
domain-specific training, can outperform larger generalist models on a
controlled dermatology domain while requiring less memory, latency, and cost.

All future benchmark analyses, reports, figures, and dissertation tables must:

- place the specialized small-model checkpoints directly alongside the larger
  comparison models whenever they share a task and evaluation protocol;
- include the small model before training, the selected specialized checkpoint,
  and the relevant larger-model baselines so that the specialization gain and
  size comparison are both visible;
- report quality and efficiency together, including parameter count and, when
  available, VRAM, latency, throughput, GPU-hours, and monetary cost;
- mark non-executed or non-comparable tasks with a dash and explain why, rather
  than silently omitting the specialized model from the table;
- distinguish strictly paired comparisons from contextual comparisons that use
  different datasets, prompts, releases, inference profiles, or judging
  protocols;
- avoid the universal claim that small models are always better: conclusions
  must remain scoped to the evaluated domain, task, data, and protocol.

## Permanent Efficiency and Agentic Benchmark Principle

Accuracy alone is insufficient for the thesis claim. Every future controlled
evaluation of local specialized and larger models should, whenever technically
available, collect inference efficiency on the same hardware and runtime:

- time to first token, end-to-end latency p50/p95/p99, tokens per second,
  requests per second, GPU-seconds per request, and peak GPU memory;
- sampled GPU power integrated over inference, Wh per request, and Wh per
  correct answer;
- quality-versus-latency, quality-versus-memory, quality-versus-energy, and
  quality-versus-cost Pareto-frontier plots.

API models may be compared on observed latency, token use, and billed cost, but
must have energy marked unavailable unless the provider exposes a verifiable
measurement. Never infer provider energy from a local GPU proxy.

Agentic evaluation begins only after the model fine-tuning and distillation
stage is frozen. It must report task success, tool-selection and argument
accuracy, executable and invalid call rates, steps and tool calls per task,
loop and recovery rates, and tokens, latency, cost, and energy per successful
task. Tool-space scaling must be evaluated with progressively larger tool sets
(for example 5, 10, 25, 50, and 100 tools), holding tasks and tool distractor
sampling controlled across compared models.
