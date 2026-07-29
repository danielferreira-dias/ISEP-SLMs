# ISEP Small Multimodal Language Models

This repository supports an ISEP thesis investigating whether a small
multimodal language model specialized with dermatology data can approach the
performance of larger teacher models.

The current codebase contains:

- normalized dermatology dataset manifests and documented source datasets;
- frozen internal and external evaluation cohorts;
- visual Top-K, paired confusion-set, and evidence-grounded diagnosis
  benchmarks;
- typed YAML model and benchmark configurations;
- local vLLM, Azure Chat Completions, and Azure Responses inference paths;
- deterministic metrics, durable results, and hash-checked resume support.

Start with the [benchmark pipeline](src/benchmark/README.md) for setup,
backend compatibility, dry-run validation, execution commands, reasoning
capture, and result interpretation. Dataset organization is described in
[the dataset configuration guide](configs/datasets/README.md).

This is research software and must not be used for clinical diagnosis or
patient-care decisions.
