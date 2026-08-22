# Training metrics

This package is the canonical home for metrics emitted or aggregated during
Student training:

- `contracts.py`: vendor-neutral scalar metric events and sinks;
- `trainer_events.py`: checkpoint contracts and Hugging Face Trainer scalar
  event bridge;
- `resources.py`: process, NVIDIA GPU, power, memory, and temperature sampling;
- `resource_metrics.py`: run-level duration, throughput, VRAM, RAM, energy, and
  checkpoint-size aggregation.

Task-quality metrics and inference-efficiency metrics remain under the
benchmark/evaluation packages because they have different denominators and
scientific protocols. Legacy imports under `src.train` are compatibility
facades only; new E3 code must import from `project.metrics`.
