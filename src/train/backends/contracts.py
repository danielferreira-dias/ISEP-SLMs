"""Typed contracts shared by training backends and execution code."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

QWEN35_4B_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

MetricValue = float | int
ParameterComponent = Literal[
    "vision",
    "attention",
    "mlp",
    "embedding",
    "language_other",
]


@dataclass(frozen=True, slots=True)
class ModelLoadSpec:
    """Describe an immutable base-model load operation."""

    model_id: str = "Qwen/Qwen3.5-4B"
    revision: str = QWEN35_4B_REVISION
    processor_id: str = "Qwen/Qwen3.5-4B"
    processor_revision: str = QWEN35_4B_REVISION
    dtype: Literal["bfloat16"] = "bfloat16"
    load_in_4bit: Literal[False] = False

    def __post_init__(self) -> None:
        """Reject floating or scientifically incompatible model revisions."""
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if not self.revision.strip() or self.revision in {"main", "latest"}:
            raise ValueError("revision must be an immutable commit")
        if self.model_id == "Qwen/Qwen3.5-4B" and self.revision != QWEN35_4B_REVISION:
            raise ValueError(
                "Qwen/Qwen3.5-4B must use the thesis-pinned revision "
                f"{QWEN35_4B_REVISION}"
            )
        if self.processor_id != self.model_id:
            raise ValueError("Processor and model repository IDs must match")
        if self.processor_revision != self.revision:
            raise ValueError("Processor and model revisions must match")
        if self.dtype != "bfloat16":
            raise ValueError("Only the declared bfloat16 recipe is supported")
        if self.load_in_4bit is not False:
            raise ValueError("QLoRA/4-bit loading is outside the E1 recipe")


@dataclass(frozen=True, slots=True)
class LoraSpec:
    """Define the controlled LoRA intervention used by E1."""

    finetune_vision_layers: bool
    finetune_language_layers: Literal[True] = True
    finetune_attention_modules: Literal[True] = True
    finetune_mlp_modules: Literal[True] = True
    rank: Literal[16] = 16
    alpha: Literal[16] = 16
    dropout: float = 0.0
    bias: Literal["none"] = "none"
    target_modules: Literal["all-linear"] = "all-linear"
    use_rslora: Literal[False] = False
    loftq_config: None = None

    def __post_init__(self) -> None:
        """Reject mutations that would invalidate the controlled ablation."""
        expected = (
            self.finetune_language_layers is True
            and self.finetune_attention_modules is True
            and self.finetune_mlp_modules is True
            and self.rank == 16
            and self.alpha == 16
            and self.dropout == 0.0
            and self.bias == "none"
            and self.target_modules == "all-linear"
            and self.use_rslora is False
            and self.loftq_config is None
        )
        if not expected:
            raise ValueError(
                "E1 permits only r16/alpha16/dropout0 all-linear LoRA; "
                "the vision flag is the sole experimental difference"
            )


@dataclass(frozen=True, slots=True)
class TrainerSpec:
    """Hold explicit SFT settings without importing TRL."""

    output_dir: Path
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 3.0
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 100
    seed: int = 3407
    max_length: int | None = None
    max_steps: int = -1
    dataset_num_proc: int = 1

    def __post_init__(self) -> None:
        """Validate numeric settings before any CUDA allocation."""
        positive_ints = {
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "dataset_num_proc": self.dataset_num_proc,
        }
        invalid = [name for name, value in positive_ints.items() if value <= 0]
        if invalid:
            raise ValueError(f"Trainer fields must be positive: {', '.join(invalid)}")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or a positive integer")
        if self.max_length is not None and self.max_length <= 0:
            raise ValueError("max_length must be positive when provided")


@dataclass(frozen=True, slots=True)
class FineTuneRequest:
    """Bundle datasets and immutable scientific settings for one fit."""

    model: ModelLoadSpec
    lora: LoraSpec
    trainer: TrainerSpec
    train_dataset: object
    eval_dataset: object


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    """Record the validated CUDA runtime used for a fit or prediction."""

    torch_version: str
    cuda_version: str
    device_name: str
    device_count: int
    bf16_supported: bool
    total_memory_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class TrainableParameter:
    """Describe one trainable tensor without retaining the tensor itself."""

    name: str
    component: ParameterComponent
    count: int


@dataclass(frozen=True, slots=True)
class TrainableParameterManifest:
    """Summarize all trainable parameters by tensor and component."""

    parameters: tuple[TrainableParameter, ...]
    total_trainable: int
    by_component: dict[ParameterComponent, int]


@dataclass(frozen=True, slots=True)
class MetricEvent:
    """Represent one scalar observation emitted during training."""

    name: str
    value: MetricValue
    step: int
    epoch: float | None = None
    timestamp_utc: str | None = None

    def __post_init__(self) -> None:
        """Reject non-finite metrics and invalid event coordinates."""
        if not self.name:
            raise ValueError("MetricEvent name must not be empty")
        if not math.isfinite(float(self.value)):
            raise ValueError("MetricEvent value must be finite")
        if self.step < 0:
            raise ValueError("MetricEvent step must be non-negative")
        if self.epoch is not None and not math.isfinite(self.epoch):
            raise ValueError("MetricEvent epoch must be finite when present")


@dataclass(frozen=True, slots=True)
class CheckpointEvent:
    """Describe a checkpoint just after the trainer has persisted it."""

    path: Path
    global_step: int
    epoch: float | None


@dataclass(frozen=True, slots=True)
class BackendFitResult:
    """Return the durable outputs of one backend fit."""

    global_step: int
    training_loss: float | None
    metrics: dict[str, MetricValue]
    checkpoints: tuple[CheckpointEvent, ...]
    final_adapter_dir: Path
    trainable_parameters: TrainableParameterManifest
    runtime: RuntimeInfo

    def __post_init__(self) -> None:
        """Reject incomplete or non-finite backend summaries."""
        if self.global_step < 0:
            raise ValueError("Backend global_step must be non-negative")
        if self.training_loss is not None and not math.isfinite(self.training_loss):
            raise ValueError("Backend training_loss must be finite")
        invalid = [
            name
            for name, value in self.metrics.items()
            if not math.isfinite(float(value))
        ]
        if invalid:
            raise ValueError(
                "Backend metrics must be finite: " + ", ".join(sorted(invalid))
            )


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Contain backend-owned model state required for inference."""

    model: object
    processor: object
    runtime: RuntimeInfo
    checkpoint_path: Path | None


@dataclass(frozen=True, slots=True)
class PredictionSample:
    """Provide one already-selected image and closed-label prompt."""

    sample_id: str
    image: object
    prompt: str


@dataclass(frozen=True, slots=True)
class GenerationSpec:
    """Define deterministic label generation settings."""

    max_new_tokens: int = 32

    def __post_init__(self) -> None:
        """Reject a non-positive output budget."""
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


@dataclass(frozen=True, slots=True)
class BackendPrediction:
    """Store one raw deterministic model answer."""

    sample_id: str
    text: str


class MetricSink(Protocol):
    """Receive scalar metrics without depending on a tracking vendor."""

    def write(self, event: MetricEvent) -> None:
        """Persist one scalar metric event."""

    def close(self) -> None:
        """Flush resources held by the sink."""


class CheckpointObserver(Protocol):
    """Receive checkpoint events while a fit is still running."""

    def on_checkpoint(self, event: CheckpointEvent) -> None:
        """Record a newly persisted checkpoint."""


class FineTuningBackend(Protocol):
    """Backend boundary implemented by Unsloth and CPU test doubles."""

    @property
    def name(self) -> str:
        """Return a stable backend identifier."""

    def validate_runtime(self) -> RuntimeInfo:
        """Fail unless a compatible single-node NVIDIA runtime exists."""

    def fit(
        self,
        request: FineTuneRequest,
        *,
        metric_sink: MetricSink,
        checkpoint_observer: CheckpointObserver,
        resume_from_checkpoint: Path | None = None,
    ) -> BackendFitResult:
        """Fit one LoRA adapter, optionally resuming a validated checkpoint."""

    def load_base(self, model: ModelLoadSpec) -> LoadedCheckpoint:
        """Load the pinned base model for the pre-update baseline."""

    def load_checkpoint(
        self,
        *,
        model: ModelLoadSpec,
        checkpoint_path: Path,
    ) -> LoadedCheckpoint:
        """Load one saved adapter over the pinned base model."""

    def predict(
        self,
        loaded: LoadedCheckpoint,
        samples: Sequence[PredictionSample],
        *,
        generation: GenerationSpec | None = None,
    ) -> tuple[BackendPrediction, ...]:
        """Generate deterministic label responses for ordered samples."""

    def release(self, loaded: LoadedCheckpoint) -> None:
        """Release GPU memory owned by a loaded inference checkpoint."""
