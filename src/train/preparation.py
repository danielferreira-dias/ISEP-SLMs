"""Dataset, backend-spec, and run-directory preparation for E1."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import overload

from src.train.artifacts import ArtifactStore
from src.train.backends import LoraSpec, TrainerSpec
from src.train.config import TrainingConfig
from src.train.data import build_lazy_phase_dataset
from src.train.domain import PreparedRelease, ReleaseSubset
from src.train.run_io import load_execution_profile, load_run_config
from src.train.scientific import config_hash


class PrefixDataset(Sequence[dict[str, object]]):
    """Deterministic prefix used only for the documented GPU smoke test."""

    def __init__(self, dataset: Sequence[dict[str, object]], length: int) -> None:
        """Expose at most ``length`` records from the ordered source dataset."""
        self._dataset = dataset
        self._length = min(len(dataset), length)

    def __len__(self) -> int:
        """Return the bounded smoke-test cardinality."""

        return self._length

    @overload
    def __getitem__(self, index: int) -> dict[str, object]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, object]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> dict[str, object] | list[dict[str, object]]:
        """Return one record or a materialized slice from the prefix."""

        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(self._length))]
        normalized = index if index >= 0 else self._length + index
        if normalized < 0 or normalized >= self._length:
            raise IndexError("Smoke dataset index out of range")
        return self._dataset[normalized]


def training_datasets(
    config: TrainingConfig,
    release: PreparedRelease,
    run_directory: Path,
    *,
    smoke: bool,
) -> tuple[Sequence[dict[str, object]], Sequence[dict[str, object]]]:
    """Build memory-mapped train/dev datasets and optional smoke prefixes."""

    cache = run_directory / "logs" / "hf_cache"
    train = build_lazy_phase_dataset(
        config,
        release,
        ReleaseSubset.SFT_TRAIN,
        cache_directory=cache,
    )
    dev = build_lazy_phase_dataset(
        config,
        release,
        ReleaseSubset.SFT_DEV,
        cache_directory=cache,
    )
    if smoke:
        return PrefixDataset(train, 64), PrefixDataset(dev, 32)
    return train, dev


def trainer_spec(
    config: TrainingConfig,
    run_directory: Path,
    *,
    train_size: int,
    smoke: bool,
) -> TrainerSpec:
    """Translate the fixed optimizer budget and derive four evals per epoch."""

    steps_per_epoch = math.ceil(train_size / config.trainer.effective_batch_size)
    eval_steps = max(
        1,
        math.ceil(steps_per_epoch / config.evaluation.evals_per_epoch),
    )
    return TrainerSpec(
        output_dir=run_directory / "checkpoints",
        per_device_train_batch_size=config.trainer.micro_batch_size,
        per_device_eval_batch_size=config.trainer.micro_batch_size,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        num_train_epochs=float(config.trainer.epochs),
        learning_rate=config.trainer.learning_rate,
        weight_decay=config.trainer.weight_decay,
        warmup_ratio=config.trainer.warmup_ratio,
        max_grad_norm=config.trainer.max_grad_norm,
        logging_steps=config.trainer.logging_steps,
        eval_steps=eval_steps,
        seed=config.trainer.seed,
        max_length=config.trainer.max_length,
        max_steps=30 if smoke else -1,
    )


def lora_spec(config: TrainingConfig) -> LoraSpec:
    """Translate the fixed, vision-controlled LoRA intervention."""

    return LoraSpec(
        finetune_vision_layers=config.lora.finetune_vision_layers,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        bias=config.lora.bias,
        target_modules=config.lora.target_modules,
        use_rslora=config.lora.use_rslora,
    )


def open_run_store(
    config: TrainingConfig,
    resume_from: Path | None,
    smoke: bool,
    requested_run_id: str | None,
) -> ArtifactStore:
    """Create a unique run or reopen the exact owner of a checkpoint."""

    if resume_from is not None:
        checkpoint = resume_from.resolve()
        if checkpoint.parent.name != "checkpoints":
            raise ValueError("Resume checkpoint must be inside a checkpoints directory")
        run_directory = checkpoint.parent.parent
        stored = load_run_config(run_directory)
        if config_hash(stored) != config_hash(config):
            raise ValueError("Resume config differs from the original run")
        expected_profile = "smoke" if smoke else "full"
        if load_execution_profile(run_directory) != expected_profile:
            raise ValueError("Cannot resume a smoke run as full training or vice versa")
        return ArtifactStore.at(run_directory)
    identifier = requested_run_id or _new_run_id(config, smoke)
    root = config.resolve_path(config.artifacts.output_directory)
    return ArtifactStore.create(root, config.experiment.id, identifier)


def _new_run_id(config: TrainingConfig, smoke: bool) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    kind = "smoke" if smoke else "run"
    return f"{kind}-{timestamp}-s{config.trainer.seed}-{config_hash(config)[:8]}"
