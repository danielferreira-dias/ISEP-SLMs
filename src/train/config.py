"""Strict configuration contract for reproducible Unsloth training runs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from src.train.domain import TrainingPhaseName, VisionTuningProfile

QWEN35_4B_REPO = "Qwen/Qwen3.5-4B"
QWEN35_4B_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


class TrainingConfigError(ValueError):
    """Raised when a training YAML does not satisfy the frozen contract."""


def _path_from_yaml(value: object) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value)
    raise ValueError("Expected a non-empty filesystem path")


StrictPath = Annotated[Path, BeforeValidator(_path_from_yaml)]


class StrictConfigModel(BaseModel):
    """Base for immutable configuration sections with no unknown keys."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class ExperimentConfig(StrictConfigModel):
    """Scientific identity of one controlled experiment."""

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    phase: TrainingPhaseName
    vision_profile: VisionTuningProfile

    @field_validator("phase", mode="before")
    @classmethod
    def _parse_phase(cls, value: object) -> TrainingPhaseName:
        if isinstance(value, TrainingPhaseName):
            return value
        if value == TrainingPhaseName.E1_LABEL.value:
            return TrainingPhaseName.E1_LABEL
        raise ValueError("Only the e1_label phase is implemented")

    @field_validator("vision_profile", mode="before")
    @classmethod
    def _parse_vision_profile(cls, value: object) -> VisionTuningProfile:
        if isinstance(value, VisionTuningProfile):
            return value
        try:
            return VisionTuningProfile(str(value))
        except ValueError as exc:
            raise ValueError("Unknown E1 vision profile") from exc


class SplitConfig(StrictConfigModel):
    """Deterministic group-safe split and development-panel settings."""

    train_ratio: float = 0.85
    dev_ratio: float = 0.15
    seed: int = 42
    secondary_feature_weight: float = 0.25
    panel_groups_per_class: int = Field(default=10, gt=0)
    panel_seed: int = 1042

    @model_validator(mode="after")
    def _validate_ratios(self) -> SplitConfig:
        if abs((self.train_ratio + self.dev_ratio) - 1.0) > 1e-9:
            raise ValueError("train_ratio and dev_ratio must sum to 1.0")
        if self.train_ratio <= 0.0 or self.dev_ratio <= 0.0:
            raise ValueError("Both split ratios must be positive")
        if not 0.0 <= self.secondary_feature_weight <= 1.0:
            raise ValueError("secondary_feature_weight must be in [0, 1]")
        return self


class ExpectedDatasetConfig(StrictConfigModel):
    """Cardinalities pinned to the audited ISEPDermData v1.3.0 pool."""

    image_count: int = 7541
    group_count: int = 5671
    class_count: int = 21
    source_count: int = 4
    train_image_count: int = 6312
    train_group_count: int = 4820
    dev_image_count: int = 1229
    dev_group_count: int = 851


class ImageConfig(StrictConfigModel):
    """Lossless deterministic image normalization for training."""

    max_edge_pixels: Literal[512] = 512
    mode: Literal["RGB"] = "RGB"
    correct_exif_orientation: Literal[True] = True
    preserve_aspect_ratio: Literal[True] = True
    allow_upscale: Literal[False] = False


class DatasetConfig(StrictConfigModel):
    """Pinned source pool and destination of its frozen assignments."""

    source_directory: StrictPath
    source_version: str = "1.3.0"
    hub_repo_id: str = "danielfdias98/ISEPDermData"
    hub_revision: str = "f7403f817376de0dea0048bd3c490e294a0ccaca"
    release_id: str = "e1_label_v1"
    release_directory: StrictPath
    source_release_file: StrictPath = Path("release.json")
    taxonomy_file: StrictPath = Path("metadata/taxonomy.json")
    split: SplitConfig = SplitConfig()
    expected: ExpectedDatasetConfig = ExpectedDatasetConfig()
    image: ImageConfig = ImageConfig()

    @model_validator(mode="after")
    def _validate_pinned_e1_source(self) -> DatasetConfig:
        if re.fullmatch(r"[0-9a-f]{40}", self.hub_revision) is None:
            raise ValueError("hub_revision must be a lowercase 40-hex commit")
        if self.expected.image_count == 7541:
            expected_identity = {
                "source_version": (self.source_version, "1.3.0"),
                "hub_repo_id": (
                    self.hub_repo_id,
                    "danielfdias98/ISEPDermData",
                ),
                "hub_revision": (
                    self.hub_revision,
                    "f7403f817376de0dea0048bd3c490e294a0ccaca",
                ),
                "release_id": (self.release_id, "e1_label_v1"),
            }
            changed = [
                name
                for name, (value, expected) in expected_identity.items()
                if value != expected
            ]
            if changed:
                raise ValueError(
                    "Production E1 dataset identity changed: " + ", ".join(changed)
                )
            if self.split != SplitConfig() or self.expected != ExpectedDatasetConfig():
                raise ValueError(
                    "Production E1 split, panel, and cardinalities are frozen"
                )
        return self


class ModelConfig(StrictConfigModel):
    """Immutable model and processor identities."""

    repo_id: Literal["Qwen/Qwen3.5-4B"] = "Qwen/Qwen3.5-4B"
    revision: Literal["851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"] = (
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    )
    processor_repo_id: Literal["Qwen/Qwen3.5-4B"] = "Qwen/Qwen3.5-4B"
    processor_revision: Literal["851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"] = (
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    )
    dtype: Literal["bfloat16"] = "bfloat16"
    load_in_4bit: Literal[False] = False
    trust_remote_code: Literal[False] = False
    enable_thinking: Literal[False] = False


class LoraConfig(StrictConfigModel):
    """Unsloth PEFT adapter configuration."""

    rank: Literal[16] = 16
    alpha: Literal[16] = 16
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    bias: Literal["none"] = "none"
    target_modules: Literal["all-linear"] = "all-linear"
    use_rslora: Literal[False] = False
    use_loftq: Literal[False] = False
    finetune_vision_layers: bool
    finetune_language_layers: Literal[True] = True
    finetune_attention_modules: Literal[True] = True
    finetune_mlp_modules: Literal[True] = True

    @field_validator("dropout")
    @classmethod
    def _fixed_dropout(cls, value: float) -> float:
        if value != 0.0:
            raise ValueError("E1 fixes LoRA dropout at 0.0")
        return value


class TrainerConfig(StrictConfigModel):
    """Fixed SFT optimization budget."""

    epochs: Literal[3] = 3
    micro_batch_size: Literal[2] = 2
    gradient_accumulation_steps: Literal[4] = 4
    learning_rate: float = Field(default=2e-4, gt=0.0)
    optimizer: Literal["adamw_8bit"] = "adamw_8bit"
    weight_decay: float = Field(default=0.001, ge=0.0)
    warmup_ratio: float = Field(default=0.05, ge=0.0, lt=1.0)
    scheduler: Literal["linear"] = "linear"
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    gradient_checkpointing: Literal["unsloth"] = "unsloth"
    bf16: Literal[True] = True
    packing: Literal[False] = False
    max_length: None = None
    early_stopping: Literal[False] = False
    seed: int = 3407
    logging_steps: Literal[10] = 10
    save_strategy: Literal["epoch"] = "epoch"

    @property
    def effective_batch_size(self) -> int:
        """Return the number of samples contributing to one optimizer step."""

        return self.micro_batch_size * self.gradient_accumulation_steps

    @model_validator(mode="after")
    def _validate_fixed_recipe(self) -> TrainerConfig:
        fixed = {
            "learning_rate": (self.learning_rate, 2e-4),
            "weight_decay": (self.weight_decay, 0.001),
            "warmup_ratio": (self.warmup_ratio, 0.05),
            "max_grad_norm": (self.max_grad_norm, 1.0),
        }
        changed = [
            name for name, (value, expected) in fixed.items() if value != expected
        ]
        if changed:
            raise ValueError(
                "E1 training recipe changed fixed fields: " + ", ".join(changed)
            )
        if self.seed not in {42, 3407, 2026}:
            raise ValueError("E1 seed must be one of 42, 3407, or 2026")
        return self


class EvaluationConfig(StrictConfigModel):
    """Checkpoint evaluation and final confirmation protocol."""

    evals_per_epoch: Literal[4] = 4
    generation_do_sample: Literal[False] = False
    generation_temperature: float = Field(default=0.0, ge=0.0)
    selection_metric: Literal["macro_f1"] = "macro_f1"
    tie_break_metrics: tuple[
        Literal["balanced_accuracy", "eval_loss", "earlier_epoch"], ...
    ] = ("balanced_accuracy", "eval_loss", "earlier_epoch")
    confirmation_seeds: tuple[int, ...] = (42, 3407, 2026)

    @field_validator("tie_break_metrics", "confirmation_seeds", mode="before")
    @classmethod
    def _tuples_from_yaml(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_fixed_evaluation(self) -> EvaluationConfig:
        if self.generation_temperature != 0.0:
            raise ValueError("E1 evaluation is greedy and fixes temperature at 0")
        if self.tie_break_metrics != (
            "balanced_accuracy",
            "eval_loss",
            "earlier_epoch",
        ):
            raise ValueError("E1 checkpoint tie-break order is frozen")
        if self.confirmation_seeds != (42, 3407, 2026):
            raise ValueError("E1 confirmation requires seeds 42, 3407, and 2026")
        return self


class CheckpointHubConfig(StrictConfigModel):
    """Private Hugging Face destination for resumable epoch checkpoints."""

    enabled: bool = False
    repo_id: Literal["danielfdias98/ISEP-training-checkpoints"] = (
        "danielfdias98/ISEP-training-checkpoints"
    )
    revision: Literal["main"] = "main"
    repo_type: Literal["model"] = "model"
    private: Literal[True] = True
    upload_smoke: Literal[False] = False


class ArtifactsConfig(StrictConfigModel):
    """Thesis-ready local and remote artifact policy."""

    output_directory: StrictPath = Path("outputs/training")
    tensorboard: Literal[True] = True
    save_png: Literal[True] = True
    save_svg: Literal[True] = True
    save_predictions_parquet: Literal[True] = True
    include_clinical_images: Literal[False] = False
    resource_sample_interval_seconds: float = Field(default=5.0, gt=0.0)
    thesis_export_directory: StrictPath | None = None
    checkpoint_hub: CheckpointHubConfig = Field(default_factory=CheckpointHubConfig)


class TrainingConfig(StrictConfigModel):
    """Complete validated configuration consumed by every CLI command."""

    schema_version: Literal[1] = 1
    experiment: ExperimentConfig
    dataset: DatasetConfig
    model: ModelConfig
    lora: LoraConfig
    trainer: TrainerConfig
    evaluation: EvaluationConfig = EvaluationConfig()
    artifacts: ArtifactsConfig = ArtifactsConfig()
    project_root: StrictPath = Field(default=Path("."), exclude=True)
    source_config_path: StrictPath | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _validate_scientific_condition(self) -> TrainingConfig:
        expected_vision = (
            self.experiment.vision_profile is VisionTuningProfile.UNSLOTH_ALL
        )
        if self.lora.finetune_vision_layers is not expected_vision:
            raise ValueError(
                "vision_profile and finetune_vision_layers describe different "
                "experimental conditions"
            )
        return self

    def resolve_path(self, path: Path) -> Path:
        """Resolve a repository-relative path without requiring CWD state."""

        return path if path.is_absolute() else (self.project_root / path).resolve()


def load_training_config(path: Path) -> TrainingConfig:
    """Load and strictly validate a training YAML file.

    Args:
        path: YAML file containing the full training contract.

    Returns:
        Immutable normalized training configuration.

    Raises:
        TrainingConfigError: If the file cannot be read or is invalid.
    """

    config_path = path.resolve()
    try:
        loaded: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        document = _string_mapping(loaded)
        document["project_root"] = _find_project_root(config_path)
        document["source_config_path"] = config_path
        return TrainingConfig.model_validate(document, strict=True)
    except (OSError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise TrainingConfigError(
            f"Invalid training configuration {config_path}: {exc}"
        ) from exc


def _string_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Training YAML root must be a mapping")
    document: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("Training YAML keys must be strings")
        document[key] = item
    return document


def _find_project_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()
