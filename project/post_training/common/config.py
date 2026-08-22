"""Strict project configuration contracts for post-training stages.

This module deliberately keeps the student identity separate from the Stage C
training recipe.  The separation makes it impossible to hide dataset or
optimizer drift inside the reusable base-model YAML.
"""

from __future__ import annotations

import hashlib
import json
import re
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

QWEN35_4B_REPO = "Qwen/Qwen3.5-4B"
QWEN35_4B_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

E3_DATASET_REPO = "danielfdias98/ISEPDistillDataset"
E3_TRAIN_REVISION = "4437aff671af4f4e32a2ebf006fdd3f4e72dea4f"
E3_DEV_REVISION = "b215f0474e4931b5951da768e79a0d579d26919d"

E3_TRAIN_TASK_COUNTS: dict[str, int] = {
    "caption": 6312,
    "diagnosis": 6312,
    "grounded_differential": 6127,
    "morphology": 6312,
    "request_new_image": 21,
}


class PostTrainingConfigError(ValueError):
    """Raised when a post-training YAML violates the frozen contract."""


def _path_from_yaml(value: object) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value)
    raise ValueError("Expected a non-empty filesystem path")


StrictPath = Annotated[Path, BeforeValidator(_path_from_yaml)]


class StrictConfigModel(BaseModel):
    """Immutable Pydantic base that rejects every unknown key."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    def canonical_json(self) -> str:
        """Return a stable JSON representation suitable for manifests."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )

    @property
    def sha256(self) -> str:
        """Hash the complete validated contract, not the source formatting."""

        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def as_manifest(self) -> dict[str, object]:
        """Return a JSON-serializable manifest fragment with its digest."""

        payload = self.model_dump(mode="json")
        return {"sha256": self.sha256, "config": payload}


class StudentModelRecipe(StrictConfigModel):
    """Pinned official multimodal base model."""

    id: Literal["Qwen/Qwen3.5-4B"]
    revision: Literal["851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"]
    link_unsloth_qwen: str = Field(min_length=1)
    canonical_id: Literal["Qwen/Qwen3.5-4B"]
    link_qwen: str = Field(min_length=1)
    architecture: Literal["vision_language_model"]
    parameter_count: Literal["4B"]
    multimodal: Literal[True]
    max_seq_length: None
    fast_inference: Literal[False]
    full_finetuning: Literal[False]


class StudentPrecisionRecipe(StrictConfigModel):
    """Frozen BF16 LoRA load policy; no quantized fallback is admitted."""

    dtype: Literal["bfloat16"]
    load_in_16bit: Literal[True]
    load_in_4bit: Literal[False]
    load_in_8bit: Literal[False]
    bf16: Literal[True]
    fp16: Literal[False]
    allow_precision_fallback: Literal[False]


class StudentImagePreprocessingRecipe(StrictConfigModel):
    """Student-side image preprocessing declared by the frozen base recipe."""

    recommended_min_dimension_px: int = Field(gt=0)
    recommended_max_dimension_px: int = Field(gt=0)
    resize: Literal["max_edge_preserve_aspect_no_upscale"]
    correct_exif_orientation: Literal[True]

    @model_validator(mode="after")
    def _validate_dimensions(self) -> StudentImagePreprocessingRecipe:
        if self.recommended_min_dimension_px > self.recommended_max_dimension_px:
            raise ValueError("Minimum image dimension exceeds the maximum")
        return self


class StudentVisionRecipe(StrictConfigModel):
    """Vision/language modules admitted to the E3 LoRA intervention."""

    enabled: Literal[True]
    finetune_vision_layers: Literal[True]
    finetune_language_layers: Literal[True]
    finetune_attention_modules: Literal[True]
    finetune_mlp_modules: Literal[True]
    data_collator: Literal["UnslothVisionDataCollator"]
    preprocessing: StudentImagePreprocessingRecipe


class StudentLoraRecipe(StrictConfigModel):
    """Controlled r16/alpha16/dropout0 all-linear LoRA topology."""

    enabled: Literal[True]
    r: Literal[16]
    lora_alpha: Literal[16]
    lora_dropout: float
    bias: Literal["none"]
    target_modules: Literal["all-linear"]
    use_rslora: Literal[False]
    loftq_config: None
    use_gradient_checkpointing: Literal["unsloth"]
    random_state: Literal[42]

    @field_validator("lora_dropout")
    @classmethod
    def _validate_dropout(cls, value: float) -> float:
        if value != 0.0:
            raise ValueError("E3 LoRA dropout is frozen at zero")
        return value


class StudentReproducibilityRecipe(StrictConfigModel):
    """Experiment-wide deterministic student seed."""

    seed: Literal[42]


class StudentDefinition(StrictConfigModel):
    """Complete immutable student definition without stage settings."""

    name: Literal["qwen3_5_4b_dermatology"]
    backend: Literal["unsloth"]
    model: StudentModelRecipe
    precision: StudentPrecisionRecipe
    vision: StudentVisionRecipe
    lora: StudentLoraRecipe
    reproducibility: StudentReproducibilityRecipe


class StudentRecipe(StrictConfigModel):
    """Top-level student YAML contract."""

    student: StudentDefinition


class StageIdentity(StrictConfigModel):
    """Scientific identity of the primary Stage C run."""

    id: Literal["e3_qwen3_5_4b_sft"]
    kind: Literal["sft"]
    experiment: Literal["E3"]
    initialization: Literal["official_base"]
    backend: Literal["unsloth"]


class StudentReference(StrictConfigModel):
    """Reference from a stage recipe to the immutable student YAML."""

    config_path: StrictPath


class DatasetReference(StrictConfigModel):
    """Immutable Hugging Face dataset view and its audited cardinalities."""

    repo_id: Literal["danielfdias98/ISEPDistillDataset"]
    revision: str = Field(min_length=40, max_length=40)
    config: str = Field(min_length=1)
    split: str = Field(min_length=1)
    expected_rows: int = Field(gt=0)
    expected_task_counts: dict[str, int] = Field(default_factory=dict)

    @field_validator("revision")
    @classmethod
    def _validate_revision(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError("Dataset revision must be a lowercase 40-hex commit")
        return value

    @field_validator("expected_task_counts")
    @classmethod
    def _validate_task_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not key or count <= 0 for key, count in value.items()):
            raise ValueError("Expected task names and counts must be positive")
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def _validate_count_sum(self) -> DatasetReference:
        if self.expected_task_counts and (
            sum(self.expected_task_counts.values()) != self.expected_rows
        ):
            raise ValueError("Expected task counts do not sum to expected_rows")
        return self

    @property
    def identity(self) -> str:
        """Return an unambiguous immutable Hub view identifier."""

        return f"{self.repo_id}@{self.revision}:{self.config}/{self.split}"


class DatasetPair(StrictConfigModel):
    """Frozen E3 train view and human-only development view."""

    train: DatasetReference
    dev: DatasetReference

    @model_validator(mode="after")
    def _validate_frozen_views(self) -> DatasetPair:
        expected_train = {
            "repo_id": E3_DATASET_REPO,
            "revision": E3_TRAIN_REVISION,
            "config": "e3_multitask_sft_v1",
            "split": "sft_train",
            "expected_rows": 25084,
            "expected_task_counts": E3_TRAIN_TASK_COUNTS,
        }
        expected_dev = {
            "repo_id": E3_DATASET_REPO,
            "revision": E3_DEV_REVISION,
            "config": "diagnosis",
            "split": "sft_dev",
            "expected_rows": 1229,
            "expected_task_counts": {},
        }
        observed_train = self.train.model_dump(mode="python")
        observed_dev = self.dev.model_dump(mode="python")
        if observed_train != expected_train:
            drift = sorted(
                key
                for key, expected in expected_train.items()
                if observed_train.get(key) != expected
            )
            raise ValueError("Frozen E3 train view changed: " + ", ".join(drift))
        if observed_dev != expected_dev:
            drift = sorted(
                key
                for key, expected in expected_dev.items()
                if observed_dev.get(key) != expected
            )
            raise ValueError("Frozen sft_dev view changed: " + ", ".join(drift))
        return self


class SFTTrainingRecipe(StrictConfigModel):
    """Stage C trainer settings that preserve the E1/E2 comparison."""

    num_train_epochs: Literal[4]
    per_device_train_batch_size: int = Field(gt=0)
    per_device_eval_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    learning_rate: float
    optimizer: Literal["adamw_8bit"]
    weight_decay: float = Field(ge=0.0)
    warmup_ratio: float = Field(ge=0.0, lt=1.0)
    lr_scheduler_type: Literal["linear"]
    max_grad_norm: float = Field(gt=0.0)
    max_length: None
    dataset_num_proc: int = Field(gt=0)
    seed: Literal[42]

    @field_validator("learning_rate")
    @classmethod
    def _validate_learning_rate(cls, value: float) -> float:
        if value != 1.0e-4:
            raise ValueError("E3 learning_rate is frozen at 1e-4")
        return value


class SFTEvaluationRecipe(StrictConfigModel):
    """Development-only checkpoint-selection policy."""

    enabled: Literal[True]
    strategy: Literal["epoch"]
    selection_metric: Literal["macro_f1"]
    greater_is_better: Literal[True]
    tie_break_metrics: list[Literal["balanced_accuracy", "eval_loss", "earlier_epoch"]]
    external_benchmark_selection: Literal["forbidden"]

    @field_validator("tie_break_metrics")
    @classmethod
    def _validate_tie_break_order(cls, value: list[str]) -> list[str]:
        expected = ["balanced_accuracy", "eval_loss", "earlier_epoch"]
        if value != expected:
            raise ValueError("Checkpoint tie-break order is frozen")
        return value


class SFTCheckpointRecipe(StrictConfigModel):
    """Epoch checkpoint preservation policy."""

    output_dir: StrictPath
    save_strategy: Literal["epoch"]
    save_total_limit: Literal[4]
    save_lora_only: Literal[True]


class SFTLoggingRecipe(StrictConfigModel):
    """Vendor-neutral logging settings."""

    logging_steps: int = Field(gt=0)
    report_to: Literal["none"]


class E3SFTStageConfig(StrictConfigModel):
    """Complete frozen configuration for the primary E3 SFT arm."""

    schema_version: Literal[1]
    stage: StageIdentity
    student: StudentReference
    datasets: DatasetPair
    training: SFTTrainingRecipe
    evaluation: SFTEvaluationRecipe
    checkpointing: SFTCheckpointRecipe
    logging: SFTLoggingRecipe

    @property
    def dataset_contract_sha256(self) -> str:
        """Hash both immutable dataset references and their expected counts."""

        return self.datasets.sha256


def _load_yaml(path: Path, *, label: str) -> object:
    resolved = path.expanduser().resolve()
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PostTrainingConfigError(f"Cannot read {label} YAML: {resolved}") from exc
    except yaml.YAMLError as exc:
        raise PostTrainingConfigError(f"Invalid {label} YAML: {resolved}") from exc
    if not isinstance(raw, dict):
        raise PostTrainingConfigError(f"{label} YAML must contain one mapping")
    return raw


def load_student_recipe(path: str | Path) -> StudentRecipe:
    """Load and validate the immutable Qwen3.5-4B student recipe."""

    resolved = Path(path)
    try:
        return StudentRecipe.model_validate(_load_yaml(resolved, label="student"))
    except ValidationError as exc:
        raise PostTrainingConfigError(f"Invalid student recipe: {resolved}") from exc


def load_sft_stage_config(path: str | Path) -> E3SFTStageConfig:
    """Load and validate the frozen Stage C E3 SFT recipe."""

    resolved = Path(path)
    try:
        return E3SFTStageConfig.model_validate(
            _load_yaml(resolved, label="E3 SFT stage")
        )
    except ValidationError as exc:
        raise PostTrainingConfigError(
            f"Invalid E3 SFT stage recipe: {resolved}"
        ) from exc


def configuration_contract_sha256(
    stage: E3SFTStageConfig,
    student: StudentRecipe,
) -> str:
    """Bind the stage recipe and reusable student into one run identity."""

    payload = json.dumps(
        {
            "stage": stage.model_dump(mode="json"),
            "student": student.model_dump(mode="json"),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
