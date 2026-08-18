"""Strict configuration for versioned E3 teacher generation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from src.config.models import AzureModelConfig, load_model_config
from src.train.e3.terminology import (
    DermatologyTerminology,
    load_dermatology_terminology,
)


class _ConfigModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class E3CampaignConfig(_ConfigModel):
    id: str = Field(min_length=1)


class E3TeacherModelConfig(_ConfigModel):
    config: str = Field(min_length=1)
    required_model_id: str = Field(min_length=1)
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"]
    structured_output_mode: Literal["json_schema"]


class E3SelectionConfig(_ConfigModel):
    strategy: Literal["stratified_round_robin"]
    seed: int
    pilot_samples: int = Field(gt=0)
    max_per_leakage_group: Literal[1]


class E3DatasetConfig(_ConfigModel):
    release_root: str = Field(min_length=1)
    release_manifest: str = Field(min_length=1)
    taxonomy: str = Field(min_length=1)
    config: Literal["diagnosis"]
    splits: Annotated[
        tuple[Literal["sft_train", "sft_dev"], ...],
        BeforeValidator(
            lambda value: tuple(value) if isinstance(value, list) else value
        ),
    ] = Field(min_length=1)
    selection: E3SelectionConfig

    @model_validator(mode="after")
    def _unique_splits(self) -> E3DatasetConfig:
        if len(self.splits) != len(set(self.splits)):
            raise ValueError("dataset.splits must be unique")
        return self


class E3PromptConfig(_ConfigModel):
    stage_a: str = Field(min_length=1)
    stage_b: str = Field(min_length=1)


class E3TerminologyConfig(_ConfigModel):
    resource: str = Field(min_length=1)
    required_lexicon_id: str = Field(min_length=1)


class E3GenerationRuntimeConfig(_ConfigModel):
    stage_a_max_output_tokens: int = Field(gt=0)
    stage_b_max_output_tokens: int = Field(gt=0)
    sequential: Literal[True]
    retries: Literal[0]
    stop_on_transport_error: bool


class E3IntegrityConfig(_ConfigModel):
    verify_selected_shard_sha256: bool
    verify_image_sha256: bool


class E3TeacherGenerationConfig(_ConfigModel):
    schema_version: Literal[1]
    campaign: E3CampaignConfig
    model: E3TeacherModelConfig
    dataset: E3DatasetConfig
    prompts: E3PromptConfig
    terminology: E3TerminologyConfig
    generation: E3GenerationRuntimeConfig
    integrity: E3IntegrityConfig
    config_path: Path
    project_root: Path

    def path(self, value: str) -> Path:
        """Resolve one project-relative path without accepting escape paths."""

        candidate = (self.project_root / value).resolve()
        try:
            candidate.relative_to(self.project_root)
        except ValueError:
            raise ValueError(f"E3 path escapes the project root: {value}") from None
        return candidate

    @property
    def config_sha256(self) -> str:
        return _sha256_file(self.config_path)

    def load_teacher_model(self) -> AzureModelConfig:
        """Load and verify the exact teacher runtime selected for E3."""

        model = load_model_config(
            self.path(self.model.config),
            root=self.project_root,
        )
        if model.model.id != self.model.required_model_id:
            raise ValueError(
                "E3 teacher model ID mismatch: "
                f"expected {self.model.required_model_id!r}, got {model.model.id!r}"
            )
        if model.generation.reasoning_effort != self.model.reasoning_effort:
            raise ValueError(
                "E3 reasoning effort mismatch between experiment and model config"
            )
        if (
            self.model.structured_output_mode
            not in model.capabilities.structured_output_modes
        ):
            raise ValueError("E3 teacher lacks the required JSON-schema capability")
        profile = model.backend.active_profile
        if profile.api_style != "responses":
            raise ValueError("E3 GPT teacher must use the Responses API")
        if not isinstance(model, AzureModelConfig):
            raise ValueError("E3 GPT teacher must use an API model configuration")
        return model

    def load_terminology(self) -> DermatologyTerminology:
        """Load and verify the exact frozen Stage-A terminology resource."""

        terminology = load_dermatology_terminology(
            self.path(self.terminology.resource)
        )
        if terminology.lexicon_id != self.terminology.required_lexicon_id:
            raise ValueError(
                "E3 terminology ID mismatch: "
                f"expected {self.terminology.required_lexicon_id!r}, "
                f"got {terminology.lexicon_id!r}"
            )
        return terminology


def load_e3_teacher_generation_config(
    path: str | Path,
    *,
    root: Path | None = None,
) -> E3TeacherGenerationConfig:
    """Load the frozen E3 campaign YAML and resolve its project identity."""

    project_root = (root or Path(__file__).resolve().parents[3]).resolve()
    raw_path = Path(path)
    config_path = (
        raw_path.resolve()
        if raw_path.is_absolute()
        else (project_root / raw_path).resolve()
    )
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"E3 generation config must be an object: {config_path}")
    return E3TeacherGenerationConfig.model_validate(
        {
            **document,
            "config_path": config_path,
            "project_root": project_root,
        }
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
