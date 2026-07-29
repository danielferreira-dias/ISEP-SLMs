from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

@dataclass(frozen=True, slots=True)
class GenerationConfig:
    temperature: float = 0.0
    max_new_tokens: int = 512
    seed: int | None = 42

@dataclass(frozen=True, slots=True)
class LocalModelConfig:
    backend_type: Literal["local"] = "local"
    model_id: str = ""
    device: str = "cuda"
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    generation: GenerationConfig = field(
        default_factory=GenerationConfig
    )

@dataclass(frozen=True, slots=True)
class AzureModelConfig:
    backend_type: Literal["azure"] = "azure"
    model_id: str = ""
    endpoint_env: str = ""
    api_key_env: str = ""
    deployment_name: str = ""
    api_version: str = ""
    generation: GenerationConfig = field(
        default_factory=GenerationConfig
    )