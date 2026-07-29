"""Backend construction from normalized runtime model configuration."""

from __future__ import annotations

from src.benchmark.runner import ModelBackend
from src.config.models import AzureModelConfig, LocalModelConfig
from src.inference.azure import AzureBackend
from src.inference.local import LocalBackend


ModelConfig = LocalModelConfig | AzureModelConfig


def create_backend(config: ModelConfig) -> ModelBackend:
    if config.backend_type == "local":
        return LocalBackend(config)
    if config.backend_type == "azure":
        return AzureBackend(config)
    raise ValueError(f"Unsupported backend type: {config.backend_type}")
