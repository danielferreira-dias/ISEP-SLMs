"""Backend construction from normalized runtime model configuration."""

from __future__ import annotations

from typing import Any

from src.config.models import AzureModelConfig, LocalModelConfig
from src.inference.azure import AzureBackend
from src.inference.base import InferenceBackend
from src.inference.local import LocalBackend
from src.inference.transformers import TransformersBackend


ModelConfig = LocalModelConfig | AzureModelConfig


def create_backend(
    config: ModelConfig,
    *,
    client: Any | None = None,
    reasoning_capture: str = "available",
    use_json_schema: bool = False,
) -> InferenceBackend:
    """Create a lazy inference backend without importing provider SDKs."""

    profile = _active_profile(config)
    engine = _attr(profile, "engine")
    if engine == "vllm":
        return LocalBackend(
            config,
            client=client,
            reasoning_capture=reasoning_capture,
            use_json_schema=use_json_schema,
        )
    if engine == "transformers":
        return TransformersBackend(
            config,
            reasoning_capture=reasoning_capture,
        )
    if engine in {"azure_openai", "openrouter", "vllm_endpoint"}:
        return AzureBackend(
            config,
            client=client,
            reasoning_capture=reasoning_capture,
            use_json_schema=use_json_schema,
        )

    backend_type = _attr(config, "backend_type")
    if backend_type == "local":
        return LocalBackend(
            config,
            client=client,
            reasoning_capture=reasoning_capture,
            use_json_schema=use_json_schema,
        )
    if backend_type in {"azure", "api"}:
        return AzureBackend(
            config,
            client=client,
            reasoning_capture=reasoning_capture,
            use_json_schema=use_json_schema,
        )
    raise ValueError(
        f"Unsupported inference backend engine: {engine or backend_type}"
    )


def _active_profile(config: Any) -> Any:
    backend = _attr(config, "backend")
    return _attr(backend, "active_profile")


def _attr(
    value: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)
