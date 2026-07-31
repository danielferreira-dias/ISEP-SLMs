"""Local-model facade backed by a vLLM OpenAI-compatible server."""

from __future__ import annotations

import os
from typing import Any

from src.config.models import LocalModelConfig
from src.inference.base import (
    InferenceBackend,
    InferenceRequest,
    InferenceResult,
    PreflightResult,
)
from src.inference.vllm import VllmBackend


class LocalBackend(InferenceBackend):
    """Adapt normalized local model configuration to :class:`VllmBackend`.

    vLLM runs as a separate process, so importing this module does not import
    vLLM, PyTorch, Transformers, or the OpenAI SDK.
    """

    def __init__(
        self,
        config: LocalModelConfig,
        client: Any | None = None,
        *,
        base_url: str | None = None,
        health_probe: Any | None = None,
        reasoning_capture: str = "available",
        use_json_schema: bool = False,
    ) -> None:
        self.config = config
        profile = _active_profile(config)
        base_url_env = (
            _optional_attr(profile, "base_url_env")
            or _optional_attr(config, "base_url_env")
            or "VLLM_BASE_URL"
        )
        configured_base_url = (
            base_url
            or _optional_attr(profile, "base_url")
            or _optional_attr(config, "base_url")
            or os.environ.get(base_url_env)
            or "http://127.0.0.1:8000/v1"
        )
        reasoning = _optional_attr(config, "reasoning")
        configured_capture = (
            reasoning_capture
            if reasoning_capture != "available"
            else _optional_attr(config, "reasoning_capture")
            or _optional_attr(reasoning, "capture")
            or _optional_attr(reasoning, "capture_mode")
            or "full"
        )
        chat_template_kwargs = _optional_attr(
            reasoning,
            "chat_template_kwargs",
        )
        request_model = (
            _value_from_env(
                _optional_attr(profile, "model_env")
            )
            or _optional_attr(config, "request_model")
            or _optional_attr(config, "repo_id")
            or _nested_attr(
                config,
                "source",
                "repo_id",
                default=None,
            )
            or config.model_id
        )
        self._backend = VllmBackend(
            model_id=config.model_id,
            request_model=request_model,
            base_url=configured_base_url,
            api_key_env=_optional_attr(
                profile,
                "api_key_env",
            )
            or _optional_attr(
                config,
                "api_key_env",
                default="VLLM_API_KEY",
            ),
            client=client,
            health_probe=health_probe,
            generation=config.generation,
            reasoning_capture=configured_capture,
            embedded_reasoning_parser=_optional_attr(
                reasoning,
                "content_parser",
            ),
            use_json_schema=use_json_schema,
            chat_template_kwargs=chat_template_kwargs,
            thinking_control=_optional_attr(
                profile,
                "thinking_control",
            ),
            timeout_seconds=(
                _optional_attr(
                    profile,
                    "request_timeout_seconds",
                    default=300.0,
                )
                or 300.0
            ),
            supports_system_role=(
                _nested_attr(
                    config,
                    "model",
                    "family",
                    default=None,
                )
                != "medgemma"
            ),
        )

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def client(self) -> Any:
        return self._backend.client

    def complete(self, request: InferenceRequest) -> InferenceResult:
        return self._backend.complete(request)

    async def acomplete(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        return await self._backend.acomplete(request)

    async def aclose(self) -> None:
        await self._backend.aclose()

    def preflight(self) -> PreflightResult:
        return self._backend.preflight()

    def require_ready(self) -> None:
        self._backend.require_ready()


def _optional_attr(
    value: Any,
    name: str,
    *,
    default: Any = None,
) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _nested_attr(
    value: Any,
    parent: str,
    child: str,
    *,
    default: Any = None,
) -> Any:
    nested = _optional_attr(value, parent)
    if nested is None:
        return default
    return _optional_attr(nested, child, default=default)


def _active_profile(config: Any) -> Any:
    backend = _optional_attr(config, "backend")
    return _optional_attr(backend, "active_profile")


def _value_from_env(env_name: Any) -> str | None:
    if not isinstance(env_name, str) or not env_name:
        return None
    return os.environ.get(env_name)
