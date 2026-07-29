"""Azure/provider API facade for chat-completions and Responses transports."""

from __future__ import annotations

import os
from typing import Any

from src.config.models import AzureModelConfig
from src.inference.base import (
    InferenceBackend,
    InferenceRequest,
    InferenceResult,
)
from src.inference.openai_compatible import (
    OpenAICompatibleChatBackend,
)
from src.inference.responses import AzureResponsesBackend


class AzureBackend(InferenceBackend):
    """Select the configured Azure/provider API contract.

    ``openai_compatible`` is suitable for Kimi-style chat endpoints.
    ``openai_responses`` uses the Responses API for Azure OpenAI deployments.
    Injected clients bypass environment validation, which keeps tests and
    offline configuration inspection independent of provider credentials.
    """

    def __init__(
        self,
        config: AzureModelConfig,
        client: Any | None = None,
        *,
        reasoning_capture: str = "available",
    ) -> None:
        self.config = config
        profile = _active_profile(config)
        engine = _attr(profile, "engine")
        raw_api_style = (
            _attr(profile, "api_style")
            or _attr(config, "api_style")
            or "responses"
        )
        api_style = {
            "chat_completions": "openai_compatible",
            "responses": "openai_responses",
        }.get(raw_api_style, raw_api_style)
        endpoint_env = (
            _attr(profile, "base_url_env")
            or _attr(profile, "endpoint_env")
            or _attr(config, "base_url_env")
            or _attr(config, "endpoint_env")
        )
        endpoint = os.environ.get(endpoint_env) if endpoint_env else None
        api_key_env = (
            _attr(profile, "api_key_env")
            or _attr(config, "api_key_env")
        )
        api_key = (
            os.environ.get(api_key_env) if api_key_env else None
        )
        reasoning = _attr(config, "reasoning")
        configured_capture = (
            reasoning_capture
            if reasoning_capture != "available"
            else _attr(config, "reasoning_capture")
            or _attr(reasoning, "capture")
            or _attr(reasoning, "capture_mode")
            or (
                "summary"
                if api_style == "openai_responses"
                else "full"
            )
        )
        if (
            api_style == "openai_responses"
            and configured_capture == "full"
        ):
            # The Responses API exposes official reasoning summaries but not
            # raw chain of thought. Reflect the level that can be retained.
            configured_capture = "summary"
        request_model = (
            _value_from_env(_attr(profile, "deployment_env"))
            or _value_from_env(_attr(profile, "model_env"))
            or (
                _attr(config, "deployment_name")
                if profile is None
                else None
            )
            or _attr(config, "request_model")
            or _attr(_attr(config, "source"), "model_name")
            or config.model_id
        )
        use_json_schema = bool(
            _attr(config, "use_json_schema", default=False)
        )

        if (
            api_style == "openai_compatible"
            or engine == "vllm_endpoint"
        ):
            self._backend: InferenceBackend = (
                OpenAICompatibleChatBackend(
                    model_id=config.model_id,
                    request_model=request_model,
                    base_url=endpoint,
                    base_url_env=endpoint_env,
                    api_key=api_key,
                    api_key_env=api_key_env,
                    client=client,
                    generation=config.generation,
                    reasoning_capture=configured_capture,
                    use_json_schema=use_json_schema,
                    chat_template_kwargs=_attr(
                        config,
                        "chat_template_kwargs",
                    ),
                )
            )
        elif api_style == "openai_responses":
            self._backend = AzureResponsesBackend(
                model_id=config.model_id,
                deployment_name=request_model,
                endpoint=endpoint,
                endpoint_env=endpoint_env,
                api_key=api_key,
                api_key_env=api_key_env,
                api_version=(
                    _value_from_env(
                        _attr(profile, "api_version_env")
                    )
                    or (
                        _attr(config, "api_version")
                        if profile is None
                        else None
                    )
                ),
                client=client,
                generation=config.generation,
                reasoning_capture=configured_capture,
                use_json_schema=use_json_schema,
                reasoning_summary=_attr(
                    config,
                    "reasoning_summary",
                    default="auto",
                ),
                supports_sampling_parameters=bool(
                    _attr(
                        config,
                        "supports_sampling_parameters",
                        default=False,
                    )
                ),
            )
        else:
            raise ValueError(f"Unsupported API style: {api_style}")

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def client(self) -> Any:
        return getattr(self._backend, "client")

    def complete(self, request: InferenceRequest) -> InferenceResult:
        return self._backend.complete(request)


def _attr(
    value: Any,
    name: str,
    *,
    default: Any = None,
) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _active_profile(config: Any) -> Any:
    backend = _attr(config, "backend")
    return _attr(backend, "active_profile")


def _value_from_env(env_name: Any) -> str | None:
    if not isinstance(env_name, str) or not env_name:
        return None
    return os.environ.get(env_name)
