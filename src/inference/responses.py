"""Azure/OpenAI Responses API backend for multimodal benchmark requests."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from src.inference.base import (
    InferenceBackend,
    InferenceConfigurationError,
    InferenceRequest,
    InferenceResult,
    InferenceTransportError,
    TokenUsage,
    build_reasoning_trace,
    extract_text,
    image_data_url,
    merge_generation,
    read_field,
    safe_optional_int,
    validate_reasoning_capture,
)


class AzureResponsesBackend(InferenceBackend):
    """Use an Azure OpenAI deployment through the Responses API."""

    def __init__(
        self,
        *,
        model_id: str,
        deployment_name: str | None = None,
        endpoint: str | None = None,
        endpoint_env: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        api_version: str | None = None,
        client: Any | None = None,
        generation: Any | None = None,
        reasoning_capture: str = "none",
        use_json_schema: bool = False,
        reasoning_summary: str = "auto",
        supports_sampling_parameters: bool = False,
        timeout_seconds: float = 300.0,
    ) -> None:
        self._model_id = model_id
        self.deployment_name = deployment_name or model_id
        self._endpoint = endpoint
        self._endpoint_env = endpoint_env
        self._api_key = api_key
        self._api_key_env = api_key_env
        self.api_version = api_version
        self._client = client
        self.default_generation = generation
        self.reasoning_capture = validate_reasoning_capture(
            reasoning_capture
        )
        self.use_json_schema = use_json_schema
        self.reasoning_summary = reasoning_summary
        self.supports_sampling_parameters = (
            supports_sampling_parameters
        )
        self.timeout_seconds = timeout_seconds

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def complete(self, request: InferenceRequest) -> InferenceResult:
        payload = self._build_payload(request)
        try:
            response = self.client.responses.create(**payload)
        except Exception as error:
            provider_detail = _provider_error_detail(error)
            detail_suffix = (
                f" ({provider_detail})" if provider_detail else ""
            )
            raise InferenceTransportError(
                f"Responses API request failed for model "
                f"{self.model_id!r}{detail_suffix}"
            ) from None
        return self._parse_response(response, request)

    def _build_client(self) -> Any:
        endpoint = self._resolve_secret_or_endpoint(
            direct=self._endpoint,
            env_name=self._endpoint_env,
            label="endpoint",
        )
        api_key = self._resolve_secret_or_endpoint(
            direct=self._api_key,
            env_name=self._api_key_env,
            label="API key",
        )
        try:
            from openai import AzureOpenAI, OpenAI
        except ImportError:
            raise InferenceConfigurationError(
                "The optional 'openai' package is required for Azure "
                "Responses inference"
            ) from None
        if self.api_version:
            return AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=self.api_version,
                timeout=self.timeout_seconds,
            )
        base_url = endpoint.rstrip("/")
        if not base_url.endswith("/openai/v1"):
            base_url = f"{base_url}/openai/v1"
        return OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=self.timeout_seconds,
        )

    def _resolve_secret_or_endpoint(
        self,
        *,
        direct: str | None,
        env_name: str | None,
        label: str,
    ) -> str:
        value = direct
        if value is None and env_name:
            value = os.environ.get(env_name)
        if value:
            return value
        if env_name:
            raise InferenceConfigurationError(
                f"Environment variable {env_name!r} is required for "
                f"the Azure {label}"
            )
        raise InferenceConfigurationError(
            f"An Azure {label} must be configured"
        )

    def _build_payload(
        self,
        request: InferenceRequest,
    ) -> dict[str, Any]:
        generation = merge_generation(
            self.default_generation,
            request.generation,
        )
        payload: dict[str, Any] = {
            "model": self.deployment_name,
            "instructions": request.system_prompt,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": image_data_url(
                                request.image_bytes,
                                request.image_mime_type,
                            ),
                        },
                        {
                            "type": "input_text",
                            "text": request.user_prompt,
                        },
                    ],
                }
            ],
        }
        max_output_tokens = generation.get("max_output_tokens")
        if max_output_tokens is None:
            max_output_tokens = generation.get("max_new_tokens")
        if max_output_tokens is None:
            max_output_tokens = generation.get("max_tokens")
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens

        # Some Responses deployments accept sampling controls while reasoning
        # models may reject them. Only explicit request overrides are sent.
        if self.supports_sampling_parameters:
            requested_generation = merge_generation(
                None,
                request.generation,
            )
            for field_name in ("temperature", "top_p"):
                value = requested_generation.get(field_name)
                if value is not None:
                    payload[field_name] = value

        if self.use_json_schema:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "benchmark_response",
                    "strict": True,
                    "schema": dict(request.schema),
                }
            }
        reasoning_options: dict[str, Any] = {}
        reasoning_effort = generation.get("reasoning_effort")
        if reasoning_effort is not None:
            reasoning_options["effort"] = reasoning_effort
        if self.reasoning_capture in {"full", "summary"}:
            reasoning_options["summary"] = self.reasoning_summary
        if reasoning_options:
            payload["reasoning"] = reasoning_options
        return payload

    def _parse_response(
        self,
        response: Any,
        request: InferenceRequest,
    ) -> InferenceResult:
        final_text = extract_text(read_field(response, "output_text"))
        if final_text is None:
            final_text = _responses_final_text(
                read_field(response, "output", ())
            )
        usage = _responses_usage(read_field(response, "usage"))
        summary, summary_source = _responses_reasoning(response)
        # The Responses API exposes an official reasoning summary, not raw
        # chain of thought. In full mode, retain all reasoning information
        # actually returned by the provider: the same official summary.
        full = summary if self.reasoning_capture == "full" else None
        full_source = (
            summary_source if self.reasoning_capture == "full" else None
        )
        reasoning = build_reasoning_trace(
            mode=self.reasoning_capture,
            full_text=full,
            summary_text=summary,
            token_count=usage.reasoning_tokens,
            full_source=full_source,
            summary_source=summary_source,
        )
        status = read_field(response, "status")
        incomplete_details = read_field(response, "incomplete_details")
        incomplete_reason = _optional_string(
            read_field(incomplete_details, "reason")
        )
        truncated = (
            status == "incomplete"
            and incomplete_reason == "max_output_tokens"
        )
        provider_model = read_field(response, "model")
        metadata: dict[str, Any] = {}
        if isinstance(provider_model, str):
            metadata["provider_model"] = provider_model
        if incomplete_reason:
            metadata["incomplete_reason"] = incomplete_reason
        if truncated:
            metadata["truncated"] = True

        return InferenceResult(
            model_id=self.model_id,
            final_text=final_text or "",
            reasoning=reasoning,
            usage=usage,
            request_id=request.request_id,
            provider_response_id=_optional_string(
                read_field(response, "id")
            ),
            finish_reason=(
                "length" if truncated else _optional_string(status)
            ),
            metadata=metadata,
        )


def _provider_error_detail(error: Exception) -> str | None:
    """Return safe structured provider error fields without request metadata."""

    details = [f"type={type(error).__name__}"]
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        details.append(f"status={status_code}")

    body = getattr(error, "body", None)
    if isinstance(body, dict):
        provider_error = body.get("error", body)
        if isinstance(provider_error, dict):
            code = provider_error.get("code")
            if isinstance(code, str) and code:
                details.append(f"code={_compact_error_field(code)}")
            message = provider_error.get("message")
            if isinstance(message, str) and message:
                details.append(
                    f"message={_compact_error_field(message)}"
                )

    if len(details) == 1:
        message = getattr(error, "message", None)
        if isinstance(message, str) and message:
            details.append(f"message={_compact_error_field(message)}")

    cause = error.__cause__
    if cause is not None:
        details.append(f"cause={type(cause).__name__}")
        cause_message = str(cause)
        if cause_message:
            details.append(
                f"cause_message={_compact_error_field(cause_message)}"
            )

    return "; ".join(details) or None


def _compact_error_field(value: str, *, limit: int = 500) -> str:
    """Normalize one provider-controlled field for concise result records."""

    compact = " ".join(value.split())
    return compact[:limit]


def _responses_final_text(output: Any) -> str | None:
    if not isinstance(output, Sequence) or isinstance(
        output,
        (str, bytes, bytearray),
    ):
        return None
    text_parts: list[str] = []
    for item in output:
        if read_field(item, "type") != "message":
            continue
        content = read_field(item, "content", ())
        if not isinstance(content, Sequence) or isinstance(
            content,
            (str, bytes, bytearray),
        ):
            continue
        for part in content:
            if read_field(part, "type") not in {
                "output_text",
                "text",
            }:
                continue
            if (text := extract_text(part)) is not None:
                text_parts.append(text)
    return "\n".join(text_parts) or None


def _responses_reasoning(
    response: Any,
) -> tuple[str | None, str | None]:
    direct_reasoning = read_field(response, "reasoning")
    summary = extract_text(read_field(direct_reasoning, "summary"))
    summary_source = "reasoning.summary" if summary else None

    output = read_field(response, "output", ())
    if isinstance(output, Sequence) and not isinstance(
        output,
        (str, bytes, bytearray),
    ):
        for item in output:
            if read_field(item, "type") != "reasoning":
                continue
            item_summary = extract_text(read_field(item, "summary"))
            if summary is None and item_summary:
                summary = item_summary
                summary_source = "output.reasoning.summary"

    return summary, summary_source


def _responses_usage(value: Any) -> TokenUsage:
    input_tokens = safe_optional_int(read_field(value, "input_tokens"))
    output_tokens = safe_optional_int(read_field(value, "output_tokens"))
    total_tokens = safe_optional_int(read_field(value, "total_tokens"))
    details = read_field(value, "output_tokens_details")
    reasoning_tokens = safe_optional_int(
        read_field(details, "reasoning_tokens")
    )
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
