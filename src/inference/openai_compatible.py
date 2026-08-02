"""OpenAI-compatible chat-completions transport for multimodal models."""

from __future__ import annotations

import inspect
import os
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass, replace
from typing import Any

from src.inference.base import (
    InferenceBackend,
    InferenceConfigurationError,
    InferenceRequest,
    InferenceRequestError,
    InferenceResult,
    InferenceSafetyRefusal,
    InferenceTransportError,
    ReasoningCaptureMode,
    TokenUsage,
    build_reasoning_trace,
    extract_text,
    image_data_url,
    is_safety_refusal,
    merge_generation,
    provider_error_details,
    provider_error_summary,
    provider_json_schema,
    read_field,
    safe_optional_int,
    validate_reasoning_capture,
)
from src.inference.reasoning_parsing import separate_embedded_reasoning


class OpenAICompatibleChatBackend(InferenceBackend):
    """Call a vLLM or provider-hosted chat-completions endpoint.

    The OpenAI SDK is imported only when a client was not injected. This keeps
    configuration inspection and unit tests usable on machines without the
    optional inference dependencies.
    """

    def __init__(
        self,
        *,
        model_id: str,
        request_model: str | None = None,
        base_url: str | None = None,
        base_url_env: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        client: Any | None = None,
        async_client: Any | None = None,
        generation: Any | None = None,
        reasoning_capture: str = "none",
        embedded_reasoning_parser: str | None = None,
        use_json_schema: bool = False,
        chat_template_kwargs: Any | None = None,
        thinking_control: str | None = None,
        supports_system_role: bool = True,
        timeout_seconds: float = 300.0,
        max_retries: int = 2,
        stream_responses: bool = False,
        image_first: bool = True,
        include_extended_sampling: bool = True,
    ) -> None:
        self._model_id = model_id
        self.request_model = request_model or model_id
        self._base_url = base_url
        self._base_url_env = base_url_env
        self._api_key = api_key
        self._api_key_env = api_key_env
        self._client = client
        self._async_client = async_client
        self.default_generation = generation
        self.reasoning_capture = validate_reasoning_capture(
            reasoning_capture
        )
        self.embedded_reasoning_parser = embedded_reasoning_parser
        self.use_json_schema = use_json_schema
        self.chat_template_kwargs = _mapping_values(
            chat_template_kwargs
        )
        if thinking_control not in {
            None,
            "chat_template",
            "reasoning_effort",
            "openrouter_reasoning",
        }:
            raise InferenceConfigurationError(
                "thinking_control must be 'chat_template', "
                "'reasoning_effort', or 'openrouter_reasoning'"
            )
        self.thinking_control = thinking_control
        self.supports_system_role = supports_system_role
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.stream_responses = stream_responses
        self.image_first = image_first
        self.include_extended_sampling = include_extended_sampling

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def client(self) -> Any:
        """Return the injected client or lazily construct an OpenAI client."""

        if self._client is None:
            self._client = self._build_client()
        return self._client

    @property
    def async_client(self) -> Any:
        """Return the injected client or lazily construct an AsyncOpenAI client."""

        if self._async_client is None:
            self._async_client = self._build_async_client()
        return self._async_client

    def complete(self, request: InferenceRequest) -> InferenceResult:
        payload = self._build_payload(request)
        if self.stream_responses:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        try:
            response = self.client.chat.completions.create(**payload)
        except Exception as error:
            details = provider_error_details(error)
            provider_detail = provider_error_summary(details)
            if is_safety_refusal(details):
                raise InferenceSafetyRefusal(
                    f"Chat-completions safety refusal for model "
                    f"{self.model_id!r} ({provider_detail})",
                    details=details,
                ) from None
            raise InferenceTransportError(
                f"Chat-completions request failed for model "
                f"{self.model_id!r} ({provider_detail})"
            ) from None
        if self.stream_responses:
            return self._parse_stream_response(response, request)
        return self._parse_response(response, request)

    async def acomplete(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        """Run chat completion with the native asynchronous OpenAI client."""

        payload = self._build_payload(request)
        if self.stream_responses:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        try:
            response = await self.async_client.chat.completions.create(
                **payload
            )
        except Exception as error:
            details = provider_error_details(error)
            provider_detail = provider_error_summary(details)
            if is_safety_refusal(details):
                raise InferenceSafetyRefusal(
                    f"Chat-completions safety refusal for model "
                    f"{self.model_id!r} ({provider_detail})",
                    details=details,
                ) from None
            raise InferenceTransportError(
                f"Chat-completions request failed for model "
                f"{self.model_id!r} ({provider_detail})"
            ) from None
        if self.stream_responses:
            return await self._parse_async_stream_response(
                response,
                request,
            )
        result = self._parse_response(response, request)
        return replace(
            result,
            metadata={
                **dict(result.metadata),
                "async_transport": True,
            },
        )

    async def aclose(self) -> None:
        client = self._async_client
        if client is None:
            return
        close = getattr(client, "close", None)
        if callable(close):
            close_result = close()
            if inspect.isawaitable(close_result):
                await close_result

    def _build_client(self) -> Any:
        base_url = self._resolve_value(
            direct=self._base_url,
            env_name=self._base_url_env,
            label="base URL",
        )
        api_key = self._resolve_value(
            direct=self._api_key,
            env_name=self._api_key_env,
            label="API key",
        )
        try:
            from openai import OpenAI
        except ImportError:
            raise InferenceConfigurationError(
                "The optional 'openai' package is required for "
                "OpenAI-compatible inference"
            ) from None
        return OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )

    def _build_async_client(self) -> Any:
        base_url = self._resolve_value(
            direct=self._base_url,
            env_name=self._base_url_env,
            label="base URL",
        )
        api_key = self._resolve_value(
            direct=self._api_key,
            env_name=self._api_key_env,
            label="API key",
        )
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise InferenceConfigurationError(
                "The optional 'openai' package is required for "
                "asynchronous OpenAI-compatible inference"
            ) from None
        return AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )

    def _resolve_value(
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
                f"the {label}"
            )
        raise InferenceConfigurationError(
            f"An inference {label} must be configured"
        )

    def _build_payload(
        self,
        request: InferenceRequest,
    ) -> dict[str, Any]:
        generation = merge_generation(
            self.default_generation,
            request.generation,
        )
        data_url = image_data_url(
            request.image_bytes,
            request.image_mime_type,
        )
        user_text = request.user_prompt
        messages: list[dict[str, Any]] = []
        if self.supports_system_role:
            messages.append(
                {
                    "role": "system",
                    "content": request.system_prompt,
                }
            )
        else:
            user_text = "\n\n".join(
                part
                for part in (
                    request.system_prompt.strip(),
                    request.user_prompt.strip(),
                )
                if part
            )
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            },
            {
                "type": "text",
                "text": user_text,
            },
        ]
        if not self.image_first:
            user_content.reverse()
        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )
        payload: dict[str, Any] = {
            "model": self.request_model,
            "messages": messages,
        }
        _apply_chat_generation(
            payload,
            generation,
            include_reasoning_effort=(
                self.thinking_control != "openrouter_reasoning"
            ),
            include_extended_sampling=self.include_extended_sampling,
        )
        extra_body = dict(payload.get("extra_body", {}))
        template_kwargs = dict(self.chat_template_kwargs)
        configured_template_kwargs = generation.get(
            "chat_template_kwargs"
        )
        template_kwargs.update(
            _mapping_values(configured_template_kwargs)
        )
        thinking_mode = generation.get("thinking_mode")
        if thinking_mode not in {None, "enabled", "disabled"}:
            raise InferenceRequestError(
                "thinking_mode must be 'enabled' or 'disabled'"
            )
        if (
            thinking_mode is not None
            and self.thinking_control == "chat_template"
        ):
            template_kwargs["thinking"] = thinking_mode == "enabled"
        elif (
            thinking_mode is not None
            and self.thinking_control == "reasoning_effort"
        ):
            payload["reasoning_effort"] = (
                "high" if thinking_mode == "enabled" else "none"
            )
        if self.thinking_control == "openrouter_reasoning":
            reasoning: dict[str, Any] = {}
            reasoning_effort = generation.get("reasoning_effort")
            if reasoning_effort is not None:
                reasoning["effort"] = reasoning_effort
            if thinking_mode is not None:
                reasoning["enabled"] = thinking_mode == "enabled"
            if reasoning:
                reasoning["exclude"] = False
                extra_body["reasoning"] = reasoning
        if template_kwargs:
            extra_body["chat_template_kwargs"] = template_kwargs
        if extra_body:
            payload["extra_body"] = extra_body
        if self.use_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "benchmark_response",
                    "strict": True,
                    "schema": provider_json_schema(request.schema),
                },
            }
        return payload

    def _parse_response(
        self,
        response: Any,
        request: InferenceRequest,
    ) -> InferenceResult:
        choices = read_field(response, "choices", ())
        if not choices:
            raise InferenceTransportError(
                f"Chat-completions response for model {self.model_id!r} "
                "did not contain a choice"
            )
        choice = choices[0]
        message = read_field(choice, "message")
        final_text = extract_text(read_field(message, "content"))
        if final_text is None:
            final_text = ""

        usage = _chat_usage(read_field(response, "usage"))
        full_reasoning, full_source, summary, summary_source = (
            _chat_reasoning(message)
        )
        embedded = separate_embedded_reasoning(
            final_text,
            parser=self.embedded_reasoning_parser,
        )
        final_text = embedded.final_text
        if full_reasoning is None and embedded.reasoning_text is not None:
            full_reasoning = embedded.reasoning_text
            full_source = embedded.reasoning_source
        reasoning = build_reasoning_trace(
            mode=self.reasoning_capture,
            full_text=full_reasoning,
            summary_text=summary,
            token_count=usage.reasoning_tokens,
            full_source=full_source,
            summary_source=summary_source,
        )
        provider_model = read_field(response, "model")
        metadata: dict[str, Any] = {}
        if isinstance(provider_model, str):
            metadata["provider_model"] = provider_model
        if embedded.parser is not None:
            metadata["embedded_reasoning_parser"] = embedded.parser
            metadata["embedded_reasoning_block_complete"] = (
                embedded.complete_block
            )

        return InferenceResult(
            model_id=self.model_id,
            final_text=final_text,
            reasoning=reasoning,
            usage=usage,
            request_id=request.request_id,
            provider_response_id=_optional_string(
                read_field(response, "id")
            ),
            finish_reason=_optional_string(
                read_field(choice, "finish_reason")
            ),
            metadata=metadata,
        )

    def _parse_stream_response(
        self,
        response: Any,
        request: InferenceRequest,
    ) -> InferenceResult:
        """Collect an OpenAI-compatible stream into one normalized result."""

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        summary_parts: list[str] = []
        usage = TokenUsage()
        finish_reason: str | None = None
        provider_response_id: str | None = None
        provider_model: str | None = None
        close = getattr(response, "close", None)
        try:
            for chunk in response:
                provider_response_id = (
                    provider_response_id
                    or _optional_string(read_field(chunk, "id"))
                )
                provider_model = (
                    provider_model
                    or _optional_string(read_field(chunk, "model"))
                )
                chunk_usage = read_field(chunk, "usage")
                if chunk_usage is not None:
                    usage = _chat_usage(chunk_usage)
                choices = read_field(chunk, "choices", ())
                if not choices:
                    continue
                choice = choices[0]
                current_finish = _optional_string(
                    read_field(choice, "finish_reason")
                )
                if current_finish is not None:
                    finish_reason = current_finish
                delta = read_field(choice, "delta")
                content = _stream_text(read_field(delta, "content"))
                if content is not None:
                    content_parts.append(content)
                reasoning = _stream_text(
                    read_field(delta, "reasoning")
                    or read_field(delta, "reasoning_content")
                )
                summary = _stream_text(
                    read_field(delta, "reasoning_summary")
                )
                if reasoning is not None:
                    reasoning_parts.append(reasoning)
                if summary is not None:
                    summary_parts.append(summary)
        except Exception as error:
            details = provider_error_details(error)
            provider_detail = provider_error_summary(details)
            if is_safety_refusal(details):
                raise InferenceSafetyRefusal(
                    f"Chat-completions stream safety refusal for model "
                    f"{self.model_id!r} ({provider_detail})",
                    details=details,
                ) from None
            raise InferenceTransportError(
                f"Chat-completions stream failed for model "
                f"{self.model_id!r} ({provider_detail})"
            ) from None
        finally:
            if callable(close):
                close()

        final_text = "".join(content_parts)
        embedded = separate_embedded_reasoning(
            final_text,
            parser=self.embedded_reasoning_parser,
        )
        final_text = embedded.final_text
        full_reasoning = "".join(reasoning_parts) or None
        if full_reasoning is None and embedded.reasoning_text is not None:
            full_reasoning = embedded.reasoning_text
        full_source = (
            "stream.delta.reasoning"
            if reasoning_parts
            else embedded.reasoning_source
        )
        summary = "".join(summary_parts) or None
        reasoning = build_reasoning_trace(
            mode=self.reasoning_capture,
            full_text=full_reasoning,
            summary_text=summary,
            token_count=usage.reasoning_tokens,
            full_source=full_source,
            summary_source=(
                "stream.delta.reasoning_summary"
                if summary is not None
                else None
            ),
        )
        metadata: dict[str, Any] = {"streamed": True}
        if provider_model is not None:
            metadata["provider_model"] = provider_model
        if embedded.parser is not None:
            metadata["embedded_reasoning_parser"] = embedded.parser
            metadata["embedded_reasoning_block_complete"] = (
                embedded.complete_block
            )
        return InferenceResult(
            model_id=self.model_id,
            final_text=final_text,
            reasoning=reasoning,
            usage=usage,
            request_id=request.request_id,
            provider_response_id=provider_response_id,
            finish_reason=finish_reason,
            metadata=metadata,
        )

    async def _parse_async_stream_response(
        self,
        response: Any,
        request: InferenceRequest,
    ) -> InferenceResult:
        """Collect an AsyncOpenAI stream into one normalized result."""

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        summary_parts: list[str] = []
        usage = TokenUsage()
        finish_reason: str | None = None
        provider_response_id: str | None = None
        provider_model: str | None = None
        close = getattr(response, "close", None)
        try:
            async for chunk in response:
                provider_response_id = (
                    provider_response_id
                    or _optional_string(read_field(chunk, "id"))
                )
                provider_model = (
                    provider_model
                    or _optional_string(read_field(chunk, "model"))
                )
                chunk_usage = read_field(chunk, "usage")
                if chunk_usage is not None:
                    usage = _chat_usage(chunk_usage)
                choices = read_field(chunk, "choices", ())
                if not choices:
                    continue
                choice = choices[0]
                current_finish = _optional_string(
                    read_field(choice, "finish_reason")
                )
                if current_finish is not None:
                    finish_reason = current_finish
                delta = read_field(choice, "delta")
                content = _stream_text(read_field(delta, "content"))
                if content is not None:
                    content_parts.append(content)
                reasoning = _stream_text(
                    read_field(delta, "reasoning")
                    or read_field(delta, "reasoning_content")
                )
                summary = _stream_text(
                    read_field(delta, "reasoning_summary")
                )
                if reasoning is not None:
                    reasoning_parts.append(reasoning)
                if summary is not None:
                    summary_parts.append(summary)
        except Exception as error:
            details = provider_error_details(error)
            provider_detail = provider_error_summary(details)
            if is_safety_refusal(details):
                raise InferenceSafetyRefusal(
                    f"Chat-completions stream safety refusal for model "
                    f"{self.model_id!r} ({provider_detail})",
                    details=details,
                ) from None
            raise InferenceTransportError(
                f"Chat-completions stream failed for model "
                f"{self.model_id!r} ({provider_detail})"
            ) from None
        finally:
            if callable(close):
                close_result = close()
                if inspect.isawaitable(close_result):
                    await close_result

        final_text = "".join(content_parts)
        embedded = separate_embedded_reasoning(
            final_text,
            parser=self.embedded_reasoning_parser,
        )
        final_text = embedded.final_text
        full_reasoning = "".join(reasoning_parts) or None
        if full_reasoning is None and embedded.reasoning_text is not None:
            full_reasoning = embedded.reasoning_text
        full_source = (
            "stream.delta.reasoning"
            if reasoning_parts
            else embedded.reasoning_source
        )
        summary = "".join(summary_parts) or None
        reasoning = build_reasoning_trace(
            mode=self.reasoning_capture,
            full_text=full_reasoning,
            summary_text=summary,
            token_count=usage.reasoning_tokens,
            full_source=full_source,
            summary_source=(
                "stream.delta.reasoning_summary"
                if summary is not None
                else None
            ),
        )
        metadata: dict[str, Any] = {
            "streamed": True,
            "async_transport": True,
        }
        if provider_model is not None:
            metadata["provider_model"] = provider_model
        if embedded.parser is not None:
            metadata["embedded_reasoning_parser"] = embedded.parser
            metadata["embedded_reasoning_block_complete"] = (
                embedded.complete_block
            )
        return InferenceResult(
            model_id=self.model_id,
            final_text=final_text,
            reasoning=reasoning,
            usage=usage,
            request_id=request.request_id,
            provider_response_id=provider_response_id,
            finish_reason=finish_reason,
            metadata=metadata,
        )


def _apply_chat_generation(
    payload: dict[str, Any],
    generation: dict[str, Any],
    *,
    include_reasoning_effort: bool = True,
    include_extended_sampling: bool = True,
) -> None:
    direct_fields = (
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "stop",
    )
    if include_reasoning_effort:
        direct_fields = ("reasoning_effort", *direct_fields)
    for field_name in direct_fields:
        value = generation.get(field_name)
        if value is not None:
            payload[field_name] = value

    max_tokens = generation.get("max_output_tokens")
    if max_tokens is None:
        max_tokens = generation.get("max_new_tokens")
    if max_tokens is None:
        max_tokens = generation.get("max_tokens")
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    extra_body: dict[str, Any] = {}
    if include_extended_sampling:
        for field_name in ("top_k", "min_p", "repetition_penalty"):
            value = generation.get(field_name)
            if value is not None:
                extra_body[field_name] = value
    if extra_body:
        payload["extra_body"] = extra_body


def _chat_reasoning(
    message: Any,
) -> tuple[str | None, str | None, str | None, str | None]:
    reasoning = read_field(message, "reasoning")
    legacy = read_field(message, "reasoning_content")

    full_text: str | None = None
    full_source: str | None = None
    summary_text: str | None = None
    summary_source: str | None = None

    if reasoning is not None:
        if isinstance(reasoning, str):
            full_text = extract_text(reasoning)
        else:
            full_text = extract_text(
                read_field(reasoning, "content")
                or read_field(reasoning, "text")
            )
            summary_text = extract_text(
                read_field(reasoning, "summary")
            )
        if full_text:
            full_source = "reasoning"
        if summary_text:
            summary_source = "reasoning.summary"

    if full_text is None:
        full_text = extract_text(legacy)
        if full_text:
            full_source = "reasoning_content"

    direct_summary = extract_text(read_field(message, "reasoning_summary"))
    if direct_summary:
        summary_text = direct_summary
        summary_source = "reasoning_summary"

    return full_text, full_source, summary_text, summary_source


def _chat_usage(value: Any) -> TokenUsage:
    input_tokens = safe_optional_int(
        read_field(value, "prompt_tokens")
        or read_field(value, "input_tokens")
    )
    output_tokens = safe_optional_int(
        read_field(value, "completion_tokens")
        or read_field(value, "output_tokens")
    )
    total_tokens = safe_optional_int(read_field(value, "total_tokens"))
    details = (
        read_field(value, "completion_tokens_details")
        or read_field(value, "output_tokens_details")
    )
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


def _stream_text(value: Any) -> str | None:
    """Retain whitespace in string deltas while normalizing rich content."""

    if isinstance(value, str):
        return value
    return extract_text(value)


def _provider_error_detail(error: Exception) -> str:
    """Return safe provider fields without headers or request objects."""

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
    return "; ".join(details)


def _compact_error_field(value: str, *, limit: int = 500) -> str:
    return " ".join(value.split())[:limit]


def _mapping_values(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return {}
