"""Vertex / Gemini Enterprise Agent Platform client for one teacher stage."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, cast

from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.nap import sleep as tenacity_sleep

if TYPE_CHECKING:
    from project.teacher.client import TeacherResponse

from project.teacher.schemas import UsageInfo
from project.teacher.teacher import (
    TeacherModel,
    TeacherPricing,
    TeacherProvider,
    VertexAPI,
)

DEFAULT_TIMEOUT_S = 180.0
_JPEG_DATA_URL = "data:image/jpeg;base64,"
LOGGER = logging.getLogger("project.teacher.vertex")


class _VertexModels(Protocol):
    def generate_content(self, **kwargs: object) -> object:
        """Run one generateContent call."""


class _VertexSdkClient(Protocol):
    models: _VertexModels


def _require_vertex_api(teacher: TeacherModel) -> VertexAPI:
    """Return the Vertex project block or reject a mismatched teacher."""
    if teacher.provider is not TeacherProvider.VERTEX:
        raise TypeError("VertexTeacherClient requires provider=vertex")
    if not isinstance(teacher.api, VertexAPI):
        raise TypeError("VertexTeacherClient requires teacher.api.project")
    return teacher.api


def jpeg_bytes_from_data_url(url: str) -> bytes:
    """Decode the local JPEG data URL produced by image preprocessing."""
    if not url.startswith(_JPEG_DATA_URL):
        raise ValueError("Vertex teacher requires a JPEG data URL")
    encoded = url.removeprefix(_JPEG_DATA_URL)
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("Vertex teacher received an invalid JPEG data URL") from exc


def split_stage_messages(
    messages: Sequence[Mapping[str, object]],
) -> tuple[str, str, bytes]:
    """Extract system text, user text, and JPEG bytes from OpenAI-shaped messages."""
    system = ""
    user_text = ""
    image_url: str | None = None

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system" and isinstance(content, str):
            system = content
            continue
        if role != "user" or not isinstance(content, list):
            raise ValueError("Vertex teacher expected a system string and user parts")
        for part in content:
            if not isinstance(part, Mapping):
                raise ValueError("Vertex teacher received a non-mapping user part")
            part_type = part.get("type")
            if part_type == "text":
                text = part.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Vertex teacher user text is empty")
                user_text = text
            elif part_type == "image_url":
                image = part.get("image_url")
                if not isinstance(image, Mapping):
                    raise ValueError("Vertex teacher image_url is not an object")
                url = image.get("url")
                if not isinstance(url, str) or not url:
                    raise ValueError("Vertex teacher image URL is empty")
                image_url = url
            else:
                raise ValueError(f"Unsupported Vertex teacher part type: {part_type!r}")

    if not system.strip():
        raise ValueError("Vertex teacher is missing the system instruction")
    if not user_text.strip():
        raise ValueError("Vertex teacher is missing the user text")
    if image_url is None:
        raise ValueError("Vertex teacher is missing the image")
    return system, user_text, jpeg_bytes_from_data_url(image_url)


def _enum_name(value: object) -> str | None:
    """Normalize SDK enums and strings to an uppercase finish-reason name."""
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _usage_from_response(
    response: object,
    pricing: TeacherPricing | None = None,
    request_attempts: int = 1,
) -> UsageInfo | None:
    """Copy Vertex tokens and optionally attach a pinned list-price estimate."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None
    prompt = getattr(usage, "prompt_token_count", None)
    completion = getattr(usage, "candidates_token_count", None)
    total = getattr(usage, "total_token_count", None)
    thoughts = getattr(usage, "thoughts_token_count", None)
    normalized = UsageInfo(
        prompt_tokens=prompt if isinstance(prompt, int) else None,
        completion_tokens=completion if isinstance(completion, int) else None,
        total_tokens=total if isinstance(total, int) else None,
        thoughts_tokens=thoughts if isinstance(thoughts, int) else None,
        request_attempts=request_attempts,
        cost=None,
    )
    return pricing.estimate_usage(normalized) if pricing is not None else normalized


def _block_reason(response: object) -> str | None:
    """Return a prompt-level block reason when Vertex refused the request."""
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is None:
        return None
    return _enum_name(getattr(feedback, "block_reason", None))


def response_from_vertex(
    response: object,
    *,
    pricing: TeacherPricing | None = None,
    request_attempts: int = 1,
) -> TeacherResponse:
    """Validate Vertex output and map it onto TeacherResponse."""
    from project.teacher.client import TeacherCompletionError, TeacherResponse

    usage = _usage_from_response(response, pricing, request_attempts)
    blocked = _block_reason(response)
    if blocked:
        raise TeacherCompletionError(
            f"provider_safety_refusal:{blocked}", usage=usage
        )

    candidates = getattr(response, "candidates", None)
    if not candidates:
        raise TeacherCompletionError("Vertex returned no candidates", usage=usage)

    choice = candidates[0]
    finish_reason = _enum_name(getattr(choice, "finish_reason", None))
    if finish_reason in {"MAX_TOKENS", "LENGTH"}:
        raise TeacherCompletionError(
            "finish_reason is length; output was truncated", usage=usage
        )
    if finish_reason in {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT"}:
        raise TeacherCompletionError(
            f"provider_safety_refusal:{finish_reason}", usage=usage
        )

    raw_content = getattr(response, "text", None)
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise TeacherCompletionError("assistant content is empty", usage=usage)

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise TeacherCompletionError(
            "assistant content is not JSON", usage=usage
        ) from exc
    if not isinstance(parsed, dict):
        raise TeacherCompletionError("assistant JSON must be an object", usage=usage)

    return TeacherResponse(
        content_json=parsed,
        raw_content=raw_content,
        usage=usage,
        finish_reason=finish_reason.lower() if finish_reason else None,
        native_finish_reason=finish_reason,
    )


def _vertex_error_code(exc: BaseException) -> str:
    """Return a stable provider outcome without response bodies or secrets."""
    name = type(exc).__name__
    normalized = str(exc).casefold()
    if any(marker in normalized for marker in ("safety", "guardrail", "filtered")):
        return f"provider_safety_refusal:{name}"
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return f"provider_http_error:{code}:{name}"
    return f"provider_error:{name}"


def _vertex_status_code(exc: BaseException) -> int | None:
    """Extract the HTTP status used by Google Gen AI API exceptions."""
    for attribute in ("code", "status_code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    return None


def _is_retryable_vertex_error(
    exc: BaseException,
    *,
    status_codes: tuple[int, ...],
) -> bool:
    """Retry only configured transient Google API HTTP failures."""
    from google.genai import errors as genai_errors

    return (
        isinstance(exc, genai_errors.APIError)
        and _vertex_status_code(exc) in status_codes
    )


def _log_retry(state: RetryCallState, *, max_attempts: int) -> None:
    """Log a stable, non-sensitive retry event before Tenacity sleeps."""
    outcome = state.outcome
    error = outcome.exception() if outcome is not None else None
    next_action = state.next_action
    delay = next_action.sleep if next_action is not None else 0.0
    code = _vertex_error_code(error) if error is not None else "provider_error"
    LOGGER.warning(
        "Transient Vertex failure %s; retrying attempt %d/%d in %.1fs",
        code,
        state.attempt_number + 1,
        max_attempts,
        delay,
    )


class VertexTeacherClient:
    """Call Gemini on Vertex using Application Default Credentials."""

    def __init__(
        self,
        teacher: TeacherModel,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: object | None = None,
        sleep: Callable[[float], None] = tenacity_sleep,
    ) -> None:
        """Build a client. Does not send a request.

        Args:
            teacher: Loaded YAML with ``provider: vertex``.
            timeout_s: Per-request timeout forwarded to the Gen AI SDK.
            client: Optional injected SDK client for tests.
            sleep: Tenacity sleep function; injectable for deterministic tests.
        """
        api = _require_vertex_api(teacher)
        self._teacher = teacher
        self._timeout_s = timeout_s
        self._sleep = sleep
        if client is not None:
            self._client = cast(_VertexSdkClient, client)
            return
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "The google-genai package is required for provider=vertex"
            ) from exc
        self._client = cast(
            _VertexSdkClient,
            genai.Client(
                enterprise=True,
                project=api.project,
                location=api.location,
                http_options=types.HttpOptions(
                    timeout=int(timeout_s * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            ),
        )

    def complete_stage(
        self,
        stage_key: Literal["A", "B"],
        messages: list[dict[str, object]],
    ) -> TeacherResponse:
        """Run one generateContent call and parse JSON content."""
        from google.genai import errors as genai_errors
        from google.genai import types

        from project.teacher.client import TeacherCompletionError

        system, user_text, image_bytes = split_stage_messages(messages)
        config_fields = self._teacher.vertex_generate_config(stage_key)
        thinking = config_fields["thinking_config"]
        assert isinstance(thinking, dict)
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=config_fields["max_output_tokens"],
            seed=config_fields["seed"],
            response_mime_type=config_fields["response_mime_type"],
            response_json_schema=config_fields["response_json_schema"],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            thinking_config=types.ThinkingConfig(
                thinking_level=thinking["thinking_level"],
                include_thoughts=thinking["include_thoughts"],
            ),
        )
        request = {
            "model": self._teacher.model.id,
            "contents": [
                types.Part.from_text(text=user_text),
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            ],
            "config": config,
        }
        retrying: Retrying | None = None
        try:
            if self._teacher.retry is None:
                response = self._client.models.generate_content(**request)
            else:
                policy = self._teacher.retry
                retrying = Retrying(
                    sleep=self._sleep,
                    stop=stop_after_attempt(policy.max_attempts),
                    wait=wait_exponential_jitter(
                        initial=policy.initial_delay_seconds,
                        max=policy.max_delay_seconds,
                        exp_base=policy.exponential_base,
                        jitter=policy.jitter_seconds,
                    ),
                    retry=retry_if_exception(
                        lambda exc: _is_retryable_vertex_error(
                            exc,
                            status_codes=policy.retryable_status_codes,
                        )
                    ),
                    before_sleep=lambda state: _log_retry(
                        state,
                        max_attempts=policy.max_attempts,
                    ),
                    reraise=True,
                )
                response = retrying(self._client.models.generate_content, **request)
        except genai_errors.APIError as exc:
            attempts = _retry_attempts(retrying)
            raise TeacherCompletionError(
                _vertex_error_code(exc),
                usage=UsageInfo(request_attempts=attempts),
            ) from exc
        return response_from_vertex(
            response,
            pricing=self._teacher.pricing,
            request_attempts=_retry_attempts(retrying),
        )


def _retry_attempts(retrying: Retrying | None) -> int:
    """Return the number of physical requests made by one logical call."""
    if retrying is None:
        return 1
    value = retrying.statistics.get("attempt_number", 1)
    return value if isinstance(value, int) and value >= 1 else 1
