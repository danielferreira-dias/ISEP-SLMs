"""Shared request, result, and backend contracts for multimodal inference.

The benchmark runner historically consumed a ``generate(...)->str`` method.
The richer interface introduced here keeps that method as a compatibility
wrapper while making reasoning, usage, and request identity explicit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal


ReasoningCaptureMode = Literal[
    "full",
    "summary",
    "tokens_only",
    "none",
]

_REASONING_CAPTURE_MODES = {
    "full",
    "summary",
    "tokens_only",
    "none",
}


class InferenceError(RuntimeError):
    """Base exception for safe, user-facing inference failures."""


class InferenceConfigurationError(InferenceError):
    """Raised when an inference backend is missing required configuration."""


class InferenceRequestError(InferenceError):
    """Raised when an inference request cannot be represented by a backend."""


class InferenceTransportError(InferenceError):
    """Raised when a provider call fails without exposing provider secrets."""


class InferenceSafetyRefusal(InferenceTransportError):
    """Raised when a provider safety system blocks an otherwise valid call."""

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.details = dict(details or {})


class InferencePreflightError(InferenceError):
    """Raised when an inference endpoint fails its readiness checks."""


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Normalized token accounting returned by an inference provider."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ReasoningTrace:
    """Reasoning retained according to an explicit capture policy.

    ``text`` contains either the full provider-returned trace or a provider
    supplied summary. The library never synthesizes a summary from a full
    chain of thought: if the provider does not expose a summary, ``text`` is
    ``None`` in ``summary`` mode.
    """

    capture_mode: ReasoningCaptureMode
    text: str | None = None
    token_count: int | None = None
    source_field: str | None = None


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """One image-and-text request sent to a multimodal model."""

    system_prompt: str
    user_prompt: str
    image_bytes: bytes
    schema: Mapping[str, Any]
    generation: Any | None = None
    request_id: str | None = None
    image_mime_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image_bytes, bytes) or not self.image_bytes:
            raise InferenceRequestError(
                "image_bytes must contain a non-empty byte string"
            )
        if not isinstance(self.schema, Mapping):
            raise InferenceRequestError("schema must be a mapping")


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Normalized model output with final answer and reasoning separated."""

    model_id: str
    final_text: str
    reasoning: ReasoningTrace
    usage: TokenUsage = field(default_factory=TokenUsage)
    request_id: str | None = None
    provider_response_id: str | None = None
    finish_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """Sanitized endpoint readiness result."""

    healthy: bool
    model_available: bool
    checks: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.healthy and self.model_available and not self.errors


class InferenceBackend(ABC):
    """Common synchronous backend API used by benchmark execution."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable benchmark model identifier."""

    @abstractmethod
    def complete(self, request: InferenceRequest) -> InferenceResult:
        """Run one normalized request."""

    async def acomplete(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        """Run one request asynchronously.

        Synchronous-only providers use a worker thread by default. Native
        asynchronous transports override this method.
        """

        return await asyncio.to_thread(self.complete, request)

    async def aclose(self) -> None:
        """Release asynchronous transport resources, when any exist."""

    def generate_result(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        schema: dict[str, Any],
        generation: Any | None = None,
        request_id: str | None = None,
    ) -> InferenceResult:
        """Build and execute one request using the public keyword interface."""

        return self.complete(
            InferenceRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_bytes=image_bytes,
                schema=schema,
                generation=generation,
                request_id=request_id,
            )
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        schema: dict[str, Any],
        generation: Any | None = None,
        request_id: str | None = None,
    ) -> str:
        """Return only final text for the legacy benchmark runner contract."""

        return self.generate_result(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_bytes=image_bytes,
            schema=schema,
            generation=generation,
            request_id=request_id,
        ).final_text

    def generate_batch(
        self,
        requests: Sequence[InferenceRequest],
    ) -> list[InferenceResult]:
        """Execute a batch sequentially.

        This stable API leaves room for a concurrent or provider-native batch
        implementation without changing the executor.
        """

        return [self.complete(request) for request in requests]

    async def generate_batch_async(
        self,
        requests: Sequence[InferenceRequest],
    ) -> list[InferenceResult]:
        """Execute a batch concurrently through the asynchronous contract."""

        return list(
            await asyncio.gather(
                *(self.acomplete(request) for request in requests)
            )
        )


def validate_reasoning_capture(
    mode: str,
) -> ReasoningCaptureMode:
    """Validate and narrow an externally configured reasoning policy."""

    if mode not in _REASONING_CAPTURE_MODES:
        allowed = ", ".join(sorted(_REASONING_CAPTURE_MODES))
        raise InferenceConfigurationError(
            f"Invalid reasoning capture mode {mode!r}; expected one of {allowed}"
        )
    return mode  # type: ignore[return-value]


def build_reasoning_trace(
    *,
    mode: ReasoningCaptureMode,
    full_text: str | None,
    summary_text: str | None,
    token_count: int | None,
    full_source: str | None = None,
    summary_source: str | None = None,
) -> ReasoningTrace:
    """Apply a capture policy without promoting full reasoning to a summary."""

    if mode == "none":
        return ReasoningTrace(capture_mode=mode)
    if mode == "tokens_only":
        return ReasoningTrace(
            capture_mode=mode,
            token_count=token_count,
        )
    if mode == "summary":
        return ReasoningTrace(
            capture_mode=mode,
            text=_clean_optional_text(summary_text),
            token_count=token_count,
            source_field=(
                summary_source if _clean_optional_text(summary_text) else None
            ),
        )
    return ReasoningTrace(
        capture_mode=mode,
        text=_clean_optional_text(full_text),
        token_count=token_count,
        source_field=full_source if _clean_optional_text(full_text) else None,
    )


def generation_values(config: Any | None) -> dict[str, Any]:
    """Convert a generation dataclass, mapping, or object into plain values."""

    if config is None:
        return {}
    if isinstance(config, Mapping):
        return {
            str(key): value
            for key, value in config.items()
            if value is not None
        }
    if is_dataclass(config) and not isinstance(config, type):
        return {
            key: value
            for key, value in asdict(config).items()
            if value is not None
        }
    values: dict[str, Any] = {}
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "do_sample",
        "max_new_tokens",
        "max_output_tokens",
        "max_tokens",
        "seed",
        "stop",
        "chat_template_kwargs",
        "reasoning_effort",
        "thinking_mode",
    ):
        value = getattr(config, key, None)
        if value is not None:
            values[key] = value
    return values


def merge_generation(
    defaults: Any | None,
    override: Any | None,
) -> dict[str, Any]:
    """Merge per-model defaults with per-request generation overrides."""

    merged = generation_values(defaults)
    merged.update(generation_values(override))
    return merged


def image_data_url(
    image_bytes: bytes,
    mime_type: str | None = None,
) -> str:
    """Encode local image bytes as an OpenAI-compatible data URL."""

    resolved_mime = mime_type or detect_image_mime_type(image_bytes)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{resolved_mime};base64,{encoded}"


def detect_image_mime_type(image_bytes: bytes) -> str:
    """Infer common dermatology image MIME types from file signatures."""

    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes.startswith(b"BM"):
        return "image/bmp"
    return "image/jpeg"


def read_field(value: Any, name: str, default: Any = None) -> Any:
    """Read one field from an SDK object or a plain mapping."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def extract_text(value: Any) -> str | None:
    """Normalize provider text represented as strings, objects, or lists."""

    if value is None:
        return None
    if isinstance(value, str):
        return _clean_optional_text(value)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        parts = [
            part
            for item in value
            if (part := extract_text(item)) is not None
        ]
        return "\n".join(parts) or None
    for field_name in (
        "text",
        "content",
        "value",
        "output_text",
        "summary_text",
    ):
        nested = read_field(value, field_name)
        if nested is not None and nested is not value:
            text = extract_text(nested)
            if text:
                return text
    return None


def safe_optional_int(value: Any) -> int | None:
    """Return an integer token count without accepting booleans."""

    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def provider_error_details(error: Exception) -> dict[str, Any]:
    """Extract only safe diagnostic fields from a provider SDK exception."""

    details: dict[str, Any] = {"type": type(error).__name__}
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        details["status_code"] = status_code

    body = getattr(error, "body", None)
    provider_error = body.get("error", body) if isinstance(body, dict) else {}
    if isinstance(provider_error, Mapping):
        for key in ("code", "message", "source_type"):
            value = provider_error.get(key)
            if isinstance(value, str) and value:
                details[key] = _compact_provider_text(value)
        safety = _safe_filter_tree(provider_error)
        if safety:
            details["content_filter"] = safety

    if "message" not in details:
        message = getattr(error, "message", None)
        if isinstance(message, str) and message:
            details["message"] = _compact_provider_text(message)

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if isinstance(headers, Mapping):
        for header in ("x-request-id", "apim-request-id"):
            value = headers.get(header)
            if isinstance(value, str) and value:
                details["request_id"] = _compact_provider_text(value)
                break
    return details


def provider_json_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the project schema to the provider Structured Outputs subset.

    Azure/OpenAI Structured Outputs rejects ``uniqueItems``. The benchmark's
    deterministic validator still enforces its complete local contract after
    generation, so removing this generation-time keyword does not weaken
    scoring.
    """

    normalized = deepcopy(dict(schema))

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("uniqueItems", None)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(normalized)
    return normalized


def provider_error_summary(details: Mapping[str, Any]) -> str:
    """Render safe provider error fields as a concise diagnostic string."""

    fields = []
    for key in ("type", "status_code", "code", "message", "request_id"):
        value = details.get(key)
        if value is not None:
            label = "status" if key == "status_code" else key
            fields.append(f"{label}={value}")
    return "; ".join(fields)


def is_safety_refusal(details: Mapping[str, Any]) -> bool:
    """Identify provider content-filter and policy refusals."""

    code = str(details.get("code", "")).casefold()
    if any(
        token in code
        for token in ("content_filter", "content_policy", "safety")
    ):
        return True
    return bool(details.get("content_filter"))


def _safe_filter_tree(value: Mapping[str, Any]) -> dict[str, Any]:
    """Retain category/severity decisions while dropping arbitrary metadata."""

    result: dict[str, Any] = {}
    scalar_keys = {
        "blocked",
        "detected",
        "filtered",
        "severity",
        "source_type",
    }
    nested_keys = {
        "content_filter_result",
        "content_filter_results",
        "inner_error",
        "innererror",
        "hate",
        "jailbreak",
        "protected_material_code",
        "protected_material_text",
        "self_harm",
        "sexual",
        "violence",
    }
    for key, nested in value.items():
        normalized_key = str(key).casefold()
        if normalized_key in scalar_keys and isinstance(
            nested, (str, bool, int, float)
        ):
            result[str(key)] = nested
        elif normalized_key in nested_keys and isinstance(nested, Mapping):
            safe_nested = _safe_filter_tree(nested)
            if safe_nested:
                result[str(key)] = safe_nested
    return result


def _compact_provider_text(value: str, *, limit: int = 500) -> str:
    return " ".join(value.split())[:limit]


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None
