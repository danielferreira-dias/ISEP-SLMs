"""OpenRouter chat client for one teacher stage."""

import json
from dataclasses import dataclass
from typing import Literal, Protocol

from openai import OpenAI

from project.teacher.schemas import UsageInfo
from project.teacher.teacher import TeacherModel

DEFAULT_TIMEOUT_S = 180.0
DEFAULT_MAX_RETRIES = 3


class TeacherCompletionError(RuntimeError):
    """The teacher HTTP call succeeded but the payload is unusable."""


@dataclass(slots=True, kw_only=True, frozen=True)
class TeacherResponse:
    """Parsed OpenRouter chat completion for one stage."""

    content_json: dict[str, object]
    raw_content: str
    usage: UsageInfo | None
    finish_reason: str | None
    native_finish_reason: str | None


class StageCompleter(Protocol):
    """Anything that can complete Stage A or Stage B messages."""

    def complete_stage(
        self,
        stage_key: Literal["A", "B"],
        messages: list[dict[str, object]],
    ) -> TeacherResponse:
        """Send one stage and return parsed JSON content."""


class TeacherClient:
    """Call OpenRouter using ``TeacherModel.openrouter_body``."""

    def __init__(
        self,
        teacher: TeacherModel,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client: object | None = None,
    ) -> None:
        """Build a client. Does not send a request.

        Args:
            teacher: Loaded YAML config.
            timeout_s: Per-request timeout forwarded to the SDK.
            max_retries: SDK retries for 429/5xx. Do not add a second loop.
            client: Optional injected OpenAI client for tests.
        """
        self._teacher = teacher
        self._client = client or OpenAI(
            base_url=teacher.api.base_url,
            api_key=teacher.api.api_key(),
            timeout=timeout_s,
            max_retries=max_retries,
        )

    def complete_stage(
        self,
        stage_key: Literal["A", "B"],
        messages: list[dict[str, object]],
    ) -> TeacherResponse:
        """Run one chat completion and parse JSON content.

        ``provider`` and ``reasoning`` are sent in ``extra_body`` so the
        OpenAI SDK does not drop OpenRouter extensions.

        Args:
            stage_key: ``A`` or ``B``.
            messages: Chat messages including the image part.

        Returns:
            Parsed JSON plus usage and finish reasons.

        Raises:
            TeacherCompletionError: Empty choices, null content, invalid JSON,
                or ``finish_reason == "length"``.
        """
        body = self._teacher.openrouter_body(stage_key, messages)
        completion = self._client.chat.completions.create(
            model=body["model"],
            messages=body["messages"],
            max_tokens=body["max_tokens"],
            seed=body["seed"],
            response_format=body["response_format"],
            extra_body={
                "reasoning": body["reasoning"],
                "provider": body["provider"],
            },
        )
        return _response_from_completion(completion)


def _response_from_completion(completion: object) -> TeacherResponse:
    """Map an OpenAI completion object to TeacherResponse."""
    choices = getattr(completion, "choices", None)
    if not choices:
        raise TeacherCompletionError("OpenRouter returned no choices")

    choice = choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        raise TeacherCompletionError("finish_reason is length; output was truncated")

    message = getattr(choice, "message", None)
    raw_content = getattr(message, "content", None) if message is not None else None
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise TeacherCompletionError("assistant content is empty")

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise TeacherCompletionError("assistant content is not JSON") from exc

    if not isinstance(parsed, dict):
        raise TeacherCompletionError("assistant JSON must be an object")

    native = getattr(choice, "native_finish_reason", None)
    return TeacherResponse(
        content_json=parsed,
        raw_content=raw_content,
        usage=_usage_from_completion(completion),
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        native_finish_reason=native if isinstance(native, str) else None,
    )


def _usage_from_completion(completion: object) -> UsageInfo | None:
    """Copy usage fields when OpenRouter provides them."""
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None

    cost = getattr(usage, "cost", None)
    return UsageInfo(
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        cost=cost if isinstance(cost, int | float) else None,
    )
