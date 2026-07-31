"""Strict and explicitly recoverable JSON parsing for benchmark outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable


_SINGLE_JSON_FENCE = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```\s*\Z",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class JsonParseResult:
    """Result of strict parsing plus one deliberately narrow recovery rule."""

    decoded: Any | None
    raw_valid: bool
    recoverable_valid: bool
    error: str | None = None
    recovery: str | None = None


def parse_json_output(
    raw_text: str,
    *,
    parse_constant: Callable[[str], Any] | None = None,
    object_pairs_hook: Callable[[list[tuple[str, Any]]], Any] | None = None,
) -> JsonParseResult:
    """Parse raw JSON and audit a single Markdown JSON fence separately.

    Recovery is intentionally limited to one complete fenced JSON value with
    no text before or after the fence. A recovered value remains ``raw_valid
    == False`` so the prompt-only benchmark does not conceal format failures.
    """

    load_options: dict[str, Any] = {}
    if parse_constant is not None:
        load_options["parse_constant"] = parse_constant
    if object_pairs_hook is not None:
        load_options["object_pairs_hook"] = object_pairs_hook
    try:
        decoded = json.loads(raw_text, **load_options)
    except (TypeError, ValueError, json.JSONDecodeError) as raw_error:
        fence = (
            _SINGLE_JSON_FENCE.fullmatch(raw_text)
            if isinstance(raw_text, str)
            else None
        )
        if fence is None:
            return JsonParseResult(
                decoded=None,
                raw_valid=False,
                recoverable_valid=False,
                error=str(raw_error),
            )
        try:
            decoded = json.loads(fence.group("body"), **load_options)
        except (TypeError, ValueError, json.JSONDecodeError):
            return JsonParseResult(
                decoded=None,
                raw_valid=False,
                recoverable_valid=False,
                error=str(raw_error),
            )
        return JsonParseResult(
            decoded=decoded,
            raw_valid=False,
            recoverable_valid=True,
            error="wrapped_in_single_json_fence",
            recovery="single_json_fence",
        )
    return JsonParseResult(
        decoded=decoded,
        raw_valid=True,
        recoverable_valid=True,
    )
