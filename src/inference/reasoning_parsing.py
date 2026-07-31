"""Deterministic separation of model-embedded reasoning and final content."""

from __future__ import annotations

from dataclasses import dataclass
import re


MEDGEMMA_SPECIAL_TOKENS = "medgemma_special_tokens"

_MEDGEMMA_THINKING_BLOCK = re.compile(
    r"<unused94>(?P<reasoning>[\s\S]*?)<unused95>",
)


@dataclass(frozen=True, slots=True)
class EmbeddedReasoningResult:
    """Content and reasoning extracted from one model message."""

    final_text: str
    reasoning_text: str | None = None
    reasoning_source: str | None = None
    parser: str | None = None
    complete_block: bool = False


def separate_embedded_reasoning(
    text: str,
    *,
    parser: str | None,
) -> EmbeddedReasoningResult:
    """Separate a configured model-specific reasoning block.

    The function never derives a final answer from reasoning. If a complete
    MedGemma block has no content after its closing token, ``final_text`` is
    empty and the benchmark correctly treats the answer as invalid.
    """

    if parser is None:
        return EmbeddedReasoningResult(final_text=text)
    if parser != MEDGEMMA_SPECIAL_TOKENS:
        raise ValueError(f"Unsupported embedded reasoning parser: {parser}")

    match = _MEDGEMMA_THINKING_BLOCK.search(text)
    if match is None:
        return EmbeddedReasoningResult(
            final_text=text,
            parser=parser,
        )

    reasoning = _strip_medgemma_thought_label(
        match.group("reasoning")
    )
    before = text[: match.start()].strip()
    after = text[match.end() :].strip()
    final_text = "\n\n".join(
        part for part in (before, after) if part
    )
    return EmbeddedReasoningResult(
        final_text=final_text,
        reasoning_text=reasoning or None,
        reasoning_source="content.medgemma_special_tokens",
        parser=parser,
        complete_block=True,
    )


def _strip_medgemma_thought_label(value: str) -> str:
    """Remove MedGemma's optional literal ``thought`` channel label."""

    return re.sub(
        r"\A[ \t]*thought(?:[ \t]*\r?\n)?",
        "",
        value,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
