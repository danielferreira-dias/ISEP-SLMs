"""Frozen natural-language templates for deterministic E3 open responses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OpenResponseContext:
    """Validated semantic clauses available to every surface template."""

    diagnosis: str
    confidence: str
    evidence: str
    alternatives: str | None
    limitations: str | None
    action: str


type TemplateRenderer = Callable[[OpenResponseContext], str]


@dataclass(frozen=True, slots=True)
class OpenResponseTemplate:
    """Stable template identity and its pure rendering function."""

    template_id: str
    render: TemplateRenderer


def _sentences(*clauses: str | None) -> str:
    return " ".join(clause.strip() for clause in clauses if clause).strip()


def _alternatives(context: OpenResponseContext, lead: str) -> str | None:
    if context.alternatives is None:
        return None
    return f"{lead} {context.alternatives}."


def _limitations(context: OpenResponseContext, lead: str) -> str | None:
    if context.limitations is None:
        return None
    return f"{lead} {context.limitations}."


def _t01(context: OpenResponseContext) -> str:
    return _sentences(
        f"The most likely diagnosis is {context.diagnosis}.",
        f"This is supported by {context.evidence}.",
        _alternatives(context, "Relevant differential diagnoses include"),
        _limitations(context, "The image does not establish"),
        context.action,
    )


def _t02(context: OpenResponseContext) -> str:
    return _sentences(
        f"The leading diagnosis is {context.diagnosis}, based on {context.evidence}.",
        _alternatives(context, "Other considerations are"),
        _limitations(context, "Features that remain unassessable include"),
        context.action,
    )


def _t03(context: OpenResponseContext) -> str:
    return _sentences(
        f"The visible findings, particularly {context.evidence}, are most "
        f"consistent with {context.diagnosis}.",
        _alternatives(context, "The differential also includes"),
        _limitations(context, "This photograph cannot determine"),
        context.action,
    )


def _t04(context: OpenResponseContext) -> str:
    return _sentences(
        f"{context.diagnosis.capitalize()} is the primary diagnostic consideration.",
        f"The supporting visual evidence is {context.evidence}.",
        _alternatives(context, "Alternative diagnoses to consider are"),
        _limitations(context, "Unresolved features are"),
        context.action,
    )


def _t05(context: OpenResponseContext) -> str:
    return _sentences(
        f"A {context.confidence}-confidence interpretation favors {context.diagnosis}.",
        f"The image demonstrates {context.evidence}.",
        _alternatives(context, "Possible alternatives include"),
        _limitations(context, "Not assessable from this image are"),
        context.action,
    )


def _t06(context: OpenResponseContext) -> str:
    return _sentences(
        f"Overall, {context.diagnosis} is favored because of {context.evidence}.",
        _alternatives(context, "Competing diagnoses are"),
        _limitations(context, "The available view does not resolve"),
        context.action,
    )


def _t07(context: OpenResponseContext) -> str:
    return _sentences(
        f"The image shows {context.evidence}.",
        f"Together, these findings favor {context.diagnosis} with "
        f"{context.confidence} confidence.",
        _alternatives(context, "The main alternatives are"),
        _limitations(context, "Additional uncertainty concerns"),
        context.action,
    )


def _t08(context: OpenResponseContext) -> str:
    return _sentences(
        f"On visual assessment, {context.evidence} is present.",
        f"The leading interpretation is {context.diagnosis}.",
        _alternatives(context, "The differential includes"),
        _limitations(context, "The photograph is insufficient to assess"),
        context.action,
    )


def _t09(context: OpenResponseContext) -> str:
    return _sentences(
        f"Visual assessment identifies {context.evidence}, supporting "
        f"{context.diagnosis} as the most likely diagnosis.",
        _alternatives(context, "Reasonable alternatives are"),
        _limitations(context, "Remaining limitations include"),
        context.action,
    )


def _t10(context: OpenResponseContext) -> str:
    return _sentences(
        f"The key visible features are {context.evidence}.",
        f"They support a {context.confidence}-confidence diagnosis of "
        f"{context.diagnosis}.",
        _alternatives(context, "Other diagnoses in the differential are"),
        _limitations(context, "The image alone cannot evaluate"),
        context.action,
    )


def _t11(context: OpenResponseContext) -> str:
    return _sentences(
        f"Based only on the supplied image, {context.evidence} supports "
        f"{context.diagnosis}.",
        _alternatives(context, "Other possibilities include"),
        _limitations(context, "Unavailable discriminators are"),
        context.action,
    )


def _t12(context: OpenResponseContext) -> str:
    return _sentences(
        f"The observed pattern is most compatible with {context.diagnosis}; "
        f"the principal evidence is {context.evidence}.",
        _alternatives(context, "Diagnostic alternatives include"),
        _limitations(context, "The current image cannot clarify"),
        context.action,
    )


OPEN_RESPONSE_TEMPLATES: tuple[OpenResponseTemplate, ...] = (
    OpenResponseTemplate("open_response_v1_t01", _t01),
    OpenResponseTemplate("open_response_v1_t02", _t02),
    OpenResponseTemplate("open_response_v1_t03", _t03),
    OpenResponseTemplate("open_response_v1_t04", _t04),
    OpenResponseTemplate("open_response_v1_t05", _t05),
    OpenResponseTemplate("open_response_v1_t06", _t06),
    OpenResponseTemplate("open_response_v1_t07", _t07),
    OpenResponseTemplate("open_response_v1_t08", _t08),
    OpenResponseTemplate("open_response_v1_t09", _t09),
    OpenResponseTemplate("open_response_v1_t10", _t10),
    OpenResponseTemplate("open_response_v1_t11", _t11),
    OpenResponseTemplate("open_response_v1_t12", _t12),
)
