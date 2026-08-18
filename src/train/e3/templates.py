"""Frozen complete-response templates for E3 grounded differentials."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GroundedDifferentialContext:
    """Validated semantic clauses available to every surface template."""

    description: str
    diagnosis: str
    confidence: str
    evidence: str
    alternatives: str | None
    limitations: str | None


type TemplateRenderer = Callable[[GroundedDifferentialContext], str]


@dataclass(frozen=True, slots=True)
class GroundedDifferentialTemplate:
    """Stable template identity and its pure rendering function."""

    template_id: str
    render: TemplateRenderer


def _sentences(*clauses: str | None) -> str:
    return " ".join(clause.strip() for clause in clauses if clause).strip()


def _alternatives(
    context: GroundedDifferentialContext,
    lead: str,
) -> str | None:
    if context.alternatives is None:
        return None
    return f"{lead} {context.alternatives}."


def _limitations(
    context: GroundedDifferentialContext,
    lead: str,
) -> str | None:
    if context.limitations is None:
        return None
    return f"{lead} {context.limitations}."


def _t01(context: GroundedDifferentialContext) -> str:
    return _sentences(
        context.description,
        f"The most likely diagnosis is {context.diagnosis} with "
        f"{context.confidence} confidence.",
        f"This is supported by {context.evidence}.",
        _alternatives(context, "Relevant differential diagnoses include"),
        _limitations(context, "The image does not establish"),
    )


def _t02(context: GroundedDifferentialContext) -> str:
    return _sentences(
        context.description,
        f"The leading diagnosis is {context.diagnosis}, based on "
        f"{context.evidence}.",
        f"Diagnostic confidence is {context.confidence}.",
        _alternatives(context, "Other considerations are"),
        _limitations(context, "Features that remain unassessable include"),
    )


def _t03(context: GroundedDifferentialContext) -> str:
    return _sentences(
        context.description,
        f"The visible evidence, particularly {context.evidence}, is most "
        f"consistent with {context.diagnosis}.",
        f"Confidence in the leading diagnosis is {context.confidence}.",
        _alternatives(context, "The differential also includes"),
        _limitations(context, "This photograph cannot determine"),
    )


def _t04(context: GroundedDifferentialContext) -> str:
    return _sentences(
        context.description,
        f"{context.diagnosis.capitalize()} is the primary diagnostic "
        f"consideration at {context.confidence} confidence.",
        f"The supporting visual evidence is {context.evidence}.",
        _alternatives(context, "Alternative diagnoses to consider are"),
        _limitations(context, "Unresolved discriminators are"),
    )


def _t05(context: GroundedDifferentialContext) -> str:
    return _sentences(
        context.description,
        f"A {context.confidence}-confidence interpretation favors "
        f"{context.diagnosis}.",
        f"The relevant evidence is {context.evidence}.",
        _alternatives(context, "Possible alternatives include"),
        _limitations(context, "Not assessable from this image are"),
    )


def _t06(context: GroundedDifferentialContext) -> str:
    return _sentences(
        context.description,
        f"Overall, {context.diagnosis} is favored because of "
        f"{context.evidence}.",
        f"This assessment has {context.confidence} confidence.",
        _alternatives(context, "Competing diagnoses are"),
        _limitations(context, "The available view does not resolve"),
    )


def _t07(context: GroundedDifferentialContext) -> str:
    return _sentences(
        context.description,
        f"Together, {context.evidence} favors {context.diagnosis} with "
        f"{context.confidence} confidence.",
        _alternatives(context, "The main alternatives are"),
        _limitations(context, "Additional uncertainty concerns"),
    )


def _t08(context: GroundedDifferentialContext) -> str:
    return _sentences(
        context.description,
        f"The leading interpretation is {context.diagnosis} with "
        f"{context.confidence} confidence.",
        f"Supporting findings are {context.evidence}.",
        _alternatives(context, "The differential includes"),
        _limitations(context, "The photograph is insufficient to assess"),
    )


def _t09(context: GroundedDifferentialContext) -> str:
    return _sentences(
        context.description,
        f"Visual evidence including {context.evidence} supports "
        f"{context.diagnosis} as the most likely diagnosis.",
        f"Diagnostic confidence is {context.confidence}.",
        _alternatives(context, "Reasonable alternatives are"),
        _limitations(context, "Remaining limitations include"),
    )


def _t10(context: GroundedDifferentialContext) -> str:
    return _sentences(
        context.description,
        f"The key supporting features are {context.evidence}.",
        f"They support a {context.confidence}-confidence diagnosis of "
        f"{context.diagnosis}.",
        _alternatives(context, "Other diagnoses in the differential are"),
        _limitations(context, "The image alone cannot evaluate"),
    )


def _t11(context: GroundedDifferentialContext) -> str:
    return _sentences(
        context.description,
        f"Based only on the supplied image, {context.evidence} supports "
        f"{context.diagnosis} with {context.confidence} confidence.",
        _alternatives(context, "Other possibilities include"),
        _limitations(context, "Unavailable discriminators are"),
    )


def _t12(context: GroundedDifferentialContext) -> str:
    return _sentences(
        context.description,
        f"The observed pattern is most compatible with {context.diagnosis}; "
        f"the principal evidence is {context.evidence}.",
        f"Confidence in this ranking is {context.confidence}.",
        _alternatives(context, "Diagnostic alternatives include"),
        _limitations(context, "The current image cannot clarify"),
    )


GROUNDED_DIFFERENTIAL_TEMPLATES: tuple[GroundedDifferentialTemplate, ...] = (
    GroundedDifferentialTemplate("grounded_differential_v1_t01", _t01),
    GroundedDifferentialTemplate("grounded_differential_v1_t02", _t02),
    GroundedDifferentialTemplate("grounded_differential_v1_t03", _t03),
    GroundedDifferentialTemplate("grounded_differential_v1_t04", _t04),
    GroundedDifferentialTemplate("grounded_differential_v1_t05", _t05),
    GroundedDifferentialTemplate("grounded_differential_v1_t06", _t06),
    GroundedDifferentialTemplate("grounded_differential_v1_t07", _t07),
    GroundedDifferentialTemplate("grounded_differential_v1_t08", _t08),
    GroundedDifferentialTemplate("grounded_differential_v1_t09", _t09),
    GroundedDifferentialTemplate("grounded_differential_v1_t10", _t10),
    GroundedDifferentialTemplate("grounded_differential_v1_t11", _t11),
    GroundedDifferentialTemplate("grounded_differential_v1_t12", _t12),
)
