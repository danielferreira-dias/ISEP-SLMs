"""Canonical E3 JSON targets and complete differential rendering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from src.train.domain import Taxonomy
from src.train.e3.domain import (
    ContextDecision,
    Observation,
    ObservationStatus,
    StageATarget,
    StageBTarget,
)
from src.train.e3.templates import (
    GROUNDED_DIFFERENTIAL_TEMPLATES,
    GroundedDifferentialContext,
    GroundedDifferentialTemplate,
)

RENDERER_VERSION = "e3_grounded_differential_v1"
CONTEXT_POLICY_RENDERER_VERSION = "e3_context_policy_v1"


@dataclass(frozen=True, slots=True)
class RenderedGroundedDifferential:
    """Natural-language target with complete deterministic provenance."""

    text: str
    template_id: str
    renderer_version: str
    target_sha256: str


@dataclass(frozen=True, slots=True)
class RenderedContextPolicy:
    """Byte-stable policy target with human-readable disease labels."""

    text: str
    renderer_version: str
    target_sha256: str


def canonical_stage_a_json(target: StageATarget) -> str:
    """Serialize the non-caption Stage-A fields into byte-stable JSON."""

    payload = target.model_dump(mode="json", exclude={"clinical_caption"})
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


class DeterministicContextPolicyRenderer:
    """Render only the accepted Stage-B context decision as canonical JSON."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        *,
        renderer_version: str = CONTEXT_POLICY_RENDERER_VERSION,
    ) -> None:
        disease_labels = dict(zip(taxonomy.disease_ids, taxonomy.labels, strict=True))
        if not disease_labels:
            raise ValueError("A non-empty taxonomy is required")
        if not renderer_version.strip():
            raise ValueError("renderer_version must be non-empty")
        self._disease_labels = disease_labels
        self._renderer_version = renderer_version

    def render(self, decision: ContextDecision) -> RenderedContextPolicy:
        """Map private disease IDs to labels and serialize deterministically."""

        requests = [
            {
                "context_type": request.context_type,
                "rationale": request.rationale,
                "discriminates_between": [
                    self._label(disease_id)
                    for disease_id in request.discriminates_between
                ],
                "priority": request.priority,
                "question": request.question,
                "required_source": request.required_source,
            }
            for request in decision.requests
        ]
        payload = {
            "decision_rationale": decision.decision_rationale,
            "information_sufficiency": decision.information_sufficiency.value,
            "requests": requests,
            "response_policy": decision.response_policy.value,
        }
        text = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        return RenderedContextPolicy(
            text=text,
            renderer_version=self._renderer_version,
            target_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )

    def _label(self, disease_id: str) -> str:
        try:
            return self._disease_labels[disease_id]
        except KeyError as exc:
            raise ValueError(
                f"Context request disease {disease_id!r} is outside the taxonomy"
            ) from exc


class DeterministicGroundedDifferentialRenderer:
    """Render accepted diagnostic facts without exposing context policy."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        *,
        templates: tuple[
            GroundedDifferentialTemplate, ...
        ] = GROUNDED_DIFFERENTIAL_TEMPLATES,
        renderer_version: str = RENDERER_VERSION,
    ) -> None:
        """Validate immutable rendering dependencies."""

        if not templates:
            raise ValueError("At least one grounded-differential template is required")
        template_ids = tuple(template.template_id for template in templates)
        if len(template_ids) != len(set(template_ids)):
            raise ValueError("Grounded-differential template IDs must be unique")
        if not renderer_version.strip():
            raise ValueError("renderer_version must be non-empty")
        disease_labels = dict(zip(taxonomy.disease_ids, taxonomy.labels, strict=True))
        if not disease_labels:
            raise ValueError("A non-empty taxonomy is required")
        self._disease_labels = disease_labels
        self._templates = templates
        self._renderer_version = renderer_version

    @property
    def template_ids(self) -> tuple[str, ...]:
        """Return frozen template IDs in selection order."""

        return tuple(template.template_id for template in self._templates)

    def template_for(self, sample_id: str) -> GroundedDifferentialTemplate:
        """Select one template deterministically from sample ID and version."""

        if not sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        digest = hashlib.sha256(
            f"{self._renderer_version}\x00{sample_id}".encode()
        ).digest()
        index = int.from_bytes(digest[:8], byteorder="big") % len(self._templates)
        return self._templates[index]

    def render(
        self,
        sample_id: str,
        stage_a: StageATarget,
        stage_b: StageBTarget,
    ) -> RenderedGroundedDifferential:
        """Render a complete response containing only accepted target facts."""

        context = self._context(stage_a, stage_b)
        template = self.template_for(sample_id)
        text = template.render(context)
        return RenderedGroundedDifferential(
            text=text,
            template_id=template.template_id,
            renderer_version=self._renderer_version,
            target_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )

    def _context(
        self,
        stage_a: StageATarget,
        stage_b: StageBTarget,
    ) -> GroundedDifferentialContext:
        differential = stage_b.diagnostic_assessment.differential
        labels = tuple(self._label(item.disease_id) for item in differential)
        leading = differential[0]
        observations = {item.id: item for item in stage_a.observations}
        evidence_concepts = tuple(
            _observation_text(observations[item_id])
            for item_id in leading.supporting_observation_ids
            if observations[item_id].status is ObservationStatus.PRESENT
        )
        if not evidence_concepts:
            evidence_concepts = tuple(
                _observation_text(item)
                for item in stage_a.observations
                if item.status is ObservationStatus.PRESENT
            )
        evidence = _natural_list(tuple(_humanize(value) for value in evidence_concepts))
        if not evidence:
            raise ValueError(
                "Grounded differentials require a present supporting observation"
            )

        limitation_values = list(stage_a.not_assessable_features)
        limitation_values.extend(
            discriminator.feature for discriminator in leading.missing_discriminators
        )
        limitations = _natural_list(
            _unique(tuple(_humanize(value) for value in limitation_values))
        )
        alternatives = _natural_list(labels[1:])
        return GroundedDifferentialContext(
            description=stage_a.clinical_caption,
            diagnosis=labels[0],
            confidence=leading.diagnostic_confidence.value,
            evidence=evidence,
            alternatives=alternatives or None,
            limitations=limitations or None,
        )

    def _label(self, disease_id: str) -> str:
        try:
            return self._disease_labels[disease_id]
        except KeyError as exc:
            raise ValueError(
                f"Differential disease {disease_id!r} is outside the taxonomy"
            ) from exc


def _observation_text(observation: Observation) -> str:
    if observation.concept_detail is None:
        return observation.concept_label
    return f"{observation.concept_label}: {observation.concept_detail}"


def _humanize(value: str) -> str:
    return " ".join(value.replace("_", " ").split())


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _natural_list(values: tuple[str, ...]) -> str:
    clean = _unique(values)
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"
