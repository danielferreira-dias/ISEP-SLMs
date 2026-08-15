"""Canonical JSON and deterministic natural-language rendering for E3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from src.train.domain import Taxonomy
from src.train.e3.domain import (
    ClinicalAction,
    ObservationStatus,
    StructuredClinicalTarget,
)
from src.train.e3.templates import (
    OPEN_RESPONSE_TEMPLATES,
    OpenResponseContext,
    OpenResponseTemplate,
)

RENDERER_VERSION = "e3_open_response_v1"


@dataclass(frozen=True, slots=True)
class RenderedOpenResponse:
    """Natural-language target with complete deterministic provenance."""

    text: str
    template_id: str
    renderer_version: str
    target_sha256: str


def canonical_structured_json(target: StructuredClinicalTarget) -> str:
    """Serialize a structured target into compact, byte-stable JSON."""

    return json.dumps(
        target.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


class DeterministicOpenResponseRenderer:
    """Render one of several frozen templates selected by stable sample hash."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        *,
        templates: tuple[OpenResponseTemplate, ...] = OPEN_RESPONSE_TEMPLATES,
        renderer_version: str = RENDERER_VERSION,
    ) -> None:
        """Validate immutable rendering dependencies.

        Args:
            taxonomy: Canonical disease IDs and display labels.
            templates: Frozen set of semantically equivalent surface templates.
            renderer_version: Version included in selection and provenance hashes.

        Raises:
            ValueError: If taxonomy, templates, or version are ambiguous.
        """

        if not templates:
            raise ValueError("At least one open-response template is required")
        template_ids = tuple(template.template_id for template in templates)
        if len(template_ids) != len(set(template_ids)):
            raise ValueError("Open-response template IDs must be unique")
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

    def template_for(self, sample_id: str) -> OpenResponseTemplate:
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
        target: StructuredClinicalTarget,
    ) -> RenderedOpenResponse:
        """Render a natural response containing only target-supported facts."""

        context = self._context(target)
        template = self.template_for(sample_id)
        text = template.render(context)
        return RenderedOpenResponse(
            text=text,
            template_id=template.template_id,
            renderer_version=self._renderer_version,
            target_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )

    def _context(self, target: StructuredClinicalTarget) -> OpenResponseContext:
        labels = tuple(self._label(item.disease_id) for item in target.differential)
        leading = target.differential[0]
        observations = {item.id: item for item in target.observations}
        evidence_concepts = tuple(
            observations[item_id].concept
            for item_id in leading.supporting_observation_ids
            if observations[item_id].status is ObservationStatus.PRESENT
        )
        if not evidence_concepts:
            evidence_concepts = tuple(
                item.concept
                for item in target.observations
                if item.status is ObservationStatus.PRESENT
            )
        evidence = _natural_list(tuple(_humanize(value) for value in evidence_concepts))
        if not evidence:
            raise ValueError(
                "Open responses require at least one present supporting observation"
            )

        limitation_values = list(target.not_assessable_features)
        limitation_values.extend(
            discriminator.feature for discriminator in leading.missing_discriminators
        )
        limitations = _natural_list(
            _unique(tuple(_humanize(value) for value in limitation_values))
        )
        alternatives = _natural_list(labels[1:])
        return OpenResponseContext(
            diagnosis=labels[0],
            confidence=leading.diagnostic_confidence.value,
            evidence=evidence,
            alternatives=alternatives or None,
            limitations=limitations or None,
            action=_action_sentence(
                target.action,
                requested_information=target.requested_information,
            ),
        )

    def _label(self, disease_id: str) -> str:
        try:
            return self._disease_labels[disease_id]
        except KeyError as exc:
            raise ValueError(
                f"Differential disease {disease_id!r} is outside the taxonomy"
            ) from exc


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


def _action_sentence(
    action: ClinicalAction,
    *,
    requested_information: str | None,
) -> str:
    requested = _humanize(requested_information) if requested_information else None
    sentences: dict[ClinicalAction, str] = {
        ClinicalAction.DIAGNOSE_PROVISIONALLY: (
            "A provisional diagnosis is appropriate on the available evidence."
        ),
        ClinicalAction.REQUEST_OVERVIEW_IMAGE: (
            "An overview image should be obtained before refining the assessment."
        ),
        ClinicalAction.REQUEST_CLOSEUP_IMAGE: (
            "A focused close-up image should be obtained before refining the "
            "assessment."
        ),
        ClinicalAction.REQUEST_SCALE_OR_PROFILE: (
            "An image with scale or a lateral profile should be obtained."
        ),
        ClinicalAction.REQUEST_CLINICAL_CONTEXT: (
            f"Additional clinical context should be obtained"
            f"{f', particularly {requested}' if requested else ''}."
        ),
        ClinicalAction.REQUEST_DERMOSCOPY: (
            "Dermoscopic assessment should be considered to refine the diagnosis."
        ),
        ClinicalAction.REQUEST_IN_PERSON_EXAM: (
            "An in-person clinical examination is appropriate."
        ),
        ClinicalAction.RECOMMEND_CONFIRMATORY_TEST: (
            "An appropriate confirmatory test should be considered."
        ),
        ClinicalAction.ABSTAIN_POOR_QUALITY: (
            "No diagnosis should be made from this image because its quality is "
            "insufficient."
        ),
        ClinicalAction.ABSTAIN_OUT_OF_DOMAIN: (
            "No supported diagnosis should be assigned because the case is out of "
            "domain."
        ),
    }
    return sentences[action]
