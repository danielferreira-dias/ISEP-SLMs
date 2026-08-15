"""Multimodal conversation formatting for E3 teacher supervision."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from PIL import Image

from src.train.domain import (
    ChatMessage,
    ImageMessageContent,
    Taxonomy,
    TextMessageContent,
)
from src.train.e3.domain import StructuredClinicalTarget
from src.train.e3.rendering import (
    DeterministicOpenResponseRenderer,
    RenderedOpenResponse,
    canonical_structured_json,
)

STRUCTURED_PROMPT_ID = "e3_structured_diagnosis_v1"
OPEN_RESPONSE_PROMPT_ID = "e3_open_diagnosis_v1"

STRUCTURED_PROMPT = (
    "Assess the dermatology image using only visible evidence. Return one compact "
    "JSON object containing image_assessment, dominant_visual_pattern, "
    "observations, not_assessable_features, differential, action, "
    "action_urgency, requested_information, and concise_clinical_rationale. "
    "Every supporting or contradicting evidence link must reference an observation "
    "ID. Do not invent history, palpation findings, evolution, tests, or metadata."
    "\n\n/no_think"
)

OPEN_RESPONSE_PROMPT = (
    "Assess the dermatology image using only visible evidence. Provide a concise "
    "clinical response stating the leading diagnosis, supporting visual findings, "
    "relevant alternatives, limitations of the image, and the appropriate next "
    "action. Do not invent clinical history, tests, or non-visible findings."
    "\n\n/no_think"
)


class E3TrainingVariant(StrEnum):
    """Separately measurable E3 output formats."""

    STRUCTURED = "structured"
    OPEN_RESPONSE = "open_response"


@dataclass(frozen=True, slots=True)
class E3StructuredSample:
    """One accepted image-target pair for E3 training."""

    sample_id: str
    leakage_group_id: str
    disease_id: str
    label: str
    image: Image.Image
    target: StructuredClinicalTarget


@dataclass(frozen=True, slots=True)
class E3FormattedExample:
    """Rendered E3 example with template provenance when applicable."""

    sample_id: str
    leakage_group_id: str
    disease_id: str
    label: str
    task_id: str
    target_variant: E3TrainingVariant
    messages: tuple[ChatMessage, ...]
    template_id: str | None
    renderer_version: str | None
    target_sha256: str

    def as_record(self) -> dict[str, object]:
        """Convert the example to a TRL/Hugging Face-compatible record."""

        return {
            "sample_id": self.sample_id,
            "leakage_group_id": self.leakage_group_id,
            "disease_id": self.disease_id,
            "label": self.label,
            "task_id": self.task_id,
            "target_variant": self.target_variant.value,
            "template_id": self.template_id,
            "renderer_version": self.renderer_version,
            "target_sha256": self.target_sha256,
            "messages": [message.as_record() for message in self.messages],
        }


@dataclass(frozen=True, slots=True)
class E3StructuredPhase:
    """Format accepted Stage-A/B targets without exposing gold metadata."""

    taxonomy: Taxonomy
    variant: E3TrainingVariant

    def __post_init__(self) -> None:
        """Reject empty or ambiguous taxonomies."""

        if not self.taxonomy.classes:
            raise ValueError("E3 requires a non-empty taxonomy")
        if len(set(self.taxonomy.disease_ids)) != len(self.taxonomy.disease_ids):
            raise ValueError("E3 taxonomy contains duplicate disease IDs")

    def format_example(self, sample: E3StructuredSample) -> E3FormattedExample:
        """Render one structured or natural-language E3 conversation.

        The gold label is retained as private row metadata for evaluation, but it
        is never inserted in the user message. The leading accepted differential
        must agree with that gold identity before the target can enter E3.
        """

        labels = dict(zip(self.taxonomy.disease_ids, self.taxonomy.labels, strict=True))
        if labels.get(sample.disease_id) != sample.label:
            raise ValueError("E3 sample contains a non-canonical gold label pair")
        if sample.target.differential[0].disease_id != sample.disease_id:
            raise ValueError(
                "E3 target leading diagnosis does not match the accepted gold label"
            )

        rendered: RenderedOpenResponse | None = None
        if self.variant is E3TrainingVariant.STRUCTURED:
            prompt = STRUCTURED_PROMPT
            task_id = STRUCTURED_PROMPT_ID
            target_text = canonical_structured_json(sample.target)
            template_id = None
            renderer_version = None
            target_sha256 = hashlib.sha256(target_text.encode()).hexdigest()
        else:
            prompt = OPEN_RESPONSE_PROMPT
            task_id = OPEN_RESPONSE_PROMPT_ID
            rendered = DeterministicOpenResponseRenderer(self.taxonomy).render(
                sample.sample_id,
                sample.target,
            )
            target_text = rendered.text
            template_id = rendered.template_id
            renderer_version = rendered.renderer_version
            target_sha256 = rendered.target_sha256

        messages = (
            ChatMessage(
                role="user",
                content=(
                    ImageMessageContent(image=sample.image),
                    TextMessageContent(text=prompt),
                ),
            ),
            ChatMessage(
                role="assistant",
                content=(TextMessageContent(text=target_text),),
            ),
        )
        return E3FormattedExample(
            sample_id=sample.sample_id,
            leakage_group_id=sample.leakage_group_id,
            disease_id=sample.disease_id,
            label=sample.label,
            task_id=task_id,
            target_variant=self.variant,
            messages=messages,
            template_id=template_id,
            renderer_version=renderer_version,
            target_sha256=target_sha256,
        )
