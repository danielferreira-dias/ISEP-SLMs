"""Human-only multimodal formatting for E2 diagnosis, morphology, and captions."""

from __future__ import annotations

from dataclasses import dataclass

from src.train.domain import (
    ChatMessage,
    ImageMessageContent,
    Taxonomy,
    TextMessageContent,
)
from src.train.e2.domain import (
    E2FormattedExample,
    E2HumanSample,
    E2TaskName,
    SkinConOntology,
)
from src.train.phases.label_only import LabelOnlyPhase


def caption_prompt() -> str:
    """Build the frozen answer-blind SkinCAP observation prompt."""

    return (
        "Describe only the visible dermatological findings in the clinical "
        "image using one short clinical sentence. Do not provide a diagnosis, "
        "differential diagnosis, testing, management, prognosis, or advice."
        "\n\n/no_think"
    )


def morphology_prompt(ontology: SkinConOntology) -> str:
    """Build the frozen answer-blind SKINCON prompt."""

    concepts = "\n".join(f"- {concept}" for concept in ontology.concepts)
    return (
        "Identify every morphology concept that is visibly present in the "
        "clinical dermatology image. Use only the SKINCON ontology below. "
        "Return one compact JSON object with keys positive_concepts and "
        "all_concepts_annotated; do not diagnose the disease or add prose."
        f"\n\nAllowed concepts:\n{concepts}\n\n/no_think"
    )


@dataclass(frozen=True, slots=True)
class E2HumanPhase:
    """Validate and render canonical human targets without teacher content."""

    taxonomy: Taxonomy
    ontology: SkinConOntology

    @property
    def diagnosis_prompt(self) -> str:
        """Return the exact E1 diagnostic prompt reused in E2."""

        return LabelOnlyPhase(self.taxonomy).prompt

    @property
    def morphology_prompt(self) -> str:
        """Return the exact SKINCON perception prompt."""

        return morphology_prompt(self.ontology)

    @property
    def caption_prompt(self) -> str:
        """Return the exact SkinCAP observation-only prompt."""

        return caption_prompt()

    def format_example(self, sample: E2HumanSample) -> E2FormattedExample:
        """Validate one human target and produce its multimodal conversation."""

        if sample.task is E2TaskName.DIAGNOSIS:
            self._validate_diagnosis(sample)
        elif sample.task is E2TaskName.MORPHOLOGY:
            self._validate_morphology(sample)
        else:
            self._validate_caption(sample)
        messages = (
            ChatMessage(
                role="user",
                content=(
                    ImageMessageContent(image=sample.image),
                    TextMessageContent(text=sample.prompt),
                ),
            ),
            ChatMessage(
                role="assistant",
                content=(TextMessageContent(text=sample.target_text),),
            ),
        )
        return E2FormattedExample(
            sample_id=sample.sample_id,
            leakage_group_id=sample.leakage_group_id,
            task=sample.task,
            target_text=sample.target_text,
            subset=sample.subset,
            source=sample.source,
            image_sha256=sample.image_sha256,
            image_width=sample.image_width,
            image_height=sample.image_height,
            pixel_count=sample.pixel_count,
            resized_width=sample.resized_width,
            resized_height=sample.resized_height,
            annotation_availability=sample.annotation_availability,
            messages=messages,
        )

    def _validate_diagnosis(self, sample: E2HumanSample) -> None:
        labels = dict(zip(self.taxonomy.disease_ids, self.taxonomy.labels, strict=True))
        if sample.disease_id is None or sample.label is None:
            raise ValueError("Diagnosis rows require disease_id and label")
        if labels.get(sample.disease_id) != sample.label:
            raise ValueError("Diagnosis row contains a non-canonical label pair")
        if sample.target_text != sample.label:
            raise ValueError("Diagnosis target must equal the canonical label")
        if sample.prompt != self.diagnosis_prompt:
            raise ValueError("Diagnosis prompt differs from the frozen E1 prompt")
        if sample.morphology is not None:
            raise ValueError("Diagnosis rows must not carry morphology targets")
        if sample.source_caption_sha256 is not None:
            raise ValueError("Diagnosis rows must not carry caption provenance")

    def _validate_morphology(self, sample: E2HumanSample) -> None:
        target = sample.morphology
        if target is None:
            raise ValueError("Morphology rows require a SKINCON target")
        if not target.all_concepts_annotated:
            raise ValueError("E2 admits only fully annotated SKINCON rows")
        unknown = set(target.positive_concepts) - set(self.ontology.concepts)
        if unknown:
            raise ValueError(f"Unknown SKINCON concepts: {sorted(unknown)}")
        ordered = tuple(
            concept
            for concept in self.ontology.concepts
            if concept in set(target.positive_concepts)
        )
        if ordered != target.positive_concepts:
            raise ValueError("SKINCON positives must follow frozen ontology order")
        if sample.target_text != target.canonical_json():
            raise ValueError("Morphology target JSON is not canonical")
        if sample.prompt != self.morphology_prompt:
            raise ValueError("Morphology prompt differs from the frozen prompt")
        if sample.disease_id is not None or sample.label is not None:
            raise ValueError("Morphology training rows must not expose diagnosis")
        if sample.source_caption_sha256 is not None:
            raise ValueError("Morphology rows must not carry caption provenance")

    def _validate_caption(self, sample: E2HumanSample) -> None:
        if sample.prompt != self.caption_prompt:
            raise ValueError("Caption prompt differs from the frozen prompt")
        if sample.disease_id is not None or sample.label is not None:
            raise ValueError("Caption training rows must not expose diagnosis")
        if sample.morphology is not None:
            raise ValueError("Caption rows must not carry morphology targets")
        digest = sample.source_caption_sha256
        if digest is None or len(digest) != 64:
            raise ValueError("Caption row has no source-caption SHA-256")
        if sample.caption_transform_version != "skincap_observation_prefix_v1":
            raise ValueError("Caption row has an unknown transform version")
        target = sample.target_text.strip()
        if target != sample.target_text or len(target.split()) < 5:
            raise ValueError("Caption target is empty, padded, or too short")
        if "\n" in target:
            raise ValueError("Caption target must be a single short paragraph")
