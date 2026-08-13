"""E1 closed-taxonomy label-only supervision."""

from __future__ import annotations

from dataclasses import dataclass

from src.train.domain import (
    ChatMessage,
    FormattedExample,
    ImageMessageContent,
    LabeledImageSample,
    Taxonomy,
    TextMessageContent,
    TrainingPhaseName,
)


@dataclass(frozen=True, slots=True)
class LabelOnlyPhase:
    """Render an image and a closed label set with no clinical side channels."""

    taxonomy: Taxonomy
    name: TrainingPhaseName = TrainingPhaseName.E1_LABEL

    def __post_init__(self) -> None:
        """Reject empty or ambiguous taxonomies at construction time."""

        if not self.taxonomy.classes:
            raise ValueError("Label-only training requires a non-empty taxonomy")
        if len(set(self.taxonomy.labels)) != len(self.taxonomy.labels):
            raise ValueError("Label-only taxonomy contains duplicate labels")

    @property
    def prompt(self) -> str:
        """Return the frozen label-only instruction shown to every sample."""

        labels = "\n".join(f"- {label}" for label in self.taxonomy.labels)
        return (
            "Classify the clinical dermatology image using the closed taxonomy "
            "below. Return exactly one canonical label and nothing else: no "
            "explanation, reasoning, punctuation, or additional text.\n\n"
            f"Allowed labels:\n{labels}\n\n/no_think"
        )

    def format_example(self, sample: LabeledImageSample) -> FormattedExample:
        """Render one sample with the exact canonical label as assistant target.

        Args:
            sample: Decoded source image with its canonical supervision fields.

        Returns:
            Two-message multimodal example suitable for Unsloth's vision
            collator and assistant-only loss masking.

        Raises:
            ValueError: If label and disease ID are not a canonical pair.
        """

        canonical = dict(
            zip(
                self.taxonomy.disease_ids,
                self.taxonomy.labels,
                strict=True,
            )
        )
        if canonical.get(sample.disease_id) != sample.label:
            raise ValueError(
                f"Non-canonical E1 target: {sample.disease_id}/{sample.label}"
            )
        messages = (
            ChatMessage(
                role="user",
                content=(
                    ImageMessageContent(image=sample.image),
                    TextMessageContent(text=self.prompt),
                ),
            ),
            ChatMessage(
                role="assistant",
                content=(TextMessageContent(text=sample.label),),
            ),
        )
        return FormattedExample(
            sample_id=sample.sample_id,
            leakage_group_id=sample.leakage_group_id,
            disease_id=sample.disease_id,
            label=sample.label,
            messages=messages,
        )
