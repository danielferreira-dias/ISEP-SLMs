"""Task-isolated multimodal conversation formatting for E3 hard KD."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum

from PIL import Image

from src.train.domain import (
    ChatMessage,
    ImageMessageContent,
    Taxonomy,
    TextMessageContent,
)
from src.train.e3.domain import (
    StageATarget,
    StageBTarget,
    StageReviewStatus,
    TeacherTargetBundle,
)
from src.train.e3.rendering import (
    DeterministicContextPolicyRenderer,
    DeterministicGroundedDifferentialRenderer,
    RenderedContextPolicy,
    RenderedGroundedDifferential,
    canonical_stage_a_json,
)
from src.train.phases.label_only import LabelOnlyPhase

DIAGNOSIS_PROMPT_ID = "e3_diagnosis_replay_v1"
MORPHOLOGY_PROMPT_ID = "e3_stage_a_morphology_v1"
CAPTION_PROMPT_ID = "e3_stage_a_caption_v1"
GROUNDED_DIFFERENTIAL_PROMPT_ID = "e3_grounded_differential_v1"
CONTEXT_POLICY_PROMPT_ID = "e3_context_policy_v1"

MORPHOLOGY_PROMPT = (
    "Assess only what is visible in the dermatology image. Return one compact "
    "JSON object containing image_assessment, dominant_visual_pattern, "
    "observations, and not_assessable_features. Every observation must use "
    "present, absent_in_observed_scope, uncertain, not_assessable, or not_shown. "
    "Do not diagnose a disease or invent history, palpation, tests, or metadata."
    "\n\n/no_think"
)

CAPTION_PROMPT = (
    "Describe only the visible dermatological findings in one complete short "
    "clinical caption. End at a sentence boundary. Do not provide a diagnosis, "
    "differential, clinical history, testing, management, prognosis, or advice."
    "\n\n/no_think"
)

GROUNDED_DIFFERENTIAL_PROMPT = (
    "Assess the dermatology image using only visible evidence. Begin with a "
    "complete visual description, then provide the leading diagnosis, relevant "
    "ranked alternatives, supporting findings, confidence, image limitations, "
    "and missing diagnostic discriminators. Do not invent clinical history, "
    "tests, or non-visible findings. Do not choose a clinical action or ask a "
    "follow-up question in this E3 task.\n\n/no_think"
)

CONTEXT_POLICY_PROMPT = (
    "Decide whether the dermatology image contains enough information to give "
    "a grounded differential or whether specific additional context is needed. "
    "Return one compact JSON object containing information_sufficiency, "
    "response_policy, decision_rationale, and requests. Use only "
    "ANSWER_DIFFERENTIAL or REQUEST_CONTEXT. Every request must contain one "
    "explicit complete question, its required source, the diagnoses it helps "
    "distinguish, and a short rationale. Do not invent clinical history, reveal "
    "a private gold label, or combine both response policies.\n\n/no_think"
)


class E3TrainingVariant(StrEnum):
    """Five separately measurable E3 student behaviors."""

    DIAGNOSIS = "diagnosis"
    MORPHOLOGY = "morphology"
    CAPTION = "caption"
    GROUNDED_DIFFERENTIAL = "grounded_differential"
    CONTEXT_POLICY = "context_policy"


@dataclass(frozen=True, slots=True)
class E3TrainingSample:
    """One image, private gold label, and independently reviewed A/B targets."""

    sample_id: str
    leakage_group_id: str
    disease_id: str
    label: str
    image: Image.Image
    teacher_targets: TeacherTargetBundle


@dataclass(frozen=True, slots=True)
class E3FormattedExample:
    """One rendered E3 row ready for assistant-only multimodal training."""

    sample_id: str
    leakage_group_id: str
    disease_id: str
    label: str
    task_id: str
    target_variant: E3TrainingVariant
    target_text: str
    target_source_fields: tuple[str, ...]
    messages: tuple[ChatMessage, ...]
    template_id: str | None
    renderer_version: str | None
    target_sha256: str
    stage_a_generation_id: str | None
    stage_b_generation_id: str | None

    def as_record(self) -> dict[str, object]:
        """Convert the example to a TRL/Hugging Face-compatible record."""

        return {
            "sample_id": self.sample_id,
            "leakage_group_id": self.leakage_group_id,
            "disease_id": self.disease_id,
            "label": self.label,
            "phase": "e3_hard_kd",
            "task": self.target_variant.value,
            "task_id": self.task_id,
            "target_variant": self.target_variant.value,
            "target_text": self.target_text,
            "target_source_fields": list(self.target_source_fields),
            "template_id": self.template_id,
            "renderer_version": self.renderer_version,
            "target_sha256": self.target_sha256,
            "stage_a_generation_id": self.stage_a_generation_id,
            "stage_b_generation_id": self.stage_b_generation_id,
            "messages": [message.as_record() for message in self.messages],
        }


@dataclass(frozen=True, slots=True)
class E3HardKDPhase:
    """Render one task without leaking gold or cross-task output contracts."""

    taxonomy: Taxonomy
    variant: E3TrainingVariant

    def __post_init__(self) -> None:
        """Reject empty or ambiguous taxonomies."""

        if not self.taxonomy.classes:
            raise ValueError("E3 requires a non-empty taxonomy")
        if len(set(self.taxonomy.disease_ids)) != len(self.taxonomy.disease_ids):
            raise ValueError("E3 taxonomy contains duplicate disease IDs")
        if len(set(self.taxonomy.labels)) != len(self.taxonomy.labels):
            raise ValueError("E3 taxonomy contains duplicate labels")

    def format_example(self, sample: E3TrainingSample) -> E3FormattedExample:
        """Render one of five E3 targets from the minimum accepted sources."""

        self._validate_gold_pair(sample)
        if self.variant is E3TrainingVariant.DIAGNOSIS:
            rendered = self._diagnosis_target(sample)
        elif self.variant is E3TrainingVariant.MORPHOLOGY:
            rendered = self._morphology_target(sample)
        elif self.variant is E3TrainingVariant.CAPTION:
            rendered = self._caption_target(sample)
        elif self.variant is E3TrainingVariant.GROUNDED_DIFFERENTIAL:
            rendered = self._grounded_differential_target(sample)
        else:
            rendered = self._context_policy_target(sample)

        messages = (
            ChatMessage(
                role="user",
                content=(
                    ImageMessageContent(image=sample.image),
                    TextMessageContent(text=rendered.prompt),
                ),
            ),
            ChatMessage(
                role="assistant",
                content=(TextMessageContent(text=rendered.target_text),),
            ),
        )
        stage_a_provenance = sample.teacher_targets.stage_a_provenance
        stage_b_provenance = sample.teacher_targets.stage_b_provenance
        return E3FormattedExample(
            sample_id=sample.sample_id,
            leakage_group_id=sample.leakage_group_id,
            disease_id=sample.disease_id,
            label=sample.label,
            task_id=rendered.task_id,
            target_variant=self.variant,
            target_text=rendered.target_text,
            target_source_fields=rendered.target_source_fields,
            messages=messages,
            template_id=rendered.template_id,
            renderer_version=rendered.renderer_version,
            target_sha256=rendered.target_sha256,
            stage_a_generation_id=(
                stage_a_provenance.generation_id
                if rendered.uses_stage_a and stage_a_provenance is not None
                else None
            ),
            stage_b_generation_id=(
                stage_b_provenance.generation_id
                if rendered.uses_stage_b and stage_b_provenance is not None
                else None
            ),
        )

    def _validate_gold_pair(self, sample: E3TrainingSample) -> None:
        labels = dict(zip(self.taxonomy.disease_ids, self.taxonomy.labels, strict=True))
        if labels.get(sample.disease_id) != sample.label:
            raise ValueError("E3 sample contains a non-canonical gold label pair")

    def _diagnosis_target(self, sample: E3TrainingSample) -> _RenderedTarget:
        target_text = sample.label
        return _RenderedTarget.from_text(
            prompt=LabelOnlyPhase(self.taxonomy).prompt,
            task_id=DIAGNOSIS_PROMPT_ID,
            target_text=target_text,
            target_source_fields=("gold_diagnosis", "disease_id"),
        )

    def _morphology_target(self, sample: E3TrainingSample) -> _RenderedTarget:
        stage_a = self._accepted_stage_a(sample.teacher_targets)
        self._reject_stage_a_diagnosis_terms(stage_a)
        return _RenderedTarget.from_text(
            prompt=MORPHOLOGY_PROMPT,
            task_id=MORPHOLOGY_PROMPT_ID,
            target_text=canonical_stage_a_json(stage_a),
            target_source_fields=(
                "stage_a.image_assessment",
                "stage_a.dominant_visual_pattern",
                "stage_a.observations",
                "stage_a.not_assessable_features",
            ),
            uses_stage_a=True,
        )

    def _caption_target(self, sample: E3TrainingSample) -> _RenderedTarget:
        stage_a = self._accepted_stage_a(sample.teacher_targets)
        self._reject_stage_a_diagnosis_terms(stage_a)
        return _RenderedTarget.from_text(
            prompt=CAPTION_PROMPT,
            task_id=CAPTION_PROMPT_ID,
            target_text=stage_a.clinical_caption,
            target_source_fields=("stage_a.clinical_caption",),
            uses_stage_a=True,
        )

    def _grounded_differential_target(
        self,
        sample: E3TrainingSample,
    ) -> _RenderedTarget:
        stage_a = self._accepted_stage_a(sample.teacher_targets)
        stage_b = self._accepted_stage_b_diagnostic(sample.teacher_targets)
        self._reject_stage_a_diagnosis_terms(stage_a)
        self._validate_stage_b_taxonomy(stage_b)
        self._validate_diagnostic_for_sample(stage_b, sample)
        rendered: RenderedGroundedDifferential = (
            DeterministicGroundedDifferentialRenderer(self.taxonomy).render(
                sample.sample_id,
                stage_a,
                stage_b,
            )
        )
        return _RenderedTarget(
            prompt=GROUNDED_DIFFERENTIAL_PROMPT,
            task_id=GROUNDED_DIFFERENTIAL_PROMPT_ID,
            target_text=rendered.text,
            target_source_fields=(
                "stage_a.clinical_caption",
                "stage_a.observations",
                "stage_a.not_assessable_features",
                "stage_b.diagnostic_assessment.differential",
            ),
            template_id=rendered.template_id,
            renderer_version=rendered.renderer_version,
            target_sha256=rendered.target_sha256,
            uses_stage_a=True,
            uses_stage_b=True,
        )

    def _context_policy_target(self, sample: E3TrainingSample) -> _RenderedTarget:
        stage_a = self._accepted_stage_a(sample.teacher_targets)
        stage_b = self._accepted_stage_b_context_policy(sample.teacher_targets)
        self._reject_stage_a_diagnosis_terms(stage_a)
        self._validate_context_policy(stage_b)
        rendered: RenderedContextPolicy = DeterministicContextPolicyRenderer(
            self.taxonomy
        ).render(stage_b.context_decision)
        return _RenderedTarget(
            prompt=CONTEXT_POLICY_PROMPT,
            task_id=CONTEXT_POLICY_PROMPT_ID,
            target_text=rendered.text,
            target_source_fields=(
                "stage_b.context_decision.information_sufficiency",
                "stage_b.context_decision.response_policy",
                "stage_b.context_decision.decision_rationale",
                "stage_b.context_decision.requests",
            ),
            renderer_version=rendered.renderer_version,
            target_sha256=rendered.target_sha256,
            uses_stage_a=True,
            uses_stage_b=True,
        )

    @staticmethod
    def _accepted_stage_a(targets: TeacherTargetBundle) -> StageATarget:
        if (
            targets.stage_a_status is not StageReviewStatus.ACCEPTED
            or targets.stage_a_target is None
        ):
            raise ValueError("This E3 task requires accepted Stage A")
        return targets.stage_a_target

    @staticmethod
    def _accepted_stage_b_diagnostic(
        targets: TeacherTargetBundle,
    ) -> StageBTarget:
        if (
            targets.stage_b_diagnostic_status is not StageReviewStatus.ACCEPTED
            or targets.stage_b_target is None
        ):
            raise ValueError("This E3 task requires accepted Stage B diagnostic")
        return targets.stage_b_target

    @staticmethod
    def _accepted_stage_b_context_policy(
        targets: TeacherTargetBundle,
    ) -> StageBTarget:
        if (
            targets.stage_b_context_policy_status
            is not StageReviewStatus.ACCEPTED
            or targets.stage_b_target is None
        ):
            raise ValueError("This E3 task requires accepted Stage B context policy")
        return targets.stage_b_target

    def _reject_stage_a_diagnosis_terms(self, stage_a: StageATarget) -> None:
        texts = [stage_a.dominant_visual_pattern, stage_a.clinical_caption]
        texts.extend(item.concept_label for item in stage_a.observations)
        texts.extend(
            item.concept_detail
            for item in stage_a.observations
            if item.concept_detail is not None
        )
        for text in texts:
            for label in self.taxonomy.labels:
                if _contains_phrase(text, label):
                    raise ValueError("Stage A contains a canonical diagnosis term")

    def _validate_stage_b_taxonomy(
        self,
        stage_b: StageBTarget,
    ) -> None:
        differential = stage_b.diagnostic_assessment.differential
        unknown = {
            item.disease_id
            for item in differential
            if item.disease_id not in set(self.taxonomy.disease_ids)
        }
        if unknown:
            raise ValueError(
                "Stage B contains diseases outside the taxonomy: "
                + ", ".join(sorted(unknown))
            )

    @staticmethod
    def _validate_diagnostic_for_sample(
        stage_b: StageBTarget,
        sample: E3TrainingSample,
    ) -> None:
        differential = stage_b.diagnostic_assessment.differential
        if differential[0].disease_id != sample.disease_id:
            raise ValueError(
                "E3 target leading diagnosis does not match the accepted gold label"
            )

    def _validate_context_policy(self, stage_b: StageBTarget) -> None:
        decision = stage_b.context_decision
        requests = decision.requests
        known = set(self.taxonomy.disease_ids)
        referenced = {
            disease_id
            for request in requests
            for disease_id in request.discriminates_between
        }
        unknown = referenced - known
        if unknown:
            raise ValueError(
                "Context policy contains diseases outside the taxonomy: "
                + ", ".join(sorted(unknown))
            )
        normalized_questions = tuple(
            " ".join(request.question.casefold().split()) for request in requests
        )
        if len(normalized_questions) != len(set(normalized_questions)):
            raise ValueError("Context policy contains duplicate questions")
        if decision.response_policy.value == "REQUEST_CONTEXT":
            leading = stage_b.diagnostic_assessment.differential[0].disease_id
            if leading not in referenced or len(referenced) < 2:
                raise ValueError(
                    "Context requests must discriminate the leading diagnosis "
                    "from at least one alternative"
                )


@dataclass(frozen=True, slots=True)
class _RenderedTarget:
    prompt: str
    task_id: str
    target_text: str
    target_source_fields: tuple[str, ...]
    template_id: str | None = None
    renderer_version: str | None = None
    target_sha256: str = ""
    uses_stage_a: bool = False
    uses_stage_b: bool = False

    @classmethod
    def from_text(
        cls,
        *,
        prompt: str,
        task_id: str,
        target_text: str,
        target_source_fields: tuple[str, ...],
        uses_stage_a: bool = False,
    ) -> _RenderedTarget:
        return cls(
            prompt=prompt,
            task_id=task_id,
            target_text=target_text,
            target_source_fields=target_source_fields,
            target_sha256=hashlib.sha256(target_text.encode()).hexdigest(),
            uses_stage_a=uses_stage_a,
        )


def _contains_phrase(text: str, phrase: str) -> bool:
    """Match canonical disease phrases across punctuation and whitespace."""

    def normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

    normalized_text = f" {normalize(text)} "
    normalized_phrase = normalize(phrase)
    return bool(normalized_phrase) and f" {normalized_phrase} " in normalized_text
