"""Fail-closed two-stage teacher-generation runner for E3 hard distillation."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.inference.base import (
    InferenceBackend,
    InferenceRequest,
    InferenceResult,
    InferenceSafetyRefusal,
    InferenceTransportError,
)
from src.train.domain import Taxonomy
from src.train.e3.domain import (
    ProviderSafetyCategory,
    ResponsePolicy,
    StageATarget,
    StageBTarget,
    StageReviewStatus,
    TeacherGenerationProvenance,
    TeacherGenerationStatus,
    TeacherTargetBundle,
)
from src.train.e3.generation_config import E3TeacherGenerationConfig
from src.train.e3.generation_data import E3Selection, E3TeacherSample
from src.train.e3.generation_records import (
    E3ArtifactStore,
    E3StageArtifact,
    E3TeacherBundleRecord,
)
from src.train.e3.progress import (
    E3CampaignSpec,
    E3CampaignState,
    E3GenerationStage,
    E3ProgressStore,
)
from src.train.e3.prompts import (
    RenderedTeacherPrompt,
    StageAPromptResource,
    StageBPromptResource,
    prompt_resource_sha256,
    render_stage_a_prompt,
    render_stage_b_prompt,
    stage_a_output_schema,
    stage_b_output_schema,
)
from src.train.e3.terminology import (
    DermatologyTerminology,
    terminology_resource_sha256,
)


class E3TeacherGenerationRunner:
    """Execute A then B once per sample, with durable resume and no repair."""

    def __init__(
        self,
        *,
        config: E3TeacherGenerationConfig,
        selection: E3Selection,
        samples: tuple[E3TeacherSample, ...],
        backend: InferenceBackend,
        stage_a_prompt: StageAPromptResource,
        stage_b_prompt: StageBPromptResource,
        output_directory: Path,
        resume: bool = False,
        campaign_id: str | None = None,
        gate_first_case: bool = False,
    ) -> None:
        self.config = config
        self.selection = selection
        self.samples = samples
        self.backend = backend
        self.stage_a_prompt = stage_a_prompt
        self.stage_b_prompt = stage_b_prompt
        self.terminology = config.load_terminology()
        self.output_directory = output_directory.resolve()
        self.resume = resume
        self.campaign_id = campaign_id or config.campaign.id
        self.gate_first_case = gate_first_case
        if tuple(item.candidate for item in samples) != selection.candidates:
            raise ValueError("Loaded E3 samples do not match the frozen selection")

    def run(self) -> dict[str, Any]:
        model = self.config.load_teacher_model()
        source_model = model.source.model_name or model.model.id
        teacher_revision = (
            model.source.revision or f"provider-managed-alias:{source_model}"
        )
        profile = model.backend.active_profile
        spec = E3CampaignSpec(
            campaign_id=self.campaign_id,
            total_samples=len(self.samples),
            provider=model.source.provider or "provider_api",
            backend=f"{profile.engine}:{profile.api_style}",
            teacher_model=source_model,
            teacher_revision=teacher_revision,
            stage_a_prompt_id=self.stage_a_prompt.prompt_id,
            stage_b_prompt_id=self.stage_b_prompt.prompt_id,
            stage_a_gold_visible_to_teacher=False,
            stage_b_gold_visible_to_teacher=(
                self.stage_b_prompt.gold_visible_to_teacher
            ),
            model_config_sha256=_sha256_file(model.config_path),
            reasoning_effort=self.config.model.reasoning_effort,
            structured_output_mode=self.config.model.structured_output_mode,
            selection_sha256=self.selection.selection_sha256,
            stage_a_prompt_resource_sha256=prompt_resource_sha256(
                self.config.path(self.config.prompts.stage_a)
            ),
            stage_a_rendered_prompt_sha256=render_stage_a_prompt(
                self.stage_a_prompt,
                terminology=self.terminology,
            ).prompt_sha256,
            stage_b_prompt_resource_sha256=prompt_resource_sha256(
                self.config.path(self.config.prompts.stage_b)
            ),
            terminology_lexicon_id=self.terminology.lexicon_id,
            terminology_resource_sha256=terminology_resource_sha256(
                self.config.path(self.config.terminology.resource)
            ),
        )
        progress = E3ProgressStore.start(
            self.output_directory,
            spec,
            resume=self.resume,
        )
        artifacts = E3ArtifactStore.start(
            self.output_directory,
            resume=self.resume,
        )
        self._reconcile(progress, artifacts)

        try:
            for sample in self.samples:
                if self._bundle_for(artifacts, sample.candidate.sample_id) is not None:
                    continue
                stage_a = self._stage_for(
                    artifacts,
                    sample.candidate.sample_id,
                    E3GenerationStage.STAGE_A,
                )
                if stage_a is None:
                    stage_a = self._generate_stage_a(sample, teacher_revision)
                    self._commit_stage(progress, artifacts, stage_a)

                stage_b: E3StageArtifact | None = None
                if stage_a.review_status is StageReviewStatus.ACCEPTED:
                    stage_b = self._stage_for(
                        artifacts,
                        sample.candidate.sample_id,
                        E3GenerationStage.STAGE_B,
                    )
                    if stage_b is None:
                        assert isinstance(stage_a.target, StageATarget)
                        stage_b = self._generate_stage_b(
                            sample,
                            stage_a.target,
                            teacher_revision,
                        )
                        self._commit_stage(progress, artifacts, stage_b)

                artifacts.append_bundle(
                    self._build_bundle_record(sample, stage_a, stage_b)
                )
                if self._must_stop(stage_a) or (
                    stage_b is not None and self._must_stop(stage_b)
                ):
                    return progress.finalize(
                        E3CampaignState.FAILED,
                        error="teacher_transport_failure_no_retry",
                    )
                if (
                    self.gate_first_case
                    and sample is self.samples[0]
                    and not _quality_gate_passed(stage_a, stage_b)
                ):
                    return progress.finalize(
                        E3CampaignState.FAILED,
                        error="quality_gate_failed_no_retry",
                    )
            if len(artifacts.read_bundles()) != len(self.samples):
                raise ValueError("E3 bundle count does not match the frozen selection")
            return progress.finalize(E3CampaignState.COMPLETED)
        except Exception:
            if progress.read_snapshot()["status"] == E3CampaignState.RUNNING.value:
                progress.finalize(
                    E3CampaignState.FAILED,
                    error="internal_fail_closed_error",
                )
            raise

    def _generate_stage_a(
        self,
        sample: E3TeacherSample,
        teacher_revision: str,
    ) -> E3StageArtifact:
        prompt = render_stage_a_prompt(
            self.stage_a_prompt,
            terminology=self.terminology,
        )
        artifact = self._call(
            sample=sample,
            stage=E3GenerationStage.STAGE_A,
            prompt=prompt,
            schema=stage_a_output_schema(self.terminology),
            target_type=StageATarget,
            max_output_tokens=self.config.generation.stage_a_max_output_tokens,
            teacher_revision=teacher_revision,
        )
        if (
            artifact.provenance.generation_status
            is not TeacherGenerationStatus.SUCCEEDED
        ):
            return artifact
        assert isinstance(artifact.target, StageATarget)
        reasons = _stage_a_rejection_reasons(
            artifact.target,
            taxonomy=self.selection.taxonomy,
            terminology=self.terminology,
        )
        return E3StageArtifact.model_validate(
            {
                **artifact.model_dump(mode="json"),
                "review_status": (
                    StageReviewStatus.REJECTED.value
                    if reasons
                    else StageReviewStatus.ACCEPTED.value
                ),
                "rejection_reasons": list(reasons),
            }
        )

    def _generate_stage_b(
        self,
        sample: E3TeacherSample,
        stage_a: StageATarget,
        teacher_revision: str,
    ) -> E3StageArtifact:
        prompt = render_stage_b_prompt(
            self.stage_b_prompt,
            taxonomy=self.selection.taxonomy,
            stage_a=stage_a,
            gold_disease_id=sample.candidate.disease_id,
            gold_diagnosis=sample.candidate.gold_diagnosis,
        )
        artifact = self._call(
            sample=sample,
            stage=E3GenerationStage.STAGE_B,
            prompt=prompt,
            schema=stage_b_output_schema(),
            target_type=StageBTarget,
            max_output_tokens=self.config.generation.stage_b_max_output_tokens,
            teacher_revision=teacher_revision,
        )
        if (
            artifact.provenance.generation_status
            is not TeacherGenerationStatus.SUCCEEDED
        ):
            return artifact
        assert isinstance(artifact.target, StageBTarget)
        differential = artifact.target.diagnostic_assessment.differential
        leading_match = differential[0].disease_id == sample.candidate.disease_id
        gold_in_top3 = sample.candidate.disease_id in {
            item.disease_id for item in differential[:3]
        }
        structural_reasons = _stage_b_rejection_reasons(
            artifact.target,
            stage_a=stage_a,
            taxonomy=self.selection.taxonomy,
        )
        diagnostic_reasons = structural_reasons
        if not leading_match:
            diagnostic_reasons = (
                *diagnostic_reasons,
                "leading_diagnosis_does_not_match_private_gold",
            )
        context_policy_reasons = _context_policy_rejection_reasons(
            artifact.target,
            taxonomy=self.selection.taxonomy,
        )
        diagnostic_status = (
            StageReviewStatus.REJECTED
            if diagnostic_reasons
            else StageReviewStatus.ACCEPTED
        )
        context_policy_status = (
            StageReviewStatus.REJECTED
            if context_policy_reasons
            else StageReviewStatus.ACCEPTED
        )
        accepted_subtarget = StageReviewStatus.ACCEPTED in {
            diagnostic_status,
            context_policy_status,
        }
        aggregate_reasons = (
            ()
            if accepted_subtarget
            else tuple(dict.fromkeys((*diagnostic_reasons, *context_policy_reasons)))
        )
        return E3StageArtifact.model_validate(
            {
                **artifact.model_dump(mode="json"),
                "review_status": (
                    StageReviewStatus.ACCEPTED.value
                    if accepted_subtarget
                    else StageReviewStatus.REJECTED.value
                ),
                "rejection_reasons": list(aggregate_reasons),
                "diagnostic_review_status": diagnostic_status.value,
                "diagnostic_rejection_reasons": list(diagnostic_reasons),
                "context_policy_review_status": context_policy_status.value,
                "context_policy_rejection_reasons": list(context_policy_reasons),
                "response_policy": (
                    artifact.target.context_decision.response_policy.value
                ),
                "leading_label_match": leading_match,
                "gold_in_top3": gold_in_top3,
            }
        )

    def _call(
        self,
        *,
        sample: E3TeacherSample,
        stage: E3GenerationStage,
        prompt: RenderedTeacherPrompt,
        schema: dict[str, Any],
        target_type: type[StageATarget] | type[StageBTarget],
        max_output_tokens: int,
        teacher_revision: str,
    ) -> E3StageArtifact:
        generation_id = _generation_id(
            campaign_id=self.campaign_id,
            selection_sha256=self.selection.selection_sha256,
            sample_id=sample.candidate.sample_id,
            stage=stage,
        )
        started = time.monotonic()
        recorded_at = datetime.now(UTC).isoformat()
        request = InferenceRequest(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            image_bytes=sample.image_bytes,
            image_mime_type=sample.image_mime_type,
            schema=schema,
            generation={
                "reasoning_effort": self.config.model.reasoning_effort,
                "max_output_tokens": max_output_tokens,
            },
            request_id=generation_id,
        )
        try:
            result = self.backend.complete(request)
        except InferenceSafetyRefusal as error:
            return self._failed_artifact(
                sample=sample,
                stage=stage,
                prompt=prompt,
                generation_id=generation_id,
                teacher_revision=teacher_revision,
                generation_status=TeacherGenerationStatus.PROVIDER_SAFETY_REFUSAL,
                provider_error_code=_safe_error_code(error.details),
                safety_categories=_safety_categories(error.details),
                recorded_at=recorded_at,
                latency_seconds=time.monotonic() - started,
            )
        except (InferenceTransportError, TimeoutError) as error:
            is_timeout = (
                isinstance(error, TimeoutError) or "timeout" in str(error).casefold()
            )
            return self._failed_artifact(
                sample=sample,
                stage=stage,
                prompt=prompt,
                generation_id=generation_id,
                teacher_revision=teacher_revision,
                generation_status=(
                    TeacherGenerationStatus.TIMEOUT
                    if is_timeout
                    else TeacherGenerationStatus.TRANSPORT_ERROR
                ),
                provider_error_code=("timeout" if is_timeout else type(error).__name__),
                recorded_at=recorded_at,
                latency_seconds=time.monotonic() - started,
            )

        latency = time.monotonic() - started
        if not result.final_text.strip():
            return self._failed_from_result(
                sample=sample,
                stage=stage,
                prompt=prompt,
                generation_id=generation_id,
                teacher_revision=teacher_revision,
                result=result,
                status=TeacherGenerationStatus.EMPTY_RESPONSE,
                error_code="empty_response",
                recorded_at=recorded_at,
                latency_seconds=latency,
            )
        if result.finish_reason == "length" or result.metadata.get("truncated") is True:
            return self._failed_from_result(
                sample=sample,
                stage=stage,
                prompt=prompt,
                generation_id=generation_id,
                teacher_revision=teacher_revision,
                result=result,
                status=TeacherGenerationStatus.INVALID_SCHEMA,
                error_code="truncated_response",
                recorded_at=recorded_at,
                latency_seconds=latency,
            )
        try:
            payload = json.loads(result.final_text)
            target = target_type.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError):
            return self._failed_from_result(
                sample=sample,
                stage=stage,
                prompt=prompt,
                generation_id=generation_id,
                teacher_revision=teacher_revision,
                result=result,
                status=TeacherGenerationStatus.INVALID_SCHEMA,
                error_code="strict_schema_validation_failed",
                recorded_at=recorded_at,
                latency_seconds=latency,
            )
        provenance = self._provenance(
            stage=stage,
            prompt=prompt,
            generation_id=generation_id,
            teacher_revision=teacher_revision,
            status=TeacherGenerationStatus.SUCCEEDED,
            result=result,
        )
        response_policy = None
        leading_label_match = None
        gold_in_top3 = None
        if isinstance(target, StageBTarget):
            differential = target.diagnostic_assessment.differential
            response_policy = target.context_decision.response_policy
            leading_label_match = (
                differential[0].disease_id == sample.candidate.disease_id
            )
            gold_in_top3 = sample.candidate.disease_id in {
                item.disease_id for item in differential[:3]
            }
        return E3StageArtifact(
            event_id=generation_id,
            sample_id=sample.candidate.sample_id,
            stage=stage,
            review_status=StageReviewStatus.ACCEPTED,
            target=target,
            provenance=provenance,
            recorded_at=recorded_at,
            latency_seconds=latency,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            response_policy=response_policy,
            leading_label_match=leading_label_match,
            gold_in_top3=gold_in_top3,
        )

    def _failed_from_result(
        self,
        *,
        sample: E3TeacherSample,
        stage: E3GenerationStage,
        prompt: RenderedTeacherPrompt,
        generation_id: str,
        teacher_revision: str,
        result: InferenceResult,
        status: TeacherGenerationStatus,
        error_code: str,
        recorded_at: str,
        latency_seconds: float,
    ) -> E3StageArtifact:
        return E3StageArtifact(
            event_id=generation_id,
            sample_id=sample.candidate.sample_id,
            stage=stage,
            review_status=StageReviewStatus.NOT_APPLICABLE,
            provenance=self._provenance(
                stage=stage,
                prompt=prompt,
                generation_id=generation_id,
                teacher_revision=teacher_revision,
                status=status,
                result=result,
                provider_error_code=error_code,
            ),
            recorded_at=recorded_at,
            latency_seconds=latency_seconds,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
        )

    def _failed_artifact(
        self,
        *,
        sample: E3TeacherSample,
        stage: E3GenerationStage,
        prompt: RenderedTeacherPrompt,
        generation_id: str,
        teacher_revision: str,
        generation_status: TeacherGenerationStatus,
        provider_error_code: str,
        recorded_at: str,
        latency_seconds: float,
        safety_categories: tuple[ProviderSafetyCategory, ...] = (),
    ) -> E3StageArtifact:
        model = self.config.load_teacher_model()
        return E3StageArtifact(
            event_id=generation_id,
            sample_id=sample.candidate.sample_id,
            stage=stage,
            review_status=StageReviewStatus.NOT_APPLICABLE,
            provenance=TeacherGenerationProvenance(
                generation_id=generation_id,
                generation_status=generation_status,
                provider=model.source.provider or "provider_api",
                teacher_model=model.source.model_name or model.model.id,
                teacher_revision=teacher_revision,
                prompt_id=prompt.prompt_id,
                prompt_sha256=prompt.prompt_sha256,
                provider_error_code=provider_error_code,
                safety_categories=safety_categories,
                gold_visible_to_teacher=self._gold_visible_for_stage(stage),
            ),
            recorded_at=recorded_at,
            latency_seconds=latency_seconds,
        )

    def _provenance(
        self,
        *,
        stage: E3GenerationStage,
        prompt: RenderedTeacherPrompt,
        generation_id: str,
        teacher_revision: str,
        status: TeacherGenerationStatus,
        result: InferenceResult,
        provider_error_code: str | None = None,
    ) -> TeacherGenerationProvenance:
        model = self.config.load_teacher_model()
        provider_model = result.metadata.get("provider_model")
        return TeacherGenerationProvenance(
            generation_id=generation_id,
            generation_status=status,
            provider=model.source.provider or "provider_api",
            teacher_model=model.source.model_name or model.model.id,
            teacher_revision=teacher_revision,
            prompt_id=prompt.prompt_id,
            prompt_sha256=prompt.prompt_sha256,
            provider_response_id=result.provider_response_id,
            provider_model_reported=(
                provider_model if isinstance(provider_model, str) else None
            ),
            finish_reason=result.finish_reason,
            provider_error_code=provider_error_code,
            gold_visible_to_teacher=self._gold_visible_for_stage(stage),
        )

    def _gold_visible_for_stage(self, stage: E3GenerationStage) -> bool:
        return (
            stage is E3GenerationStage.STAGE_B
            and self.stage_b_prompt.gold_visible_to_teacher
        )

    def _build_bundle_record(
        self,
        sample: E3TeacherSample,
        stage_a: E3StageArtifact,
        stage_b: E3StageArtifact | None,
    ) -> E3TeacherBundleRecord:
        bundle = TeacherTargetBundle(
            stage_a_status=stage_a.review_status,
            stage_a_target=(
                stage_a.target if isinstance(stage_a.target, StageATarget) else None
            ),
            stage_a_provenance=stage_a.provenance,
            stage_a_rejection_reasons=stage_a.rejection_reasons,
            stage_b_status=(
                stage_b.review_status
                if stage_b is not None
                else StageReviewStatus.NOT_GENERATED
            ),
            stage_b_target=(
                stage_b.target
                if stage_b is not None and isinstance(stage_b.target, StageBTarget)
                else None
            ),
            stage_b_provenance=(stage_b.provenance if stage_b is not None else None),
            stage_b_rejection_reasons=(
                stage_b.rejection_reasons if stage_b is not None else ()
            ),
            stage_b_diagnostic_status=(
                stage_b.diagnostic_review_status
                if stage_b is not None
                else StageReviewStatus.NOT_GENERATED
            ),
            stage_b_diagnostic_rejection_reasons=(
                stage_b.diagnostic_rejection_reasons if stage_b is not None else ()
            ),
            stage_b_context_policy_status=(
                stage_b.context_policy_review_status
                if stage_b is not None
                else StageReviewStatus.NOT_GENERATED
            ),
            stage_b_context_policy_rejection_reasons=(
                stage_b.context_policy_rejection_reasons
                if stage_b is not None
                else ()
            ),
        )
        candidate = sample.candidate
        return E3TeacherBundleRecord(
            sample_id=candidate.sample_id,
            leakage_group_id=candidate.leakage_group_id,
            disease_id=candidate.disease_id,
            gold_diagnosis=candidate.gold_diagnosis,
            split=candidate.split,
            image_sha256=candidate.image_sha256,
            image_width=sample.image_width,
            image_height=sample.image_height,
            teacher_targets=bundle,
        )

    def _commit_stage(
        self,
        progress: E3ProgressStore,
        artifacts: E3ArtifactStore,
        artifact: E3StageArtifact,
    ) -> None:
        artifacts.append_stage(artifact)
        progress.record(artifact.progress_event())

    @staticmethod
    def _reconcile(
        progress: E3ProgressStore,
        artifacts: E3ArtifactStore,
    ) -> None:
        artifact_values = artifacts.read_stage_results()
        progress_values = progress.read_events()
        artifact_keys = {(item.sample_id, item.stage) for item in artifact_values}
        progress_keys = {(item.sample_id, item.stage) for item in progress_values}
        missing_private = progress_keys - artifact_keys
        if missing_private:
            raise ValueError("Progress exists without private stage target artifacts")
        for artifact in artifact_values:
            key = (artifact.sample_id, artifact.stage)
            if key not in progress_keys:
                progress.record(artifact.progress_event())

    @staticmethod
    def _stage_for(
        artifacts: E3ArtifactStore,
        sample_id: str,
        stage: E3GenerationStage,
    ) -> E3StageArtifact | None:
        return next(
            (
                item
                for item in artifacts.read_stage_results()
                if item.sample_id == sample_id and item.stage is stage
            ),
            None,
        )

    @staticmethod
    def _bundle_for(
        artifacts: E3ArtifactStore,
        sample_id: str,
    ) -> E3TeacherBundleRecord | None:
        return next(
            (item for item in artifacts.read_bundles() if item.sample_id == sample_id),
            None,
        )

    def _must_stop(self, artifact: E3StageArtifact) -> bool:
        return self.config.generation.stop_on_transport_error and (
            artifact.provenance.generation_status
            in {
                TeacherGenerationStatus.TRANSPORT_ERROR,
                TeacherGenerationStatus.TIMEOUT,
            }
        )


def _quality_gate_passed(
    stage_a: E3StageArtifact,
    stage_b: E3StageArtifact | None,
) -> bool:
    """Require parseable A+B provider output before a quality slice expands."""

    return (
        stage_a.provenance.generation_status is TeacherGenerationStatus.SUCCEEDED
        and isinstance(stage_a.target, StageATarget)
        and stage_a.review_status is StageReviewStatus.ACCEPTED
        and stage_b is not None
        and stage_b.provenance.generation_status
        is TeacherGenerationStatus.SUCCEEDED
        and isinstance(stage_b.target, StageBTarget)
    )


def _stage_a_rejection_reasons(
    target: StageATarget,
    *,
    taxonomy: Taxonomy,
    terminology: DermatologyTerminology,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not target.image_assessment.is_evaluable:
        reasons.append("stage_a_image_not_evaluable")
    text = json.dumps(target.model_dump(mode="json"), ensure_ascii=False).casefold()
    terms: list[str] = list(taxonomy.disease_ids)
    for label in taxonomy.labels:
        terms.extend((label, label.replace("_", " ")))
    leaked = sorted(
        {
            term
            for term in terms
            if re.search(
                rf"(?<![a-z0-9]){re.escape(term.casefold())}(?![a-z0-9])", text
            )
        }
    )
    if leaked:
        reasons.append("stage_a_contains_canonical_diagnosis_term")
    for observation in target.observations:
        reasons.extend(
            terminology.audit_observation(
                concept_id=observation.concept_id,
                concept_label=observation.concept_label,
                image_modality=target.image_assessment.image_modality,
            )
        )
    return tuple(dict.fromkeys(reasons))


def _stage_b_rejection_reasons(
    target: StageBTarget,
    *,
    stage_a: StageATarget,
    taxonomy: Taxonomy,
) -> tuple[str, ...]:
    reasons: list[str] = []
    known_diseases = set(taxonomy.disease_ids)
    generated_diseases = {
        item.disease_id for item in target.diagnostic_assessment.differential
    }
    if generated_diseases - known_diseases:
        reasons.append("stage_b_contains_disease_outside_closed_taxonomy")
    known_observations = {item.id for item in stage_a.observations}
    for item in target.diagnostic_assessment.differential:
        linked = set(item.supporting_observation_ids).union(
            item.contradicting_observation_ids
        )
        if linked - known_observations:
            reasons.append("stage_b_contains_unknown_stage_a_evidence_link")
            break
    if any(
        item.observation_id not in known_observations
        for item in target.stage_b_corrections
    ):
        reasons.append("stage_b_correction_has_unknown_stage_a_link")
    return tuple(dict.fromkeys(reasons))


def _context_policy_rejection_reasons(
    target: StageBTarget,
    *,
    taxonomy: Taxonomy,
) -> tuple[str, ...]:
    """Review policy structure without treating private-gold agreement as truth."""

    reasons: list[str] = []
    decision = target.context_decision
    requests = decision.requests
    normalized_questions = tuple(
        " ".join(request.question.casefold().split()) for request in requests
    )
    if len(normalized_questions) != len(set(normalized_questions)):
        reasons.append("context_policy_contains_duplicate_questions")

    known_diseases = set(taxonomy.disease_ids)
    referenced_diseases = {
        disease_id
        for request in requests
        for disease_id in request.discriminates_between
    }
    if referenced_diseases - known_diseases:
        reasons.append("context_policy_references_disease_outside_closed_taxonomy")

    if decision.response_policy is ResponsePolicy.REQUEST_CONTEXT:
        leading_disease = target.diagnostic_assessment.differential[0].disease_id
        if leading_disease not in referenced_diseases:
            reasons.append("context_policy_requests_do_not_cover_leading_diagnosis")
        if len(referenced_diseases) < 2:
            reasons.append("context_policy_requests_do_not_cover_an_alternative")

    return tuple(dict.fromkeys(reasons))


def _generation_id(
    *,
    campaign_id: str,
    selection_sha256: str,
    sample_id: str,
    stage: E3GenerationStage,
) -> str:
    payload = f"{campaign_id}\n{selection_sha256}\n{sample_id}\n{stage.value}".encode()
    return f"e3-{hashlib.sha256(payload).hexdigest()}"


def _safe_error_code(details: dict[str, Any]) -> str:
    value = details.get("code") or details.get("type") or "provider_safety_refusal"
    return str(value)[:120]


def _safety_categories(details: dict[str, Any]) -> tuple[ProviderSafetyCategory, ...]:
    tree = details.get("content_filter")
    found: dict[str, ProviderSafetyCategory] = {}

    def visit(value: Any, path: tuple[str, ...] = ()) -> None:
        if not isinstance(value, dict):
            return
        filtered = value.get("filtered")
        severity = value.get("severity")
        if path and (isinstance(filtered, bool) or isinstance(severity, str)):
            category = path[-1]
            found[category] = ProviderSafetyCategory(
                category=category,
                severity=severity if isinstance(severity, str) else None,
                filtered=filtered if isinstance(filtered, bool) else None,
            )
        for key, nested in value.items():
            if isinstance(nested, dict):
                visit(nested, (*path, str(key)))

    visit(tree)
    if not found:
        found["provider_content_filter"] = ProviderSafetyCategory(
            category="provider_content_filter",
            filtered=True,
        )
    return tuple(found[key] for key in sorted(found))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
