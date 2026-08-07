"""Task-specific prompt, validation, and metric adapters.

The execution layer only needs to load one benchmark YAML, select an adapter,
call :meth:`prepare`, send the returned prompts/schema to a backend, and pass
the backend's final text to :meth:`parse_response`. This keeps model execution
independent from benchmark-specific output formats.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from src.benchmark.evidence_metrics import (
    compute_evidence_grounded_metrics,
)
from src.benchmark.evidence_validation import (
    parse_and_validate_evidence_response,
)
from src.benchmark.metrics import (
    compute_clinical_context_ablation_metrics,
    compute_confusion_set_metrics,
    compute_metrics,
)
from src.benchmark.hallucination import (
    compute_dermatology_counterfactual_metrics,
    compute_general_visual_hallucination_metrics,
    parse_dermatology_counterfactual_response,
    parse_general_visual_hallucination_response,
)
from src.benchmark.runner import (
    BenchmarkPrediction,
    BenchmarkSample,
    ModelResponse,
    parse_and_validate_response,
)
from src.benchmark.visual_grounding import (
    compute_visual_grounding_metrics,
    parse_and_validate_visual_grounding_response,
)


@dataclass(frozen=True, slots=True)
class PreparedTask:
    """One sample rendered into backend-ready benchmark inputs."""

    benchmark_id: str
    task_id: str
    sample_id: str
    system_prompt: str
    user_prompt: str
    schema: dict[str, Any]
    allowed_disease_ids: tuple[str, ...]


class BenchmarkTaskAdapter(Protocol):
    """Contract implemented by every benchmark task."""

    @property
    def benchmark_id(self) -> str:
        ...

    def prepare(self, sample: BenchmarkSample) -> PreparedTask:
        ...

    def parse_response(
        self,
        model_id: str,
        raw_text: str,
        prepared_task: PreparedTask,
        reasoning_text: str | None = None,
    ) -> ModelResponse:
        ...

    def compute_metrics(
        self,
        predictions: Iterable[BenchmarkPrediction],
    ) -> dict[str, Any]:
        ...


class VisualTopKTaskAdapter:
    """Adapter for the existing closed-set visual Top-K task."""

    def __init__(
        self,
        *,
        benchmark_id: str,
        system_prompt_template: str,
        user_prompt_template: str,
        schema: Mapping[str, Any],
        disease_taxonomy_items: Sequence[Mapping[str, Any]],
        top_k: int,
        minimum_subgroup_unique_groups: int = 30,
        minimum_per_disease_unique_groups: int = 10,
        subgroup_confidence_level: float = 0.95,
    ) -> None:
        self._benchmark_id = benchmark_id
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.schema = deepcopy(dict(schema))
        self.disease_taxonomy_items = _validate_taxonomy_items(
            disease_taxonomy_items,
            taxonomy_name="disease",
        )
        self.top_k = top_k
        self.minimum_subgroup_unique_groups = (
            minimum_subgroup_unique_groups
        )
        self.minimum_per_disease_unique_groups = (
            minimum_per_disease_unique_groups
        )
        self.subgroup_confidence_level = subgroup_confidence_level
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k > len(self.disease_taxonomy_items):
            raise ValueError("top_k exceeds the disease taxonomy size")
        self._taxonomy_by_id = {
            str(item["id"]): item
            for item in self.disease_taxonomy_items
        }

    @property
    def benchmark_id(self) -> str:
        return self._benchmark_id

    def prepare(self, sample: BenchmarkSample) -> PreparedTask:
        candidate_ids = self._candidate_ids(sample)
        rendered_taxonomy = _render_taxonomy(
            candidate_ids,
            taxonomy_by_id=self._taxonomy_by_id,
        )
        values = {
            "top_k": self.top_k,
            "disease_taxonomy": rendered_taxonomy,
        }
        return PreparedTask(
            benchmark_id=self.benchmark_id,
            task_id=sample.task_id or sample.sample_id,
            sample_id=sample.sample_id,
            system_prompt=_render_template(
                self.system_prompt_template,
                **values,
            ),
            user_prompt=_render_template(
                self.user_prompt_template,
                **values,
            ),
            schema=_narrow_ranked_schema(
                self.schema,
                candidate_ids=candidate_ids,
                prediction_count=self.top_k,
                array_field="predictions",
            ),
            allowed_disease_ids=tuple(candidate_ids),
        )

    def parse_response(
        self,
        model_id: str,
        raw_text: str,
        prepared_task: PreparedTask,
        reasoning_text: str | None = None,
    ) -> ModelResponse:
        # Reasoning is an unscored backend channel. Only final_text is parsed.
        del reasoning_text
        _require_matching_benchmark(prepared_task, self.benchmark_id)
        return parse_and_validate_response(
            model_id=model_id,
            raw_text=raw_text,
            allowed_disease_ids=set(
                prepared_task.allowed_disease_ids
            ),
            top_k=self.top_k,
        )

    def compute_metrics(
        self,
        predictions: Iterable[BenchmarkPrediction],
    ) -> dict[str, Any]:
        return compute_metrics(
            predictions,
            allowed_disease_ids=list(self._taxonomy_by_id),
            minimum_subgroup_unique_groups=(
                self.minimum_subgroup_unique_groups
            ),
            minimum_per_disease_unique_groups=(
                self.minimum_per_disease_unique_groups
            ),
            confidence_level=self.subgroup_confidence_level,
        )

    def _candidate_ids(self, sample: BenchmarkSample) -> list[str]:
        candidate_ids = (
            list(sample.candidate_disease_ids)
            if sample.candidate_disease_ids is not None
            else list(self._taxonomy_by_id)
        )
        _validate_candidates(
            candidate_ids,
            allowed_ids=set(self._taxonomy_by_id),
        )
        if sample.candidate_disease_ids is not None and (
            len(candidate_ids) != self.top_k
        ):
            raise ValueError(
                f"Task must contain exactly {self.top_k} candidates"
            )
        return candidate_ids


class ConfusionSetTaskAdapter(VisualTopKTaskAdapter):
    """Adapter for paired three-way visual confusion-set tasks."""

    def __init__(
        self,
        *,
        bootstrap_resamples: int = 10000,
        bootstrap_seed: int = 42,
        confidence_level: float = 0.95,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.bootstrap_resamples = bootstrap_resamples
        self.bootstrap_seed = bootstrap_seed
        self.confidence_level = confidence_level

    def prepare(self, sample: BenchmarkSample) -> PreparedTask:
        if sample.candidate_disease_ids is None:
            raise ValueError(
                "Confusion-set tasks require candidate_disease_ids"
            )
        return super().prepare(sample)

    def compute_metrics(
        self,
        predictions: Iterable[BenchmarkPrediction],
    ) -> dict[str, Any]:
        return compute_confusion_set_metrics(
            predictions,
            allowed_disease_ids=list(self._taxonomy_by_id),
            bootstrap_resamples=self.bootstrap_resamples,
            bootstrap_seed=self.bootstrap_seed,
            confidence_level=self.confidence_level,
        )


class ClinicalContextAblationTaskAdapter(VisualTopKTaskAdapter):
    """Adapter for paired image-only and patient-context diagnosis tasks."""

    def __init__(
        self,
        *,
        bootstrap_resamples: int = 10000,
        bootstrap_seed: int = 42,
        confidence_level: float = 0.95,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.bootstrap_resamples = bootstrap_resamples
        self.bootstrap_seed = bootstrap_seed
        self.confidence_level = confidence_level

    def prepare(self, sample: BenchmarkSample) -> PreparedTask:
        candidate_ids = self._candidate_ids(sample)
        context = sample.metadata.get("clinical_context")
        if not isinstance(context, str) or not context.strip():
            raise ValueError("Context-ablation task requires clinical_context")
        values = {
            "top_k": self.top_k,
            "disease_taxonomy": _render_taxonomy(
                candidate_ids,
                taxonomy_by_id=self._taxonomy_by_id,
            ),
            "clinical_context": context.strip(),
        }
        return PreparedTask(
            benchmark_id=self.benchmark_id,
            task_id=sample.task_id or sample.sample_id,
            sample_id=sample.sample_id,
            system_prompt=_render_template(self.system_prompt_template, **values),
            user_prompt=_render_template(self.user_prompt_template, **values),
            schema=_narrow_ranked_schema(
                self.schema,
                candidate_ids=candidate_ids,
                prediction_count=self.top_k,
                array_field="predictions",
            ),
            allowed_disease_ids=tuple(candidate_ids),
        )

    def compute_metrics(
        self,
        predictions: Iterable[BenchmarkPrediction],
    ) -> dict[str, Any]:
        return compute_clinical_context_ablation_metrics(
            predictions,
            allowed_disease_ids=list(self._taxonomy_by_id),
            bootstrap_resamples=self.bootstrap_resamples,
            bootstrap_seed=self.bootstrap_seed,
            confidence_level=self.confidence_level,
        )


class EvidenceGroundedTaskAdapter:
    """Adapter for morphology, description, diagnosis, and evidence links."""

    def __init__(
        self,
        *,
        benchmark_id: str,
        system_prompt_template: str,
        user_prompt_template: str,
        schema: Mapping[str, Any],
        disease_taxonomy_items: Sequence[Mapping[str, Any]],
        morphology_taxonomy_items: Sequence[Mapping[str, Any]],
        top_k: int,
        minimum_positive_cases_per_concept: int = 20,
        calibration_bins: int = 10,
    ) -> None:
        self._benchmark_id = benchmark_id
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.schema = deepcopy(dict(schema))
        self.disease_taxonomy_items = _validate_taxonomy_items(
            disease_taxonomy_items,
            taxonomy_name="disease",
        )
        self.morphology_taxonomy_items = _validate_taxonomy_items(
            morphology_taxonomy_items,
            taxonomy_name="morphology",
        )
        self.top_k = top_k
        self.minimum_positive_cases_per_concept = (
            minimum_positive_cases_per_concept
        )
        self.calibration_bins = calibration_bins
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k > len(self.disease_taxonomy_items):
            raise ValueError("top_k exceeds the disease taxonomy size")
        self._disease_by_id = {
            str(item["id"]): item
            for item in self.disease_taxonomy_items
        }
        self._morphology_by_id = {
            str(item["id"]): item
            for item in self.morphology_taxonomy_items
        }
        self._disease_terms = _disease_terms(
            self.disease_taxonomy_items
        )

    @property
    def benchmark_id(self) -> str:
        return self._benchmark_id

    def prepare(self, sample: BenchmarkSample) -> PreparedTask:
        candidate_ids = (
            list(sample.candidate_disease_ids)
            if sample.candidate_disease_ids is not None
            else list(self._disease_by_id)
        )
        _validate_candidates(
            candidate_ids,
            allowed_ids=set(self._disease_by_id),
        )
        if len(candidate_ids) < self.top_k:
            raise ValueError(
                f"Evidence task requires at least {self.top_k} diseases"
            )
        values = {
            "top_k": self.top_k,
            "disease_taxonomy": _render_taxonomy(
                candidate_ids,
                taxonomy_by_id=self._disease_by_id,
            ),
            "morphology_taxonomy": _render_taxonomy(
                list(self._morphology_by_id),
                taxonomy_by_id=self._morphology_by_id,
            ),
        }
        return PreparedTask(
            benchmark_id=self.benchmark_id,
            task_id=sample.task_id or sample.sample_id,
            sample_id=sample.sample_id,
            system_prompt=_render_template(
                self.system_prompt_template,
                **values,
            ),
            user_prompt=_render_template(
                self.user_prompt_template,
                **values,
            ),
            schema=_narrow_evidence_schema(
                self.schema,
                candidate_ids=candidate_ids,
                concept_ids=list(self._morphology_by_id),
                prediction_count=self.top_k,
            ),
            allowed_disease_ids=tuple(candidate_ids),
        )

    def parse_response(
        self,
        model_id: str,
        raw_text: str,
        prepared_task: PreparedTask,
        reasoning_text: str | None = None,
    ) -> ModelResponse:
        _require_matching_benchmark(prepared_task, self.benchmark_id)
        return parse_and_validate_evidence_response(
            model_id=model_id,
            raw_text=raw_text,
            reasoning_text=reasoning_text,
            allowed_disease_ids=set(
                prepared_task.allowed_disease_ids
            ),
            allowed_concept_ids=set(self._morphology_by_id),
            top_k=self.top_k,
            disease_terms=self._disease_terms,
        )

    def compute_metrics(
        self,
        predictions: Iterable[BenchmarkPrediction],
    ) -> dict[str, Any]:
        return compute_evidence_grounded_metrics(
            predictions,
            allowed_disease_ids=list(self._disease_by_id),
            allowed_concept_ids=list(self._morphology_by_id),
            minimum_positive_cases_per_concept=(
                self.minimum_positive_cases_per_concept
            ),
            calibration_bins=self.calibration_bins,
        )


class OpenEndedDiagnosisTaskAdapter:
    """Adapter that preserves a natural-language clinical response verbatim."""

    def __init__(
        self,
        *,
        benchmark_id: str,
        system_prompt_template: str,
        user_prompt_template: str,
    ) -> None:
        self._benchmark_id = benchmark_id
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template

    @property
    def benchmark_id(self) -> str:
        return self._benchmark_id

    def prepare(self, sample: BenchmarkSample) -> PreparedTask:
        return PreparedTask(
            benchmark_id=self.benchmark_id,
            task_id=sample.task_id or sample.sample_id,
            sample_id=sample.sample_id,
            system_prompt=self.system_prompt_template,
            user_prompt=self.user_prompt_template,
            schema={},
            allowed_disease_ids=(),
        )

    def parse_response(
        self,
        model_id: str,
        raw_text: str,
        prepared_task: PreparedTask,
        reasoning_text: str | None = None,
    ) -> ModelResponse:
        del reasoning_text
        _require_matching_benchmark(prepared_task, self.benchmark_id)
        text = raw_text.strip()
        if not text:
            return ModelResponse(
                model_id=model_id,
                raw_text=raw_text,
                parsed_output=None,
                json_valid=False,
                schema_valid=False,
                validation_errors=["empty_free_text_response"],
                metadata={"output_mode": "free_text"},
            )
        # The generic executor uses these two booleans as output-contract
        # success flags. In free-text mode they mean non-empty text accepted;
        # no JSON parsing or schema validation has occurred.
        return ModelResponse(
            model_id=model_id,
            raw_text=raw_text,
            parsed_output=None,
            json_valid=True,
            schema_valid=True,
            recoverable_json_valid=True,
            metadata={
                "output_mode": "free_text",
                "free_text_valid": True,
            },
        )

    def compute_metrics(
        self,
        predictions: Iterable[BenchmarkPrediction],
    ) -> dict[str, Any]:
        values = list(predictions)
        nonempty = [
            item
            for item in values
            if bool(item.response.raw_text.strip())
        ]
        return {
            "total": len(values),
            "free_text_response_rate": (
                len(nonempty) / len(values) if values else 0.0
            ),
            "mean_response_characters": (
                sum(len(item.response.raw_text) for item in nonempty)
                / len(nonempty)
                if nonempty
                else 0.0
            ),
            "judging_status": "pending",
        }


class VisualGroundingNoImageTaskAdapter:
    """Adapter for the validation-only uniform-gray grounding control."""

    def __init__(
        self,
        *,
        benchmark_id: str,
        system_prompt_template: str,
        user_prompt_template: str,
        schema: Mapping[str, Any],
        disease_taxonomy_items: Sequence[Mapping[str, Any]],
    ) -> None:
        self._benchmark_id = benchmark_id
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.schema = deepcopy(dict(schema))
        self.disease_taxonomy_items = _validate_taxonomy_items(
            disease_taxonomy_items,
            taxonomy_name="disease",
        )
        self._taxonomy_by_id = {
            str(item["id"]): item for item in self.disease_taxonomy_items
        }

    @property
    def benchmark_id(self) -> str:
        return self._benchmark_id

    def prepare(self, sample: BenchmarkSample) -> PreparedTask:
        candidate_ids = (
            list(sample.candidate_disease_ids)
            if sample.candidate_disease_ids is not None
            else list(self._taxonomy_by_id)
        )
        _validate_candidates(
            candidate_ids,
            allowed_ids=set(self._taxonomy_by_id),
        )
        values = {
            "disease_taxonomy": _render_taxonomy(
                candidate_ids,
                taxonomy_by_id=self._taxonomy_by_id,
            ),
        }
        return PreparedTask(
            benchmark_id=self.benchmark_id,
            task_id=sample.task_id or sample.sample_id,
            sample_id=sample.sample_id,
            system_prompt=_render_template(
                self.system_prompt_template,
                **values,
            ),
            user_prompt=_render_template(
                self.user_prompt_template,
                **values,
            ),
            schema=deepcopy(self.schema),
            allowed_disease_ids=tuple(candidate_ids),
        )

    def parse_response(
        self,
        model_id: str,
        raw_text: str,
        prepared_task: PreparedTask,
        reasoning_text: str | None = None,
    ) -> ModelResponse:
        _require_matching_benchmark(prepared_task, self.benchmark_id)
        return parse_and_validate_visual_grounding_response(
            model_id=model_id,
            raw_text=raw_text,
            reasoning_text=reasoning_text,
            allowed_disease_ids=set(prepared_task.allowed_disease_ids),
        )

    def compute_metrics(
        self,
        predictions: Iterable[BenchmarkPrediction],
    ) -> dict[str, Any]:
        return compute_visual_grounding_metrics(predictions)


class GeneralVisualHallucinationTaskAdapter:
    """Adapter for the fixed HaloQuest answerability audit."""

    def __init__(
        self,
        *,
        benchmark_id: str,
        system_prompt_template: str,
        user_prompt_template: str,
        schema: Mapping[str, Any],
    ) -> None:
        self._benchmark_id = benchmark_id
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.schema = deepcopy(dict(schema))

    @property
    def benchmark_id(self) -> str:
        return self._benchmark_id

    def prepare(self, sample: BenchmarkSample) -> PreparedTask:
        question = sample.metadata.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("General hallucination task requires a question")
        return PreparedTask(
            benchmark_id=self.benchmark_id,
            task_id=sample.task_id or sample.sample_id,
            sample_id=sample.sample_id,
            system_prompt=self.system_prompt_template,
            user_prompt=_render_template(
                self.user_prompt_template,
                question=question.strip(),
            ),
            schema=deepcopy(self.schema),
            allowed_disease_ids=(),
        )

    def parse_response(
        self,
        model_id: str,
        raw_text: str,
        prepared_task: PreparedTask,
        reasoning_text: str | None = None,
    ) -> ModelResponse:
        _require_matching_benchmark(prepared_task, self.benchmark_id)
        return parse_general_visual_hallucination_response(
            model_id=model_id,
            raw_text=raw_text,
            reasoning_text=reasoning_text,
        )

    def compute_metrics(
        self,
        predictions: Iterable[BenchmarkPrediction],
    ) -> dict[str, Any]:
        return compute_general_visual_hallucination_metrics(predictions)


class DermatologyCounterfactualTaskAdapter(VisualGroundingNoImageTaskAdapter):
    """Adapter for pixel-corruption and hard-negative dermatology cases."""

    def parse_response(
        self,
        model_id: str,
        raw_text: str,
        prepared_task: PreparedTask,
        reasoning_text: str | None = None,
    ) -> ModelResponse:
        _require_matching_benchmark(prepared_task, self.benchmark_id)
        return parse_dermatology_counterfactual_response(
            model_id=model_id,
            raw_text=raw_text,
            reasoning_text=reasoning_text,
            allowed_disease_ids=set(prepared_task.allowed_disease_ids),
        )

    def compute_metrics(
        self,
        predictions: Iterable[BenchmarkPrediction],
    ) -> dict[str, Any]:
        return compute_dermatology_counterfactual_metrics(predictions)


def build_task_adapter(
    *,
    benchmark_config: Mapping[str, Any],
    prompt_config: Mapping[str, Any],
    schema: Mapping[str, Any],
    disease_taxonomy_items: Sequence[Mapping[str, Any]],
    morphology_taxonomy_items: (
        Sequence[Mapping[str, Any]] | None
    ) = None,
) -> BenchmarkTaskAdapter:
    """Construct the adapter selected by a loaded benchmark YAML."""

    benchmark = _mapping_value(
        benchmark_config,
        "benchmark",
        context="benchmark config",
    )
    benchmark_id = str(benchmark["id"])
    task = str(benchmark["task"])
    common = {
        "benchmark_id": benchmark_id,
        "system_prompt_template": str(prompt_config["system_prompt"]),
        "user_prompt_template": str(prompt_config["user_template"]),
        "schema": schema,
        "disease_taxonomy_items": disease_taxonomy_items,
    }
    if task == "visual_disease_ranking":
        subgroup = benchmark_config.get("subgroup_evaluation", {})
        subgroup = subgroup if isinstance(subgroup, Mapping) else {}
        comparison = benchmark_config.get("comparison", {})
        comparison = comparison if isinstance(comparison, Mapping) else {}
        interval = comparison.get("confidence_interval", {})
        interval = interval if isinstance(interval, Mapping) else {}
        return VisualTopKTaskAdapter(
            **common,
            top_k=int(benchmark["top_k"]),
            minimum_subgroup_unique_groups=int(
                subgroup.get("minimum_subgroup_unique_groups", 30)
            ),
            minimum_per_disease_unique_groups=int(
                subgroup.get("minimum_per_disease_unique_groups", 10)
            ),
            subgroup_confidence_level=float(
                interval.get("confidence_level", 0.95)
            ),
        )
    if task == "visual_disease_contrast_ranking":
        comparison = benchmark_config.get("comparison", {})
        interval = (
            comparison.get("confidence_interval", {})
            if isinstance(comparison, Mapping)
            else {}
        )
        return ConfusionSetTaskAdapter(
            **common,
            top_k=int(benchmark["ranking_count"]),
            bootstrap_resamples=int(interval.get("resamples", 10000)),
            bootstrap_seed=int(interval.get("seed", 42)),
            confidence_level=float(
                interval.get("confidence_level", 0.95)
            ),
        )
    if task == "clinical_context_ablation":
        comparison = benchmark_config.get("comparison", {})
        interval = (
            comparison.get("confidence_interval", {})
            if isinstance(comparison, Mapping)
            else {}
        )
        return ClinicalContextAblationTaskAdapter(
            **common,
            top_k=int(benchmark["top_k"]),
            bootstrap_resamples=int(interval.get("resamples", 10000)),
            bootstrap_seed=int(interval.get("seed", 42)),
            confidence_level=float(interval.get("confidence_level", 0.95)),
        )
    if task == "evidence_grounded_visual_diagnosis":
        morphology_items = morphology_taxonomy_items
        if morphology_items is None:
            taxonomy = _mapping_value(
                benchmark_config,
                "taxonomy",
                context="benchmark config",
            )
            morphology = _mapping_value(
                taxonomy,
                "morphology",
                context="benchmark taxonomy",
            )
            configured = morphology.get("concepts")
            if not isinstance(configured, list):
                raise ValueError(
                    "Evidence benchmark requires morphology concepts"
                )
            morphology_items = configured
        return EvidenceGroundedTaskAdapter(
            **common,
            morphology_taxonomy_items=morphology_items,
            top_k=int(benchmark["top_k"]),
            minimum_positive_cases_per_concept=(
                _configured_metric_int(
                    benchmark_config,
                    category="morphology",
                    metric_id="macro_f1_supported_concepts",
                    field="minimum_positive_cases_per_concept",
                    default=20,
                )
            ),
            calibration_bins=_configured_metric_int(
                benchmark_config,
                category="calibration",
                metric_id="top_1_expected_calibration_error",
                field="bins",
                default=10,
            ),
        )
    if task == "open_ended_clinical_diagnosis":
        return OpenEndedDiagnosisTaskAdapter(
            benchmark_id=benchmark_id,
            system_prompt_template=str(prompt_config["system_prompt"]),
            user_prompt_template=str(prompt_config["user_template"]),
        )
    if task == "visual_grounding_no_image_ablation":
        return VisualGroundingNoImageTaskAdapter(**common)
    if task == "general_visual_hallucination_audit":
        return GeneralVisualHallucinationTaskAdapter(
            benchmark_id=benchmark_id,
            system_prompt_template=str(prompt_config["system_prompt"]),
            user_prompt_template=str(prompt_config["user_template"]),
            schema=schema,
        )
    if task == "dermatology_counterfactual_hallucination":
        return DermatologyCounterfactualTaskAdapter(**common)
    raise ValueError(f"Unsupported benchmark task: {task}")


# A readable alias for callers that use "create" terminology.
create_task_adapter = build_task_adapter


def _validate_taxonomy_items(
    items: Sequence[Mapping[str, Any]],
    *,
    taxonomy_name: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"{taxonomy_name} taxonomy item {index} must be a mapping"
            )
        identifier = item.get("id")
        display_name = item.get("display_name")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(
                f"{taxonomy_name} taxonomy item {index} has no valid id"
            )
        if not isinstance(display_name, str) or not display_name:
            raise ValueError(
                f"{taxonomy_name} taxonomy item {index} has no display_name"
            )
        if identifier in seen:
            raise ValueError(
                f"Duplicate {taxonomy_name} taxonomy ID: {identifier}"
            )
        seen.add(identifier)
        normalized.append(dict(item))
    if not normalized:
        raise ValueError(f"{taxonomy_name} taxonomy must not be empty")
    return normalized


def _validate_candidates(
    candidate_ids: list[str],
    *,
    allowed_ids: set[str],
) -> None:
    if not candidate_ids:
        raise ValueError("Task candidate disease IDs must not be empty")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Task candidate disease IDs must be unique")
    unknown = set(candidate_ids) - allowed_ids
    if unknown:
        raise ValueError(
            "Task contains candidates outside the benchmark taxonomy: "
            + ", ".join(sorted(unknown))
        )


def _render_taxonomy(
    ids: list[str],
    *,
    taxonomy_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    return "\n".join(
        f"- {identifier}: {taxonomy_by_id[identifier]['display_name']}"
        for identifier in ids
    )


def _render_template(template: str, **values: Any) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{ {key} }}}}", str(value))
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered


def _narrow_ranked_schema(
    schema: Mapping[str, Any],
    *,
    candidate_ids: list[str],
    prediction_count: int,
    array_field: str,
) -> dict[str, Any]:
    narrowed = deepcopy(dict(schema))
    predictions = narrowed["properties"][array_field]
    predictions["minItems"] = prediction_count
    predictions["maxItems"] = prediction_count
    item_properties = predictions["items"]["properties"]
    item_properties["rank"]["minimum"] = 1
    item_properties["rank"]["maximum"] = prediction_count
    item_properties["disease_id"]["enum"] = list(candidate_ids)
    return narrowed


def _narrow_evidence_schema(
    schema: Mapping[str, Any],
    *,
    candidate_ids: list[str],
    concept_ids: list[str],
    prediction_count: int,
) -> dict[str, Any]:
    narrowed = _narrow_ranked_schema(
        schema,
        candidate_ids=candidate_ids,
        prediction_count=prediction_count,
        array_field="differential",
    )
    finding_properties = narrowed["properties"]["findings"]["items"][
        "properties"
    ]
    finding_properties["concept_id"]["enum"] = list(concept_ids)
    return narrowed


def _disease_terms(
    items: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    terms: set[str] = set()
    for item in items:
        for key in ("display_name", "canonical_name"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                terms.add(value)
                terms.add(value.replace("_", " "))
        aliases = item.get("aliases", ())
        if isinstance(aliases, list):
            terms.update(
                str(alias)
                for alias in aliases
                if isinstance(alias, str) and alias.strip()
            )
    return tuple(sorted(terms, key=str.casefold))


def _require_matching_benchmark(
    prepared_task: PreparedTask,
    benchmark_id: str,
) -> None:
    if prepared_task.benchmark_id != benchmark_id:
        raise ValueError(
            "Prepared task belongs to a different benchmark: "
            f"{prepared_task.benchmark_id}"
        )


def _mapping_value(
    mapping: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}.{key} must be a mapping")
    return value


def _configured_metric_int(
    config: Mapping[str, Any],
    *,
    category: str,
    metric_id: str,
    field: str,
    default: int,
) -> int:
    metrics = config.get("metrics")
    if not isinstance(metrics, Mapping):
        return default
    category_metrics = metrics.get(category)
    if not isinstance(category_metrics, list):
        return default
    for metric in category_metrics:
        if (
            isinstance(metric, Mapping)
            and metric.get("id") == metric_id
        ):
            return int(metric.get(field, default))
    return default
