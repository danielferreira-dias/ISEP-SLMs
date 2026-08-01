"""Core execution and validation for the visual top-k benchmark."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any, Iterable, Protocol

from src.benchmark.json_parsing import parse_json_output


class ModelBackend(Protocol):
    """Minimal backend contract shared by local and API model adapters."""

    @property
    def model_id(self) -> str:
        ...

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        schema: dict[str, Any],
    ) -> str:
        ...


@dataclass(slots=True)
class ModelResponse:
    model_id: str
    raw_text: str
    parsed_output: dict[str, Any] | None
    json_valid: bool
    schema_valid: bool
    recoverable_json_valid: bool = False
    canonical_output: dict[str, Any] | None = None
    canonical_schema_valid: bool = False
    canonicalization_rules: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.json_valid and self.schema_valid


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    sample_id: str
    image_uri: str
    disease_id: str
    metadata: dict[str, Any]
    task_id: str | None = None
    candidate_disease_ids: tuple[str, ...] | None = None
    image_bytes: bytes | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    response_schema: dict[str, Any] | None = None


@dataclass(slots=True)
class BenchmarkPrediction:
    sample_id: str
    model_id: str
    ground_truth_disease_id: str
    response: ModelResponse
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Render prompts, call one backend, and validate every response."""

    def __init__(
        self,
        *,
        backend: ModelBackend,
        system_prompt: str,
        user_prompt_template: str,
        schema: dict[str, Any],
        taxonomy_items: list[dict[str, str]],
        top_k: int,
        image_loader: Any,
    ) -> None:
        self.backend = backend
        self.system_prompt_template = system_prompt
        self.user_prompt_template = user_prompt_template
        self.schema = schema
        self.taxonomy_items = taxonomy_items
        self.top_k = top_k
        self.image_loader = image_loader
        self.allowed_disease_ids = {
            item["id"]
            for item in taxonomy_items
        }

    def run_sample(
        self,
        sample: BenchmarkSample,
    ) -> BenchmarkPrediction:
        candidate_ids = (
            list(sample.candidate_disease_ids)
            if sample.candidate_disease_ids is not None
            else [item["id"] for item in self.taxonomy_items]
        )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Task candidate disease IDs must be unique")
        unknown_candidates = (
            set(candidate_ids) - self.allowed_disease_ids
        )
        if unknown_candidates:
            raise ValueError(
                "Task contains candidates outside the benchmark taxonomy: "
                + ", ".join(sorted(unknown_candidates))
            )
        if sample.candidate_disease_ids is not None and (
            len(candidate_ids) != self.top_k
        ):
            raise ValueError(
                f"Task must contain exactly {self.top_k} candidates"
            )
        taxonomy_by_id = {
            item["id"]: item
            for item in self.taxonomy_items
        }
        taxonomy = "\n".join(
            f"- {item['id']}: {item['display_name']}"
            for item in (
                taxonomy_by_id[disease_id]
                for disease_id in candidate_ids
            )
        )
        system_prompt = _render_template(
            self.system_prompt_template,
            top_k=self.top_k,
            disease_taxonomy=taxonomy,
        )
        user_prompt = _render_template(
            self.user_prompt_template,
            top_k=self.top_k,
            disease_taxonomy=taxonomy,
        )
        image_bytes = self.image_loader(sample.image_uri)
        task_schema = _schema_for_candidates(
            self.schema,
            candidate_ids=candidate_ids,
            prediction_count=self.top_k,
        )
        try:
            raw_text = self.backend.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_bytes=image_bytes,
                schema=task_schema,
            )
        except Exception as exc:
            response = ModelResponse(
                model_id=self.backend.model_id,
                raw_text="",
                parsed_output=None,
                json_valid=False,
                schema_valid=False,
                validation_errors=[f"backend_error:{exc}"],
            )
        else:
            response = parse_and_validate_response(
                model_id=self.backend.model_id,
                raw_text=raw_text,
                allowed_disease_ids=set(candidate_ids),
                top_k=self.top_k,
            )
        prediction_metadata = dict(sample.metadata)
        prediction_metadata.setdefault(
            "candidate_disease_ids",
            list(candidate_ids),
        )
        return BenchmarkPrediction(
            sample_id=sample.sample_id,
            model_id=self.backend.model_id,
            ground_truth_disease_id=sample.disease_id,
            response=response,
            task_id=sample.task_id or sample.sample_id,
            metadata=prediction_metadata,
        )

    def run(
        self,
        samples: Iterable[BenchmarkSample],
    ) -> list[BenchmarkPrediction]:
        return [self.run_sample(sample) for sample in samples]


def parse_and_validate_response(
    *,
    model_id: str,
    raw_text: str,
    allowed_disease_ids: set[str],
    top_k: int,
) -> ModelResponse:
    """Parse JSON and enforce benchmark rules not expressible as basic types."""

    parse_result = parse_json_output(raw_text)
    if not parse_result.recoverable_valid:
        return ModelResponse(
            model_id=model_id,
            raw_text=raw_text,
            parsed_output=None,
            json_valid=False,
            schema_valid=False,
            recoverable_json_valid=False,
            validation_errors=[f"invalid_json:{parse_result.error}"],
        )
    parsed = parse_result.decoded

    errors: list[str] = []
    if not parse_result.raw_valid:
        errors.append(f"invalid_json:{parse_result.error}")
    if not isinstance(parsed, dict):
        errors.append("root_must_be_object")
        predictions: Any = None
    else:
        extra_root_fields = set(parsed) - {"predictions"}
        if extra_root_fields:
            errors.append("unexpected_root_fields")
        predictions = parsed.get("predictions")
    if not isinstance(predictions, list):
        errors.append("predictions_must_be_array")
        predictions = []
    if len(predictions) != top_k:
        errors.append(f"prediction_count_must_equal_{top_k}")

    ranks: list[int] = []
    disease_ids: list[str] = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            errors.append(f"prediction_{index}_must_be_object")
            continue
        if set(prediction) != {"rank", "disease_id"}:
            errors.append(f"prediction_{index}_fields_invalid")
        rank = prediction.get("rank")
        disease_id = prediction.get("disease_id")
        if not isinstance(rank, int) or isinstance(rank, bool):
            errors.append(f"prediction_{index}_rank_invalid")
        else:
            ranks.append(rank)
        if not isinstance(disease_id, str):
            errors.append(f"prediction_{index}_disease_id_invalid")
        else:
            disease_ids.append(disease_id)
            if disease_id not in allowed_disease_ids:
                errors.append(f"prediction_{index}_disease_id_unknown")
    if ranks != list(range(1, top_k + 1)):
        errors.append("ranks_must_be_consecutive")
    if len(disease_ids) != len(set(disease_ids)):
        errors.append("disease_ids_must_be_unique")

    (
        canonical_output,
        canonical_errors,
        canonicalization_rules,
    ) = _canonicalize_ranked_output(
        parsed,
        allowed_disease_ids=allowed_disease_ids,
        top_k=top_k,
    )
    if parse_result.recovery is not None:
        canonicalization_rules.insert(0, parse_result.recovery)

    return ModelResponse(
        model_id=model_id,
        raw_text=raw_text,
        parsed_output=parsed,
        json_valid=parse_result.raw_valid,
        schema_valid=not errors,
        recoverable_json_valid=parse_result.recoverable_valid,
        canonical_output=canonical_output,
        canonical_schema_valid=not canonical_errors,
        canonicalization_rules=canonicalization_rules,
        validation_errors=errors,
        metadata={"json_recovery": parse_result.recovery},
    )


def _canonicalize_ranked_output(
    parsed: Any,
    *,
    allowed_disease_ids: set[str],
    top_k: int,
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """Project equivalent ranked-list forms into the benchmark schema.

    This projection is deliberately narrow. It accepts the requested object
    representation or a JSON array of disease IDs whose order unambiguously
    defines rank. It never extracts diagnoses from prose or reasoning.
    """

    rules: list[str] = []
    if not isinstance(parsed, dict) or set(parsed) != {"predictions"}:
        return None, ["canonical_root_invalid"], rules
    predictions = parsed.get("predictions")
    if not isinstance(predictions, list):
        return None, ["canonical_predictions_invalid"], rules

    if predictions and all(
        isinstance(item, str) for item in predictions
    ):
        canonical_predictions = [
            {"rank": rank, "disease_id": disease_id}
            for rank, disease_id in enumerate(predictions, start=1)
        ]
        rules.append("ranked_disease_id_list_to_objects")
    else:
        canonical_predictions = predictions

    errors: list[str] = []
    if len(canonical_predictions) != top_k:
        errors.append(f"prediction_count_must_equal_{top_k}")
    ranks: list[int] = []
    disease_ids: list[str] = []
    for index, prediction in enumerate(canonical_predictions):
        if not isinstance(prediction, dict):
            errors.append(f"prediction_{index}_must_be_object")
            continue
        if set(prediction) != {"rank", "disease_id"}:
            errors.append(f"prediction_{index}_fields_invalid")
        rank = prediction.get("rank")
        disease_id = prediction.get("disease_id")
        if not isinstance(rank, int) or isinstance(rank, bool):
            errors.append(f"prediction_{index}_rank_invalid")
        else:
            ranks.append(rank)
        if not isinstance(disease_id, str):
            errors.append(f"prediction_{index}_disease_id_invalid")
        else:
            disease_ids.append(disease_id)
            if disease_id not in allowed_disease_ids:
                errors.append(f"prediction_{index}_disease_id_unknown")
    if ranks != list(range(1, top_k + 1)):
        errors.append("ranks_must_be_consecutive")
    if len(disease_ids) != len(set(disease_ids)):
        errors.append("disease_ids_must_be_unique")
    if errors:
        return None, errors, rules
    return {"predictions": canonical_predictions}, [], rules


def _render_template(template: str, **values: Any) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{ {key} }}}}", str(value))
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered


def _schema_for_candidates(
    schema: dict[str, Any],
    *,
    candidate_ids: list[str],
    prediction_count: int,
) -> dict[str, Any]:
    """Return a per-task schema restricted to the rendered candidates."""

    narrowed = deepcopy(schema)
    predictions = narrowed["properties"]["predictions"]
    predictions["minItems"] = prediction_count
    predictions["maxItems"] = prediction_count
    item_properties = predictions["items"]["properties"]
    item_properties["rank"]["minimum"] = 1
    item_properties["rank"]["maximum"] = prediction_count
    item_properties["disease_id"]["enum"] = list(candidate_ids)
    return narrowed
