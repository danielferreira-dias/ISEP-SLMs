"""Core execution and validation for the visual top-k benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterable, Protocol


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


@dataclass(slots=True)
class BenchmarkPrediction:
    sample_id: str
    model_id: str
    ground_truth_disease_id: str
    response: ModelResponse


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
        taxonomy = "\n".join(
            f"- {item['id']}: {item['display_name']}"
            for item in self.taxonomy_items
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
        try:
            raw_text = self.backend.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_bytes=image_bytes,
                schema=self.schema,
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
                allowed_disease_ids=self.allowed_disease_ids,
                top_k=self.top_k,
            )
        return BenchmarkPrediction(
            sample_id=sample.sample_id,
            model_id=self.backend.model_id,
            ground_truth_disease_id=sample.disease_id,
            response=response,
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

    try:
        parsed = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError) as exc:
        return ModelResponse(
            model_id=model_id,
            raw_text=raw_text,
            parsed_output=None,
            json_valid=False,
            schema_valid=False,
            validation_errors=[f"invalid_json:{exc}"],
        )

    errors: list[str] = []
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

    return ModelResponse(
        model_id=model_id,
        raw_text=raw_text,
        parsed_output=parsed,
        json_valid=True,
        schema_valid=not errors,
        validation_errors=errors,
    )


def _render_template(template: str, **values: Any) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{ {key} }}}}", str(value))
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered
