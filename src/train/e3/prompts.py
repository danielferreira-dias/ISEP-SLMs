"""Versioned teacher prompts and strict output schemas for E3."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.train.domain import Taxonomy
from src.train.e3.domain import StageATarget, StageBTarget
from src.train.e3.terminology import DermatologyTerminology


class _PromptResource(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class StageAPromptResource(_PromptResource):
    schema_version: Literal[1, 2]
    prompt_id: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    user_prompt: str | None = Field(default=None, min_length=1)
    user_prompt_template: str | None = Field(default=None, min_length=1)
    required_lexicon_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_versioned_contract(self) -> StageAPromptResource:
        if self.schema_version == 1:
            if self.user_prompt is None:
                raise ValueError("Stage-A prompt v1 requires user_prompt")
            if self.user_prompt_template is not None or self.required_lexicon_id:
                raise ValueError("Stage-A prompt v1 cannot require terminology")
            return self
        if self.user_prompt is not None:
            raise ValueError("Stage-A prompt v2 must use user_prompt_template")
        if self.user_prompt_template is None or self.required_lexicon_id is None:
            raise ValueError(
                "Stage-A prompt v2 requires a template and frozen lexicon ID"
            )
        required = {"{lexicon_id}", "{lexicon_json}"}
        missing = tuple(
            token
            for token in required
            if token not in self.user_prompt_template
        )
        if missing:
            raise ValueError(
                "Stage-A prompt v2 is missing placeholders: " + ", ".join(missing)
            )
        return self


class StageBPromptResource(_PromptResource):
    schema_version: Literal[1, 2]
    prompt_id: str = Field(min_length=1)
    gold_conditioning: Literal["none", "private_gold"] = "none"
    system_prompt: str = Field(min_length=1)
    user_prompt_template: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_versioned_contract(self) -> StageBPromptResource:
        required = {"{taxonomy_json}", "{stage_a_json}"}
        missing = tuple(
            token for token in required if token not in self.user_prompt_template
        )
        if missing:
            raise ValueError(
                "Stage-B teacher prompt is missing placeholders: "
                + ", ".join(missing)
            )
        has_gold_placeholder = "{gold_anchor_json}" in self.user_prompt_template
        if self.schema_version == 1:
            if self.gold_conditioning != "none" or has_gold_placeholder:
                raise ValueError("Stage-B prompt v1 must remain answer-blind")
        elif self.gold_conditioning != "private_gold" or not has_gold_placeholder:
            raise ValueError(
                "Stage-B prompt v2 requires private_gold and {gold_anchor_json}"
            )
        return self

    @property
    def gold_visible_to_teacher(self) -> bool:
        return self.gold_conditioning == "private_gold"


class RenderedTeacherPrompt(BaseModel):
    """Exact text identity sent for one teacher call, excluding the image."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    prompt_id: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def load_stage_a_prompt(path: Path) -> StageAPromptResource:
    return StageAPromptResource.model_validate(_load_yaml_object(path))


def load_stage_b_prompt(path: Path) -> StageBPromptResource:
    return StageBPromptResource.model_validate(_load_yaml_object(path))


def render_stage_a_prompt(
    resource: StageAPromptResource,
    *,
    terminology: DermatologyTerminology | None = None,
) -> RenderedTeacherPrompt:
    if resource.schema_version == 1:
        assert resource.user_prompt is not None
        return _rendered(
            resource.prompt_id,
            resource.system_prompt,
            resource.user_prompt,
        )
    if terminology is None:
        raise ValueError("Stage-A prompt v2 requires frozen terminology")
    if terminology.lexicon_id != resource.required_lexicon_id:
        raise ValueError(
            "Stage-A prompt/terminology identity mismatch: "
            f"expected {resource.required_lexicon_id!r}, "
            f"got {terminology.lexicon_id!r}"
        )
    assert resource.user_prompt_template is not None
    user_prompt = resource.user_prompt_template.format(
        lexicon_id=terminology.lexicon_id,
        lexicon_json=terminology.prompt_catalog_json(),
    )
    return _rendered(resource.prompt_id, resource.system_prompt, user_prompt)


def render_stage_b_prompt(
    resource: StageBPromptResource,
    *,
    taxonomy: Taxonomy,
    stage_a: StageATarget,
    gold_disease_id: str | None = None,
    gold_diagnosis: str | None = None,
) -> RenderedTeacherPrompt:
    """Render Stage B with versioned answer-blind or private-gold inputs."""

    taxonomy_payload = [
        {"disease_id": item.disease_id, "label": item.label}
        for item in taxonomy.classes
    ]
    values = {
        "taxonomy_json": _canonical_json(taxonomy_payload),
        "stage_a_json": _canonical_json(stage_a.model_dump(mode="json")),
        "gold_anchor_json": "",
    }
    if resource.gold_visible_to_teacher:
        if gold_disease_id is None or gold_diagnosis is None:
            raise ValueError("Gold-conditioned Stage B requires the private gold")
        labels = {item.disease_id: item.label for item in taxonomy.classes}
        if labels.get(gold_disease_id) != gold_diagnosis:
            raise ValueError("Stage-B private gold does not match the taxonomy")
        values["gold_anchor_json"] = _canonical_json(
            {"disease_id": gold_disease_id, "diagnosis": gold_diagnosis}
        )
    user_prompt = resource.user_prompt_template.format(**values)
    return _rendered(resource.prompt_id, resource.system_prompt, user_prompt)


def stage_a_output_schema(
    terminology: DermatologyTerminology | None = None,
) -> dict[str, Any]:
    schema = StageATarget.model_json_schema()
    if terminology is not None:
        definitions = schema.get("$defs")
        if not isinstance(definitions, dict):
            raise ValueError("Stage-A schema is missing definitions")
        observation = definitions.get("Observation")
        if not isinstance(observation, dict):
            raise ValueError("Stage-A schema is missing Observation")
        properties = observation.get("properties")
        if not isinstance(properties, dict):
            raise ValueError("Stage-A Observation schema is missing properties")
        concept_id = properties.get("concept_id")
        if not isinstance(concept_id, dict):
            raise ValueError("Stage-A Observation schema is missing concept_id")
        concept_id["enum"] = list(terminology.concept_ids)
    return strict_provider_schema(schema)


def stage_b_output_schema() -> dict[str, Any]:
    return strict_provider_schema(StageBTarget.model_json_schema())


def strict_provider_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make every object field explicit for provider strict structured output.

    Pydantic omits fields with defaults from ``required``. OpenAI strict
    Structured Outputs expects every property to be required; nullable fields
    therefore remain required and use ``null`` when absent. Local Pydantic
    validation remains the final fail-closed contract.
    """

    normalized = deepcopy(schema)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("default", None)
            value.pop("uniqueItems", None)
            properties = value.get("properties")
            if isinstance(properties, dict):
                value["required"] = list(properties)
                value["additionalProperties"] = False
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(normalized)
    return normalized


def prompt_resource_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rendered(
    prompt_id: str,
    system_prompt: str,
    user_prompt: str,
) -> RenderedTeacherPrompt:
    payload = _canonical_json(
        {
            "prompt_id": prompt_id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
    ).encode("utf-8")
    return RenderedTeacherPrompt(
        prompt_id=prompt_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _load_yaml_object(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Prompt resource must be an object: {path}")
    return value
