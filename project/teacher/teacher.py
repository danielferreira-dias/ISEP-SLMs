"""Load a teacher YAML into typed dataclasses.

Public entry point: ``TeacherModel.from_yaml()``.

The extra functions are not extra behaviour. They exist so a broken config
fails at load time with a field path, instead of later when a provider is
called. There are three groups:

* ``_require_*`` — coerce one YAML value to a Python type.
* ``_resolve_project_path`` / ``_parse_prompt_markdown`` — read files the
  YAML only references (``prompt_ref``, ``schema_ref``).
* dataclass methods — look up a stage or build a provider request body.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from project.teacher.schemas import UsageInfo, teacher_output_schema

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "teacher_configs" / "gemini_3_7_flash.yaml"

_HEADING_RE = re.compile(r"(?im)^##\s+(system|user)\s*$")

type ReasoningEffort = Literal["high", "medium", "low"]


def _require_mapping(value: object, path: str) -> dict[str, Any]:
    """Return ``value`` as a dict, or raise if the YAML node is not an object."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return dict(value)


def _require_str(value: object, path: str) -> str:
    """Return a non-empty string, or raise if the YAML node is missing or blank."""
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path} must be a non-empty string")
    return value


def _require_sha256(value: object, path: str) -> str:
    """Return a lowercase SHA-256 digest or fail config loading."""
    digest = _require_str(value, path)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return digest


def _require_bool(value: object, path: str) -> bool:
    """Return a real boolean. Rejects 0/1 so YAML integers cannot sneak in."""
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean")
    return value


def _require_int(value: object, path: str) -> int:
    """Return an integer. Rejects bool because ``True`` is a subclass of ``int``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    return value


def _require_positive_number(value: object, path: str) -> float:
    """Return a positive finite YAML number, rejecting booleans and strings."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{path} must be a number")
    parsed = float(value)
    if not 0 < parsed < float("inf"):
        raise ValueError(f"{path} must be positive and finite")
    return parsed


def _require_nonnegative_number(value: object, path: str) -> float:
    """Return a finite YAML number greater than or equal to zero."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{path} must be a number")
    parsed = float(value)
    if not 0 <= parsed < float("inf"):
        raise ValueError(f"{path} must be non-negative and finite")
    return parsed


def _resolve_project_path(raw: str, *, project_root: Path) -> Path:
    """Resolve a prompt/schema reference and require an existing file."""
    path = Path(raw)
    resolved = path if path.is_absolute() else project_root / path
    if not resolved.is_file():
        raise FileNotFoundError(f"Referenced file does not exist: {resolved}")
    return resolved


def _parse_prompt_markdown(text: str, *, source: Path) -> tuple[str, str]:
    """Split a stage prompt file into the ``## system`` and ``## user`` bodies."""
    matches = list(_HEADING_RE.finditer(text))
    found = {match.group(1).lower() for match in matches}
    if found != {"system", "user"}:
        raise ValueError(
            f"{source} must contain exactly one '## system' and one '## user' section"
        )

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        body = re.sub(r"(?m)^---\s*$", "", body).strip()
        sections[match.group(1).lower()] = body

    if not sections["system"] or not sections["user"]:
        raise ValueError(f"{source} has an empty system or user section")
    return sections["system"], sections["user"]


class TeacherProvider(StrEnum):
    """Backends that can complete Stage A and Stage B."""

    OPENROUTER = "openrouter"
    VERTEX = "vertex"


@dataclass(frozen=True)
class TeacherAPI:
    """OpenRouter endpoint and the env var that holds the API key."""

    base_url: str
    api_key_env: str

    def api_key(self) -> str:
        """Read the secret from the environment. Not stored on the dataclass."""
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise OSError(f"Missing environment variable {self.api_key_env}")
        return key


@dataclass(frozen=True)
class VertexAPI:
    """Gemini Enterprise Agent Platform project. Auth is ADC, not an API key."""

    project: str
    location: str


@dataclass(frozen=True)
class TeacherModelSpec:
    """Which model the provider should serve, and whether it accepts images."""

    id: str
    multimodal: bool


@dataclass(frozen=True)
class TeacherRouting:
    """OpenRouter ``provider`` object. Pins Vertex and forbids AI Studio fallback."""

    only: tuple[str, ...]
    allow_fallbacks: bool
    require_parameters: bool

    def as_openrouter_provider(self) -> dict[str, Any]:
        """Shape expected in the chat-completions ``provider`` field."""
        return {
            "only": list(self.only),
            "allow_fallbacks": self.allow_fallbacks,
            "require_parameters": self.require_parameters,
        }


def _load_api(
    provider: TeacherProvider,
    api: dict[str, Any],
) -> TeacherAPI | VertexAPI:
    """Parse provider-specific API identity. Secrets stay in the environment."""
    if provider is TeacherProvider.OPENROUTER:
        return TeacherAPI(
            base_url=_require_str(api.get("base_url"), "teacher.api.base_url"),
            api_key_env=_require_str(api.get("api_key_env"), "teacher.api.api_key_env"),
        )
    return VertexAPI(
        project=_require_str(api.get("project"), "teacher.api.project"),
        location=_require_str(api.get("location"), "teacher.api.location"),
    )


def _load_routing(provider: TeacherProvider, raw: object) -> TeacherRouting | None:
    """OpenRouter must pin a provider list. Vertex uses the GCP project instead."""
    if provider is TeacherProvider.VERTEX:
        return None
    routing = _require_mapping(raw, "teacher.routing")
    only = routing.get("only")
    if (
        not isinstance(only, list)
        or not only
        or not all(isinstance(item, str) for item in only)
    ):
        raise TypeError("teacher.routing.only must be a non-empty list of strings")
    return TeacherRouting(
        only=tuple(only),
        allow_fallbacks=_require_bool(
            routing.get("allow_fallbacks"),
            "teacher.routing.allow_fallbacks",
        ),
        require_parameters=_require_bool(
            routing.get("require_parameters"),
            "teacher.routing.require_parameters",
        ),
    )


@dataclass(frozen=True)
class TeacherGeneration:
    """Sampling limits shared by Stage A and Stage B. No temperature on Vertex."""

    max_tokens: int
    seed: int


@dataclass(frozen=True)
class TeacherReasoning:
    """Gemini thinking settings. ``exclude`` keeps CoT out of the saved target."""

    effort: ReasoningEffort
    exclude: bool

    def as_openrouter_reasoning(self) -> dict[str, Any]:
        """Shape expected in the chat-completions ``reasoning`` field."""
        return {"effort": self.effort, "exclude": self.exclude}

    def as_vertex_thinking(self) -> dict[str, Any]:
        """Shape expected in Vertex ``GenerateContentConfig.thinking_config``."""
        return {
            "thinking_level": self.effort.upper(),
            "include_thoughts": not self.exclude,
        }


@dataclass(frozen=True)
class TeacherRetry:
    """Bounded application-level retry policy for transient provider errors."""

    max_attempts: int
    initial_delay_seconds: float
    max_delay_seconds: float
    exponential_base: float
    jitter_seconds: float
    retryable_status_codes: tuple[int, ...]


def _load_retry(raw: dict[str, Any]) -> TeacherRetry:
    """Validate a bounded retry policy before constructing any client."""
    max_attempts = _require_int(
        raw.get("max_attempts"),
        "teacher.retry.max_attempts",
    )
    if max_attempts < 1:
        raise ValueError("teacher.retry.max_attempts must be at least 1")

    initial_delay = _require_positive_number(
        raw.get("initial_delay_seconds"),
        "teacher.retry.initial_delay_seconds",
    )
    max_delay = _require_positive_number(
        raw.get("max_delay_seconds"),
        "teacher.retry.max_delay_seconds",
    )
    if max_delay < initial_delay:
        raise ValueError(
            "teacher.retry.max_delay_seconds must be greater than or equal to "
            "teacher.retry.initial_delay_seconds"
        )

    exponential_base = _require_positive_number(
        raw.get("exponential_base"),
        "teacher.retry.exponential_base",
    )
    if exponential_base <= 1:
        raise ValueError("teacher.retry.exponential_base must be greater than 1")

    status_codes = raw.get("retryable_status_codes")
    if not isinstance(status_codes, list) or not status_codes:
        raise TypeError(
            "teacher.retry.retryable_status_codes must be a non-empty list"
        )
    if any(
        isinstance(code, bool) or not isinstance(code, int)
        for code in status_codes
    ):
        raise TypeError(
            "teacher.retry.retryable_status_codes must contain integers"
        )
    if any(not 100 <= code <= 599 for code in status_codes):
        raise ValueError(
            "teacher.retry.retryable_status_codes contains an invalid HTTP status"
        )
    if len(status_codes) != len(set(status_codes)):
        raise ValueError(
            "teacher.retry.retryable_status_codes must not contain duplicates"
        )

    return TeacherRetry(
        max_attempts=max_attempts,
        initial_delay_seconds=initial_delay,
        max_delay_seconds=max_delay,
        exponential_base=exponential_base,
        jitter_seconds=_require_nonnegative_number(
            raw.get("jitter_seconds"),
            "teacher.retry.jitter_seconds",
        ),
        retryable_status_codes=tuple(status_codes),
    )


@dataclass(frozen=True)
class TeacherPricing:
    """Pinned list-price inputs used only for transparent local estimation."""

    input_per_million_tokens_usd: float
    output_per_million_tokens_usd: float
    traffic_type: str
    effective_through: str
    source_url: str

    def estimate_usage(self, usage: UsageInfo) -> UsageInfo:
        """Attach a USD estimate, billing hidden thinking as output tokens."""
        prompt = usage.prompt_tokens
        if prompt is None:
            return usage
        if usage.total_tokens is not None:
            output = max(0, usage.total_tokens - prompt)
        elif usage.completion_tokens is not None:
            output = usage.completion_tokens + (usage.thoughts_tokens or 0)
        else:
            return usage
        cost = (
            prompt * self.input_per_million_tokens_usd
            + output * self.output_per_million_tokens_usd
        ) / 1_000_000
        return usage.model_copy(
            update={
                "cost": cost,
                "cost_currency": "USD",
                "cost_basis": "estimated_list_price",
            }
        )


@dataclass(frozen=True)
class TeacherPrompt:
    """One stage prompt, already split into system and user text."""

    system: str
    user: str
    source_path: Path
    version: str
    sha256: str

    def render_user(self, **placeholders: str) -> str:
        """Replace ``{{name}}`` tokens in the user prompt. Stage B needs this."""
        rendered = self.user
        for name, value in placeholders.items():
            rendered = rendered.replace(f"{{{{{name}}}}}", value)
        missing = re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", rendered)
        if missing:
            raise ValueError(
                f"{self.source_path} user prompt is missing placeholders: "
                + ", ".join(sorted(set(missing)))
            )
        return rendered


@dataclass(frozen=True)
class JsonSchemaSpec:
    """Loaded JSON Schema for one stage, plus the OpenRouter wrapper fields."""

    name: str
    strict: bool
    schema: dict[str, Any]
    source_path: Path

    def as_openrouter_json_schema(self) -> dict[str, Any]:
        """Inner ``json_schema`` object sent in ``response_format``."""
        return {
            "name": self.name,
            "strict": self.strict,
            "schema": self.schema,
        }


@dataclass(frozen=True)
class TeacherStage:
    """One teacher invocation: prompt + structured-output schema."""

    key: str
    name: str
    prompt: TeacherPrompt
    response_format_type: str
    json_schema: JsonSchemaSpec

    def response_format(self) -> dict[str, Any]:
        """Full ``response_format`` block for this stage."""
        return {
            "type": self.response_format_type,
            "json_schema": self.json_schema.as_openrouter_json_schema(),
        }


@dataclass(frozen=True)
class TeacherModel:
    """Complete teacher config after YAML, prompts, and schemas have been loaded."""

    name: str
    provider: TeacherProvider
    description: str
    api: TeacherAPI | VertexAPI
    model: TeacherModelSpec
    routing: TeacherRouting | None
    generation: TeacherGeneration
    reasoning: TeacherReasoning
    retry: TeacherRetry | None
    pricing: TeacherPricing | None
    structured: bool
    stages: dict[str, TeacherStage]
    config_path: Path
    project_root: Path

    @classmethod
    def from_yaml(
        cls,
        path: str | Path | None = None,
        *,
        project_root: Path | None = None,
    ) -> TeacherModel:
        """Load a teacher YAML, defaulting to the project Gemini config."""
        root = (project_root or PROJECT_ROOT).resolve()
        config_path = Path(path) if path is not None else DEFAULT_CONFIG
        if not config_path.is_absolute():
            config_path = (root / config_path).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Teacher config not found: {config_path}")

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload = _require_mapping(raw, str(config_path))
        teacher = _require_mapping(payload.get("teacher"), "teacher")
        return cls._from_mapping(
            teacher,
            description=str(payload.get("description") or "").strip(),
            config_path=config_path,
            project_root=root,
        )

    def stage(self, key: str) -> TeacherStage:
        """Return a stage or raise with the configured stage keys."""
        try:
            return self.stages[key]
        except KeyError as exc:
            known = ", ".join(sorted(self.stages))
            raise KeyError(
                f"Unknown teacher stage {key!r}. Known stages: {known}"
            ) from exc

    def openrouter_body(
        self,
        stage_key: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build one chat-completions body without sending it."""
        if self.provider is not TeacherProvider.OPENROUTER or self.routing is None:
            raise TypeError("openrouter_body requires provider=openrouter")
        stage = self.stage(stage_key)
        return {
            "model": self.model.id,
            "messages": messages,
            "max_tokens": self.generation.max_tokens,
            "seed": self.generation.seed,
            "reasoning": self.reasoning.as_openrouter_reasoning(),
            "provider": self.routing.as_openrouter_provider(),
            "response_format": stage.response_format(),
        }

    def vertex_generate_config(self, stage_key: str) -> dict[str, Any]:
        """Build Vertex ``GenerateContentConfig`` fields without sending them."""
        if self.provider is not TeacherProvider.VERTEX:
            raise TypeError("vertex_generate_config requires provider=vertex")
        stage = self.stage(stage_key)
        return {
            "max_output_tokens": self.generation.max_tokens,
            "seed": self.generation.seed,
            "response_mime_type": "application/json",
            "response_json_schema": stage.json_schema.schema,
            "thinking_config": self.reasoning.as_vertex_thinking(),
        }

    @classmethod
    def _from_mapping(
        cls,
        teacher: dict[str, Any],
        *,
        description: str,
        config_path: Path,
        project_root: Path,
    ) -> TeacherModel:
        """Validate the ``teacher:`` mapping and construct nested dataclasses."""
        api = _require_mapping(teacher.get("api"), "teacher.api")
        model = _require_mapping(teacher.get("model"), "teacher.model")
        generation = _require_mapping(teacher.get("generation"), "teacher.generation")
        reasoning = _require_mapping(teacher.get("reasoning"), "teacher.reasoning")
        retry_raw = teacher.get("retry")
        retry = (
            None
            if retry_raw is None
            else _require_mapping(retry_raw, "teacher.retry")
        )
        pricing_raw = teacher.get("pricing")
        pricing = (
            None
            if pricing_raw is None
            else _require_mapping(pricing_raw, "teacher.pricing")
        )
        output = _require_mapping(teacher.get("output"), "teacher.output")
        stages_raw = _require_mapping(output.get("stages"), "teacher.output.stages")
        if not stages_raw:
            raise ValueError("teacher.output.stages must define at least one stage")

        provider_value = _require_str(teacher.get("provider"), "teacher.provider")
        try:
            provider = TeacherProvider(provider_value)
        except ValueError as exc:
            known = ", ".join(item.value for item in TeacherProvider)
            raise ValueError(f"teacher.provider must be one of: {known}") from exc

        effort = _require_str(reasoning.get("effort"), "teacher.reasoning.effort")
        if effort not in {"high", "medium", "low"}:
            raise ValueError(
                "teacher.reasoning.effort must be one of: high, medium, low"
            )

        stages = {
            str(key): cls._load_stage(
                str(key),
                _require_mapping(value, f"teacher.output.stages.{key}"),
                project_root=project_root,
            )
            for key, value in stages_raw.items()
        }

        return cls(
            name=_require_str(teacher.get("name"), "teacher.name"),
            provider=provider,
            description=description,
            api=_load_api(provider, api),
            model=TeacherModelSpec(
                id=_require_str(model.get("id"), "teacher.model.id"),
                multimodal=_require_bool(
                    model.get("multimodal"), "teacher.model.multimodal"
                ),
            ),
            routing=_load_routing(provider, teacher.get("routing")),
            generation=TeacherGeneration(
                max_tokens=_require_int(
                    generation.get("max_tokens"),
                    "teacher.generation.max_tokens",
                ),
                seed=_require_int(generation.get("seed"), "teacher.generation.seed"),
            ),
            reasoning=TeacherReasoning(
                effort=cast(ReasoningEffort, effort),
                exclude=_require_bool(
                    reasoning.get("exclude"), "teacher.reasoning.exclude"
                ),
            ),
            retry=None if retry is None else _load_retry(retry),
            pricing=(
                None
                if pricing is None
                else TeacherPricing(
                    input_per_million_tokens_usd=_require_positive_number(
                        pricing.get("input_per_million_tokens_usd"),
                        "teacher.pricing.input_per_million_tokens_usd",
                    ),
                    output_per_million_tokens_usd=_require_positive_number(
                        pricing.get("output_per_million_tokens_usd"),
                        "teacher.pricing.output_per_million_tokens_usd",
                    ),
                    traffic_type=_require_str(
                        pricing.get("traffic_type"),
                        "teacher.pricing.traffic_type",
                    ),
                    effective_through=_require_str(
                        pricing.get("effective_through"),
                        "teacher.pricing.effective_through",
                    ),
                    source_url=_require_str(
                        pricing.get("source_url"),
                        "teacher.pricing.source_url",
                    ),
                )
            ),
            structured=_require_bool(
                output.get("structured"), "teacher.output.structured"
            ),
            stages=stages,
            config_path=config_path,
            project_root=project_root,
        )

    @classmethod
    def _load_stage(
        cls,
        key: str,
        raw: dict[str, Any],
        *,
        project_root: Path,
    ) -> TeacherStage:
        """Load one stage: read its prompt markdown and JSON Schema from disk."""
        prompt_path = _resolve_project_path(
            _require_str(
                raw.get("prompt_ref"), f"teacher.output.stages.{key}.prompt_ref"
            ),
            project_root=project_root,
        )
        prompt_bytes = prompt_path.read_bytes()
        system, user = _parse_prompt_markdown(
            prompt_bytes.decode("utf-8"),
            source=prompt_path,
        )
        prompt_version = _require_str(
            raw.get("prompt_version"),
            f"teacher.output.stages.{key}.prompt_version",
        )
        expected_prompt_sha256 = _require_sha256(
            raw.get("prompt_sha256"),
            f"teacher.output.stages.{key}.prompt_sha256",
        )
        actual_prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
        if actual_prompt_sha256 != expected_prompt_sha256:
            raise ValueError(
                f"{prompt_path} frozen prompt SHA-256 mismatch for "
                f"{prompt_version}: expected {expected_prompt_sha256}, "
                f"found {actual_prompt_sha256}. Preserve v1 and create an "
                "explicit v2 prompt/config instead of editing it in place."
            )

        response_format = _require_mapping(
            raw.get("response_format"),
            f"teacher.output.stages.{key}.response_format",
        )
        json_schema = _require_mapping(
            response_format.get("json_schema"),
            f"teacher.output.stages.{key}.response_format.json_schema",
        )
        schema_path = _resolve_project_path(
            _require_str(
                json_schema.get("schema_ref"),
                f"teacher.output.stages.{key}.response_format.json_schema.schema_ref",
            ),
            project_root=project_root,
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        if not isinstance(schema, dict):
            raise TypeError(f"{schema_path} must contain a JSON object")
        canonical_schema = teacher_output_schema(key)
        if schema != canonical_schema:
            raise ValueError(
                f"{schema_path} is stale. Run "
                "`python -m project.teacher.export_schemas` and review the diff."
            )

        return TeacherStage(
            key=key,
            name=_require_str(raw.get("name"), f"teacher.output.stages.{key}.name"),
            prompt=TeacherPrompt(
                system=system,
                user=user,
                source_path=prompt_path,
                version=prompt_version,
                sha256=actual_prompt_sha256,
            ),
            response_format_type=_require_str(
                response_format.get("type"),
                f"teacher.output.stages.{key}.response_format.type",
            ),
            json_schema=JsonSchemaSpec(
                name=_require_str(
                    json_schema.get("name"),
                    f"teacher.output.stages.{key}.response_format.json_schema.name",
                ),
                strict=_require_bool(
                    json_schema.get("strict"),
                    f"teacher.output.stages.{key}.response_format.json_schema.strict",
                ),
                schema=schema,
                source_path=schema_path,
            ),
        )
