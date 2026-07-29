"""Typed, validated configuration for benchmark model runtimes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml


ModelBackendType = Literal["local", "api"]
ModelEngine = Literal[
    "vllm",
    "transformers",
    "azure_openai",
    "vllm_endpoint",
]
ApiStyle = Literal["chat_completions", "responses"]


class ModelConfigError(ValueError):
    """Raised when a model YAML file does not satisfy the runtime contract."""


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Backend-independent decoding settings."""

    profile: str = "default"
    reasoning_effort: str | None = None
    do_sample: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class ModelMetadataConfig:
    """Human-readable and experimental model identity."""

    id: str
    display_name: str
    family: str
    evaluation_role: str
    training_stage: str
    task_adaptation_stage: str
    domain: str | None = None


@dataclass(frozen=True, slots=True)
class ModelUsageConfig:
    """Allowed project uses for a model."""

    benchmark: bool
    fine_tuning: bool


@dataclass(frozen=True, slots=True)
class ModelCapabilitiesConfig:
    """Capabilities needed to validate benchmark compatibility."""

    modalities: tuple[str, ...]
    structured_output_modes: tuple[str, ...]
    context_length_tokens: int | None = None
    maximum_output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ModelSourceConfig:
    """Immutable description of the model source."""

    type: Literal["huggingface", "provider_api"]
    repo_id: str | None = None
    revision: str | None = None
    access: str | None = None
    license: str | None = None
    terms: str | None = None
    provider: str | None = None
    model_name: str | None = None


@dataclass(frozen=True, slots=True)
class ImageProcessorConfig:
    """Optional image preprocessing profile."""

    profile: str
    downsample_mode: str
    max_slice_nums: int


@dataclass(frozen=True, slots=True)
class ProcessorConfig:
    """Processor repository and optional image settings."""

    repo_id: str
    revision: str
    image: ImageProcessorConfig | None = None


@dataclass(frozen=True, slots=True)
class ChatTemplateConfig:
    """Arguments forwarded to the model chat template."""

    enable_thinking: bool = False


@dataclass(frozen=True, slots=True)
class ReasoningConfig:
    """How a model exposes and separates its reasoning channel."""

    enabled: bool = False
    parser: str | None = None
    exclude_from_structured_output: bool = True
    chat_template_kwargs: ChatTemplateConfig = ChatTemplateConfig()


@dataclass(frozen=True, slots=True)
class SecurityConfig:
    """Security-sensitive model loading flags."""

    trust_remote_code: bool = False


@dataclass(frozen=True, slots=True)
class BackendProfileConfig:
    """One executable local or API backend profile."""

    name: str
    type: ModelBackendType
    engine: ModelEngine
    api_style: ApiStyle | None = None
    device: str | None = None
    dtype: str | None = None
    tensor_parallel_size: int | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    limit_images_per_prompt: int | None = None
    managed: bool | None = None
    managed_allowed: bool | None = None
    endpoint_env: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None
    deployment_env: str | None = None
    model_env: str | None = None
    api_version_env: str | None = None

    @property
    def limit_mm_per_prompt_image(self) -> int | None:
        """vLLM-compatible alias for the single-image prompt limit."""

        return self.limit_images_per_prompt


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """Named backend profiles and the default selected profile."""

    default_profile: str
    profiles: tuple[BackendProfileConfig, ...]
    selected_profile: str

    @property
    def active_profile(self) -> BackendProfileConfig:
        """Return the explicitly selected backend profile."""

        for profile in self.profiles:
            if profile.name == self.selected_profile:
                return profile
        raise ModelConfigError(
            f"Selected backend profile {self.selected_profile!r} is missing"
        )

    def profile(self, name: str) -> BackendProfileConfig:
        """Return a named backend profile."""

        for profile in self.profiles:
            if profile.name == name:
                return profile
        choices = ", ".join(item.name for item in self.profiles)
        raise ModelConfigError(
            f"Unknown backend profile {name!r}; available profiles: {choices}"
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Complete normalized model configuration."""

    schema_version: int
    model: ModelMetadataConfig
    usage: ModelUsageConfig
    capabilities: ModelCapabilitiesConfig
    source: ModelSourceConfig
    processor: ProcessorConfig | None
    backend: BackendConfig
    reasoning: ReasoningConfig
    generation: GenerationConfig
    security: SecurityConfig
    config_path: Path

    @property
    def model_id(self) -> str:
        """Compatibility alias used by the existing inference scaffolds."""

        return self.model.id

    @property
    def backend_type(self) -> Literal["local", "azure"]:
        """Compatibility alias until the inference factory uses profiles."""

        return (
            "local"
            if self.backend.active_profile.type == "local"
            else "azure"
        )

    @property
    def device(self) -> str:
        return self.backend.active_profile.device or "cuda"

    @property
    def dtype(self) -> str:
        return self.backend.active_profile.dtype or "bfloat16"

    @property
    def trust_remote_code(self) -> bool:
        return self.security.trust_remote_code

    @property
    def endpoint_env(self) -> str:
        return self.backend.active_profile.endpoint_env or ""

    @property
    def api_key_env(self) -> str:
        return self.backend.active_profile.api_key_env or ""

    @property
    def deployment_name(self) -> str:
        return self.backend.active_profile.deployment_env or ""

    @property
    def api_version(self) -> str:
        return self.backend.active_profile.api_version_env or ""


@dataclass(frozen=True, slots=True)
class LocalModelConfig(ModelConfig):
    """Model configuration whose selected profile runs locally."""


@dataclass(frozen=True, slots=True)
class AzureModelConfig(ModelConfig):
    """Model configuration whose selected profile is API-hosted."""


_TOP_LEVEL_KEYS = {
    "schema_version",
    "model",
    "usage",
    "capabilities",
    "source",
    "processor",
    "backend",
    "reasoning",
    "generation",
    "security",
}
_MODEL_KEYS = {
    "id",
    "display_name",
    "family",
    "evaluation_role",
    "training_stage",
    "task_adaptation_stage",
    "domain",
}
_USAGE_KEYS = {"benchmark", "fine_tuning"}
_CAPABILITY_KEYS = {
    "modalities",
    "structured_output_modes",
    "context_length_tokens",
    "maximum_output_tokens",
}
_SOURCE_KEYS = {
    "type",
    "repo_id",
    "revision",
    "access",
    "license",
    "terms",
    "provider",
    "model_name",
}
_PROCESSOR_KEYS = {"repo_id", "revision", "image"}
_IMAGE_PROCESSOR_KEYS = {
    "profile",
    "downsample_mode",
    "max_slice_nums",
}
_BACKEND_KEYS = {"default_profile", "profiles"}
_PROFILE_KEYS = {
    "type",
    "engine",
    "api_style",
    "device",
    "dtype",
    "tensor_parallel_size",
    "max_model_len",
    "gpu_memory_utilization",
    "limit_images_per_prompt",
    "managed",
    "managed_allowed",
    "endpoint_env",
    "base_url_env",
    "api_key_env",
    "deployment_env",
    "model_env",
    "api_version_env",
}
_REASONING_KEYS = {
    "enabled",
    "parser",
    "exclude_from_structured_output",
    "chat_template_kwargs",
}
_CHAT_TEMPLATE_KEYS = {"enable_thinking"}
_GENERATION_KEYS = {
    "profile",
    "reasoning_effort",
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
    "seed",
}
_SECURITY_KEYS = {"trust_remote_code"}


def load_model_config(
    id_or_path: str | Path,
    *,
    root: Path | None = None,
    backend_profile: str | None = None,
) -> ModelConfig:
    """Load one of the project model YAMLs by model ID or file path."""

    project_root = root or _project_root()
    path = _resolve_config_path(
        id_or_path,
        directory=project_root / "configs/models",
        id_path=("model", "id"),
        kind="model",
        root=project_root,
    )
    document = _load_yaml(path)
    _reject_unknown(document, _TOP_LEVEL_KEYS, str(path))
    _require_keys(
        document,
        {
            "schema_version",
            "model",
            "usage",
            "capabilities",
            "source",
            "backend",
            "generation",
            "security",
        },
        str(path),
    )
    schema_version = _integer(document["schema_version"], "schema_version")
    if schema_version != 1:
        raise ModelConfigError(
            f"schema_version must equal 1 in {path}, got {schema_version}"
        )

    model = _parse_model(_mapping(document["model"], "model"))
    usage = _parse_usage(_mapping(document["usage"], "usage"))
    capabilities = _parse_capabilities(
        _mapping(document["capabilities"], "capabilities")
    )
    if "image" not in capabilities.modalities:
        raise ModelConfigError(
            f"capabilities.modalities for {model.id} must include 'image'"
        )
    source = _parse_source(_mapping(document["source"], "source"))
    processor = (
        _parse_processor(_mapping(document["processor"], "processor"))
        if document.get("processor") is not None
        else None
    )
    backend = _parse_backend(
        _mapping(document["backend"], "backend"),
        selected_profile=backend_profile,
    )
    reasoning = _parse_reasoning(
        _mapping(document.get("reasoning", {}), "reasoning")
    )
    generation = _parse_generation(
        _mapping(document["generation"], "generation")
    )
    security = _parse_security(
        _mapping(document["security"], "security")
    )
    if backend.active_profile.type == "local":
        if source.type != "huggingface" or not source.repo_id:
            raise ModelConfigError(
                f"Local model {model.id} requires source.repo_id"
            )
        config_type: type[ModelConfig] = LocalModelConfig
        if generation.do_sample is None:
            raise ModelConfigError(
                f"Local model {model.id} requires generation.do_sample"
            )
    else:
        if source.type != "provider_api":
            raise ModelConfigError(
                f"API model {model.id} requires source.type provider_api"
            )
        config_type = AzureModelConfig
    return config_type(
        schema_version=schema_version,
        model=model,
        usage=usage,
        capabilities=capabilities,
        source=source,
        processor=processor,
        backend=backend,
        reasoning=reasoning,
        generation=generation,
        security=security,
        config_path=path,
    )


def list_model_configs(*, root: Path | None = None) -> tuple[ModelConfig, ...]:
    """Load every project model configuration in stable ID order."""

    project_root = root or _project_root()
    configs = [
        load_model_config(path, root=project_root)
        for path in sorted((project_root / "configs/models").glob("*.yaml"))
    ]
    ids = [config.model.id for config in configs]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ModelConfigError(
            "Duplicate model IDs: " + ", ".join(duplicates)
        )
    return tuple(sorted(configs, key=lambda item: item.model.id))


def select_backend_profile(
    config: ModelConfig,
    profile_name: str,
) -> ModelConfig:
    """Return a copy of a loaded config with another profile selected."""

    profile = config.backend.profile(profile_name)
    backend = replace(config.backend, selected_profile=profile.name)
    target: type[ModelConfig] = (
        LocalModelConfig if profile.type == "local" else AzureModelConfig
    )
    values = {
        field_name: getattr(config, field_name)
        for field_name in config.__dataclass_fields__
    }
    values["backend"] = backend
    return target(**values)


def _parse_model(value: dict[str, Any]) -> ModelMetadataConfig:
    _reject_unknown(value, _MODEL_KEYS, "model")
    _require_keys(
        value,
        {
            "id",
            "display_name",
            "family",
            "evaluation_role",
            "training_stage",
            "task_adaptation_stage",
        },
        "model",
    )
    return ModelMetadataConfig(
        id=_text(value["id"], "model.id"),
        display_name=_text(value["display_name"], "model.display_name"),
        family=_text(value["family"], "model.family"),
        evaluation_role=_text(
            value["evaluation_role"], "model.evaluation_role"
        ),
        training_stage=_text(
            value["training_stage"], "model.training_stage"
        ),
        task_adaptation_stage=_text(
            value["task_adaptation_stage"],
            "model.task_adaptation_stage",
        ),
        domain=_optional_text(value.get("domain"), "model.domain"),
    )


def _parse_usage(value: dict[str, Any]) -> ModelUsageConfig:
    _reject_unknown(value, _USAGE_KEYS, "usage")
    _require_keys(value, _USAGE_KEYS, "usage")
    return ModelUsageConfig(
        benchmark=_boolean(value["benchmark"], "usage.benchmark"),
        fine_tuning=_boolean(
            value["fine_tuning"], "usage.fine_tuning"
        ),
    )


def _parse_capabilities(value: dict[str, Any]) -> ModelCapabilitiesConfig:
    _reject_unknown(value, _CAPABILITY_KEYS, "capabilities")
    _require_keys(
        value,
        {"modalities", "structured_output_modes"},
        "capabilities",
    )
    modalities = _string_tuple(
        value["modalities"], "capabilities.modalities"
    )
    output_modes = _string_tuple(
        value["structured_output_modes"],
        "capabilities.structured_output_modes",
    )
    if "prompt_only" not in output_modes:
        raise ModelConfigError(
            "capabilities.structured_output_modes must include 'prompt_only'"
        )
    return ModelCapabilitiesConfig(
        modalities=modalities,
        structured_output_modes=output_modes,
        context_length_tokens=_optional_positive_integer(
            value.get("context_length_tokens"),
            "capabilities.context_length_tokens",
        ),
        maximum_output_tokens=_optional_positive_integer(
            value.get("maximum_output_tokens"),
            "capabilities.maximum_output_tokens",
        ),
    )


def _parse_source(value: dict[str, Any]) -> ModelSourceConfig:
    _reject_unknown(value, _SOURCE_KEYS, "source")
    _require_keys(value, {"type"}, "source")
    source_type = _text(value["type"], "source.type")
    if source_type not in {"huggingface", "provider_api"}:
        raise ModelConfigError(
            "source.type must be 'huggingface' or 'provider_api'"
        )
    return ModelSourceConfig(
        type=source_type,
        repo_id=_optional_text(value.get("repo_id"), "source.repo_id"),
        revision=_optional_text(
            value.get("revision"), "source.revision"
        ),
        access=_optional_text(value.get("access"), "source.access"),
        license=_optional_text(value.get("license"), "source.license"),
        terms=_optional_text(value.get("terms"), "source.terms"),
        provider=_optional_text(
            value.get("provider"), "source.provider"
        ),
        model_name=_optional_text(
            value.get("model_name"), "source.model_name"
        ),
    )


def _parse_processor(value: dict[str, Any]) -> ProcessorConfig:
    _reject_unknown(value, _PROCESSOR_KEYS, "processor")
    _require_keys(value, {"repo_id", "revision"}, "processor")
    image_value = value.get("image")
    image = None
    if image_value is not None:
        image_mapping = _mapping(image_value, "processor.image")
        _reject_unknown(
            image_mapping,
            _IMAGE_PROCESSOR_KEYS,
            "processor.image",
        )
        _require_keys(
            image_mapping,
            _IMAGE_PROCESSOR_KEYS,
            "processor.image",
        )
        image = ImageProcessorConfig(
            profile=_text(
                image_mapping["profile"], "processor.image.profile"
            ),
            downsample_mode=_text(
                image_mapping["downsample_mode"],
                "processor.image.downsample_mode",
            ),
            max_slice_nums=_positive_integer(
                image_mapping["max_slice_nums"],
                "processor.image.max_slice_nums",
            ),
        )
    return ProcessorConfig(
        repo_id=_text(value["repo_id"], "processor.repo_id"),
        revision=_text(value["revision"], "processor.revision"),
        image=image,
    )


def _parse_backend(
    value: dict[str, Any],
    *,
    selected_profile: str | None,
) -> BackendConfig:
    _reject_unknown(value, _BACKEND_KEYS, "backend")
    _require_keys(value, _BACKEND_KEYS, "backend")
    default = _text(
        value["default_profile"], "backend.default_profile"
    )
    profile_values = _mapping(value["profiles"], "backend.profiles")
    if not profile_values:
        raise ModelConfigError("backend.profiles must not be empty")
    profiles = tuple(
        _parse_backend_profile(name, profile)
        for name, profile in profile_values.items()
    )
    names = [profile.name for profile in profiles]
    if len(names) != len(set(names)):
        raise ModelConfigError("backend profile names must be unique")
    if default not in names:
        raise ModelConfigError(
            f"backend.default_profile {default!r} is not defined"
        )
    selected = selected_profile or default
    if selected not in names:
        raise ModelConfigError(
            f"Unknown backend profile {selected!r}; "
            f"available profiles: {', '.join(names)}"
        )
    return BackendConfig(
        default_profile=default,
        profiles=profiles,
        selected_profile=selected,
    )


def _parse_backend_profile(
    name_value: Any,
    profile_value: Any,
) -> BackendProfileConfig:
    name = _text(name_value, "backend.profiles key")
    value = _mapping(profile_value, f"backend.profiles.{name}")
    section = f"backend.profiles.{name}"
    _reject_unknown(value, _PROFILE_KEYS, section)
    _require_keys(value, {"type", "engine"}, section)
    backend_type = _text(value["type"], f"{section}.type")
    engine = _text(value["engine"], f"{section}.engine")
    if backend_type not in {"local", "api"}:
        raise ModelConfigError(
            f"{section}.type must be 'local' or 'api'"
        )
    if engine not in {
        "vllm",
        "transformers",
        "azure_openai",
        "vllm_endpoint",
    }:
        raise ModelConfigError(
            f"{section}.engine is unsupported: {engine!r}"
        )
    api_style_value = value.get("api_style")
    api_style = (
        _text(api_style_value, f"{section}.api_style")
        if api_style_value is not None
        else None
    )
    if backend_type == "local":
        if engine not in {"vllm", "transformers"}:
            raise ModelConfigError(
                f"{section}: local profiles must use engine 'vllm' "
                "or 'transformers'"
            )
        if api_style is not None:
            raise ModelConfigError(
                f"{section}.api_style is only valid for API profiles"
            )
    else:
        if engine in {"vllm", "transformers"}:
            raise ModelConfigError(
                f"{section}: API profiles cannot use local engine "
                f"{engine!r}"
            )
        if api_style not in {"chat_completions", "responses"}:
            raise ModelConfigError(
                f"{section}.api_style must be chat_completions or responses"
            )
        if not value.get("api_key_env"):
            raise ModelConfigError(
                f"{section}.api_key_env is required for API profiles"
            )
        if engine == "azure_openai" and not value.get("endpoint_env"):
            raise ModelConfigError(
                f"{section}.endpoint_env is required for Azure profiles"
            )
        if engine == "vllm_endpoint" and not value.get("base_url_env"):
            raise ModelConfigError(
                f"{section}.base_url_env is required for vLLM endpoints"
            )
    tensor_parallel_size = _optional_positive_integer(
        value.get("tensor_parallel_size"),
        f"{section}.tensor_parallel_size",
    )
    max_model_len = _optional_positive_integer(
        value.get("max_model_len"),
        f"{section}.max_model_len",
    )
    gpu_memory_utilization = _optional_fraction(
        value.get("gpu_memory_utilization"),
        f"{section}.gpu_memory_utilization",
    )
    limit_images_per_prompt = _optional_positive_integer(
        value.get("limit_images_per_prompt"),
        f"{section}.limit_images_per_prompt",
    )
    managed = _optional_boolean(
        value.get("managed"), f"{section}.managed"
    )
    managed_allowed = _optional_boolean(
        value.get("managed_allowed"), f"{section}.managed_allowed"
    )
    if backend_type == "local":
        max_model_len = max_model_len or 16384
        gpu_memory_utilization = (
            0.9
            if gpu_memory_utilization is None
            else gpu_memory_utilization
        )
        limit_images_per_prompt = limit_images_per_prompt or 1
        managed = True if managed is None else managed
        managed_allowed = (
            True if managed_allowed is None else managed_allowed
        )
    else:
        managed = False if managed is None else managed
        managed_allowed = (
            False if managed_allowed is None else managed_allowed
        )
    return BackendProfileConfig(
        name=name,
        type=backend_type,
        engine=engine,
        api_style=api_style,
        device=_optional_text(
            value.get("device"), f"{section}.device"
        ),
        dtype=_optional_text(value.get("dtype"), f"{section}.dtype"),
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_images_per_prompt=limit_images_per_prompt,
        managed=managed,
        managed_allowed=managed_allowed,
        endpoint_env=_optional_text(
            value.get("endpoint_env"), f"{section}.endpoint_env"
        ),
        base_url_env=_optional_text(
            value.get("base_url_env"), f"{section}.base_url_env"
        ),
        api_key_env=_optional_text(
            value.get("api_key_env"), f"{section}.api_key_env"
        ),
        deployment_env=_optional_text(
            value.get("deployment_env"), f"{section}.deployment_env"
        ),
        model_env=_optional_text(
            value.get("model_env"), f"{section}.model_env"
        ),
        api_version_env=_optional_text(
            value.get("api_version_env"),
            f"{section}.api_version_env",
        ),
    )


def _parse_reasoning(value: dict[str, Any]) -> ReasoningConfig:
    _reject_unknown(value, _REASONING_KEYS, "reasoning")
    chat_values = _mapping(
        value.get("chat_template_kwargs", {}),
        "reasoning.chat_template_kwargs",
    )
    _reject_unknown(
        chat_values,
        _CHAT_TEMPLATE_KEYS,
        "reasoning.chat_template_kwargs",
    )
    enabled = _boolean(value.get("enabled", False), "reasoning.enabled")
    parser = _optional_text(value.get("parser"), "reasoning.parser")
    if enabled and not parser:
        raise ModelConfigError(
            "reasoning.parser is required when reasoning.enabled is true"
        )
    return ReasoningConfig(
        enabled=enabled,
        parser=parser,
        exclude_from_structured_output=_boolean(
            value.get("exclude_from_structured_output", True),
            "reasoning.exclude_from_structured_output",
        ),
        chat_template_kwargs=ChatTemplateConfig(
            enable_thinking=_boolean(
                chat_values.get("enable_thinking", False),
                "reasoning.chat_template_kwargs.enable_thinking",
            )
        ),
    )


def _parse_generation(value: dict[str, Any]) -> GenerationConfig:
    _reject_unknown(value, _GENERATION_KEYS, "generation")
    _require_keys(value, {"profile"}, "generation")
    return GenerationConfig(
        profile=_text(value["profile"], "generation.profile"),
        reasoning_effort=_optional_text(
            value.get("reasoning_effort"),
            "generation.reasoning_effort",
        ),
        do_sample=_optional_boolean(
            value.get("do_sample"), "generation.do_sample"
        ),
        temperature=_optional_number(
            value.get("temperature"), "generation.temperature"
        ),
        top_p=_optional_number(
            value.get("top_p"), "generation.top_p"
        ),
        top_k=_optional_integer(
            value.get("top_k"), "generation.top_k"
        ),
        min_p=_optional_number(
            value.get("min_p"), "generation.min_p"
        ),
        presence_penalty=_optional_number(
            value.get("presence_penalty"),
            "generation.presence_penalty",
        ),
        repetition_penalty=_optional_number(
            value.get("repetition_penalty"),
            "generation.repetition_penalty",
        ),
        seed=_optional_integer(
            value.get("seed"), "generation.seed"
        ),
    )


def _parse_security(value: dict[str, Any]) -> SecurityConfig:
    _reject_unknown(value, _SECURITY_KEYS, "security")
    _require_keys(value, _SECURITY_KEYS, "security")
    return SecurityConfig(
        trust_remote_code=_boolean(
            value["trust_remote_code"],
            "security.trust_remote_code",
        )
    )


def _resolve_config_path(
    id_or_path: str | Path,
    *,
    directory: Path,
    id_path: tuple[str, str],
    kind: str,
    root: Path,
) -> Path:
    value = Path(id_or_path)
    candidates = [value] if value.is_absolute() else [root / value, value]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if value.suffix in {".yaml", ".yml"} or value.parent != Path("."):
        raise ModelConfigError(f"{kind.title()} config not found: {value}")
    matches: list[Path] = []
    for path in sorted(directory.glob("*.yaml")):
        document = _load_yaml(path)
        section = document.get(id_path[0])
        if isinstance(section, dict) and section.get(id_path[1]) == str(
            id_or_path
        ):
            matches.append(path)
    if not matches:
        raise ModelConfigError(
            f"Unknown {kind} ID {str(id_or_path)!r}"
        )
    if len(matches) > 1:
        raise ModelConfigError(
            f"Duplicate {kind} ID {str(id_or_path)!r}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0].resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ModelConfigError(
            f"Could not read model config {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ModelConfigError(
            f"Model config {path} must contain a YAML mapping"
        )
    return value


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelConfigError(f"{path} must be a mapping")
    return value


def _reject_unknown(
    value: dict[str, Any],
    allowed: set[str],
    path: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ModelConfigError(
            f"{path} contains unknown field(s): {', '.join(unknown)}"
        )


def _require_keys(
    value: dict[str, Any],
    required: set[str],
    path: str,
) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ModelConfigError(
            f"{path} is missing required field(s): {', '.join(missing)}"
        )


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelConfigError(f"{path} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _text(value, path)


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ModelConfigError(f"{path} must be a boolean")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelConfigError(f"{path} must be a number")
    return float(value)


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelConfigError(f"{path} must be an integer")
    return value


def _positive_integer(value: Any, path: str) -> int:
    result = _integer(value, path)
    if result <= 0:
        raise ModelConfigError(f"{path} must be positive")
    return result


def _optional_positive_integer(value: Any, path: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, path)


def _optional_boolean(value: Any, path: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, path)


def _optional_fraction(value: Any, path: str) -> float | None:
    if value is None:
        return None
    result = _number(value, path)
    if not 0.0 < result <= 1.0:
        raise ModelConfigError(
            f"{path} must be greater than zero and at most one"
        )
    return result


def _optional_number(value: Any, path: str) -> float | None:
    if value is None:
        return None
    return _number(value, path)


def _optional_integer(value: Any, path: str) -> int | None:
    if value is None:
        return None
    return _integer(value, path)


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ModelConfigError(f"{path} must be a non-empty list")
    result = tuple(
        _text(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise ModelConfigError(f"{path} values must be unique")
    return result


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]
