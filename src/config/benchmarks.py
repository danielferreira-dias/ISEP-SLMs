"""Typed, validated configuration for the three benchmark protocols."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


BenchmarkTask = Literal[
    "visual_disease_ranking",
    "visual_disease_contrast_ranking",
    "evidence_grounded_visual_diagnosis",
]


class BenchmarkConfigError(ValueError):
    """Raised when a benchmark YAML does not satisfy the runtime contract."""


@dataclass(frozen=True, slots=True)
class BenchmarkMetadataConfig:
    """Benchmark identity and task-level cardinality."""

    id: str
    version: str
    task: BenchmarkTask
    description: str
    status: str | None = None
    top_k: int | None = None
    candidate_count: int | None = None
    ranking_count: int | None = None


@dataclass(frozen=True, slots=True)
class MorphologyConceptConfig:
    """One controlled morphology label."""

    id: str
    display_name: str


@dataclass(frozen=True, slots=True)
class TaxonomyConfig:
    """Normalized paths and concepts used to render a benchmark taxonomy."""

    disease_path: Path
    morphology_concepts: tuple[MorphologyConceptConfig, ...] = ()
    confusion_sets_path: Path | None = None


@dataclass(frozen=True, slots=True)
class EvaluationSetConfig:
    """One named Parquet evaluation set."""

    id: str
    manifest: Path
    role: str
    description: str


@dataclass(frozen=True, slots=True)
class BenchmarkDatasetConfig:
    """Normalized dataset columns shared by benchmark task adapters."""

    default_evaluation_set: str
    evaluation_sets: tuple[EvaluationSetConfig, ...]
    image_column: str
    sample_id_column: str
    label_column: str
    group_column: str | None = None
    task_id_column: str | None = None
    pair_id_column: str | None = None
    candidate_ids_column: str | None = None
    condition_column: str | None = None
    confusion_set_column: str | None = None

    @property
    def default(self) -> EvaluationSetConfig:
        """Return the configured default evaluation set."""

        return self.evaluation_set(self.default_evaluation_set)

    def evaluation_set(self, name: str) -> EvaluationSetConfig:
        """Return one named evaluation set with a clear unknown-ID error."""

        for evaluation_set in self.evaluation_sets:
            if evaluation_set.id == name:
                return evaluation_set
        choices = ", ".join(item.id for item in self.evaluation_sets)
        raise BenchmarkConfigError(
            f"Unknown evaluation set {name!r}; available sets: {choices}"
        )


@dataclass(frozen=True, slots=True)
class BenchmarkExecutionConfig:
    """Execution behavior fixed by the benchmark protocol."""

    max_output_tokens: int
    batch_size: int
    resume: bool
    save_raw_responses: bool
    save_rendered_prompts: bool
    fail_fast_on_invalid_output: bool
    implementation_status: str | None = None


@dataclass(frozen=True, slots=True)
class ImagePreprocessingConfig:
    """Deterministic image normalization shared by every benchmark model."""

    profile: str
    max_edge_pixels: int
    max_encoded_bytes: int
    jpeg_quality: int
    minimum_jpeg_quality: int
    minimum_edge_pixels: int


@dataclass(frozen=True, slots=True)
class StructuredOutputConfig:
    """Structured-output enforcement mode."""

    mode: Literal["prompt_only"]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Complete normalized runtime benchmark configuration."""

    benchmark: BenchmarkMetadataConfig
    prompt_path: Path
    schema_path: Path
    taxonomy: TaxonomyConfig
    dataset: BenchmarkDatasetConfig
    image_preprocessing: ImagePreprocessingConfig
    execution: BenchmarkExecutionConfig
    structured_output: StructuredOutputConfig
    output_directory: Path
    config_path: Path

    @property
    def benchmark_id(self) -> str:
        return self.benchmark.id

    @property
    def task(self) -> BenchmarkTask:
        return self.benchmark.task

    @property
    def max_output_tokens(self) -> int:
        return self.execution.max_output_tokens


_TOP_LEVEL_KEYS = {
    "benchmark",
    "prompt",
    "schema",
    "taxonomy",
    "dataset",
    "metric_cohorts",
    "metrics",
    "semantic_validation",
    "subgroup_evaluation",
    "validation",
    "comparison",
    "image_preprocessing",
    "execution",
    "structured_output",
    "output",
    "release",
}
_BENCHMARK_KEYS = {
    "id",
    "version",
    "task",
    "status",
    "description",
    "top_k",
    "candidate_count",
    "ranking_count",
}
_PROMPT_KEYS = {"path"}
_SCHEMA_KEYS = {"path", "dynamic_candidate_enum"}
_EXECUTION_KEYS = {
    "implementation_status",
    "max_output_tokens",
    "batch_size",
    "resume",
    "save_raw_responses",
    "save_rendered_prompts",
    "fail_fast_on_invalid_output",
}
_IMAGE_PREPROCESSING_KEYS = {
    "profile",
    "max_edge_pixels",
    "max_encoded_bytes",
    "jpeg_quality",
    "minimum_jpeg_quality",
    "minimum_edge_pixels",
}
_STRUCTURED_OUTPUT_KEYS = {"mode"}
_OUTPUT_KEYS = {"directory"}
_EVALUATION_SET_KEYS = {"manifest", "role", "description"}
_TAXONOMY_KEYS = {
    "path",
    "id_field",
    "label_field",
    "expected_size",
    "disease",
    "morphology",
    "confusion_sets",
}
_DISEASE_TAXONOMY_KEYS = {
    "path",
    "id_field",
    "label_field",
    "expected_size",
}
_MORPHOLOGY_TAXONOMY_KEYS = {
    "source",
    "expected_size",
    "concepts",
}
_CONFUSION_TAXONOMY_KEYS = {
    "path",
    "expected_set_count",
    "expected_covered_disease_count",
}
_DATASET_KEYS = {
    "manifest",
    "task_manifest",
    "source_manifest",
    "release_manifest",
    "evaluation_origin",
    "default_evaluation_set",
    "evaluation_sets",
    "split_config",
    "manifest_schema",
    "dataset_catalog",
    "disease_inclusion_policy",
    "morphology_annotations",
    "caption_metadata",
    "image_column",
    "sample_id_column",
    "task_id_column",
    "pair_id_column",
    "source_group_column",
    "group_column",
    "source_dataset_column",
    "label_column",
    "disease_label_column",
    "diagnosis_basis_column",
    "reference_diagnoses_column",
    "reference_label_format",
    "disease_eligibility_column",
    "skin_tone_column",
    "annotation_image_id_column",
    "annotation_exclusion_column",
    "caption_image_id_column",
    "caption_source_column",
    "caption_text_column",
    "candidate_ids_column",
    "condition_column",
    "confusion_set_column",
    "conditions",
    "joins",
    "filters",
    "expected_counts",
    "expected_unique_images",
    "expected_pairs",
    "expected_tasks",
    "summary_report",
    "integrity_report",
    "reference_policy",
    "leakage_policy",
    "description",
    "development_validation",
    "internal_benchmark",
}


def load_benchmark_config(
    id_or_path: str | Path,
    *,
    root: Path | None = None,
) -> BenchmarkConfig:
    """Load a benchmark YAML by benchmark ID or file path."""

    project_root = root or _project_root()
    path = _resolve_config_path(
        id_or_path,
        directory=project_root / "configs/benchmarks",
        root=project_root,
    )
    document = _load_yaml(path)
    _reject_unknown(document, _TOP_LEVEL_KEYS, str(path))
    _require_keys(
        document,
        {
            "benchmark",
            "prompt",
            "schema",
            "taxonomy",
            "dataset",
            "image_preprocessing",
            "execution",
            "structured_output",
            "output",
        },
        str(path),
    )
    benchmark = _parse_metadata(
        _mapping(document["benchmark"], "benchmark")
    )
    prompt_path = _referenced_file(
        project_root,
        _single_path_section(
            document["prompt"], "prompt", _PROMPT_KEYS
        ),
        "prompt.path",
    )
    schema_path = _referenced_file(
        project_root,
        _single_path_section(
            document["schema"], "schema", _SCHEMA_KEYS
        ),
        "schema.path",
    )
    taxonomy = _parse_taxonomy(
        _mapping(document["taxonomy"], "taxonomy"),
        root=project_root,
    )
    dataset = _parse_dataset(
        _mapping(document["dataset"], "dataset"),
        root=project_root,
        task=benchmark.task,
    )
    image_preprocessing = _parse_image_preprocessing(
        _mapping(
            document["image_preprocessing"],
            "image_preprocessing",
        )
    )
    execution = _parse_execution(
        _mapping(document["execution"], "execution")
    )
    structured_output = _parse_structured_output(
        _mapping(document["structured_output"], "structured_output")
    )
    output_values = _mapping(document["output"], "output")
    _reject_unknown(output_values, _OUTPUT_KEYS, "output")
    _require_keys(output_values, _OUTPUT_KEYS, "output")
    output_directory = _resolve_relative(
        project_root,
        _text(output_values["directory"], "output.directory"),
    )
    if (
        benchmark.task == "visual_disease_ranking"
        and benchmark.top_k != 6
    ):
        raise BenchmarkConfigError(
            "visual_disease_ranking requires benchmark.top_k equal to 6"
        )
    if benchmark.task == "visual_disease_contrast_ranking":
        if benchmark.candidate_count != 3:
            raise BenchmarkConfigError(
                "visual_disease_contrast_ranking requires "
                "benchmark.candidate_count equal to 3"
            )
        if benchmark.ranking_count != benchmark.candidate_count:
            raise BenchmarkConfigError(
                "benchmark.ranking_count must equal candidate_count"
            )
    if (
        benchmark.task == "evidence_grounded_visual_diagnosis"
        and benchmark.top_k != 6
    ):
        raise BenchmarkConfigError(
            "evidence_grounded_visual_diagnosis requires "
            "benchmark.top_k equal to 6"
        )
    return BenchmarkConfig(
        benchmark=benchmark,
        prompt_path=prompt_path,
        schema_path=schema_path,
        taxonomy=taxonomy,
        dataset=dataset,
        image_preprocessing=image_preprocessing,
        execution=execution,
        structured_output=structured_output,
        output_directory=output_directory,
        config_path=path,
    )


def list_benchmark_configs(
    *,
    root: Path | None = None,
) -> tuple[BenchmarkConfig, ...]:
    """Load all DermaISEP runtime task configurations in stable ID order."""

    project_root = root or _project_root()
    configs = [
        load_benchmark_config(path, root=project_root)
        for path in sorted(
            (project_root / "configs/benchmarks/derma_isep").glob("*.yaml")
        )
    ]
    ids = [config.benchmark.id for config in configs]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise BenchmarkConfigError(
            "Duplicate benchmark IDs: " + ", ".join(duplicates)
        )
    return tuple(sorted(configs, key=lambda item: item.benchmark.id))


def _parse_metadata(value: dict[str, Any]) -> BenchmarkMetadataConfig:
    _reject_unknown(value, _BENCHMARK_KEYS, "benchmark")
    _require_keys(
        value,
        {"id", "version", "task", "description"},
        "benchmark",
    )
    task = _text(value["task"], "benchmark.task")
    supported_tasks = {
        "visual_disease_ranking",
        "visual_disease_contrast_ranking",
        "evidence_grounded_visual_diagnosis",
    }
    if task not in supported_tasks:
        raise BenchmarkConfigError(
            f"benchmark.task is unsupported: {task!r}"
        )
    return BenchmarkMetadataConfig(
        id=_text(value["id"], "benchmark.id"),
        version=_text(value["version"], "benchmark.version"),
        task=task,
        description=_text(value["description"], "benchmark.description"),
        status=_optional_text(value.get("status"), "benchmark.status"),
        top_k=_optional_positive_integer(
            value.get("top_k"), "benchmark.top_k"
        ),
        candidate_count=_optional_positive_integer(
            value.get("candidate_count"),
            "benchmark.candidate_count",
        ),
        ranking_count=_optional_positive_integer(
            value.get("ranking_count"),
            "benchmark.ranking_count",
        ),
    )


def _parse_taxonomy(
    value: dict[str, Any],
    *,
    root: Path,
) -> TaxonomyConfig:
    _reject_unknown(value, _TAXONOMY_KEYS, "taxonomy")
    if "path" in value:
        disease_path_value = value["path"]
    else:
        disease = _mapping(value.get("disease"), "taxonomy.disease")
        _reject_unknown(
            disease, _DISEASE_TAXONOMY_KEYS, "taxonomy.disease"
        )
        _require_keys(disease, {"path"}, "taxonomy.disease")
        disease_path_value = disease.get("path")
    disease_path = _referenced_file(
        root,
        _text(disease_path_value, "taxonomy disease path"),
        "taxonomy disease path",
    )
    morphology_concepts: tuple[MorphologyConceptConfig, ...] = ()
    morphology_value = value.get("morphology")
    if morphology_value is not None:
        morphology = _mapping(
            morphology_value, "taxonomy.morphology"
        )
        _reject_unknown(
            morphology,
            _MORPHOLOGY_TAXONOMY_KEYS,
            "taxonomy.morphology",
        )
        concepts_value = morphology.get("concepts")
        if not isinstance(concepts_value, list) or not concepts_value:
            raise BenchmarkConfigError(
                "taxonomy.morphology.concepts must be a non-empty list"
            )
        concepts: list[MorphologyConceptConfig] = []
        for index, item in enumerate(concepts_value):
            concept = _mapping(
                item, f"taxonomy.morphology.concepts[{index}]"
            )
            _reject_unknown(
                concept,
                {"id", "display_name"},
                f"taxonomy.morphology.concepts[{index}]",
            )
            _require_keys(
                concept,
                {"id", "display_name"},
                f"taxonomy.morphology.concepts[{index}]",
            )
            concepts.append(
                MorphologyConceptConfig(
                    id=_text(
                        concept["id"],
                        f"taxonomy.morphology.concepts[{index}].id",
                    ),
                    display_name=_text(
                        concept["display_name"],
                        "taxonomy.morphology.concepts"
                        f"[{index}].display_name",
                    ),
                )
            )
        concept_ids = [item.id for item in concepts]
        if len(concept_ids) != len(set(concept_ids)):
            raise BenchmarkConfigError(
                "taxonomy morphology concept IDs must be unique"
            )
        morphology_concepts = tuple(concepts)
    confusion_sets_path = None
    confusion_sets_value = value.get("confusion_sets")
    if confusion_sets_value is not None:
        confusion_sets = _mapping(
            confusion_sets_value, "taxonomy.confusion_sets"
        )
        _reject_unknown(
            confusion_sets,
            _CONFUSION_TAXONOMY_KEYS,
            "taxonomy.confusion_sets",
        )
        _require_keys(
            confusion_sets,
            {"path"},
            "taxonomy.confusion_sets",
        )
        confusion_sets_path = _referenced_file(
            root,
            _text(
                confusion_sets.get("path"),
                "taxonomy.confusion_sets.path",
            ),
            "taxonomy.confusion_sets.path",
        )
    return TaxonomyConfig(
        disease_path=disease_path,
        morphology_concepts=morphology_concepts,
        confusion_sets_path=confusion_sets_path,
    )


def _parse_dataset(
    value: dict[str, Any],
    *,
    root: Path,
    task: BenchmarkTask,
) -> BenchmarkDatasetConfig:
    _reject_unknown(value, _DATASET_KEYS, "dataset")
    _require_keys(
        value,
        {
            "default_evaluation_set",
            "evaluation_sets",
            "image_column",
            "sample_id_column",
        },
        "dataset",
    )
    sets_value = _mapping(
        value["evaluation_sets"], "dataset.evaluation_sets"
    )
    if not sets_value:
        raise BenchmarkConfigError(
            "dataset.evaluation_sets must not be empty"
        )
    evaluation_sets: list[EvaluationSetConfig] = []
    for set_id_value, set_value in sets_value.items():
        set_id = _text(set_id_value, "dataset.evaluation_sets key")
        section = f"dataset.evaluation_sets.{set_id}"
        item = _mapping(set_value, section)
        _reject_unknown(item, _EVALUATION_SET_KEYS, section)
        _require_keys(item, _EVALUATION_SET_KEYS, section)
        evaluation_sets.append(
            EvaluationSetConfig(
                id=set_id,
                manifest=_referenced_file(
                    root,
                    _text(item["manifest"], f"{section}.manifest"),
                    f"{section}.manifest",
                ),
                role=_text(item["role"], f"{section}.role"),
                description=_text(
                    item["description"], f"{section}.description"
                ),
            )
        )
    default = _text(
        value["default_evaluation_set"],
        "dataset.default_evaluation_set",
    )
    set_ids = [item.id for item in evaluation_sets]
    if default not in set_ids:
        raise BenchmarkConfigError(
            f"dataset.default_evaluation_set {default!r} is not defined"
        )
    label_key = (
        "disease_label_column"
        if task == "evidence_grounded_visual_diagnosis"
        else "label_column"
    )
    if label_key not in value:
        raise BenchmarkConfigError(
            f"dataset.{label_key} is required for task {task}"
        )
    if task == "visual_disease_contrast_ranking":
        for key in {
            "task_id_column",
            "pair_id_column",
            "candidate_ids_column",
            "condition_column",
            "confusion_set_column",
        }:
            if key not in value:
                raise BenchmarkConfigError(
                    f"dataset.{key} is required for task {task}"
                )
    return BenchmarkDatasetConfig(
        default_evaluation_set=default,
        evaluation_sets=tuple(evaluation_sets),
        image_column=_text(
            value["image_column"], "dataset.image_column"
        ),
        sample_id_column=_text(
            value["sample_id_column"], "dataset.sample_id_column"
        ),
        label_column=_text(value[label_key], f"dataset.{label_key}"),
        group_column=_optional_text(
            value.get("group_column"), "dataset.group_column"
        ),
        task_id_column=_optional_text(
            value.get("task_id_column"), "dataset.task_id_column"
        ),
        pair_id_column=_optional_text(
            value.get("pair_id_column"), "dataset.pair_id_column"
        ),
        candidate_ids_column=_optional_text(
            value.get("candidate_ids_column"),
            "dataset.candidate_ids_column",
        ),
        condition_column=_optional_text(
            value.get("condition_column"),
            "dataset.condition_column",
        ),
        confusion_set_column=_optional_text(
            value.get("confusion_set_column"),
            "dataset.confusion_set_column",
        ),
    )


def _parse_execution(value: dict[str, Any]) -> BenchmarkExecutionConfig:
    _reject_unknown(value, _EXECUTION_KEYS, "execution")
    _require_keys(
        value,
        {
            "max_output_tokens",
            "batch_size",
            "resume",
            "save_raw_responses",
            "save_rendered_prompts",
            "fail_fast_on_invalid_output",
        },
        "execution",
    )
    return BenchmarkExecutionConfig(
        max_output_tokens=_positive_integer(
            value["max_output_tokens"], "execution.max_output_tokens"
        ),
        batch_size=_positive_integer(
            value["batch_size"], "execution.batch_size"
        ),
        resume=_boolean(value["resume"], "execution.resume"),
        save_raw_responses=_boolean(
            value["save_raw_responses"],
            "execution.save_raw_responses",
        ),
        save_rendered_prompts=_boolean(
            value["save_rendered_prompts"],
            "execution.save_rendered_prompts",
        ),
        fail_fast_on_invalid_output=_boolean(
            value["fail_fast_on_invalid_output"],
            "execution.fail_fast_on_invalid_output",
        ),
        implementation_status=_optional_text(
            value.get("implementation_status"),
            "execution.implementation_status",
        ),
    )


def _parse_image_preprocessing(
    value: dict[str, Any],
) -> ImagePreprocessingConfig:
    _reject_unknown(
        value,
        _IMAGE_PREPROCESSING_KEYS,
        "image_preprocessing",
    )
    _require_keys(
        value,
        _IMAGE_PREPROCESSING_KEYS,
        "image_preprocessing",
    )
    jpeg_quality = _bounded_integer(
        value["jpeg_quality"],
        "image_preprocessing.jpeg_quality",
        minimum=1,
        maximum=95,
    )
    minimum_jpeg_quality = _bounded_integer(
        value["minimum_jpeg_quality"],
        "image_preprocessing.minimum_jpeg_quality",
        minimum=1,
        maximum=95,
    )
    if minimum_jpeg_quality > jpeg_quality:
        raise BenchmarkConfigError(
            "image_preprocessing.minimum_jpeg_quality must not exceed "
            "image_preprocessing.jpeg_quality"
        )
    max_edge_pixels = _positive_integer(
        value["max_edge_pixels"],
        "image_preprocessing.max_edge_pixels",
    )
    minimum_edge_pixels = _positive_integer(
        value["minimum_edge_pixels"],
        "image_preprocessing.minimum_edge_pixels",
    )
    if minimum_edge_pixels > max_edge_pixels:
        raise BenchmarkConfigError(
            "image_preprocessing.minimum_edge_pixels must not exceed "
            "image_preprocessing.max_edge_pixels"
        )
    return ImagePreprocessingConfig(
        profile=_text(
            value["profile"],
            "image_preprocessing.profile",
        ),
        max_edge_pixels=max_edge_pixels,
        max_encoded_bytes=_positive_integer(
            value["max_encoded_bytes"],
            "image_preprocessing.max_encoded_bytes",
        ),
        jpeg_quality=jpeg_quality,
        minimum_jpeg_quality=minimum_jpeg_quality,
        minimum_edge_pixels=minimum_edge_pixels,
    )


def _parse_structured_output(
    value: dict[str, Any],
) -> StructuredOutputConfig:
    _reject_unknown(
        value, _STRUCTURED_OUTPUT_KEYS, "structured_output"
    )
    _require_keys(value, _STRUCTURED_OUTPUT_KEYS, "structured_output")
    mode = _text(value["mode"], "structured_output.mode")
    if mode != "prompt_only":
        raise BenchmarkConfigError(
            "structured_output.mode must equal 'prompt_only'"
        )
    return StructuredOutputConfig(mode=mode)


def _single_path_section(
    value: Any,
    section: str,
    allowed: set[str],
) -> str:
    mapping = _mapping(value, section)
    _reject_unknown(mapping, allowed, section)
    _require_keys(mapping, {"path"}, section)
    return _text(mapping["path"], f"{section}.path")


def _resolve_config_path(
    id_or_path: str | Path,
    *,
    directory: Path,
    root: Path,
) -> Path:
    value = Path(id_or_path)
    candidates = [value] if value.is_absolute() else [root / value, value]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if value.suffix in {".yaml", ".yml"} or value.parent != Path("."):
        raise BenchmarkConfigError(
            f"Benchmark config not found: {value}"
        )
    matches: list[Path] = []
    for path in sorted((directory / "derma_isep").glob("*.yaml")):
        document = _load_yaml(path)
        section = document.get("benchmark")
        if isinstance(section, dict) and section.get("id") == str(
            id_or_path
        ):
            matches.append(path)
    if not matches:
        raise BenchmarkConfigError(
            f"Unknown benchmark ID {str(id_or_path)!r}"
        )
    if len(matches) > 1:
        raise BenchmarkConfigError(
            f"Duplicate benchmark ID {str(id_or_path)!r}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0].resolve()


def _referenced_file(root: Path, value: str, path: str) -> Path:
    result = _resolve_relative(root, value)
    if not result.is_file():
        raise BenchmarkConfigError(
            f"{path} references a missing file: {value}"
        )
    return result


def _resolve_relative(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise BenchmarkConfigError(
            f"Could not read benchmark config {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise BenchmarkConfigError(
            f"Benchmark config {path} must contain a YAML mapping"
        )
    return value


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkConfigError(f"{path} must be a mapping")
    return value


def _reject_unknown(
    value: dict[str, Any],
    allowed: set[str],
    path: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise BenchmarkConfigError(
            f"{path} contains unknown field(s): {', '.join(unknown)}"
        )


def _require_keys(
    value: dict[str, Any],
    required: set[str],
    path: str,
) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise BenchmarkConfigError(
            f"{path} is missing required field(s): {', '.join(missing)}"
        )


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkConfigError(f"{path} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _text(value, path)


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise BenchmarkConfigError(f"{path} must be a boolean")
    return value


def _positive_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BenchmarkConfigError(f"{path} must be a positive integer")
    return value


def _bounded_integer(
    value: Any,
    path: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise BenchmarkConfigError(
            f"{path} must be an integer between {minimum} and {maximum}"
        )
    return value


def _optional_positive_integer(value: Any, path: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, path)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]
