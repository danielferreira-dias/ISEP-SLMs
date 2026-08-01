"""Runtime access to the frozen ISEPDermaBench Hugging Face release.

The release stores model inputs and scoring references in separate Parquet
configurations.  This module joins those views only inside the evaluator and
uses the already-rendered prompts, response schema, and benchmark image bytes
from each task row.  Runtime execution therefore does not depend on the
source benchmark YAML directory or on editable prompt/schema files.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.benchmark.datasets import LoadedBenchmarkDataset
from src.benchmark.results import canonical_hash, file_sha256
from src.benchmark.runner import BenchmarkSample, ModelResponse
from src.benchmark.selection import select_units
from src.benchmark.task_adapters import PreparedTask
from src.config.benchmarks import (
    BenchmarkConfig,
    BenchmarkDatasetConfig,
    BenchmarkExecutionConfig,
    BenchmarkMetadataConfig,
    EvaluationSetConfig,
    ImagePreprocessingConfig,
    MorphologyConceptConfig,
    StructuredOutputConfig,
    TaxonomyConfig,
)


DEFAULT_REPO_ID = "danielfdias98/ISEPDermaBench"
DEFAULT_LOCAL_RELEASE = Path("data/benchmarks/ISEPDermaBench")


@dataclass(frozen=True, slots=True)
class ISEPDermaBenchSpec:
    """Stable mapping between one protocol and its Hub configurations."""

    key: str
    benchmark_id: str
    aliases: tuple[str, ...]
    default_split: str
    splits: tuple[str, ...]
    split_aliases: tuple[tuple[str, str], ...] = ()

    def normalize_split(self, value: str | None) -> str:
        selected = value or self.default_split
        aliases = dict(self.split_aliases)
        selected = aliases.get(selected, selected)
        if selected not in self.splits:
            choices = ", ".join(self.splits)
            raise ValueError(
                f"Unknown ISEPDermaBench split {selected!r} for "
                f"{self.benchmark_id!r}; expected one of {choices}"
            )
        return selected


SPECS = (
    ISEPDermaBenchSpec(
        key="visual_top_k",
        benchmark_id="visual_top_k_closed_set",
        aliases=("visual_top_k", "visual_top_k.yaml"),
        default_split="internal_benchmark",
        splits=(
            "validation",
            "internal_benchmark",
            "external_ddi",
            "external_skindisnet",
        ),
        split_aliases=(("internal_benchmark_1000", "internal_benchmark"),),
    ),
    ISEPDermaBenchSpec(
        key="visual_confusion_sets",
        benchmark_id="visual_disease_confusion_sets",
        aliases=("visual_confusion_sets", "visual_confusion_sets.yaml"),
        default_split="internal_benchmark",
        splits=("validation", "internal_benchmark"),
        split_aliases=(
            ("paired_confusion_tasks", "internal_benchmark"),
            ("validation_paired_confusion_tasks", "validation"),
        ),
    ),
    ISEPDermaBenchSpec(
        key="evidence_grounded_diagnosis",
        benchmark_id="evidence_grounded_diagnosis",
        aliases=("evidence_grounded_diagnosis.yaml",),
        default_split="internal_benchmark",
        splits=("validation", "internal_benchmark", "external_ddi"),
        split_aliases=(
            ("internal_benchmark_evidence", "internal_benchmark"),
            ("validation_fitzpatrick_evidence", "validation"),
            ("external_ddi_evidence", "external_ddi"),
        ),
    ),
)


def load_isep_dermabench_config(
    id_or_alias: str | Path,
    *,
    root: Path,
) -> BenchmarkConfig:
    """Build the typed runtime config from frozen release artifacts."""

    spec = _resolve_spec(id_or_alias)
    release_root = (root / DEFAULT_LOCAL_RELEASE).resolve()
    config_path = release_root / "artifacts/configs" / f"{spec.key}.yaml"
    raw = _load_yaml(config_path)
    benchmark_raw = _mapping(raw, "benchmark")
    execution_raw = _mapping(raw, "execution")
    preprocessing_raw = _mapping(raw, "image_preprocessing")

    task = str(benchmark_raw["task"])
    metadata = BenchmarkMetadataConfig(
        id=str(benchmark_raw["id"]),
        version=str(benchmark_raw["version"]),
        task=task,  # type: ignore[arg-type]
        description=str(benchmark_raw.get("description", "")),
        status=_optional_text(benchmark_raw.get("status")),
        top_k=_optional_int(benchmark_raw.get("top_k")),
        candidate_count=_optional_int(
            benchmark_raw.get("candidate_count")
        ),
        ranking_count=_optional_int(benchmark_raw.get("ranking_count")),
    )
    evaluation_sets = tuple(
        EvaluationSetConfig(
            id=split,
            manifest=release_root / "tasks" / spec.key,
            role=(
                "development"
                if split == "validation"
                else "sealed_evaluation"
            ),
            description=f"ISEPDermaBench {spec.key}/{split}",
        )
        for split in spec.splits
    )
    dataset = BenchmarkDatasetConfig(
        default_evaluation_set=spec.default_split,
        evaluation_sets=evaluation_sets,
        image_column="image",
        sample_id_column="sample_id",
        label_column="reference_disease_id",
        group_column="leakage_group_id",
        task_id_column="task_id",
        pair_id_column="pair_id",
        candidate_ids_column="candidate_disease_ids",
        condition_column="condition",
        confusion_set_column="confusion_set_id",
    )
    morphology = _mapping(raw, "taxonomy").get("morphology", {})
    concepts = (
        morphology.get("concepts", [])
        if isinstance(morphology, dict)
        else []
    )
    morphology_concepts = tuple(
        MorphologyConceptConfig(
            id=str(item["id"]),
            display_name=str(item["display_name"]),
        )
        for item in concepts
        if isinstance(item, dict)
        and item.get("id")
        and item.get("display_name")
    )
    prompt_path = (
        release_root / "artifacts/prompts" / _prompt_filename(spec.key)
    )
    schema_path = (
        release_root / "artifacts/schemas" / f"{spec.key}.schema.json"
    )
    confusion_path = (
        release_root
        / "artifacts/taxonomies/disease_confusion_sets.yaml"
        if spec.key == "visual_confusion_sets"
        else None
    )
    output_raw = _mapping(raw, "output")
    return BenchmarkConfig(
        benchmark=metadata,
        prompt_path=prompt_path,
        schema_path=schema_path,
        taxonomy=TaxonomyConfig(
            disease_path=(
                release_root / "artifacts/taxonomies/diseases.yaml"
            ),
            morphology_concepts=morphology_concepts,
            confusion_sets_path=confusion_path,
        ),
        dataset=dataset,
        image_preprocessing=ImagePreprocessingConfig(
            profile=str(preprocessing_raw["profile"]),
            max_edge_pixels=int(preprocessing_raw["max_edge_pixels"]),
            max_encoded_bytes=int(
                preprocessing_raw["max_encoded_bytes"]
            ),
            jpeg_quality=int(preprocessing_raw["jpeg_quality"]),
            minimum_jpeg_quality=int(
                preprocessing_raw["minimum_jpeg_quality"]
            ),
            minimum_edge_pixels=int(
                preprocessing_raw["minimum_edge_pixels"]
            ),
        ),
        execution=BenchmarkExecutionConfig(
            max_output_tokens=int(execution_raw["max_output_tokens"]),
            batch_size=int(execution_raw["batch_size"]),
            resume=bool(execution_raw.get("resume", True)),
            save_raw_responses=bool(
                execution_raw.get("save_raw_responses", True)
            ),
            save_rendered_prompts=bool(
                execution_raw.get("save_rendered_prompts", True)
            ),
            fail_fast_on_invalid_output=bool(
                execution_raw.get("fail_fast_on_invalid_output", False)
            ),
            implementation_status=_optional_text(
                execution_raw.get("implementation_status")
            ),
        ),
        structured_output=StructuredOutputConfig(mode="prompt_only"),
        output_directory=(root / str(output_raw["directory"])).resolve(),
        config_path=config_path,
    )


def list_isep_dermabench_configs(*, root: Path) -> tuple[BenchmarkConfig, ...]:
    """Return the three frozen benchmark protocol configs."""

    return tuple(
        load_isep_dermabench_config(spec.benchmark_id, root=root)
        for spec in SPECS
    )


def load_isep_dermabench_dataset(
    *,
    root: Path,
    benchmark: BenchmarkConfig,
    evaluation_set: str | None,
    limit: int | None,
    seed: int,
    source: str = "auto",
    repo_id: str = DEFAULT_REPO_ID,
) -> LoadedBenchmarkDataset:
    """Load tasks and isolated references from the local mirror or Hub."""

    spec = _resolve_spec(benchmark.benchmark.id)
    split = spec.normalize_split(evaluation_set)
    release_root = (root / DEFAULT_LOCAL_RELEASE).resolve()
    local_available = bool(
        list((release_root / "tasks" / spec.key).glob(f"{split}-*.parquet"))
    )
    if source not in {"auto", "local", "hub"}:
        raise ValueError("benchmark source must be auto, local, or hub")
    resolved_source = (
        "local" if source == "auto" and local_available else source
    )
    if resolved_source == "auto":
        resolved_source = "hub"
    if resolved_source == "local":
        if not local_available:
            raise FileNotFoundError(
                f"Local ISEPDermaBench split is unavailable: "
                f"{spec.key}/{split}"
            )
        task_frame, reference_frame, shard_paths = _load_local_frames(
            release_root=release_root,
            spec=spec,
            split=split,
        )
        manifest_path = release_root / "tasks" / spec.key
        manifest_sha256 = canonical_hash(
            [file_sha256(path) for path in shard_paths]
        )
        release_path = release_root / "release.json"
        release_sha256 = file_sha256(release_path)
    else:
        task_frame, reference_frame = _load_hub_frames(
            repo_id=repo_id,
            spec=spec,
            split=split,
        )
        manifest_path = Path("huggingface") / repo_id / spec.key / split
        manifest_sha256 = _frame_identity(task_frame)
        release_sha256 = canonical_hash(
            {"repo_id": repo_id, "release": "ISEPDermaBench/1.0.0"}
        )

    _validate_release_frames(task_frame, reference_frame)
    unit_column = (
        "pair_id"
        if benchmark.benchmark.task
        == "visual_disease_contrast_ranking"
        else "task_id"
    )
    selected, selection = select_units(
        task_frame,
        unit_column=unit_column,
        task_column="task_id",
        limit=limit,
        seed=seed,
        benchmark_release_hash=release_sha256,
    )
    selected_ids = set(selected["task_id"].astype(str))
    selected_references = reference_frame[
        reference_frame["task_id"].astype(str).isin(selected_ids)
    ]
    reference_by_id = {
        str(row["task_id"]): row
        for row in selected_references.to_dict(orient="records")
    }
    if set(reference_by_id) != selected_ids:
        missing = sorted(selected_ids - set(reference_by_id))
        raise ValueError(
            "ISEPDermaBench references are missing selected task IDs: "
            + ", ".join(missing[:10])
        )
    samples = tuple(
        _release_row_to_sample(
            row,
            reference_by_id[str(row["task_id"])],
            spec=spec,
            split=split,
            repo_id=repo_id,
        )
        for row in selected.to_dict(orient="records")
    )
    return LoadedBenchmarkDataset(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        release_sha256=release_sha256,
        evaluation_set=split,
        frame=selected,
        samples=samples,
        selection=selection
        | {
            "benchmark_source": resolved_source,
            "repo_id": repo_id,
            "task_configuration": spec.key,
            "reference_configuration": f"{spec.key}_references",
        },
    )


class FrozenISEPDermaBenchAdapter:
    """Use frozen row inputs while delegating parsing and metrics."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate

    def prepare(self, sample: BenchmarkSample) -> PreparedTask:
        if sample.system_prompt is None or sample.user_prompt is None:
            raise ValueError("ISEPDermaBench sample has no rendered prompts")
        if sample.response_schema is None:
            raise ValueError("ISEPDermaBench sample has no response schema")
        candidate_ids = tuple(sample.candidate_disease_ids or ())
        if not candidate_ids:
            raise ValueError("ISEPDermaBench sample has no candidate diseases")
        benchmark_id = str(
            sample.metadata.get("benchmark_id", "")
        )
        return PreparedTask(
            benchmark_id=benchmark_id,
            task_id=sample.task_id or sample.sample_id,
            sample_id=sample.sample_id,
            system_prompt=sample.system_prompt,
            user_prompt=sample.user_prompt,
            schema=dict(sample.response_schema),
            allowed_disease_ids=candidate_ids,
        )

    def parse_response(
        self,
        model_id: str,
        raw_text: str,
        prepared_task: PreparedTask,
        reasoning_text: str | None = None,
    ) -> ModelResponse:
        return self.delegate.parse_response(
            model_id,
            raw_text,
            prepared_task=prepared_task,
            reasoning_text=reasoning_text,
        )

    def compute_metrics(self, predictions: Iterable[Any]) -> dict[str, Any]:
        return self.delegate.compute_metrics(predictions)


def _load_local_frames(
    *,
    release_root: Path,
    spec: ISEPDermaBenchSpec,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[Path, ...]]:
    task_paths = tuple(
        sorted((release_root / "tasks" / spec.key).glob(f"{split}-*.parquet"))
    )
    reference_paths = tuple(
        sorted(
            (release_root / "references" / spec.key).glob(
                f"{split}-*.parquet"
            )
        )
    )
    if not task_paths or not reference_paths:
        raise FileNotFoundError(
            f"Incomplete local ISEPDermaBench split: {spec.key}/{split}"
        )
    tasks = pa.concat_tables([pq.read_table(path) for path in task_paths])
    references = pa.concat_tables(
        [pq.read_table(path) for path in reference_paths]
    )
    return (
        tasks.to_pandas(),
        references.to_pandas(),
        task_paths + reference_paths,
    )


def _load_hub_frames(
    *,
    repo_id: str,
    spec: ISEPDermaBenchSpec,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from datasets import Image as HFImage
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required for --benchmark-source hub"
        ) from exc
    tasks = load_dataset(repo_id, spec.key, split=split)
    tasks = tasks.cast_column("image", HFImage(decode=False))
    references = load_dataset(
        repo_id,
        f"{spec.key}_references",
        split=split,
    )
    return tasks.to_pandas(), references.to_pandas()


def _release_row_to_sample(
    task: dict[str, Any],
    reference: dict[str, Any],
    *,
    spec: ISEPDermaBenchSpec,
    split: str,
    repo_id: str,
) -> BenchmarkSample:
    task_id = str(task["task_id"])
    image_bytes = _extract_image_bytes(task["image"])
    response_schema = json.loads(str(task["response_schema_json"]))
    if not isinstance(response_schema, dict):
        raise ValueError(f"Task {task_id} response schema is not an object")
    metadata = {
        str(key): _python_value(value)
        for key, value in (task | reference).items()
        if key
        not in {
            "image",
            "system_prompt",
            "user_prompt",
            "response_schema_json",
        }
    }
    candidates = tuple(
        str(value) for value in _sequence(task["candidate_disease_ids"])
    )
    disease_value = reference.get("reference_disease_id")
    disease_id = "" if _missing(disease_value) else str(disease_value)
    return BenchmarkSample(
        sample_id=str(task["sample_id"]),
        task_id=task_id,
        image_uri=(
            f"hf://datasets/{repo_id}/{spec.key}/{split}/{task_id}"
        ),
        image_bytes=image_bytes,
        disease_id=disease_id,
        candidate_disease_ids=candidates,
        system_prompt=str(task["system_prompt"]),
        user_prompt=str(task["user_prompt"]),
        response_schema=response_schema,
        metadata=metadata,
    )


def _validate_release_frames(
    tasks: pd.DataFrame,
    references: pd.DataFrame,
) -> None:
    task_required = {
        "image",
        "task_id",
        "sample_id",
        "benchmark_id",
        "system_prompt",
        "user_prompt",
        "response_schema_json",
        "candidate_disease_ids",
    }
    reference_required = {"task_id", "reference_disease_id"}
    missing_tasks = sorted(task_required - set(tasks.columns))
    missing_references = sorted(reference_required - set(references.columns))
    if missing_tasks:
        raise ValueError(
            "ISEPDermaBench tasks are missing columns: "
            + ", ".join(missing_tasks)
        )
    if missing_references:
        raise ValueError(
            "ISEPDermaBench references are missing columns: "
            + ", ".join(missing_references)
        )
    if tasks["task_id"].astype(str).duplicated().any():
        raise ValueError("ISEPDermaBench task IDs must be unique per split")
    if references["task_id"].astype(str).duplicated().any():
        raise ValueError(
            "ISEPDermaBench reference task IDs must be unique per split"
        )


def _frame_identity(frame: pd.DataFrame) -> str:
    columns = [
        name
        for name in (
            "task_id",
            "benchmark_image_sha256",
            "prompt_sha256",
            "response_schema_sha256",
        )
        if name in frame.columns
    ]
    return canonical_hash(frame[columns].to_dict(orient="records"))


def _extract_image_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, memoryview):
            raw = raw.tobytes()
        if isinstance(raw, bytearray):
            raw = bytes(raw)
        if isinstance(raw, bytes) and raw:
            return raw
    raise ValueError("ISEPDermaBench image has no embedded bytes")


def _resolve_spec(value: str | Path) -> ISEPDermaBenchSpec:
    text = Path(value).name if isinstance(value, Path) else str(value)
    for spec in SPECS:
        if text in {spec.key, spec.benchmark_id, *spec.aliases}:
            return spec
    choices = ", ".join(spec.benchmark_id for spec in SPECS)
    raise ValueError(f"Unknown ISEPDermaBench benchmark {text!r}: {choices}")


def _prompt_filename(key: str) -> str:
    return {
        "visual_top_k": "top_k.yaml",
        "visual_confusion_sets": "confusion_sets.yaml",
        "evidence_grounded_diagnosis": "evidence_grounded_diagnosis.yaml",
    }[key]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ISEPDermaBench artifact is missing: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"ISEPDermaBench artifact requires {key!r}")
    return result


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    raise ValueError("Expected a sequence in ISEPDermaBench task")


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, bool) else False


def _python_value(value: Any) -> Any:
    if _missing(value):
        return None
    if isinstance(value, memoryview):
        return value.tobytes()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple, set)):
        return [_python_value(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _python_value(value.tolist())
        except (TypeError, ValueError):
            pass
    return value
