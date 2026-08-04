"""Build the fixed 50-case no-image visual-grounding ablation.

The cohort is a group-unique, class-balanced subset of the already frozen
100-case visual Top-K Validation screening cohort.  Every source image is
replaced by a uniform mid-gray JPEG with the same width and height.  Source
labels remain isolated in the reference configuration and are not valid
targets for the control image.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.benchmark.runner import BenchmarkSample
from src.benchmark.task_adapters import build_task_adapter
from src.data_pipeline.huggingface_benchmark_export import (
    REFERENCE_FEATURES,
    TASK_FEATURES,
    _file_sha256,
    _write_shards,
)


BENCHMARK_KEY = "visual_grounding_no_image"
BENCHMARK_ID = "visual_grounding_no_image"
SPLIT = "validation"
COHORT_SIZE = 50
MINIMUM_PER_CLASS = 2
RELEASE_VERSION = "1.6.0"
SOURCE_IDS = Path(
    "metadata/validation_screening_v1/"
    "visual_top_k_100_cases.task_ids.txt"
)


def build_visual_grounding_no_image(
    root: Path,
    *,
    release_path: Path = Path("data/benchmarks/ISEPDermaBench"),
) -> dict[str, Any]:
    """Materialize the task, references, metadata, and release manifest."""

    root = root.resolve()
    release_root = root / release_path
    tasks = _read_split(release_root / "tasks/visual_top_k", SPLIT)
    references = _read_split(
        release_root / "references/visual_top_k",
        SPLIT,
    )
    source_ids = _read_task_ids(release_root / SOURCE_IDS)
    merged = tasks.merge(
        references,
        on="task_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_reference"),
    )
    merged = merged.set_index("task_id").loc[source_ids].reset_index()
    selected = _select_group_unique_balanced(merged)

    artifact_root = release_root / "artifacts"
    config_path = artifact_root / "configs" / f"{BENCHMARK_KEY}.yaml"
    prompt_path = artifact_root / "prompts" / f"{BENCHMARK_KEY}.yaml"
    schema_path = artifact_root / "schemas" / f"{BENCHMARK_KEY}.schema.json"
    taxonomy_path = artifact_root / "taxonomies/diseases.yaml"
    config = _yaml(config_path)
    prompt = _yaml(prompt_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    taxonomy = _yaml(taxonomy_path)
    disease_items = taxonomy.get("diseases")
    if not isinstance(disease_items, list):
        raise ValueError("Disease taxonomy must contain a diseases list")
    adapter = build_task_adapter(
        benchmark_config=config,
        prompt_config=prompt,
        schema=schema,
        disease_taxonomy_items=disease_items,
    )

    config_sha = _file_sha256(config_path)
    taxonomy_sha = _file_sha256(taxonomy_path)
    task_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    embedded_bytes = 0
    for row in selected.to_dict(orient="records"):
        source_task_id = str(row["task_id"])
        sample_id = str(row["sample_id"])
        task_id = f"{BENCHMARK_KEY}:{SPLIT}:{sample_id}"
        candidate_ids = tuple(str(value) for value in row["candidate_disease_ids"])
        sample = BenchmarkSample(
            sample_id=sample_id,
            task_id=task_id,
            image_uri=f"embedded://{task_id}",
            disease_id=str(row["reference_disease_id"]),
            candidate_disease_ids=candidate_ids,
            metadata={},
        )
        prepared = adapter.prepare(sample)
        gray_image = _uniform_gray_image(row["image"])
        embedded_bytes += len(gray_image)
        rendered_schema = json.dumps(
            prepared.schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        prompt_payload = prepared.system_prompt + "\0" + prepared.user_prompt
        task_rows.append(
            {
                "image": {
                    "bytes": gray_image,
                    "path": f"{_safe_id(sample_id)}_gray.jpg",
                },
                "task_id": task_id,
                "sample_id": sample_id,
                "benchmark_id": BENCHMARK_ID,
                "benchmark_version": "1.0.0",
                "evaluation_set": SPLIT,
                "source": str(row["source"]),
                "leakage_group_id": str(row["leakage_group_id"]),
                "system_prompt": prepared.system_prompt,
                "user_prompt": prepared.user_prompt,
                "response_schema_json": rendered_schema,
                "prompt_id": str(prompt["id"]),
                "prompt_version": str(prompt["version"]),
                "top_k": 3,
                "candidate_disease_ids": list(candidate_ids),
                "pair_id": source_task_id,
                "condition": "uniform_gray_no_image",
                "confusion_set_id": None,
                "prompt_sha256": _text_sha256(prompt_payload),
                "response_schema_sha256": _text_sha256(rendered_schema),
                "benchmark_config_sha256": config_sha,
                "taxonomy_sha256": taxonomy_sha,
                "source_image_sha256": str(row["benchmark_image_sha256"]),
                "benchmark_image_sha256": sha256(gray_image).hexdigest(),
                "image_preprocessing_profile": (
                    "no_image_gray_same_dimensions_v1"
                ),
                "license_id": str(row["license_id"]),
            }
        )
        reference_rows.append(
            {
                "task_id": task_id,
                "sample_id": sample_id,
                "benchmark_id": BENCHMARK_ID,
                "evaluation_set": SPLIT,
                "source": str(row["source"]),
                "leakage_group_id": str(row["leakage_group_id"]),
                "reference_disease_id": str(row["reference_disease_id"]),
                "reference_diagnoses_json": _optional_text(
                    row.get("reference_diagnoses_json")
                ),
                "diagnosis_basis": _optional_text(row.get("diagnosis_basis")),
                "morphology_concept_ids": [],
                "reference_clinical_description": None,
                "score_morphology": False,
                "score_description": False,
                "score_diagnosis": False,
                "pair_id": source_task_id,
                "condition": "uniform_gray_no_image",
                "confusion_set_id": None,
                "age_years": _optional_int(row.get("age_years")),
                "age_group_standardized": _optional_text(
                    row.get("age_group_standardized")
                ),
                "skin_tone_system": _optional_text(
                    row.get("skin_tone_system")
                ),
                "skin_tone": _optional_text(row.get("skin_tone")),
                "sex_or_gender_system": _optional_text(
                    row.get("sex_or_gender_system")
                ),
                "sex_or_gender": _optional_text(row.get("sex_or_gender")),
                "race_ethnicity": _optional_text(row.get("race_ethnicity")),
                "license_id": str(row["license_id"]),
            }
        )

    task_directory = release_root / "tasks" / BENCHMARK_KEY
    reference_directory = release_root / "references" / BENCHMARK_KEY
    _clear_parquet_shards(task_directory, SPLIT)
    _clear_parquet_shards(reference_directory, SPLIT)
    task_shards = _write_shards(
        records=task_rows,
        features=TASK_FEATURES,
        directory=task_directory,
        split=SPLIT,
        shard_size=COHORT_SIZE,
    )
    reference_shards = _write_shards(
        records=reference_rows,
        features=REFERENCE_FEATURES,
        directory=reference_directory,
        split=SPLIT,
        shard_size=COHORT_SIZE,
    )
    for item in task_shards + reference_shards:
        item["path"] = Path(item["path"]).relative_to(release_root).as_posix()

    metadata_root = release_root / "metadata/visual_grounding_no_image_v1"
    metadata_root.mkdir(parents=True, exist_ok=True)
    task_ids_path = metadata_root / "50_cases.task_ids.txt"
    task_ids_path.write_text(
        "\n".join(
            [
                "# ISEPDermaBench no-image visual-grounding ablation",
                "# split: validation",
                "# selected_cases: 50",
                "# source_cohort: visual_top_k_100_cases",
                "# selection: two_per_class_then_balanced_fill_group_unique_v1",
                *[row["task_id"] for row in task_rows],
                "",
            ]
        ),
        encoding="utf-8",
    )
    class_counts = Counter(
        str(value) for value in selected["reference_disease_id"]
    )
    metadata = {
        "schema_version": 1,
        "id": "visual_grounding_no_image_v1",
        "purpose": "Validation-only paired reasoning and grounding ablation",
        "source_configuration": "visual_top_k",
        "source_split": "validation",
        "source_cohort": SOURCE_IDS.as_posix(),
        "case_count": COHORT_SIZE,
        "group_count": int(selected["leakage_group_id"].nunique()),
        "class_count": int(selected["reference_disease_id"].nunique()),
        "class_distribution": dict(sorted(class_counts.items())),
        "image_control": "uniform_gray_same_dimensions_v1",
        "correct_response": {
            "image_status": "not_evaluable",
            "visual_findings": [],
            "predictions": [],
            "confidence": "low",
        },
        "task_ids_file": task_ids_path.name,
        "task_ids_sha256": _file_sha256(task_ids_path),
    }
    (metadata_root / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    split_record = {
        "benchmark": BENCHMARK_KEY,
        "benchmark_id": BENCHMARK_ID,
        "split": SPLIT,
        "source_evaluation_set": "visual_top_k/validation_screening_100",
        "task_count": COHORT_SIZE,
        "sample_count": COHORT_SIZE,
        "group_count": int(selected["leakage_group_id"].nunique()),
        "embedded_image_bytes": embedded_bytes,
        "task_shards": task_shards,
        "reference_shards": reference_shards,
    }
    _update_release(
        release_root,
        split_record=split_record,
        artifact_paths=(config_path, prompt_path, schema_path),
        metadata=metadata,
    )
    return metadata | {
        "embedded_image_bytes": embedded_bytes,
        "task_shards": task_shards,
        "reference_shards": reference_shards,
    }


def validate_visual_grounding_no_image(
    root: Path,
    *,
    release_path: Path = Path("data/benchmarks/ISEPDermaBench"),
) -> dict[str, Any]:
    """Validate cohort size, gray controls, joins, and reference isolation."""

    release_root = root.resolve() / release_path
    tasks = _read_split(release_root / "tasks" / BENCHMARK_KEY, SPLIT)
    references = _read_split(
        release_root / "references" / BENCHMARK_KEY,
        SPLIT,
    )
    if len(tasks) != COHORT_SIZE or len(references) != COHORT_SIZE:
        raise ValueError("No-image ablation must contain exactly 50 rows")
    if tasks["task_id"].tolist() != references["task_id"].tolist():
        raise ValueError("No-image task/reference IDs differ")
    if tasks["leakage_group_id"].nunique() != COHORT_SIZE:
        raise ValueError("No-image cohort must contain 50 unique groups")
    if references["reference_disease_id"].nunique() != 21:
        raise ValueError("No-image cohort must cover all 21 diseases")
    if "reference_disease_id" in tasks.columns:
        raise ValueError("Task inputs expose the hidden source diagnosis")
    for row in tasks.to_dict(orient="records"):
        encoded = row["image"]["bytes"]
        if sha256(encoded).hexdigest() != row["benchmark_image_sha256"]:
            raise ValueError(f"Gray image checksum mismatch: {row['task_id']}")
        with Image.open(BytesIO(encoded)) as image:
            rgb = image.convert("RGB")
            extrema = rgb.getextrema()
            if any(low != high for low, high in extrema):
                raise ValueError(f"Control image is not uniform: {row['task_id']}")
        if row["condition"] != "uniform_gray_no_image":
            raise ValueError("Unexpected ablation condition")
    return {
        "case_count": len(tasks),
        "group_count": int(tasks["leakage_group_id"].nunique()),
        "class_count": int(references["reference_disease_id"].nunique()),
    }


def _select_group_unique_balanced(frame: Any) -> Any:
    selected_indices: list[int] = []
    selected_ids: set[str] = set()
    groups: set[str] = set()
    counts: Counter[str] = Counter()
    for index, row in frame.iterrows():
        group = str(row["leakage_group_id"])
        disease = str(row["reference_disease_id"])
        if group in groups or counts[disease] >= MINIMUM_PER_CLASS:
            continue
        selected_indices.append(index)
        selected_ids.add(str(row["task_id"]))
        groups.add(group)
        counts[disease] += 1
    if len(counts) != 21 or min(counts.values(), default=0) < MINIMUM_PER_CLASS:
        raise ValueError("Source cohort cannot provide two unique groups per class")
    for index, row in frame.iterrows():
        if len(selected_indices) >= COHORT_SIZE:
            break
        group = str(row["leakage_group_id"])
        task_id = str(row["task_id"])
        if group in groups or task_id in selected_ids:
            continue
        selected_indices.append(index)
        selected_ids.add(task_id)
        groups.add(group)
    selected = frame.loc[selected_indices].copy()
    if len(selected) != COHORT_SIZE:
        raise ValueError("Could not select 50 unique source groups")
    return selected


def _uniform_gray_image(value: Any) -> bytes:
    if not isinstance(value, dict) or not isinstance(value.get("bytes"), bytes):
        raise ValueError("Source task has no embedded image bytes")
    with Image.open(BytesIO(value["bytes"])) as source:
        width, height = source.size
    control = Image.new("RGB", (width, height), color=(127, 127, 127))
    buffer = BytesIO()
    control.save(
        buffer,
        format="JPEG",
        quality=95,
        optimize=True,
        subsampling=0,
    )
    return buffer.getvalue()


def _update_release(
    release_root: Path,
    *,
    split_record: dict[str, Any],
    artifact_paths: tuple[Path, ...],
    metadata: dict[str, Any],
) -> None:
    path = release_root / "release.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    release = document["release"]
    prior_splits = [
        item
        for item in release["splits"]
        if item.get("benchmark") == BENCHMARK_KEY
    ]
    prior_bytes = sum(int(item.get("embedded_image_bytes", 0)) for item in prior_splits)
    release["embedded_image_bytes"] = (
        int(release["embedded_image_bytes"])
        - prior_bytes
        + int(split_record["embedded_image_bytes"])
    )
    release["splits"] = [
        item
        for item in release["splits"]
        if item.get("benchmark") != BENCHMARK_KEY
    ] + [split_record]
    release["version"] = RELEASE_VERSION
    release["artifacts"] = [
        item
        for item in release["artifacts"]
        if BENCHMARK_KEY not in str(item.get("path", ""))
    ]
    for artifact in artifact_paths:
        release["artifacts"].append(
            {
                "path": artifact.relative_to(release_root).as_posix(),
                "source": artifact.relative_to(
                    root_from_release(release_root)
                ).as_posix(),
                "sha256": _file_sha256(artifact),
            }
        )
    release["visual_grounding_no_image_control"] = metadata | {
        "status": "development_validation_ablation",
        "paired_comparisons": [
            "normal_image_vs_uniform_gray",
            "thinking_off_vs_thinking_on",
        ],
        "must_not_be_reported_as_final_clinical_accuracy": True,
    }
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def root_from_release(release_root: Path) -> Path:
    return release_root.parents[2]


def _read_split(directory: Path, split: str) -> Any:
    paths = sorted(directory.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No {split} Parquet shards in {directory}")
    return pa.concat_tables([pq.read_table(path) for path in paths]).to_pandas()


def _read_task_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _clear_parquet_shards(directory: Path, split: str) -> None:
    if not directory.is_dir():
        return
    for path in directory.glob(f"{split}-*.parquet"):
        path.unlink()


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _safe_id(value: str) -> str:
    return "".join(
        character if character.isalnum() else "_" for character in value
    )[:180]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _optional_int(value: Any) -> int | None:
    text = _optional_text(value)
    return None if text is None else int(float(text))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or validate the 50-case no-image grounding ablation."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = (
        validate_visual_grounding_no_image(args.project_root)
        if args.validate_only
        else build_visual_grounding_no_image(args.project_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
