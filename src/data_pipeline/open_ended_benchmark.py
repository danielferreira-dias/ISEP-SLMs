"""Build the open-ended diagnosis extension for ISEPDermaBench.

The extension is derived only from the frozen ISEPDermaBench v1.0.0 task and
reference views. It never reopens the deleted source benchmark directories.
Model inputs remain isolated from gold references and use the already-frozen
benchmark JPEG bytes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from typing import Any

from datasets import Dataset, Features, Image as HFImage, Sequence, Value
import pandas as pd
import pyarrow.parquet as pq
import yaml


RELEASE_VERSION = "1.1.0"
SUPPORTED_RELEASE_VERSIONS = {
    RELEASE_VERSION,
    "1.2.0",
    "1.3.0",
    "1.4.0",
    "1.5.0",
    "1.6.0",
    "1.7.0",
    "1.8.0",
}
SELECTION_SEED = 42
VALIDATION_SIZE = 100
INTERNAL_SIZE = 300
MINIMUM_INTERNAL_PER_CLASS = 10
DEFAULT_SOURCE = Path("data/benchmarks/ISEPDermaBench")
DEFAULT_OUTPUT = Path("data/benchmarks/ISEPDermaBench-v1.1.0")
RESOURCE_ROOT = Path("src/benchmark/resources/open_ended_diagnosis")


TASK_FEATURES = Features(
    {
        "image": HFImage(decode=True),
        "task_id": Value("string"),
        "sample_id": Value("string"),
        "benchmark_id": Value("string"),
        "benchmark_version": Value("string"),
        "evaluation_set": Value("string"),
        "source": Value("string"),
        "leakage_group_id": Value("string"),
        "system_prompt": Value("string"),
        "user_prompt": Value("string"),
        "output_mode": Value("string"),
        "response_schema_json": Value("string"),
        "prompt_id": Value("string"),
        "prompt_version": Value("string"),
        "top_k": Value("int16"),
        "requested_differential_count": Value("int16"),
        "candidate_disease_ids": Sequence(Value("string")),
        "prompt_sha256": Value("string"),
        "response_schema_sha256": Value("string"),
        "benchmark_config_sha256": Value("string"),
        "taxonomy_sha256": Value("string"),
        "source_image_sha256": Value("string"),
        "benchmark_image_sha256": Value("string"),
        "image_preprocessing_profile": Value("string"),
        "license_id": Value("string"),
    }
)


REFERENCE_FEATURES = Features(
    {
        "task_id": Value("string"),
        "sample_id": Value("string"),
        "benchmark_id": Value("string"),
        "evaluation_set": Value("string"),
        "source": Value("string"),
        "leakage_group_id": Value("string"),
        "reference_disease_id": Value("string"),
        "reference_disease_name": Value("string"),
        "reference_diagnoses_json": Value("string"),
        "diagnosis_basis": Value("string"),
        "morphology_concept_ids": Sequence(Value("string")),
        "reference_clinical_description": Value("string"),
        "morphology_reference_source": Value("string"),
        "description_reference_source": Value("string"),
        "score_morphology": Value("bool"),
        "score_description": Value("bool"),
        "score_diagnosis": Value("bool"),
        "age_years": Value("int32"),
        "age_group_standardized": Value("string"),
        "skin_tone_system": Value("string"),
        "skin_tone": Value("string"),
        "sex_or_gender_system": Value("string"),
        "sex_or_gender": Value("string"),
        "race_ethnicity": Value("string"),
        "license_id": Value("string"),
    }
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _read_split(directory: Path, split: str) -> pd.DataFrame:
    paths = sorted(directory.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet shards for {directory}/{split}")
    return pd.concat(
        [pq.read_table(path).to_pandas() for path in paths],
        ignore_index=True,
    )


def _stable_key(task_id: str) -> str:
    return _text_sha256(f"{SELECTION_SEED}:{task_id}")


def _enriched_candidates(source: Path, split: str) -> pd.DataFrame:
    tasks = _read_split(source / "tasks/visual_top_k", split)
    references = _read_split(source / "references/visual_top_k", split)
    evidence = _read_split(
        source / "references/evidence_grounded_diagnosis",
        split,
    )
    reference_columns = [
        column
        for column in references.columns
        if column not in {"benchmark_id", "evaluation_set"}
    ]
    merged = tasks.merge(
        references[reference_columns],
        on=["task_id", "sample_id", "source", "leakage_group_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_reference"),
    )
    evidence_columns = evidence[
        [
            "sample_id",
            "morphology_concept_ids",
            "reference_clinical_description",
            "score_morphology",
            "score_description",
        ]
    ].rename(
        columns={
            "morphology_concept_ids": "open_morphology_concept_ids",
            "reference_clinical_description": "open_reference_description",
            "score_morphology": "open_score_morphology",
            "score_description": "open_score_description",
        }
    )
    merged = merged.merge(
        evidence_columns,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    merged["has_morphology_reference"] = merged[
        "open_score_morphology"
    ].fillna(False).astype(bool)
    merged["has_description_reference"] = merged[
        "open_score_description"
    ].fillna(False).astype(bool)
    merged["has_reference"] = (
        merged["has_morphology_reference"]
        | merged["has_description_reference"]
    )
    merged["stable_key"] = merged["task_id"].astype(str).map(_stable_key)
    return merged


def _one_per_group(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(
        [
            "has_morphology_reference",
            "has_description_reference",
            "stable_key",
        ],
        ascending=[False, False, True],
        kind="stable",
    )
    return ordered.drop_duplicates("leakage_group_id", keep="first").copy()


def _validation_quotas(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["reference_disease_id"].value_counts().sort_index()
    if len(counts) != 21:
        raise ValueError("Validation source must cover all 21 diseases")
    quotas = {str(identifier): 4 for identifier in counts.index}
    remaining = VALIDATION_SIZE - sum(quotas.values())
    weights = counts / counts.sum()
    exact = {str(key): float(value * remaining) for key, value in weights.items()}
    floors = {key: int(math.floor(value)) for key, value in exact.items()}
    for key, value in floors.items():
        quotas[key] += value
    left = remaining - sum(floors.values())
    order = sorted(
        exact,
        key=lambda key: (-(exact[key] - floors[key]), key),
    )
    for key in order[:left]:
        quotas[key] += 1
    return quotas


def _select_validation(frame: pd.DataFrame) -> pd.DataFrame:
    pool = _one_per_group(frame)
    quotas = _validation_quotas(pool)
    selected: list[pd.DataFrame] = []
    for disease_id, quota in sorted(quotas.items()):
        candidates = pool[pool["reference_disease_id"] == disease_id]
        candidates = candidates.sort_values(
            [
                "has_morphology_reference",
                "has_description_reference",
                "stable_key",
            ],
            ascending=[False, False, True],
            kind="stable",
        )
        if len(candidates) < quota:
            raise ValueError(f"Insufficient Validation cases for {disease_id}")
        selected.append(candidates.head(quota))
    result = pd.concat(selected, ignore_index=True)
    return result.sort_values("stable_key", kind="stable").reset_index(drop=True)


def _next_balanced_disease(
    *,
    full_counts: Counter[str],
    selected_counts: Counter[str],
    available: set[str],
    next_size: int,
) -> str:
    total = sum(full_counts.values())
    return min(
        available,
        key=lambda disease_id: (
            -(
                full_counts[disease_id] / total * next_size
                - selected_counts[disease_id]
            ),
            disease_id,
        ),
    )


def _select_internal(frame: pd.DataFrame) -> pd.DataFrame:
    pool = _one_per_group(frame)
    by_id = {
        str(row.task_id): row
        for row in pool.itertuples(index=False)
    }
    selected_ids = set(
        pool.loc[pool["has_morphology_reference"], "task_id"].astype(str)
    )
    if len(selected_ids) != 134:
        raise ValueError(
            f"Expected all 134 internal SKINCON cases, found {len(selected_ids)}"
        )

    ordered_by_class: dict[str, list[str]] = {}
    for disease_id, group in pool.groupby("reference_disease_id"):
        ordered = group.sort_values("stable_key", kind="stable")
        ordered_by_class[str(disease_id)] = ordered["task_id"].astype(str).tolist()

    def selected_counts() -> Counter[str]:
        return Counter(
            str(by_id[task_id].reference_disease_id)
            for task_id in selected_ids
        )

    counts = selected_counts()
    for disease_id in sorted(ordered_by_class):
        needed = max(0, MINIMUM_INTERNAL_PER_CLASS - counts[disease_id])
        for task_id in ordered_by_class[disease_id]:
            if needed == 0:
                break
            if task_id not in selected_ids:
                selected_ids.add(task_id)
                needed -= 1
        if needed:
            raise ValueError(f"Could not reach internal minimum for {disease_id}")

    full_counts = Counter(pool["reference_disease_id"].astype(str))
    while len(selected_ids) < INTERNAL_SIZE:
        counts = selected_counts()
        available = {
            disease_id
            for disease_id, task_ids in ordered_by_class.items()
            if any(task_id not in selected_ids for task_id in task_ids)
        }
        disease_id = _next_balanced_disease(
            full_counts=full_counts,
            selected_counts=counts,
            available=available,
            next_size=len(selected_ids) + 1,
        )
        task_id = next(
            value
            for value in ordered_by_class[disease_id]
            if value not in selected_ids
        )
        selected_ids.add(task_id)

    result = pool[pool["task_id"].astype(str).isin(selected_ids)].copy()
    return result.sort_values("stable_key", kind="stable").reset_index(drop=True)


def select_open_ended_cohorts(source: Path) -> dict[str, pd.DataFrame]:
    validation = _select_validation(
        _enriched_candidates(source, "validation")
    )
    internal = _select_internal(
        _enriched_candidates(source, "internal_benchmark")
    )
    expected = {"validation": VALIDATION_SIZE, "internal_benchmark": INTERNAL_SIZE}
    for split, frame in {"validation": validation, "internal_benchmark": internal}.items():
        if len(frame) != expected[split]:
            raise ValueError(f"Unexpected {split} size: {len(frame)}")
        if frame["leakage_group_id"].nunique() != len(frame):
            raise ValueError(f"{split} must contain one task per group")
        if frame["reference_disease_id"].nunique() != 21:
            raise ValueError(f"{split} must cover all 21 diseases")
    overlap = set(validation["leakage_group_id"].astype(str)) & set(
        internal["leakage_group_id"].astype(str)
    )
    if overlap:
        raise ValueError("Open-ended Validation/Internal groups overlap")
    return {"validation": validation, "internal_benchmark": internal}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _nullable_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)
    return text if text and text.casefold() != "nan" else None


def _nullable_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(value)


def _string_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [str(item) for item in value]


def _task_and_reference_records(
    *,
    frame: pd.DataFrame,
    split: str,
    prompt: dict[str, Any],
    disease_names: dict[str, str],
    benchmark_config_sha256: str,
    taxonomy_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system_prompt = str(prompt["system_prompt"])
    user_prompt = str(prompt["user_template"])
    prompt_hash = _text_sha256(system_prompt + "\0" + user_prompt)
    empty_schema = "{}"
    tasks: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for source in frame.to_dict(orient="records"):
        sample_id = str(source["sample_id"])
        task_id = f"open_ended_diagnosis:{split}:{sample_id}"
        image = source["image"]
        tasks.append(
            {
                "image": image,
                "task_id": task_id,
                "sample_id": sample_id,
                "benchmark_id": "open_ended_diagnosis",
                "benchmark_version": "1.0.0",
                "evaluation_set": split,
                "source": str(source["source"]),
                "leakage_group_id": str(source["leakage_group_id"]),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "output_mode": "free_text",
                "response_schema_json": empty_schema,
                "prompt_id": str(prompt["id"]),
                "prompt_version": str(prompt["version"]),
                "top_k": 3,
                "requested_differential_count": 3,
                "candidate_disease_ids": [],
                "prompt_sha256": prompt_hash,
                "response_schema_sha256": _text_sha256(empty_schema),
                "benchmark_config_sha256": benchmark_config_sha256,
                "taxonomy_sha256": taxonomy_sha256,
                "source_image_sha256": _nullable_text(source.get("source_image_sha256")),
                "benchmark_image_sha256": str(source["benchmark_image_sha256"]),
                "image_preprocessing_profile": str(source["image_preprocessing_profile"]),
                "license_id": str(source["license_id"]),
            }
        )
        disease_id = str(source["reference_disease_id"])
        morphology = _string_list(source.get("open_morphology_concept_ids"))
        description = _nullable_text(source.get("open_reference_description"))
        references.append(
            {
                "task_id": task_id,
                "sample_id": sample_id,
                "benchmark_id": "open_ended_diagnosis",
                "evaluation_set": split,
                "source": str(source["source"]),
                "leakage_group_id": str(source["leakage_group_id"]),
                "reference_disease_id": disease_id,
                "reference_disease_name": disease_names[disease_id],
                "reference_diagnoses_json": _nullable_text(source.get("reference_diagnoses_json")),
                "diagnosis_basis": _nullable_text(source.get("diagnosis_basis")),
                "morphology_concept_ids": morphology,
                "reference_clinical_description": description,
                "morphology_reference_source": "SKINCON" if morphology else None,
                "description_reference_source": "SkinCAP" if description else None,
                "score_morphology": bool(morphology),
                "score_description": description is not None,
                "score_diagnosis": True,
                "age_years": _nullable_int(source.get("age_years")),
                "age_group_standardized": _nullable_text(source.get("age_group_standardized")),
                "skin_tone_system": _nullable_text(source.get("skin_tone_system")),
                "skin_tone": _nullable_text(source.get("skin_tone")),
                "sex_or_gender_system": _nullable_text(source.get("sex_or_gender_system")),
                "sex_or_gender": _nullable_text(source.get("sex_or_gender")),
                "race_ethnicity": _nullable_text(source.get("race_ethnicity")),
                "license_id": str(source["license_id"]),
            }
        )
    return tasks, references


def _write_shard(
    *,
    records: list[dict[str, Any]],
    features: Features,
    path: Path,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records, features=features).to_parquet(path)
    return {
        "path": path.as_posix(),
        "rows": len(records),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _copy_resources(root: Path, output: Path) -> list[dict[str, Any]]:
    resources = {
        "artifacts/configs/open_ended_diagnosis.yaml": "benchmark.yaml",
        "artifacts/prompts/open_ended_diagnosis.yaml": "model_prompt.yaml",
        "artifacts/judges/open_ended_diagnosis_judge.yaml": "judge_prompt.yaml",
        "artifacts/schemas/open_ended_diagnosis_judge.schema.json": "judge.schema.json",
    }
    records: list[dict[str, Any]] = []
    for destination_value, source_name in resources.items():
        source = root / RESOURCE_ROOT / source_name
        destination = output / destination_value
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "path": destination_value,
                "source": (RESOURCE_ROOT / source_name).as_posix(),
                "sha256": _file_sha256(destination),
            }
        )
    return records


def _update_card(text: str, summaries: list[dict[str, Any]]) -> str:
    if "config_name: open_ended_diagnosis" not in text:
        marker = "---\n\n# ISEPDermaBench"
        addition = """- config_name: open_ended_diagnosis
  data_files:
  - split: validation
    path: tasks/open_ended_diagnosis/validation-*.parquet
  - split: internal_benchmark
    path: tasks/open_ended_diagnosis/internal_benchmark-*.parquet
- config_name: open_ended_diagnosis_references
  data_files:
  - split: validation
    path: references/open_ended_diagnosis/validation-*.parquet
  - split: internal_benchmark
    path: references/open_ended_diagnosis/internal_benchmark-*.parquet
"""
        text = text.replace(marker, addition + marker, 1)
    if "| open_ended_diagnosis |" not in text:
        rows = "".join(
            f"| open_ended_diagnosis | {item['split']} | {item['task_count']} | "
            f"{item['sample_count']} | {item['group_count']} |\n"
            for item in summaries
        )
        text = text.replace("\n### `visual_top_k`", rows + "\n### `visual_top_k`", 1)
    if "### `open_ended_diagnosis`" not in text:
        section = """
### `open_ended_diagnosis`

Free-text, image-only clinical assessment with an explicitly ranked Top-3
differential and concise visible-evidence rationale. The evaluated model sees
no disease taxonomy, candidate IDs, gold label, SKINCON concepts, SkinCAP
description, or JSON schema. A separate single-judge stage uses GPT-5.6 Luna
and the isolated reference configuration.

"""
        text = text.replace("\n## Input schema", "\n" + section + "## Input schema", 1)
    text = text.replace("the project's three frozen", "the project's four frozen")
    return text.replace(
        "python -m src.data_pipeline.huggingface_benchmark_export\n"
        "python -m src.data_pipeline.huggingface_benchmark_export --validate-only",
        "python -m src.data_pipeline.open_ended_benchmark \\\n"
        "  --source data/benchmarks/ISEPDermaBench-v1.0.0 \\\n"
        "  --output data/benchmarks/ISEPDermaBench-v1.1.0\n"
        "python -m src.data_pipeline.open_ended_benchmark \\\n"
        "  --output data/benchmarks/ISEPDermaBench-v1.1.0 \\\n"
        "  --validate-only",
    )


def build_open_ended_release(
    root: Path,
    *,
    source_path: Path = DEFAULT_SOURCE,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    root = root.resolve()
    source = root / source_path
    output = root / output_path
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    current = json.loads((source / "release.json").read_text(encoding="utf-8"))["release"]
    if current["version"] != "1.0.0":
        raise ValueError("Open-ended extension expects ISEPDermaBench v1.0.0")

    shutil.copytree(source, output)
    try:
        cohorts = select_open_ended_cohorts(source)
        config_path = root / RESOURCE_ROOT / "benchmark.yaml"
        prompt_path = root / RESOURCE_ROOT / "model_prompt.yaml"
        taxonomy_path = source / "artifacts/taxonomies/diseases.yaml"
        prompt = _load_yaml(prompt_path)
        taxonomy = _load_yaml(taxonomy_path)
        disease_names = {
            str(item["id"]): str(item["display_name"])
            for item in taxonomy["diseases"]
        }
        new_splits: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        added_image_bytes = 0
        for split, frame in cohorts.items():
            tasks, references = _task_and_reference_records(
                frame=frame,
                split=split,
                prompt=prompt,
                disease_names=disease_names,
                benchmark_config_sha256=_file_sha256(config_path),
                taxonomy_sha256=_file_sha256(taxonomy_path),
            )
            task_path = output / "tasks/open_ended_diagnosis" / f"{split}-00000-of-00001.parquet"
            reference_path = output / "references/open_ended_diagnosis" / f"{split}-00000-of-00001.parquet"
            task_shard = _write_shard(records=tasks, features=TASK_FEATURES, path=task_path)
            reference_shard = _write_shard(records=references, features=REFERENCE_FEATURES, path=reference_path)
            for shard in (task_shard, reference_shard):
                shard["path"] = Path(shard["path"]).relative_to(output).as_posix()
            added_image_bytes += sum(len(row["image"]["bytes"]) for row in tasks)
            record = {
                "benchmark": "open_ended_diagnosis",
                "benchmark_id": "open_ended_diagnosis",
                "split": split,
                "source_evaluation_set": split,
                "task_count": len(tasks),
                "sample_count": len({row["sample_id"] for row in tasks}),
                "group_count": len({row["leakage_group_id"] for row in tasks}),
                "reference_enrichment": {
                    "skincon": sum(bool(row["morphology_concept_ids"]) for row in references),
                    "skincap": sum(bool(row["reference_clinical_description"]) for row in references),
                },
                "task_shards": [task_shard],
                "reference_shards": [reference_shard],
            }
            new_splits.append(record)
            summaries.append({key: record[key] for key in ("split", "task_count", "sample_count", "group_count")})

        new_artifacts = _copy_resources(root, output)
        current["version"] = RELEASE_VERSION
        current["created_at"] = "2026-08-01"
        current["embedded_image_bytes"] = int(current["embedded_image_bytes"]) + added_image_bytes
        current["splits"] = [
            item for item in current["splits"] if item["benchmark"] != "open_ended_diagnosis"
        ] + new_splits
        current["artifacts"] = [
            item for item in current["artifacts"] if "open_ended_diagnosis" not in item["path"]
        ] + new_artifacts
        current["open_ended_judge"] = {
            "model_id": "gpt_5_6_luna",
            "model_config": "configs/models/gpt_5_6_luna.yaml",
            "single_judge": True,
            "second_judge": False,
            "human_review": False,
            "input_policy": "image_reference_final_response_only",
        }
        (output / "release.json").write_text(
            json.dumps({"release": current}, indent=2) + "\n",
            encoding="utf-8",
        )
        card = (output / "README.md").read_text(encoding="utf-8")
        (output / "README.md").write_text(_update_card(card, summaries), encoding="utf-8")
    except Exception:
        shutil.rmtree(output)
        raise
    return current


def validate_open_ended_release(
    root: Path,
    *,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = root.resolve() / output_path
    release = json.loads((output / "release.json").read_text(encoding="utf-8"))["release"]
    if release["version"] not in SUPPORTED_RELEASE_VERSIONS:
        raise ValueError("Unexpected ISEPDermaBench release version")
    result: dict[str, Any] = {"version": release["version"], "splits": {}}
    split_groups: dict[str, set[str]] = {}
    for split in ("validation", "internal_benchmark"):
        tasks = _read_split(output / "tasks/open_ended_diagnosis", split)
        refs = _read_split(output / "references/open_ended_diagnosis", split)
        expected = VALIDATION_SIZE if split == "validation" else INTERNAL_SIZE
        if len(tasks) != expected or len(refs) != expected:
            raise ValueError(f"Unexpected task/reference count for {split}")
        if tasks["task_id"].tolist() != refs["task_id"].tolist():
            raise ValueError(f"Task/reference order differs for {split}")
        if set(tasks.columns) & {
            "reference_disease_id",
            "morphology_concept_ids",
            "reference_clinical_description",
        }:
            raise ValueError("Scoring references leaked into task Parquet")
        if set(tasks["output_mode"]) != {"free_text"}:
            raise ValueError("Open-ended task output mode must be free_text")
        if any(len(_string_list(value)) > 0 for value in tasks["candidate_disease_ids"]):
            raise ValueError("Open-ended tasks must not expose candidates")
        if refs["reference_disease_id"].nunique() != 21:
            raise ValueError(f"{split} does not cover 21 diseases")
        groups = set(tasks["leakage_group_id"].astype(str))
        if len(groups) != expected:
            raise ValueError(f"{split} groups are not unique")
        split_groups[split] = groups
        result["splits"][split] = {
            "tasks": len(tasks),
            "groups": len(groups),
            "classes": refs["reference_disease_id"].nunique(),
            "skincon": int(refs["score_morphology"].sum()),
            "skincap": int(refs["score_description"].sum()),
        }
    if split_groups["validation"] & split_groups["internal_benchmark"]:
        raise ValueError("Validation/Internal group leakage")
    for split in release["splits"]:
        for shard in split["task_shards"] + split["reference_shards"]:
            path = output / shard["path"]
            if _file_sha256(path) != shard["sha256"]:
                raise ValueError(f"Shard checksum mismatch: {path}")
    for artifact in release["artifacts"]:
        if _file_sha256(output / artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {artifact['path']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ISEPDermaBench open-ended diagnosis v1.1.0")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = (
        validate_open_ended_release(args.project_root, output_path=args.output)
        if args.validate_only
        else build_open_ended_release(args.project_root, source_path=args.source, output_path=args.output)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
