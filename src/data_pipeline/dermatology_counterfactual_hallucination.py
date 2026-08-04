"""Build the nested 200-case dermatology counterfactual audit."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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


BENCHMARK_KEY = "dermatology_counterfactual_hallucination"
BENCHMARK_ID = BENCHMARK_KEY
SPLIT = "validation"
COHORT_SIZE = 200
PARENT_COHORT_SIZE = 50
PIXEL_SHUFFLE_SIZE = 50
HARD_NEGATIVE_SIZE = 150
RELEASE_VERSION = "1.8.0"
BENCHMARK_VERSION = "1.1.0"


def build_dermatology_counterfactual_hallucination(
    root: Path,
    *,
    release_path: Path = Path("data/benchmarks/ISEPDermaBench"),
) -> dict[str, Any]:
    """Materialize 50 shuffled controls and 150 hard-negative swaps."""

    root = root.resolve()
    release_root = root / release_path
    legacy = _load_legacy_provenance(release_root)
    selected = _load_expanded_source_cohort(release_root, legacy)
    selected["_condition"] = _assign_conditions(selected, legacy)
    records_by_task = {
        str(row["task_id"]): row
        for row in selected.to_dict(orient="records")
    }
    donor_by_task = {
        str(item["source_prompt_task_id"]): records_by_task[
            str(item["counterfactual_image_task_id"])
        ]
        for item in legacy
        if item["condition"] == "hard_negative_image_swap"
    }
    legacy_source_ids = {
        str(item["source_prompt_task_id"]) for item in legacy
    }
    new_swap_sources = selected[
        (selected["_condition"] == "hard_negative_image_swap")
        & (~selected["task_id"].astype(str).isin(legacy_source_ids))
    ].copy()
    donor_by_task.update(
        _match_unique_donors(new_swap_sources, release_root)
    )

    artifact_root = release_root / "artifacts"
    config_path = artifact_root / "configs" / f"{BENCHMARK_KEY}.yaml"
    prompt_path = artifact_root / "prompts" / f"{BENCHMARK_KEY}.yaml"
    schema_path = artifact_root / "schemas" / f"{BENCHMARK_KEY}.schema.json"
    taxonomy_path = artifact_root / "taxonomies/diseases.yaml"
    config = _yaml(config_path)
    prompt = _yaml(prompt_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    taxonomy = _yaml(taxonomy_path)
    disease_items = taxonomy["diseases"]
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
    audit_rows: list[dict[str, Any]] = []
    embedded_bytes = 0
    for source in selected.to_dict(orient="records"):
        condition = str(source["_condition"])
        source_task_id = str(source["task_id"])
        source_disease = str(source["reference_disease_id"])
        if condition == "pixel_shuffle":
            benchmark_bytes = _pixel_shuffle_image(
                _image_bytes(source),
                seed=_seed(source_task_id),
            )
            target = source
            target_disease = source_disease
            profile = "pixel_shuffle_rgb_v1"
        else:
            target = donor_by_task[source_task_id]
            benchmark_bytes = _image_bytes(target)
            target_disease = str(target["reference_disease_id"])
            profile = "hard_negative_image_swap_v1"
            if target_disease == source_disease:
                raise ValueError("Hard-negative donor has the source disease")
        embedded_bytes += len(benchmark_bytes)
        source_sample_id = str(source["sample_id"])
        sample_id = f"CF_{source_sample_id}"
        task_id = f"{BENCHMARK_KEY}:{SPLIT}:{source_sample_id}"
        candidate_ids = tuple(
            str(value) for value in source["candidate_disease_ids"]
        )
        sample = BenchmarkSample(
            sample_id=sample_id,
            task_id=task_id,
            image_uri=f"embedded://{task_id}",
            disease_id=target_disease,
            candidate_disease_ids=candidate_ids,
            metadata={},
        )
        prepared = adapter.prepare(sample)
        rendered_schema = _compact_json(prepared.schema)
        prompt_payload = prepared.system_prompt + "\0" + prepared.user_prompt
        task_rows.append(
            {
                "image": {
                    "bytes": benchmark_bytes,
                    "path": f"{_safe_id(sample_id)}_{profile}.jpg",
                },
                "task_id": task_id,
                "sample_id": sample_id,
                "benchmark_id": BENCHMARK_ID,
                "benchmark_version": BENCHMARK_VERSION,
                "evaluation_set": SPLIT,
                "source": str(target["source"]),
                "leakage_group_id": str(target["leakage_group_id"]),
                "system_prompt": prepared.system_prompt,
                "user_prompt": prepared.user_prompt,
                "response_schema_json": rendered_schema,
                "prompt_id": str(prompt["id"]),
                "prompt_version": str(prompt["version"]),
                "top_k": 3,
                "candidate_disease_ids": list(candidate_ids),
                "pair_id": source_task_id,
                "condition": condition,
                "confusion_set_id": _optional_text(
                    source.get("confusion_set_id")
                ),
                "prompt_sha256": _text_sha256(prompt_payload),
                "response_schema_sha256": _text_sha256(rendered_schema),
                "benchmark_config_sha256": config_sha,
                "taxonomy_sha256": taxonomy_sha,
                "source_image_sha256": str(
                    source["benchmark_image_sha256"]
                ),
                "benchmark_image_sha256": sha256(benchmark_bytes).hexdigest(),
                "image_preprocessing_profile": profile,
                "license_id": str(target["license_id"]),
            }
        )
        provenance = {
            "source_prompt_task_id": source_task_id,
            "source_prompt_sample_id": source_sample_id,
            "source_prompt_disease_id": source_disease,
            "counterfactual_image_task_id": str(target["task_id"]),
            "counterfactual_image_sample_id": str(target["sample_id"]),
            "counterfactual_image_disease_id": target_disease,
            "condition": condition,
            "selection_phase": (
                "parent_v1"
                if source_task_id in legacy_source_ids
                else "expansion_v2"
            ),
        }
        reference_rows.append(
            {
                "task_id": task_id,
                "sample_id": sample_id,
                "benchmark_id": BENCHMARK_ID,
                "evaluation_set": SPLIT,
                "source": str(target["source"]),
                "leakage_group_id": str(target["leakage_group_id"]),
                "reference_disease_id": target_disease,
                "reference_diagnoses_json": _compact_json(provenance),
                "diagnosis_basis": "counterfactual_image_ground_truth",
                "morphology_concept_ids": [],
                "reference_clinical_description": None,
                "score_morphology": False,
                "score_description": False,
                "score_diagnosis": condition == "hard_negative_image_swap",
                "pair_id": source_task_id,
                "condition": condition,
                "confusion_set_id": _optional_text(
                    source.get("confusion_set_id")
                ),
                "age_years": _optional_int(target.get("age_years")),
                "age_group_standardized": _optional_text(
                    target.get("age_group_standardized")
                ),
                "skin_tone_system": _optional_text(
                    target.get("skin_tone_system")
                ),
                "skin_tone": _optional_text(target.get("skin_tone")),
                "sex_or_gender_system": _optional_text(
                    target.get("sex_or_gender_system")
                ),
                "sex_or_gender": _optional_text(
                    target.get("sex_or_gender")
                ),
                "race_ethnicity": _optional_text(
                    target.get("race_ethnicity")
                ),
                "license_id": str(target["license_id"]),
            }
        )
        audit_rows.append(provenance | {"task_id": task_id})

    task_shards, reference_shards = _write_release_shards(
        release_root, task_rows, reference_rows
    )
    metadata_root = release_root / "metadata/dermatology_counterfactual_v2"
    metadata_root.mkdir(parents=True, exist_ok=True)
    audit_path = metadata_root / "case_provenance.json"
    audit_path.write_text(
        json.dumps(audit_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    condition_counts = Counter(row["condition"] for row in task_rows)
    metadata = {
        "schema_version": 2,
        "id": "dermatology_counterfactual_hallucination_v2",
        "case_count": COHORT_SIZE,
        "source_case_count": COHORT_SIZE,
        "group_count": len({row["leakage_group_id"] for row in task_rows}),
        "source_class_count": int(selected["reference_disease_id"].nunique()),
        "condition_distribution": dict(sorted(condition_counts.items())),
        "selection_policy": "nested_group_unique_200_from_visual_top_k_validation_v2",
        "parent_case_count": PARENT_COHORT_SIZE,
        "parent_selection": "dermatology_counterfactual_hallucination_v1",
        "pixel_shuffle_policy": "deterministic_rgb_pixel_permutation_v1",
        "hard_negative_policy": "unique_donor_different_disease_confusion_preferred_v1",
        "case_provenance_file": audit_path.name,
        "case_provenance_sha256": _file_sha256(audit_path),
    }
    _write_task_ids(
        metadata_root / "50_parent.task_ids.txt",
        [
            row["task_id"]
            for row in audit_rows
            if row["selection_phase"] == "parent_v1"
        ],
    )
    _write_task_ids(
        metadata_root / "150_added.task_ids.txt",
        [
            row["task_id"]
            for row in audit_rows
            if row["selection_phase"] == "expansion_v2"
        ],
    )
    (metadata_root / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    split_record = {
        "benchmark": BENCHMARK_KEY,
        "benchmark_id": BENCHMARK_ID,
        "split": SPLIT,
        "source_evaluation_set": "visual_top_k/validation_nested_200",
        "task_count": COHORT_SIZE,
        "sample_count": COHORT_SIZE,
        "group_count": metadata["group_count"],
        "embedded_image_bytes": embedded_bytes,
        "task_shards": task_shards,
        "reference_shards": reference_shards,
    }
    _update_release(
        root=root,
        release_root=release_root,
        split_record=split_record,
        artifact_paths=(config_path, prompt_path, schema_path),
        metadata=metadata,
    )
    return metadata | {"embedded_image_bytes": embedded_bytes}


def validate_dermatology_counterfactual_hallucination(
    root: Path,
    *,
    release_path: Path = Path("data/benchmarks/ISEPDermaBench"),
) -> dict[str, Any]:
    release_root = root.resolve() / release_path
    tasks = _read_split(release_root / "tasks" / BENCHMARK_KEY, SPLIT)
    refs = _read_split(release_root / "references" / BENCHMARK_KEY, SPLIT)
    if len(tasks) != COHORT_SIZE or len(refs) != COHORT_SIZE:
        raise ValueError("Dermatology counterfactual audit must contain 200 rows")
    if tasks["task_id"].tolist() != refs["task_id"].tolist():
        raise ValueError("Counterfactual task/reference IDs differ")
    expected = {
        "pixel_shuffle": PIXEL_SHUFFLE_SIZE,
        "hard_negative_image_swap": HARD_NEGATIVE_SIZE,
    }
    if tasks["condition"].value_counts().to_dict() != expected:
        raise ValueError("Counterfactual condition quotas differ")
    if "reference_disease_id" in tasks.columns:
        raise ValueError("Counterfactual task view exposes diagnosis references")
    for task, reference in zip(
        tasks.to_dict(orient="records"),
        refs.to_dict(orient="records"),
        strict=True,
    ):
        encoded = task["image"]["bytes"]
        if sha256(encoded).hexdigest() != task["benchmark_image_sha256"]:
            raise ValueError(f"Image checksum mismatch: {task['task_id']}")
        payload = json.loads(reference["reference_diagnoses_json"])
        if task["condition"] == "hard_negative_image_swap":
            if payload["source_prompt_disease_id"] == reference["reference_disease_id"]:
                raise ValueError("Hard-negative source and target diseases match")
        else:
            if reference["score_diagnosis"]:
                raise ValueError("Pixel-shuffled controls must not score diagnosis")
    return {
        "case_count": len(tasks),
        "condition_distribution": expected,
        "source_class_count": len(
            {
                json.loads(value)["source_prompt_disease_id"]
                for value in refs["reference_diagnoses_json"]
            }
        ),
    }


def _load_legacy_provenance(release_root: Path) -> list[dict[str, Any]]:
    path = (
        release_root
        / "metadata/dermatology_counterfactual_v1/case_provenance.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != PARENT_COHORT_SIZE:
        raise ValueError("Legacy counterfactual provenance must contain 50 cases")
    return value


def _load_expanded_source_cohort(
    release_root: Path,
    legacy: list[dict[str, Any]],
) -> Any:
    source_ids = [str(item["source_prompt_task_id"]) for item in legacy]
    tasks = _read_split(release_root / "tasks/visual_top_k", SPLIT)
    refs = _read_split(release_root / "references/visual_top_k", SPLIT)
    merged = tasks.merge(
        refs,
        on="task_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_reference"),
    )
    indexed = merged.set_index("task_id")
    parent = indexed.loc[source_ids].reset_index()
    parent_groups = set(parent["leakage_group_id"].astype(str))
    parent_counts = Counter(parent["reference_disease_id"].astype(str))

    unique_candidates = merged[
        ~merged["leakage_group_id"].astype(str).isin(parent_groups)
    ].copy()
    unique_candidates["_selection_hash"] = unique_candidates["task_id"].map(
        lambda value: _text_sha256(f"42|counterfactual|{value}")
    )
    unique_candidates = (
        unique_candidates.sort_values("_selection_hash")
        .drop_duplicates("leakage_group_id", keep="first")
    )
    available = Counter(parent_counts)
    available.update(unique_candidates["reference_disease_id"].astype(str))
    targets = {
        disease_id: min(count, 10)
        for disease_id, count in sorted(available.items())
    }
    while sum(targets.values()) > COHORT_SIZE:
        choices = [
            disease_id
            for disease_id, target in targets.items()
            if target > parent_counts[disease_id]
        ]
        disease_id = min(
            choices,
            key=lambda value: (
                -targets[value],
                _text_sha256(f"42|target-reduce|{value}"),
            ),
        )
        targets[disease_id] -= 1
    while sum(targets.values()) < COHORT_SIZE:
        choices = [
            disease_id
            for disease_id, target in targets.items()
            if target < available[disease_id]
        ]
        if not choices:
            raise ValueError("Insufficient group-unique Validation cases")
        disease_id = min(
            choices,
            key=lambda value: (
                targets[value],
                _text_sha256(f"42|target-grow|{value}"),
            ),
        )
        targets[disease_id] += 1

    additions = []
    for disease_id, target in sorted(targets.items()):
        needed = target - parent_counts[disease_id]
        candidates = unique_candidates[
            unique_candidates["reference_disease_id"].astype(str)
            == disease_id
        ].sort_values("_selection_hash")
        if len(candidates) < needed:
            raise ValueError(f"Insufficient Validation groups for {disease_id}")
        additions.append(candidates.head(needed))
    selected = pd.concat([parent, *additions], ignore_index=True)
    selected["_selection_phase"] = selected["task_id"].map(
        lambda value: "parent_v1" if str(value) in source_ids else "expansion_v2"
    )
    selected = selected.sort_values(
        "task_id", key=lambda column: column.map(_text_sha256)
    ).reset_index(drop=True)
    if len(selected) != COHORT_SIZE:
        raise ValueError("Expanded dermatology source cohort must contain 200 cases")
    if selected["leakage_group_id"].nunique() != COHORT_SIZE:
        raise ValueError("Expanded dermatology source cohort must be group unique")
    return selected


def _assign_conditions(
    frame: Any,
    legacy: list[dict[str, Any]],
) -> dict[int, str]:
    legacy_conditions = {
        str(item["source_prompt_task_id"]): str(item["condition"])
        for item in legacy
    }
    assignments = {
        int(index): legacy_conditions[str(row["task_id"])]
        for index, row in frame.iterrows()
        if str(row["task_id"]) in legacy_conditions
    }
    pixel_counts = Counter(
        str(frame.loc[index, "reference_disease_id"])
        for index, condition in assignments.items()
        if condition == "pixel_shuffle"
    )
    unassigned = [index for index in frame.index if index not in assignments]
    needed_shuffle = PIXEL_SHUFFLE_SIZE - sum(pixel_counts.values())
    for _ in range(needed_shuffle):
        index = min(
            unassigned,
            key=lambda candidate: (
                pixel_counts[str(frame.loc[candidate, "reference_disease_id"])],
                _text_sha256(str(frame.loc[candidate, "task_id"])),
            ),
        )
        assignments[index] = "pixel_shuffle"
        pixel_counts[str(frame.loc[index, "reference_disease_id"])] += 1
        unassigned.remove(index)
    for index in unassigned:
        assignments[index] = "hard_negative_image_swap"
    counts = Counter(assignments.values())
    if counts != Counter(
        {
            "pixel_shuffle": PIXEL_SHUFFLE_SIZE,
            "hard_negative_image_swap": HARD_NEGATIVE_SIZE,
        }
    ):
        raise ValueError(f"Counterfactual assignment imbalance: {counts}")
    return assignments


def _match_unique_donors(frame: Any, release_root: Path) -> dict[str, dict[str, Any]]:
    confusion = _confusion_membership(release_root)
    records = {
        str(row["task_id"]): row for row in frame.to_dict(orient="records")
    }
    matches: dict[str, str] = {}

    def candidates(source_id: str) -> list[str]:
        source = records[source_id]
        source_disease = str(source["reference_disease_id"])
        source_set = confusion.get(source_disease)
        choices = [
            donor_id
            for donor_id, donor in records.items()
            if donor_id != source_id
            and str(donor["reference_disease_id"]) != source_disease
        ]
        return sorted(
            choices,
            key=lambda donor_id: (
                confusion.get(str(records[donor_id]["reference_disease_id"]))
                != source_set,
                _text_sha256(f"{source_id}|{donor_id}"),
            ),
        )

    donor_owner: dict[str, str] = {}

    def assign(source_id: str, visited: set[str]) -> bool:
        for donor_id in candidates(source_id):
            if donor_id in visited:
                continue
            visited.add(donor_id)
            owner = donor_owner.get(donor_id)
            if owner is None or assign(owner, visited):
                donor_owner[donor_id] = source_id
                matches[source_id] = donor_id
                return True
        return False

    for source_id in sorted(records):
        if not assign(source_id, set()):
            raise ValueError("Could not create a unique hard-negative matching")
    if len(set(matches.values())) != len(records):
        raise ValueError("Hard-negative donors must be unique")
    return {
        source_id: records[donor_id] for source_id, donor_id in matches.items()
    }


def _confusion_membership(release_root: Path) -> dict[str, str]:
    document = _yaml(
        release_root / "artifacts/taxonomies/disease_confusion_sets.yaml"
    )
    result: dict[str, str] = {}
    for item in document["high_confusability_sets"]:
        for disease_id in item["disease_ids"]:
            result[str(disease_id)] = str(item["id"])
    return result


def _pixel_shuffle_image(encoded: bytes, *, seed: int) -> bytes:
    with Image.open(BytesIO(encoded)) as source:
        image = source.convert("RGB")
        image.thumbnail((768, 768), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.uint8).copy()
    flat = array.reshape(-1, 3)
    generator = np.random.default_rng(seed)
    shuffled = flat[generator.permutation(len(flat))].reshape(array.shape)
    image = Image.fromarray(shuffled, mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=False, subsampling=0)
    return buffer.getvalue()


def _image_bytes(row: dict[str, Any]) -> bytes:
    value = row.get("image")
    if not isinstance(value, dict) or not isinstance(value.get("bytes"), bytes):
        raise ValueError("Source task has no embedded image bytes")
    return value["bytes"]


def _seed(value: str) -> int:
    return int(_text_sha256(value)[:16], 16)


def _write_release_shards(
    release_root: Path,
    tasks: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_dir = release_root / "tasks" / BENCHMARK_KEY
    ref_dir = release_root / "references" / BENCHMARK_KEY
    _clear_shards(task_dir)
    _clear_shards(ref_dir)
    task_shards = _write_shards(
        records=tasks,
        features=TASK_FEATURES,
        directory=task_dir,
        split=SPLIT,
        shard_size=COHORT_SIZE,
    )
    ref_shards = _write_shards(
        records=references,
        features=REFERENCE_FEATURES,
        directory=ref_dir,
        split=SPLIT,
        shard_size=COHORT_SIZE,
    )
    for item in task_shards + ref_shards:
        item["path"] = Path(item["path"]).relative_to(release_root).as_posix()
    return task_shards, ref_shards


def _update_release(
    *,
    root: Path,
    release_root: Path,
    split_record: dict[str, Any],
    artifact_paths: tuple[Path, ...],
    metadata: dict[str, Any],
) -> None:
    path = release_root / "release.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    release = document["release"]
    prior = [
        item for item in release["splits"] if item.get("benchmark") == BENCHMARK_KEY
    ]
    release["embedded_image_bytes"] = (
        int(release["embedded_image_bytes"])
        - sum(int(item.get("embedded_image_bytes", 0)) for item in prior)
        + int(split_record["embedded_image_bytes"])
    )
    release["splits"] = [
        item for item in release["splits"] if item.get("benchmark") != BENCHMARK_KEY
    ] + [split_record]
    release["version"] = RELEASE_VERSION
    release["artifacts"] = [
        item
        for item in release["artifacts"]
        if BENCHMARK_KEY not in str(item.get("path", ""))
    ]
    release["artifacts"].extend(
        {
            "path": artifact.relative_to(release_root).as_posix(),
            "source": artifact.relative_to(root).as_posix(),
            "sha256": _file_sha256(artifact),
        }
        for artifact in artifact_paths
    )
    release["dermatology_counterfactual_hallucination"] = metadata | {
        "status": "development_validation",
        "must_not_be_reported_as_final_clinical_accuracy": True,
    }
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_split(directory: Path, split: str) -> Any:
    paths = sorted(directory.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No {split} Parquet shards in {directory}")
    return pa.concat_tables([pq.read_table(path) for path in paths]).to_pandas()


def _clear_shards(directory: Path) -> None:
    if directory.is_dir():
        for path in directory.glob(f"{SPLIT}-*.parquet"):
            path.unlink()


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return value


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)[:180]


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


def _write_task_ids(path: Path, task_ids: list[str]) -> None:
    path.write_text("".join(f"{task_id}\n" for task_id in task_ids), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = (
        validate_dermatology_counterfactual_hallucination(args.project_root)
        if args.validate_only
        else build_dermatology_counterfactual_hallucination(args.project_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
