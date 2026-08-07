"""Build the paired SCIN clinical-context ablation in ISEPDermaBench."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from datasets import Features, Value
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


BENCHMARK_KEY = "clinical_context_ablation"
BENCHMARK_ID = BENCHMARK_KEY
RELEASE_VERSION = "1.9.0"
BENCHMARK_VERSION = "1.0.0"
SPLITS = ("validation", "internal_benchmark")
CONDITIONS = ("image_only", "image_plus_context")
SHARD_SIZE = 512

CONTEXT_TASK_FEATURES = Features(
    dict(TASK_FEATURES)
    | {
        "clinical_context": Value("string"),
        "context_present": Value("bool"),
    }
)
CONTEXT_REFERENCE_FEATURES = Features(
    dict(REFERENCE_FEATURES)
    | {
        "clinical_context_json": Value("string"),
        "context_source": Value("string"),
    }
)

TEXTURE_FIELDS = {
    "textures_raised_or_bumpy": "raised or bumpy",
    "textures_flat": "flat",
    "textures_rough_or_flaky": "rough or flaky",
    "textures_fluid_filled": "fluid-filled",
}
BODY_FIELDS = {
    "body_parts_head_or_neck": "head or neck",
    "body_parts_arm": "arm",
    "body_parts_palm": "palm",
    "body_parts_back_of_hand": "back of hand",
    "body_parts_torso_front": "front of torso",
    "body_parts_torso_back": "back of torso",
    "body_parts_genitalia_or_groin": "genital or groin area",
    "body_parts_buttocks": "buttocks",
    "body_parts_leg": "leg",
    "body_parts_foot_top_or_side": "top or side of foot",
    "body_parts_foot_sole": "sole of foot",
    "body_parts_other": "other body site",
}
SYMPTOM_FIELDS = {
    "condition_symptoms_bothersome_appearance": "bothersome appearance",
    "condition_symptoms_bleeding": "bleeding",
    "condition_symptoms_increasing_size": "increasing size",
    "condition_symptoms_darkening": "darkening",
    "condition_symptoms_itching": "itching",
    "condition_symptoms_burning": "burning",
    "condition_symptoms_pain": "pain",
}
OTHER_SYMPTOM_FIELDS = {
    "other_symptoms_fever": "fever",
    "other_symptoms_chills": "chills",
    "other_symptoms_fatigue": "fatigue",
    "other_symptoms_joint_pain": "joint pain",
    "other_symptoms_mouth_sores": "mouth sores",
    "other_symptoms_shortness_of_breath": "shortness of breath",
}
DURATION_LABELS = {
    "ONE_DAY": "one day or less",
    "LESS_THAN_ONE_WEEK": "less than one week",
    "ONE_TO_FOUR_WEEKS": "one to four weeks",
    "ONE_TO_THREE_MONTHS": "one to three months",
    "THREE_TO_TWELVE_MONTHS": "three to twelve months",
    "MORE_THAN_ONE_YEAR": "more than one year",
    "MORE_THAN_FIVE_YEARS": "more than five years",
    "SINCE_CHILDHOOD": "since childhood",
    "UNKNOWN": "unknown",
}


def build_clinical_context_ablation(
    root: Path,
    *,
    release_path: Path = Path("data/benchmarks/ISEPDermaBench"),
) -> dict[str, Any]:
    """Materialize paired image-only/context tasks from held-out SCIN cases."""

    root = root.resolve()
    release_root = root / release_path
    artifacts = release_root / "artifacts"
    config_path = artifacts / "configs/clinical_context_ablation.yaml"
    prompt_path = artifacts / "prompts/clinical_context_ablation.yaml"
    schema_path = artifacts / "schemas/clinical_context_ablation.schema.json"
    taxonomy_path = artifacts / "taxonomies/diseases.yaml"
    config = _yaml(config_path)
    prompt = _yaml(prompt_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    taxonomy = _yaml(taxonomy_path)
    adapter = build_task_adapter(
        benchmark_config=config,
        prompt_config=prompt,
        schema=schema,
        disease_taxonomy_items=taxonomy["diseases"],
    )
    config_sha = _file_sha256(config_path)
    taxonomy_sha = _file_sha256(taxonomy_path)
    scin_manifest = _scin_manifest_index(root)
    train_hashes, train_groups = _training_identity(root)

    split_records: list[dict[str, Any]] = []
    selected_metadata: dict[str, list[dict[str, Any]]] = {}
    total_embedded_bytes = 0
    for split in SPLITS:
        source_tasks = _read_split(release_root / "tasks/visual_top_k", split)
        source_refs = _read_split(release_root / "references/visual_top_k", split)
        reference_by_id = {str(row["task_id"]): row for row in source_refs}
        task_rows: list[dict[str, Any]] = []
        reference_rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        for source_task in source_tasks:
            if str(source_task["source"]) != "scin":
                continue
            sample_id = str(source_task["sample_id"])
            normalized = scin_manifest.get(sample_id)
            if normalized is None:
                raise ValueError(f"SCIN manifest is missing {sample_id}")
            source_context = _source_context(root, normalized)
            if not _has_explicit_condition_symptom_response(source_context):
                continue
            context_data = _context_data(source_context)
            context_text = _render_context(context_data)
            source_reference = reference_by_id[str(source_task["task_id"])]
            image_hash = str(source_task["source_image_sha256"])
            group_id = str(source_task["leakage_group_id"])
            if image_hash in train_hashes or group_id in train_groups:
                raise ValueError(
                    f"Context-ablation case overlaps ISEPDermData Train: {sample_id}"
                )
            pair_id = f"{BENCHMARK_KEY}:{split}:{sample_id}"
            for condition in CONDITIONS:
                clinical_context = (
                    "No patient-reported context was provided."
                    if condition == "image_only"
                    else context_text
                )
                task_id = f"{pair_id}:{condition}"
                sample = BenchmarkSample(
                    sample_id=sample_id,
                    task_id=task_id,
                    image_uri=f"embedded://{task_id}",
                    disease_id=str(source_reference["reference_disease_id"]),
                    candidate_disease_ids=None,
                    metadata={
                        "clinical_context": clinical_context,
                        "condition": condition,
                        "pair_id": pair_id,
                    },
                )
                prepared = adapter.prepare(sample)
                rendered_schema = _compact_json(prepared.schema)
                image = _image_value(source_task["image"])
                total_embedded_bytes += len(image["bytes"])
                task_rows.append(
                    {
                        **{
                            key: source_task.get(key)
                            for key in TASK_FEATURES
                        },
                        "image": image,
                        "task_id": task_id,
                        "sample_id": sample_id,
                        "benchmark_id": BENCHMARK_ID,
                        "benchmark_version": BENCHMARK_VERSION,
                        "evaluation_set": split,
                        "system_prompt": prepared.system_prompt,
                        "user_prompt": prepared.user_prompt,
                        "response_schema_json": rendered_schema,
                        "prompt_id": str(prompt["id"]),
                        "prompt_version": str(prompt["version"]),
                        "pair_id": pair_id,
                        "condition": condition,
                        "prompt_sha256": _text_sha256(
                            prepared.system_prompt + "\0" + prepared.user_prompt
                        ),
                        "response_schema_sha256": _text_sha256(rendered_schema),
                        "benchmark_config_sha256": config_sha,
                        "taxonomy_sha256": taxonomy_sha,
                        "clinical_context": clinical_context,
                        "context_present": condition == "image_plus_context",
                    }
                )
                reference_rows.append(
                    {
                        **{
                            key: source_reference.get(key)
                            for key in REFERENCE_FEATURES
                        },
                        "task_id": task_id,
                        "sample_id": sample_id,
                        "benchmark_id": BENCHMARK_ID,
                        "evaluation_set": split,
                        "pair_id": pair_id,
                        "condition": condition,
                        "clinical_context_json": _compact_json(context_data),
                        "context_source": "SCIN participant self-report",
                    }
                )
            audit_rows.append(
                {
                    "pair_id": pair_id,
                    "sample_id": sample_id,
                    "leakage_group_id": group_id,
                    "reference_disease_id": str(
                        source_reference["reference_disease_id"]
                    ),
                    "context_fields": context_data,
                }
            )
        task_rows.sort(key=lambda row: (str(row["pair_id"]), str(row["condition"])))
        reference_rows.sort(
            key=lambda row: (str(row["pair_id"]), str(row["condition"]))
        )
        selected_metadata[split] = audit_rows
        task_shards, reference_shards = _write_release_shards(
            release_root=release_root,
            split=split,
            tasks=task_rows,
            references=reference_rows,
        )
        split_records.append(
            {
                "benchmark": BENCHMARK_KEY,
                "benchmark_id": BENCHMARK_ID,
                "split": split,
                "source_evaluation_set": f"visual_top_k/{split}",
                "task_count": len(task_rows),
                "sample_count": len(audit_rows),
                "pair_count": len(audit_rows),
                "group_count": len(
                    {str(row["leakage_group_id"]) for row in audit_rows}
                ),
                "condition_counts": dict(
                    sorted(Counter(str(row["condition"]) for row in task_rows).items())
                ),
                "embedded_image_bytes": sum(
                    len(_image_value(row["image"])["bytes"]) for row in task_rows
                ),
                "task_shards": task_shards,
                "reference_shards": reference_shards,
            }
        )

    metadata_root = release_root / "metadata/clinical_context_ablation_v1"
    metadata_root.mkdir(parents=True, exist_ok=True)
    manifest_path = metadata_root / "selected_cases.json"
    manifest_path.write_text(
        json.dumps(selected_metadata, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": 1,
        "id": "clinical_context_ablation_v1",
        "source": "SCIN participant self-report",
        "conditions": list(CONDITIONS),
        "selection": (
            "Held-out SCIN Visual Top-K cases with an explicit response in at "
            "least one condition_symptoms field"
        ),
        "included_context_categories": [
            "condition_duration",
            "self_reported_texture",
            "self_reported_body_location",
            "condition_symptoms",
            "other_symptoms",
        ],
        "excluded_context_categories": [
            "related_category",
            "dermatologist_labels",
            "dermatologist_confidence",
            "demographics",
        ],
        "null_policy": "not_reported_not_negative",
        "validation_pair_count": next(
            item["pair_count"] for item in split_records if item["split"] == "validation"
        ),
        "internal_benchmark_pair_count": next(
            item["pair_count"]
            for item in split_records
            if item["split"] == "internal_benchmark"
        ),
        "selected_cases_path": manifest_path.relative_to(release_root).as_posix(),
        "selected_cases_sha256": _file_sha256(manifest_path),
        "reference_caveat": (
            "SCIN labels are retrospective dermatologist differentials created "
            "with access to self-report and are not uniformly pathology-confirmed."
        ),
    }
    _update_release(
        release_root=release_root,
        split_records=split_records,
        artifact_paths=(config_path, prompt_path, schema_path),
        metadata=metadata,
    )
    validate_clinical_context_ablation(root, release_path=release_path)
    return {
        "benchmark": BENCHMARK_ID,
        "release_version": RELEASE_VERSION,
        "splits": split_records,
        "metadata": metadata,
    }


def validate_clinical_context_ablation(
    root: Path,
    *,
    release_path: Path = Path("data/benchmarks/ISEPDermaBench"),
) -> dict[str, Any]:
    """Validate pair completeness, image equality, checksums, and isolation."""

    release_root = root.resolve() / release_path
    release = json.loads((release_root / "release.json").read_text(encoding="utf-8"))
    records = [
        item
        for item in release["release"]["splits"]
        if item.get("benchmark") == BENCHMARK_KEY
    ]
    if {item["split"] for item in records} != set(SPLITS):
        raise ValueError("Context ablation does not contain both frozen splits")
    for record in records:
        tasks = _read_split(release_root / "tasks" / BENCHMARK_KEY, record["split"])
        refs = _read_split(
            release_root / "references" / BENCHMARK_KEY, record["split"]
        )
        if len(tasks) != int(record["task_count"]) or len(refs) != len(tasks):
            raise ValueError("Context-ablation task/reference count mismatch")
        ref_ids = {str(row["task_id"]) for row in refs}
        if ref_ids != {str(row["task_id"]) for row in tasks}:
            raise ValueError("Context-ablation task/reference IDs do not join")
        pairs: dict[str, list[dict[str, Any]]] = {}
        for row in tasks:
            pairs.setdefault(str(row["pair_id"]), []).append(row)
        for pair_id, pair in pairs.items():
            if {str(row["condition"]) for row in pair} != set(CONDITIONS):
                raise ValueError(f"Incomplete context-ablation pair: {pair_id}")
            if len({str(row["benchmark_image_sha256"]) for row in pair}) != 1:
                raise ValueError(f"Paired images differ: {pair_id}")
            if len({_image_value(row["image"])["bytes"] for row in pair}) != 1:
                raise ValueError(f"Paired image bytes differ: {pair_id}")
        for shard in record["task_shards"] + record["reference_shards"]:
            path = release_root / shard["path"]
            if _file_sha256(path) != shard["sha256"]:
                raise ValueError(f"Context-ablation checksum mismatch: {path}")
    return {"split_count": len(records), "status": "valid"}


def _scin_manifest_index(root: Path) -> dict[str, dict[str, Any]]:
    table = pq.read_table(
        root / "data/manifests/scin_v3.parquet",
        columns=["sample_id", "source_metadata"],
    )
    return {
        str(row["sample_id"]): json.loads(str(row["source_metadata"]))
        for row in table.to_pylist()
    }


@lru_cache(maxsize=None)
def _raw_scin_shard(path: str) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _source_context(root: Path, normalized: dict[str, Any]) -> dict[str, Any]:
    shard = root / str(normalized["source_shard"])
    rows = _raw_scin_shard(str(shard))
    return rows[int(normalized["source_row"])]


def _has_explicit_condition_symptom_response(row: dict[str, Any]) -> bool:
    fields = [*SYMPTOM_FIELDS, "condition_symptoms_no_relevant_experience"]
    return any(_selected(row.get(field)) for field in fields)


def _context_data(row: dict[str, Any]) -> dict[str, Any]:
    lesion_symptoms = _selected_labels(row, SYMPTOM_FIELDS)
    if _selected(row.get("condition_symptoms_no_relevant_experience")):
        lesion_symptoms = ["no listed lesion symptoms"]
    other_symptoms = _selected_labels(row, OTHER_SYMPTOM_FIELDS)
    if _selected(row.get("other_symptoms_no_relevant_symptoms")):
        other_symptoms = ["no listed systemic symptoms"]
    duration = row.get("condition_duration")
    return {
        "duration": DURATION_LABELS.get(str(duration), None),
        "reported_texture": _selected_labels(row, TEXTURE_FIELDS),
        "reported_body_location": _selected_labels(row, BODY_FIELDS),
        "reported_lesion_symptoms": lesion_symptoms,
        "reported_other_symptoms": other_symptoms,
    }


def _render_context(context: dict[str, Any]) -> str:
    lines: list[str] = []
    if context["duration"]:
        lines.append(f"- Reported duration: {context['duration']}.")
    for key, label in (
        ("reported_body_location", "Reported body location"),
        ("reported_texture", "Reported lesion texture"),
        ("reported_lesion_symptoms", "Reported lesion symptoms"),
        ("reported_other_symptoms", "Reported other symptoms"),
    ):
        values = context[key]
        if values:
            lines.append(f"- {label}: {', '.join(values)}.")
    if not lines:
        raise ValueError("Selected SCIN context unexpectedly contains no fields")
    return "\n".join(lines)


def _selected_labels(row: dict[str, Any], mapping: dict[str, str]) -> list[str]:
    return [label for field, label in mapping.items() if _selected(row.get(field))]


def _selected(value: Any) -> bool:
    return str(value).strip().casefold() == "yes"


def _training_identity(root: Path) -> tuple[set[str], set[str]]:
    hashes: set[str] = set()
    groups: set[str] = set()
    for path in (root / "data/training/ISEPDermData/data").glob("*.parquet"):
        table = pq.read_table(path, columns=["image_sha256", "leakage_group_id"])
        for row in table.to_pylist():
            hashes.add(str(row["image_sha256"]))
            groups.add(str(row["leakage_group_id"]))
    return hashes, groups


def _read_split(directory: Path, split: str) -> list[dict[str, Any]]:
    paths = sorted(directory.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No {split} Parquet shards in {directory}")
    return pa.concat_tables([pq.read_table(path) for path in paths]).to_pylist()


def _write_release_shards(
    *,
    release_root: Path,
    split: str,
    tasks: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_dir = release_root / "tasks" / BENCHMARK_KEY
    ref_dir = release_root / "references" / BENCHMARK_KEY
    for directory in (task_dir, ref_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob(f"{split}-*.parquet"):
            path.unlink()
    task_shards = _write_shards(
        records=tasks,
        features=CONTEXT_TASK_FEATURES,
        directory=task_dir,
        split=split,
        shard_size=SHARD_SIZE,
    )
    reference_shards = _write_shards(
        records=references,
        features=CONTEXT_REFERENCE_FEATURES,
        directory=ref_dir,
        split=split,
        shard_size=SHARD_SIZE,
    )
    for shard in task_shards + reference_shards:
        shard["path"] = Path(shard["path"]).relative_to(release_root).as_posix()
    return task_shards, reference_shards


def _update_release(
    *,
    release_root: Path,
    split_records: list[dict[str, Any]],
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
        + sum(int(item["embedded_image_bytes"]) for item in split_records)
    )
    release["splits"] = [
        item for item in release["splits"] if item.get("benchmark") != BENCHMARK_KEY
    ] + split_records
    release["version"] = RELEASE_VERSION
    release["artifacts"] = [
        item
        for item in release["artifacts"]
        if BENCHMARK_KEY not in str(item.get("path", ""))
    ]
    release["artifacts"].extend(
        {
            "path": artifact.relative_to(release_root).as_posix(),
            "source": artifact.relative_to(release_root).as_posix(),
            "sha256": _file_sha256(artifact),
        }
        for artifact in artifact_paths
    )
    release[BENCHMARK_KEY] = metadata | {
        "status": "paired_development_and_internal_evaluation",
        "primary_analysis": "within_model_paired_context_delta",
    }
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _image_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("bytes"), bytes):
        raise ValueError("Frozen benchmark row has no embedded image bytes")
    return {"bytes": value["bytes"], "path": value.get("path")}


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


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
        validate_clinical_context_ablation(args.project_root)
        if args.validate_only
        else build_clinical_context_ablation(args.project_root)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
