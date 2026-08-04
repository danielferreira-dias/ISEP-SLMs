"""Build the nested 300-case HaloQuest visual hallucination audit."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd
from PIL import Image, ImageOps
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


BENCHMARK_KEY = "general_visual_hallucination_audit"
BENCHMARK_ID = BENCHMARK_KEY
SPLIT = "validation"
COHORT_SIZE = 300
PARENT_COHORT_SIZE = 100
RELEASE_VERSION = "1.8.0"
BENCHMARK_VERSION = "1.1.0"
SELECTION_SEED = 42
PARENT_QUOTAS = {
    ("false premises", "generated"): 34,
    ("false premises", "real"): 16,
    ("visual challenge", "generated"): 18,
    ("visual challenge", "real"): 12,
    ("insufficient context", "generated"): 12,
    ("insufficient context", "real"): 8,
}
CONDITION_QUOTAS = {
    "false premises": 100,
    "visual challenge": 100,
    "insufficient context": 100,
}
CONDITION_NAMES = {
    "false premises": "false_premises",
    "visual challenge": "visual_challenge",
    "insufficient context": "insufficient_context",
}


def build_general_visual_hallucination(
    root: Path,
    *,
    source_csv: Path,
    release_path: Path = Path("data/benchmarks/ISEPDermaBench"),
    download_retries: int = 3,
) -> dict[str, Any]:
    """Select, download, normalize, and materialize 300 HaloQuest cases."""

    root = root.resolve()
    release_root = root / release_path
    frame = pd.read_csv(source_csv)
    selected = select_haloquest_cases(frame)
    artifact_root = release_root / "artifacts"
    config_path = artifact_root / "configs" / f"{BENCHMARK_KEY}.yaml"
    prompt_path = artifact_root / "prompts" / f"{BENCHMARK_KEY}.yaml"
    schema_path = artifact_root / "schemas" / f"{BENCHMARK_KEY}.schema.json"
    taxonomy_path = artifact_root / "taxonomies/diseases.yaml"
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
    task_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    source_manifest_rows: list[dict[str, Any]] = []
    embedded_bytes = 0
    existing_by_sample: dict[str, dict[str, Any]] = {}
    existing_task_dir = release_root / "tasks" / BENCHMARK_KEY
    if list(existing_task_dir.glob(f"{SPLIT}-*.parquet")):
        existing = _read_split(existing_task_dir, SPLIT)
        existing_by_sample = {
            str(row["sample_id"]): row
            for row in existing.to_dict(orient="records")
        }
    downloaded: dict[str, bytes] = {}
    unavailable_images: set[str] = set()
    download_errors: dict[str, str] = {}
    for _ in range(20):
        selected = select_haloquest_cases(
            frame, excluded_expansion_images=unavailable_images
        )
        rows = selected.to_dict(orient="records")
        rows_to_download = [
            row
            for row in rows
            if f"HALOQUEST_EVAL_{int(row['_source_index']):04d}"
            not in existing_by_sample
            and str(row["url"]) not in downloaded
        ]
        urls = sorted({str(row["url"]) for row in rows_to_download})
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(
                executor.map(
                    lambda url: _try_download(
                        url, retries=download_retries
                    ),
                    urls,
                )
            )
        errors_by_url: dict[str, str] = {}
        for url, (value, error) in zip(urls, results, strict=True):
            if value is None:
                errors_by_url[url] = str(error)
            else:
                downloaded[url] = value
        if not errors_by_url:
            break
        for row in rows_to_download:
            url = str(row["url"])
            if url in errors_by_url:
                image_name = str(row["image_name"])
                unavailable_images.add(image_name)
                download_errors[image_name] = errors_by_url[url]
    else:
        raise RuntimeError("HaloQuest URL replacement did not converge")
    for row in rows:
        source_index = int(row["_source_index"])
        sample_id = f"HALOQUEST_EVAL_{source_index:04d}"
        task_id = f"{BENCHMARK_KEY}:{SPLIT}:{sample_id}"
        existing_row = existing_by_sample.get(sample_id)
        if existing_row is None:
            raw = downloaded[str(row["url"])]
            image_bytes = _normalize_image(raw)
            source_image_sha256 = sha256(raw).hexdigest()
        else:
            image_bytes = _image_bytes(existing_row)
            source_image_sha256 = str(existing_row["source_image_sha256"])
        embedded_bytes += len(image_bytes)
        condition = CONDITION_NAMES[str(row["hallucination type"])]
        sample = BenchmarkSample(
            sample_id=sample_id,
            task_id=task_id,
            image_uri=f"embedded://{task_id}",
            disease_id="",
            candidate_disease_ids=(),
            metadata={"question": str(row["question"])},
        )
        prepared = adapter.prepare(sample)
        rendered_schema = _compact_json(prepared.schema)
        prompt_payload = prepared.system_prompt + "\0" + prepared.user_prompt
        image_name = str(row["image_name"])
        task_rows.append(
            {
                "image": {
                    "bytes": image_bytes,
                    "path": f"{_safe_id(image_name)}.jpg",
                },
                "task_id": task_id,
                "sample_id": sample_id,
                "benchmark_id": BENCHMARK_ID,
                "benchmark_version": BENCHMARK_VERSION,
                "evaluation_set": SPLIT,
                "source": "haloquest",
                "leakage_group_id": f"HALOQUEST::{image_name}",
                "system_prompt": prepared.system_prompt,
                "user_prompt": prepared.user_prompt,
                "response_schema_json": rendered_schema,
                "prompt_id": str(prompt["id"]),
                "prompt_version": str(prompt["version"]),
                "top_k": 0,
                "candidate_disease_ids": [],
                "pair_id": f"haloquest_eval_row_{source_index}",
                "condition": condition,
                "confusion_set_id": str(row["image type"]),
                "prompt_sha256": _text_sha256(prompt_payload),
                "response_schema_sha256": _text_sha256(rendered_schema),
                "benchmark_config_sha256": config_sha,
                "taxonomy_sha256": taxonomy_sha,
                "source_image_sha256": source_image_sha256,
                "benchmark_image_sha256": sha256(image_bytes).hexdigest(),
                "image_preprocessing_profile": "haloquest_rgb_jpeg_v1",
                "license_id": "HALOQUEST_APACHE_2_0_UPSTREAM_IMAGE_TERMS",
            }
        )
        provenance = {
            "source_row_index": source_index,
            "image_name": image_name,
            "image_url": str(row["url"]),
            "image_type": str(row["image type"]),
            "haloquest_condition": str(row["hallucination type"]),
            "selection_phase": str(row["_selection_phase"]),
        }
        reference_rows.append(
            {
                "task_id": task_id,
                "sample_id": sample_id,
                "benchmark_id": BENCHMARK_ID,
                "evaluation_set": SPLIT,
                "source": "haloquest",
                "leakage_group_id": f"HALOQUEST::{image_name}",
                "reference_disease_id": "",
                "reference_diagnoses_json": _compact_json(provenance),
                "diagnosis_basis": "official_haloquest_eval_groundtruth",
                "morphology_concept_ids": [],
                "reference_clinical_description": str(
                    row["groundtruth responses"]
                ),
                "score_morphology": False,
                "score_description": False,
                "score_diagnosis": False,
                "pair_id": f"haloquest_eval_row_{source_index}",
                "condition": condition,
                "confusion_set_id": str(row["image type"]),
                "age_years": None,
                "age_group_standardized": None,
                "skin_tone_system": None,
                "skin_tone": None,
                "sex_or_gender_system": None,
                "sex_or_gender": None,
                "race_ethnicity": None,
                "license_id": "HALOQUEST_APACHE_2_0_UPSTREAM_IMAGE_TERMS",
            }
        )
        source_manifest_rows.append(provenance | {"task_id": task_id})

    task_shards, reference_shards = _write_release_shards(
        release_root, task_rows, reference_rows
    )
    metadata_root = release_root / "metadata/general_visual_hallucination_v2"
    metadata_root.mkdir(parents=True, exist_ok=True)
    manifest_path = metadata_root / "selected_cases.json"
    manifest_path.write_text(
        json.dumps(source_manifest_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    condition_counts = Counter(str(row["condition"]) for row in task_rows)
    image_type_counts = Counter(
        str(row["confusion_set_id"]) for row in task_rows
    )
    metadata = {
        "schema_version": 2,
        "id": "general_visual_hallucination_audit_v2",
        "source": "google/haloquest eval",
        "source_csv_sha256": _file_sha256(source_csv),
        "selection_seed": SELECTION_SEED,
        "selection_policy": "nested_condition_balanced_max_unique_image_sha256_v2",
        "parent_case_count": PARENT_COHORT_SIZE,
        "parent_selection": "general_visual_hallucination_audit_v1",
        "case_count": COHORT_SIZE,
        "unique_image_count": len(
            {row["leakage_group_id"] for row in task_rows}
        ),
        "condition_distribution": dict(sorted(condition_counts.items())),
        "image_type_distribution": dict(sorted(image_type_counts.items())),
        "unavailable_expansion_image_count": len(unavailable_images),
        "unavailable_expansion_images": dict(sorted(download_errors.items())),
        "selected_cases_file": manifest_path.name,
        "selected_cases_sha256": _file_sha256(manifest_path),
        "primary_scope": "answerability_and_premise_grounding",
        "free_text_answer_correctness": "not_scored_without_semantic_judge",
    }
    parent_ids = [
        row["task_id"]
        for row in source_manifest_rows
        if row["selection_phase"] == "parent_v1"
    ]
    added_ids = [
        row["task_id"]
        for row in source_manifest_rows
        if row["selection_phase"] != "parent_v1"
    ]
    _write_task_ids(metadata_root / "100_parent.task_ids.txt", parent_ids)
    _write_task_ids(metadata_root / "200_added.task_ids.txt", added_ids)
    (metadata_root / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    split_record = {
        "benchmark": BENCHMARK_KEY,
        "benchmark_id": BENCHMARK_ID,
        "split": SPLIT,
        "source_evaluation_set": "haloquest/eval",
        "task_count": COHORT_SIZE,
        "sample_count": COHORT_SIZE,
        "group_count": metadata["unique_image_count"],
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


def select_haloquest_cases(
    frame: pd.DataFrame,
    *,
    excluded_expansion_images: set[str] | None = None,
) -> pd.DataFrame:
    """Extend the original 100 cases to 300 without repeating an image."""

    required = {
        "image_name",
        "url",
        "image type",
        "hallucination type",
        "question",
        "groundtruth responses",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("HaloQuest CSV missing columns: " + ", ".join(sorted(missing)))
    working = frame.reset_index(drop=True).copy()
    excluded_expansion_images = excluded_expansion_images or set()
    working["_source_index"] = working.index
    working["_selection_hash"] = working.apply(
        lambda row: _text_sha256(
            f"{SELECTION_SEED}|{row['_source_index']}|{row['image_name']}|{row['question']}"
        ),
        axis=1,
    )
    selected: list[pd.Series] = []
    used_images: set[str] = set()
    for (condition, image_type), quota in PARENT_QUOTAS.items():
        candidates = working[
            (working["hallucination type"] == condition)
            & (working["image type"] == image_type)
        ].sort_values("_selection_hash")
        chosen = []
        for _, row in candidates.iterrows():
            image_name = str(row["image_name"])
            if image_name in used_images:
                continue
            chosen.append(row)
            used_images.add(image_name)
            if len(chosen) == quota:
                break
        if len(chosen) != quota:
            raise ValueError(f"Could not fill HaloQuest stratum {(condition, image_type)}")
        selected.extend(chosen)
    for row in selected:
        row["_selection_phase"] = "parent_v1"

    parent_counts = Counter(
        str(row["hallucination type"]) for row in selected
    )
    needed = {
        condition: quota - parent_counts[condition]
        for condition, quota in CONDITION_QUOTAS.items()
    }
    candidates: dict[str, dict[str, pd.Series]] = {}
    for condition in CONDITION_QUOTAS:
        condition_rows = working[
            (working["hallucination type"] == condition)
            & (~working["image_name"].astype(str).isin(used_images))
            & (
                ~working["image_name"]
                .astype(str)
                .isin(excluded_expansion_images)
            )
        ]
        candidates[condition] = {
            str(image_name): group.sort_values("_selection_hash").iloc[0]
            for image_name, group in condition_rows.groupby(
                condition_rows["image_name"].astype(str), sort=False
            )
        }

    slots = [
        (condition, position)
        for condition, quota in needed.items()
        for position in range(quota)
    ]
    slots.sort(
        key=lambda slot: (
            len(candidates[slot[0]]) / needed[slot[0]],
            _text_sha256(f"{slot[0]}|{slot[1]}"),
        )
    )
    owner_by_image: dict[str, tuple[str, int]] = {}
    image_by_slot: dict[tuple[str, int], str] = {}

    def assign(slot: tuple[str, int], visited: set[str]) -> bool:
        condition = slot[0]
        ordered_images = sorted(
            candidates[condition],
            key=lambda image_name: (
                str(candidates[condition][image_name]["_selection_hash"]),
                image_name,
            ),
        )
        for image_name in ordered_images:
            if image_name in visited:
                continue
            visited.add(image_name)
            owner = owner_by_image.get(image_name)
            if owner is None or assign(owner, visited):
                owner_by_image[image_name] = slot
                image_by_slot[slot] = image_name
                return True
        return False

    unmatched_slots: list[tuple[str, int]] = []
    for slot in slots:
        if not assign(slot, set()):
            unmatched_slots.append(slot)
    for slot, image_name in image_by_slot.items():
        row = candidates[slot[0]][image_name].copy()
        row["_selection_phase"] = "expansion_v2"
        selected.append(row)

    selected_source_indices = {
        int(row["_source_index"]) for row in selected
    }
    for slot in unmatched_slots:
        condition = slot[0]
        fallback = working[
            (working["hallucination type"] == condition)
            & (~working["image_name"].astype(str).isin(excluded_expansion_images))
            & (~working["_source_index"].isin(selected_source_indices))
        ].sort_values("_selection_hash")
        if fallback.empty:
            raise ValueError(
                f"Could not fill HaloQuest condition {condition}"
            )
        row = fallback.iloc[0].copy()
        row["_selection_phase"] = "expansion_v2_reused_image"
        selected_source_indices.add(int(row["_source_index"]))
        selected.append(row)

    result = (
        pd.DataFrame(selected)
        .sort_values("_selection_hash")
        .reset_index(drop=True)
    )
    if len(result) != COHORT_SIZE:
        raise ValueError("HaloQuest selection must contain 300 cases")
    if result["image_name"].nunique() < 295:
        raise ValueError("HaloQuest selection unexpectedly reuses too many images")
    observed = result["hallucination type"].value_counts().to_dict()
    if observed != CONDITION_QUOTAS:
        raise ValueError(f"HaloQuest condition quotas differ: {observed}")
    return result


def validate_general_visual_hallucination(
    root: Path,
    *,
    release_path: Path = Path("data/benchmarks/ISEPDermaBench"),
) -> dict[str, Any]:
    release_root = root.resolve() / release_path
    tasks = _read_split(release_root / "tasks" / BENCHMARK_KEY, SPLIT)
    refs = _read_split(release_root / "references" / BENCHMARK_KEY, SPLIT)
    if len(tasks) != COHORT_SIZE or len(refs) != COHORT_SIZE:
        raise ValueError("General hallucination audit must contain 300 rows")
    if tasks["task_id"].tolist() != refs["task_id"].tolist():
        raise ValueError("General hallucination task/reference IDs differ")
    if tasks["leakage_group_id"].nunique() < 295:
        raise ValueError("General hallucination audit reuses too many images")
    expected = {
        CONDITION_NAMES[key]: value
        for key, value in CONDITION_QUOTAS.items()
    }
    if tasks["condition"].value_counts().to_dict() != expected:
        raise ValueError("General hallucination condition quotas differ")
    if "reference_clinical_description" in tasks.columns:
        raise ValueError("Task view exposes HaloQuest reference answers")
    for row in tasks.to_dict(orient="records"):
        encoded = row["image"]["bytes"]
        if sha256(encoded).hexdigest() != row["benchmark_image_sha256"]:
            raise ValueError(f"Image checksum mismatch: {row['task_id']}")
        json.loads(row["response_schema_json"])
    return {
        "case_count": len(tasks),
        "unique_image_count": int(tasks["leakage_group_id"].nunique()),
        "condition_distribution": expected,
    }


def _download(url: str, *, retries: int) -> bytes:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "ISEPDermaBench/1.0"})
            with urlopen(request, timeout=45) as response:
                value = response.read()
            if not value:
                raise ValueError("downloaded image is empty")
            return value
        except Exception as exc:  # network errors are reported with source URL
            error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Could not download HaloQuest image {url}: {error}")


def _try_download(
    url: str,
    *,
    retries: int,
) -> tuple[bytes | None, Exception | None]:
    try:
        return _download(url, retries=retries), None
    except Exception as exc:
        return None, exc


def _normalize_image(raw: bytes) -> bytes:
    with Image.open(BytesIO(raw)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        for quality in (92, 88, 84, 80):
            buffer = BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=quality,
                optimize=True,
                subsampling=0,
            )
            encoded = buffer.getvalue()
            if len(encoded) <= 1_000_000:
                return encoded
    raise ValueError("Could not normalize HaloQuest image below 1 MB")


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
    release["general_visual_hallucination_audit"] = metadata | {
        "status": "development_validation",
        "must_not_be_reported_as_dermatology_accuracy": True,
    }
    license_path = release_root / "metadata/source_licenses.json"
    licenses = json.loads(license_path.read_text(encoding="utf-8"))
    licenses["sources"]["haloquest"] = {
        "license_id": "APACHE_2_0",
        "notes": (
            "HaloQuest metadata is released under Apache-2.0. Images "
            "originate from Open Images and Midjourney; preserve the "
            "upstream URL and comply with the applicable upstream image "
            "terms."
        ),
    }
    license_path.write_text(
        json.dumps(licenses, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    release["source_licenses"] = {
        "path": "metadata/source_licenses.json",
        "sha256": _file_sha256(license_path),
    }
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_split(directory: Path, split: str) -> pd.DataFrame:
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


def _image_bytes(row: dict[str, Any]) -> bytes:
    value = row.get("image")
    if not isinstance(value, dict) or not isinstance(value.get("bytes"), bytes):
        raise ValueError("Existing task has no embedded image bytes")
    return value["bytes"]


def _write_task_ids(path: Path, task_ids: list[str]) -> None:
    path.write_text("".join(f"{task_id}\n" for task_id in task_ids), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--source-csv", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = validate_general_visual_hallucination(args.project_root)
    else:
        if args.source_csv is None:
            parser.error("--source-csv is required when building")
        result = build_general_visual_hallucination(
            args.project_root,
            source_csv=args.source_csv,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
