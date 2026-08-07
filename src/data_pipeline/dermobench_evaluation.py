"""Build the training-leakage-filtered DermoBench evaluation view.

The official gated release remains immutable under ``release/``.  This module
creates a derived ``evaluation/tasks`` tree after excluding every task whose
image is present in ISEPDermData Train.  Exact image SHA-256 is the primary
test; source-level identities and leakage groups provide an independent audit
for SCIN, Fitzpatrick17k-C, PAD-UFES-20, and HIBA/ISIC.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
from typing import Any

import pyarrow.parquet as pq

from src.data_pipeline.dermobench import _file_sha256


DEFAULT_BENCHMARK_ROOT = Path("data/benchmarks/DermoBench")
DEFAULT_TRAIN_ROOT = Path("data/training/ISEPDermData")
JUDGE_CONFIG = "configs/models/gemini_3_5_flash_lite_openrouter.yaml"
JUDGE_MODEL = "google/gemini-3.5-flash-lite"
OPEN_ENDED_TASKS = {
    "task1/1_1_description_wo_morph/task1_1_final.jsonl",
    "task1/1_2_description_w_morph/task1_2_final.jsonl",
    "task3/3_1/task3_1_final.jsonl",
    "task3/3_2/task3_2_final.jsonl",
}


def build_filtered_dermobench(
    root: Path,
    *,
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT,
    train_root: Path = DEFAULT_TRAIN_ROOT,
) -> dict[str, Any]:
    """Materialize and validate the leakage-filtered DermoBench tasks."""

    root = root.resolve()
    benchmark = root / benchmark_root
    upstream = benchmark / "release"
    evaluation = benchmark / "evaluation"
    task_output = evaluation / "tasks"
    index_path = upstream / "image_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            "DermoBench image_index.json is missing; run "
            "python -m src.data_pipeline.dermobench --extract first"
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    image_paths = index.get("image_paths")
    if not isinstance(image_paths, dict):
        raise ValueError("DermoBench image index has no image_paths mapping")

    train = _load_train_identity(root / train_root)
    scin_cases = _load_scin_case_index(root)
    hiba_patients = _load_hiba_patient_index(root)
    overlap_by_image = _audit_images(
        upstream=upstream,
        image_paths=image_paths,
        train=train,
        scin_cases=scin_cases,
        hiba_patients=hiba_patients,
    )

    annotation_paths = _annotation_paths(upstream)
    file_records: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []
    total_rows = kept_rows = 0
    for source_path in annotation_paths:
        relative = source_path.relative_to(upstream)
        rows = _read_rows(source_path)
        kept: list[dict[str, Any]] = []
        for row in rows:
            image = str(row["image"])
            audit = overlap_by_image[image]
            total_rows += 1
            if audit["reasons"]:
                excluded_records.append(
                    {
                        "annotation_file": relative.as_posix(),
                        "task_id": str(row.get("id", "")),
                        "image": image,
                        "image_sha256": audit["image_sha256"],
                        "reasons": audit["reasons"],
                    }
                )
            else:
                kept.append(row)
                kept_rows += 1
        output_path = task_output / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_rows(output_path, kept)
        identifiers = [str(row.get("id", "")) for row in kept]
        file_records.append(
            {
                "path": relative.as_posix(),
                "upstream_rows": len(rows),
                "excluded_rows": len(rows) - len(kept),
                "evaluation_rows": len(kept),
                "duplicate_task_id_count": (
                    len(identifiers) - len(set(identifiers))
                ),
                "upstream_sha256": _file_sha256(source_path),
                "evaluation_sha256": _file_sha256(output_path),
                "scoring": (
                    "llm_as_a_judge"
                    if relative.as_posix() in OPEN_ENDED_TASKS
                    else "deterministic_exact_choice"
                ),
            }
        )

    evaluation.mkdir(parents=True, exist_ok=True)
    exclusions_path = evaluation / "excluded_cases.jsonl"
    exclusions_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in excluded_records
        ),
        encoding="utf-8",
    )
    overlapping_images = {
        image: audit
        for image, audit in overlap_by_image.items()
        if audit["reasons"]
    }
    source_counts = Counter(image.split("/", 1)[0] for image in overlapping_images)
    reason_counts = Counter(
        reason
        for audit in overlapping_images.values()
        for reason in audit["reasons"]
    )
    document = {
        "schema_version": 1,
        "benchmark": "DermoBench",
        "upstream": {
            "repo_id": "mendicant04/DermoBench",
            "release_root": "../release",
            "annotation_file_count": len(annotation_paths),
            "task_count": total_rows,
            "unique_image_references": len(image_paths),
            "image_archive_sha256": str(index["archive"]["sha256"]),
        },
        "evaluation": {
            "policy": "exclude_isepdermdata_train_image_or_source_group_v1",
            "task_count": kept_rows,
            "excluded_task_count": total_rows - kept_rows,
            "excluded_unique_image_count": len(overlapping_images),
            "excluded_image_source_counts": dict(sorted(source_counts.items())),
            "overlap_reason_counts": dict(sorted(reason_counts.items())),
            "task_root": "tasks",
            "files": file_records,
            "excluded_cases": {
                "path": exclusions_path.name,
                "sha256": _file_sha256(exclusions_path),
                "rows": len(excluded_records),
            },
        },
        "judge": {
            "model_config": JUDGE_CONFIG,
            "model": JUDGE_MODEL,
            "provider": "google-ai-studio",
            "applies_to": sorted(OPEN_ENDED_TASKS),
            "protocol": "upstream_text_only_task_specific_judge_prompts",
            "upstream_baseline_judge": "gemini-2.5-pro",
            "comparability": (
                "Judge-based scores must be reported as a new Flash-Lite "
                "protocol and are not directly comparable with the paper's "
                "Gemini 2.5 Pro judge scores."
            ),
        },
        "training_release": {
            "path": str(train_root),
            "release_sha256": _file_sha256(root / train_root / "release.json"),
            "image_count": train["row_count"],
            "unique_image_sha256_count": len(train["hashes"]),
            "leakage_group_count": len(train["groups"]),
        },
    }
    release_path = evaluation / "release.json"
    release_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_filtered_dermobench(root, benchmark_root=benchmark_root)
    return document


def validate_filtered_dermobench(
    root: Path,
    *,
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT,
) -> dict[str, Any]:
    """Verify filtered counts, checksums, joins, and absence of excluded rows."""

    evaluation = root.resolve() / benchmark_root / "evaluation"
    release = json.loads((evaluation / "release.json").read_text(encoding="utf-8"))
    expected_total = 0
    for record in release["evaluation"]["files"]:
        path = evaluation / "tasks" / record["path"]
        if _file_sha256(path) != record["evaluation_sha256"]:
            raise ValueError(f"Filtered DermoBench checksum mismatch: {path}")
        rows = _read_rows(path)
        if len(rows) != int(record["evaluation_rows"]):
            raise ValueError(f"Filtered DermoBench row-count mismatch: {path}")
        expected_total += len(rows)
        identifiers = [str(row.get("id", "")) for row in rows]
        duplicate_count = len(identifiers) - len(set(identifiers))
        if duplicate_count != int(record["duplicate_task_id_count"]):
            raise ValueError(f"Filtered duplicate-ID audit mismatch: {path}")
    if expected_total != int(release["evaluation"]["task_count"]):
        raise ValueError("Filtered DermoBench total task count is inconsistent")
    excluded = {
        (row["annotation_file"], row["task_id"])
        for row in _read_jsonl(evaluation / "excluded_cases.jsonl")
    }
    for record in release["evaluation"]["files"]:
        for row in _read_rows(evaluation / "tasks" / record["path"]):
            if (record["path"], str(row.get("id", ""))) in excluded:
                raise ValueError("Excluded DermoBench task remains in evaluation view")
    return release


def _load_train_identity(train_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted((train_root / "data").glob("train-*.parquet")):
        rows.extend(
            pq.read_table(
                path,
                columns=[
                    "source",
                    "source_image_id",
                    "leakage_group_id",
                    "image_sha256",
                ],
            ).to_pylist()
        )
    if not rows:
        raise FileNotFoundError(f"No ISEPDermData Train shards in {train_root}")
    source_ids: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_ids[str(row["source"])].add(str(row["source_image_id"]).casefold())
    return {
        "row_count": len(rows),
        "hashes": {str(row["image_sha256"]) for row in rows},
        "groups": {str(row["leakage_group_id"]) for row in rows},
        "source_ids": source_ids,
    }


def _load_scin_case_index(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted((root / "configs/datasets/scin/data/data").glob("*.parquet")):
        table = pq.read_table(
            path,
            columns=["case_id", "image_1_path", "image_2_path", "image_3_path"],
        )
        for row in table.to_pylist():
            for column in ("image_1_path", "image_2_path", "image_3_path"):
                value = row[column]
                if value and value.get("path"):
                    result[Path(str(value["path"])).name] = (
                        f"SCIN_CASE_{row['case_id']}"
                    )
    return result


def _load_hiba_patient_index(root: Path) -> dict[str, str]:
    path = root / "configs/datasets/hiba/data/hiba-skin-lesions.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            str(row["isic_id"]): f"HIBA_PATIENT_{row['patient_id']}"
            for row in csv.DictReader(handle)
            if row.get("isic_id") and row.get("patient_id")
        }


def _audit_images(
    *,
    upstream: Path,
    image_paths: dict[str, str],
    train: dict[str, Any],
    scin_cases: dict[str, str],
    hiba_patients: dict[str, str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for annotation_path, archive_path in sorted(image_paths.items()):
        path = upstream / "images" / archive_path
        image_hash = _file_sha256(path)
        prefix = annotation_path.split("/", 1)[0].casefold()
        name = Path(annotation_path).name
        stem = Path(annotation_path).stem
        reasons: set[str] = set()
        if image_hash in train["hashes"]:
            reasons.add("exact_image_sha256")
        if prefix == "f17k" and stem.casefold() in train["source_ids"]["fitzpatrick17k_c"]:
            reasons.add("fitzpatrick17k_c_source_image_id")
        if prefix == "scin" and scin_cases.get(name) in train["groups"]:
            reasons.add("scin_case_group")
        if prefix == "pad":
            if name.casefold() in train["source_ids"]["pad_ufes_20"]:
                reasons.add("pad_ufes_20_source_image_id")
            match = re.match(r"(PAT_[^_]+)_", stem, flags=re.IGNORECASE)
            if match and (
                f"PAD_UFES_20_PATIENT_{match.group(1).upper()}" in train["groups"]
            ):
                reasons.add("pad_ufes_20_patient_group")
        if prefix == "isic" and hiba_patients.get(stem) in train["groups"]:
            reasons.add("hiba_patient_group")
        result[annotation_path] = {
            "image_sha256": image_hash,
            "reasons": sorted(reasons),
        }
    return result


def _annotation_paths(upstream: Path) -> list[Path]:
    return sorted(
        path
        for path in upstream.rglob("*")
        if path.suffix in {".json", ".jsonl"} and path.name != "image_index.json"
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.suffix == ".jsonl":
        value = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    else:
        value = json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    path.write_text(value, encoding="utf-8")


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
        validate_filtered_dermobench(args.project_root)
        if args.validate_only
        else build_filtered_dermobench(args.project_root)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
