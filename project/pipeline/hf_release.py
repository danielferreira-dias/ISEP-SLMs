"""Build a sanitized, sharded Hugging Face release for E3 multitask SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from project.teacher.teacher import PROJECT_ROOT

REPO_ID = "danielfdias98/ISEPDistillDataset"
CONFIG_NAME = "e3_multitask_sft_v1"
SPLIT = "sft_train"
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "sft" / "e3_multitask" / "sft_train.parquet"
DEFAULT_SOURCE_MANIFEST = DEFAULT_SOURCE.with_suffix(".manifest.json")
DEFAULT_STAGE_A_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "morphology"
    / "frozen"
    / "e3_stage_a_v1_20260822"
    / "freeze_manifest.json"
)
DEFAULT_STAGE_B_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "reasoning"
    / "frozen"
    / "e3_stage_b_v1_20260822"
    / "freeze_manifest.json"
)
DEFAULT_STAGING = PROJECT_ROOT / "data" / "hf_staging" / CONFIG_NAME
DEFAULT_MAX_SHARD_BYTES = 512 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return raw


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _compressed_row_group_bytes(metadata: Any) -> int:
    columns = metadata.num_columns
    return sum(
        int(metadata.column(index).total_compressed_size) for index in range(columns)
    )


def _shard_plan(parquet_file: Any, max_shard_bytes: int) -> list[list[int]]:
    metadata = parquet_file.metadata
    groups: list[list[int]] = []
    current: list[int] = []
    current_bytes = 0
    for index in range(metadata.num_row_groups):
        size = _compressed_row_group_bytes(metadata.row_group(index))
        if current and current_bytes + size > max_shard_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(index)
        current_bytes += size
    if current:
        groups.append(current)
    return groups


def _sanitized_stage_a(manifest: dict[str, Any]) -> dict[str, object]:
    accepted = manifest["accepted_release"]
    return {
        "freeze_id": manifest["freeze_id"],
        "rows": accepted["rows"],
        "sha256": accepted["sha256"],
        "prompt_version": manifest["protocol"]["prompt_version"],
        "prompt_sha256": manifest["protocol"]["prompt_sha256"],
        "schema_sha256": manifest["protocol"]["schema_sha256"],
        "teacher": manifest["teacher"],
    }


def _sanitized_stage_b(manifest: dict[str, Any]) -> dict[str, object]:
    accepted = manifest["accepted_release"]
    rejected = manifest["rejected_release"]
    return {
        "freeze_id": manifest["freeze_id"],
        "accepted_rows": accepted["rows"],
        "accepted_sha256": accepted["sha256"],
        "rejected_rows": rejected["rows"],
        "rejected_sha256": rejected["sha256"],
        "terminal_coverage": manifest["terminal_coverage"],
        "prompt_version": manifest["protocol"]["prompt_version"],
        "prompt_sha256": manifest["protocol"]["prompt_sha256"],
        "schema_sha256": manifest["protocol"]["schema_sha256"],
        "teacher": manifest["teacher"],
    }


def build_release(
    *,
    source: Path,
    source_manifest_path: Path,
    stage_a_manifest_path: Path,
    stage_b_manifest_path: Path,
    staging_dir: Path,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> dict[str, object]:
    """Shard one verified E3 Parquet and write sanitized publication metadata."""
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    source_manifest = _load_json(source_manifest_path)
    stage_a_manifest = _load_json(stage_a_manifest_path)
    stage_b_manifest = _load_json(stage_b_manifest_path)
    expected_source_sha = source_manifest["sha256"]
    if not isinstance(expected_source_sha, str):
        raise TypeError("Source manifest SHA-256 is missing")
    if _sha256_file(source) != expected_source_sha:
        raise ValueError("Source Parquet SHA-256 differs from its manifest")

    final_data_dir = staging_dir / "data" / CONFIG_NAME
    final_release_dir = staging_dir / "releases" / CONFIG_NAME
    if final_data_dir.exists() or final_release_dir.exists():
        raise FileExistsError("E3 Hub staging release already exists")
    staging_dir.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".e3-hf-release-", dir=staging_dir))
    temporary_data = temporary_root / "data" / CONFIG_NAME
    temporary_release = temporary_root / "releases" / CONFIG_NAME
    temporary_data.mkdir(parents=True)
    temporary_release.mkdir(parents=True)

    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(source)
        if parquet_file.metadata.num_rows != source_manifest["rows"]:
            raise ValueError("Source Parquet row count differs from its manifest")
        plans = _shard_plan(parquet_file, max_shard_bytes)
        shard_count = len(plans)
        shard_records: list[dict[str, object]] = []
        row_ids: set[str] = set()
        sample_ids: set[str] = set()
        task_counts: Counter[str] = Counter()
        prompt_registry: dict[str, dict[str, object]] = {}
        total_rows = 0

        for shard_index, row_groups in enumerate(plans):
            name = f"{SPLIT}-{shard_index:05d}-of-{shard_count:05d}.parquet"
            path = temporary_data / name
            writer = pq.ParquetWriter(
                path,
                parquet_file.schema_arrow,
                compression="zstd",
                use_dictionary=True,
            )
            shard_rows = 0
            try:
                for row_group in row_groups:
                    table = parquet_file.read_row_group(row_group)
                    writer.write_table(table, row_group_size=table.num_rows)
                    shard_rows += table.num_rows
                    metadata = table.select(
                        [
                            "row_id",
                            "sample_id",
                            "task",
                            "prompt",
                            "prompt_sha256",
                            "target_source",
                        ]
                    ).to_pylist()
                    for record in metadata:
                        row_id = record["row_id"]
                        sample_id = record["sample_id"]
                        task = record["task"]
                        prompt = record["prompt"]
                        prompt_sha = record["prompt_sha256"]
                        if not all(
                            isinstance(value, str)
                            for value in (row_id, sample_id, task, prompt, prompt_sha)
                        ):
                            raise TypeError("E3 publication metadata must be strings")
                        if row_id in row_ids:
                            raise ValueError(f"Duplicate row_id: {row_id}")
                        if _sha256_text(prompt) != prompt_sha:
                            raise ValueError(f"Prompt hash mismatch: {row_id}")
                        row_ids.add(row_id)
                        sample_ids.add(sample_id)
                        task_counts[task] += 1
                        entry = prompt_registry.setdefault(
                            task,
                            {
                                "prompt_sha256": prompt_sha,
                                "prompt": prompt,
                                "target_source": record["target_source"],
                            },
                        )
                        if entry["prompt_sha256"] != prompt_sha:
                            raise ValueError(f"Task {task} uses multiple prompt hashes")
            finally:
                writer.close()
            written = pq.ParquetFile(path)
            if written.metadata.num_rows != shard_rows:
                raise ValueError(f"Shard row count mismatch: {name}")
            if written.schema_arrow != parquet_file.schema_arrow:
                raise ValueError(f"Shard schema mismatch: {name}")
            total_rows += shard_rows
            shard_records.append(
                {
                    "path": f"data/{CONFIG_NAME}/{name}",
                    "rows": shard_rows,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                    "source_row_groups": row_groups,
                }
            )
            print(
                f"shard {shard_index + 1}/{shard_count}: "
                f"rows={shard_rows} bytes={path.stat().st_size}"
            )

        if total_rows != source_manifest["rows"]:
            raise ValueError("Published shard rows do not equal source rows")
        if dict(sorted(task_counts.items())) != source_manifest["task_counts"]:
            raise ValueError("Published task counts differ from source manifest")
        if len(row_ids) != total_rows:
            raise ValueError("Published row IDs are not unique")
        if len(sample_ids) != source_manifest["source_samples"]:
            raise ValueError("Published sample coverage differs from source manifest")

        created_at = datetime.now(UTC).isoformat()
        release: dict[str, object] = {
            "schema_version": "e3_multitask_hf_release_v1",
            "release_id": CONFIG_NAME,
            "status": "completed",
            "created_at": created_at,
            "hub": {
                "repo_id": REPO_ID,
                "config": CONFIG_NAME,
                "split": SPLIT,
                "path_pattern": f"data/{CONFIG_NAME}/{SPLIT}-*.parquet",
            },
            "source_dataset": stage_a_manifest["dataset"],
            "source_artifact": {
                "rows": source_manifest["rows"],
                "bytes": source_manifest["bytes"],
                "sha256": source_manifest["sha256"],
                "schema_version": source_manifest["schema_version"],
            },
            "rows": total_rows,
            "source_samples": len(sample_ids),
            "task_counts": dict(sorted(task_counts.items())),
            "stage_b_coverage": source_manifest["stage_b_coverage"],
            "stage_a": _sanitized_stage_a(stage_a_manifest),
            "stage_b": _sanitized_stage_b(stage_b_manifest),
            "sharding": {
                "max_source_compressed_bytes_per_shard": max_shard_bytes,
                "shard_count": shard_count,
                "total_bytes": sum(int(item["bytes"]) for item in shard_records),
                "shards": shard_records,
            },
            "publication_policy": {
                "trainer_visible_rows_only": True,
                "raw_provider_outputs_published": False,
                "failed_attempts_published": False,
                "rejected_stage_b_targets_published": False,
                "local_absolute_paths_published": False,
                "credentials_published": False,
                "sft_dev_distilled": False,
            },
        }
        quality_summary: dict[str, object] = {
            "schema_version": "e3_multitask_quality_v1",
            "release_id": CONFIG_NAME,
            "source_samples": source_manifest["source_samples"],
            "trainer_rows": total_rows,
            "task_counts": dict(sorted(task_counts.items())),
            "stage_a_accepted": source_manifest["stage_a_ok"],
            "stage_b_accepted": source_manifest["stage_b_ok"],
            "stage_b_coverage": source_manifest["stage_b_coverage"],
            "stage_b_rejected_targets_excluded": len(
                source_manifest["stage_b_rejected_attempts"]
            ),
            "stage_b_error_attempts_excluded": len(
                source_manifest["stage_b_error_attempts"]
            ),
            "stage_b_missing_attempts": len(
                source_manifest["stage_b_missing_attempt_ids"]
            ),
            "normalized_overlap_rows": stage_b_manifest["normalization"][
                "normalized_rows"
            ],
            "all_rows_quality_status": "accepted",
        }
        schema: dict[str, object] = {
            "schema_version": source_manifest["schema_version"],
            "columns": {
                field.name: str(field.type) for field in parquet_file.schema_arrow
            },
            "tasks": dict(sorted(task_counts.items())),
            "trainer_input_columns": ["image", "messages"],
            "target_contract": (
                "messages contains one user turn with image plus task prompt and "
                "one assistant turn equal to target_text"
            ),
        }
        normalization: dict[str, object] = {
            "normalization_version": stage_b_manifest["protocol"][
                "normalization_version"
            ],
            "policy": stage_b_manifest["protocol"]["normalization_policy"],
            "normalized_rows": stage_b_manifest["normalization"]["normalized_rows"],
            "diagnosis_changed": stage_b_manifest["protocol"][
                "diagnosis_changed_by_normalization"
            ],
            "clinical_reasoning_changed": stage_b_manifest["protocol"][
                "clinical_reasoning_changed_by_normalization"
            ],
            "audit_sha256": stage_b_manifest["normalization"]["artifacts"][0]["sha256"],
            "raw_provider_outputs_published": False,
        }
        _write_json(temporary_release / "release.json", release)
        _write_json(temporary_release / "quality_summary.json", quality_summary)
        _write_json(temporary_release / "schema.json", schema)
        _write_json(
            temporary_release / "prompt_registry.json",
            {
                "schema_version": "e3_multitask_prompt_registry_v1",
                "tasks": dict(sorted(prompt_registry.items())),
            },
        )
        _write_json(
            temporary_release / "normalization_summary.json",
            normalization,
        )

        final_data_dir.parent.mkdir(parents=True, exist_ok=True)
        final_release_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_data, final_data_dir)
        os.replace(temporary_release, final_release_dir)
        temporary_data.parent.rmdir()
        temporary_release.parent.rmdir()
        temporary_root.rmdir()
        return release
    except Exception:
        print(f"Incomplete build preserved at {temporary_root}")
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sanitized E3 Hugging Face dataset shards."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument(
        "--stage-a-manifest", type=Path, default=DEFAULT_STAGE_A_MANIFEST
    )
    parser.add_argument(
        "--stage-b-manifest", type=Path, default=DEFAULT_STAGE_B_MANIFEST
    )
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--max-shard-bytes", type=int, default=DEFAULT_MAX_SHARD_BYTES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_release(
        source=args.source,
        source_manifest_path=args.source_manifest,
        stage_a_manifest_path=args.stage_a_manifest,
        stage_b_manifest_path=args.stage_b_manifest,
        staging_dir=args.staging_dir,
        max_shard_bytes=args.max_shard_bytes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
