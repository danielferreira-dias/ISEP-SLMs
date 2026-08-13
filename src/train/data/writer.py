"""Serialization of a prepared assignment release and thesis audit tables."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa
import pyarrow.parquet as pq

from src.train.config import TrainingConfig
from src.train.data.integrity import ASSIGNMENT_COLUMNS, sha256_file
from src.train.domain import ReleaseAudit, ReleaseSubset, Taxonomy

RELEASE_SCHEMA_VERSION = "1.0.0"
SPLIT_ALGORITHM_VERSION = "greedy_multilabel_group_stratification_v1"


def write_release_files(
    root: Path,
    *,
    config: TrainingConfig,
    assignments: pd.DataFrame,
    taxonomy: Taxonomy,
    audit: ReleaseAudit,
    data_config_sha: str,
) -> None:
    """Write ID-only manifests, distributions, hashes, and release metadata."""

    _write_parquet(assignments, root / "assignments.parquet")
    _write_parquet(
        assignments[assignments["split"] == ReleaseSubset.SFT_TRAIN.value],
        root / "sft_train.parquet",
    )
    _write_parquet(
        assignments[assignments["split"] == ReleaseSubset.SFT_DEV.value],
        root / "sft_dev.parquet",
    )
    _write_parquet(
        assignments[assignments["is_dev_panel"].astype(bool)],
        root / "dev_panel.parquet",
    )
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    _distribution(assignments, "label").to_csv(
        metadata / "class_distribution.csv", index=False
    )
    _distribution(assignments, "source").to_csv(
        metadata / "source_distribution.csv", index=False
    )
    _write_json(metadata / "audit.json", {"audit": asdict(audit)})
    artifact_paths = sorted(path for path in root.rglob("*") if path.is_file())
    artifacts = {
        str(path.relative_to(root)): sha256_file(path) for path in artifact_paths
    }
    manifest = {
        "release": {
            "id": config.dataset.release_id,
            "schema_version": RELEASE_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "phase": config.experiment.phase.value,
            "split_algorithm": SPLIT_ALGORITHM_VERSION,
            "split_seed": config.dataset.split.seed,
            "panel_seed": config.dataset.split.panel_seed,
            "panel_groups_per_class": config.dataset.split.panel_groups_per_class,
            "taxonomy_id": taxonomy.taxonomy_id,
            "hub_repo_id": config.dataset.hub_repo_id,
            "hub_revision": config.dataset.hub_revision,
            "identity": {
                "source_release_sha256": audit.source_release_sha256,
                "assignment_sha256": audit.assignment_sha256,
                "data_config_sha256": data_config_sha,
            },
            "audit": asdict(audit),
            "artifacts": artifacts,
        }
    }
    _write_json(root / "release.json", manifest)


def _distribution(assignments: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        assignments.groupby(["split", column], sort=True)
        .agg(
            image_count=("sample_id", "size"),
            group_count=("leakage_group_id", "nunique"),
            dev_panel_count=("is_dev_panel", "sum"),
        )
        .reset_index()
    )


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(frame[list(ASSIGNMENT_COLUMNS)], preserve_index=False)
    pq.write_table(  # type: ignore[no-untyped-call]
        table, path, compression="zstd"
    )


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
