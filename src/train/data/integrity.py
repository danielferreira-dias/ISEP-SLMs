"""Integrity primitives for frozen, image-free assignment manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pyarrow.parquet as pq

from src.train.domain import ReleaseAudit, ReleaseSubset

ASSIGNMENT_COLUMNS = (
    "sample_id",
    "leakage_group_id",
    "disease_id",
    "label",
    "source",
    "split",
    "is_dev_panel",
)


def inspect_data_release(release_directory: Path) -> ReleaseAudit:
    """Verify hashes, cardinalities, leakage safety, and panel composition."""

    root = release_directory.resolve()
    manifest = load_json_mapping(root / "release.json", "release manifest")
    release = child_mapping(manifest, "release")
    release_id = required_string(release, "id")
    identity = child_mapping(release, "identity")
    source_sha = required_string(identity, "source_release_sha256")
    assignments = pq.read_table(  # type: ignore[no-untyped-call]
        root / "assignments.parquet"
    ).to_pandas()
    validate_assignment_columns(assignments)
    audit = audit_assignments(
        assignments,
        release_id=release_id,
        source_sha=source_sha,
    )
    if audit.assignment_sha256 != required_string(identity, "assignment_sha256"):
        raise ValueError("Semantic assignment hash does not match release")
    _verify_artifact_hashes(root, child_mapping(release, "artifacts"))
    _verify_subset_manifests(root, assignments)
    _verify_declared_audit(release, audit)
    panel_groups_per_class = required_int(release, "panel_groups_per_class")
    panel = assignments[assignments["is_dev_panel"].astype(bool)]
    panel_counts = panel.groupby("label")["leakage_group_id"].nunique()
    if len(panel_counts) != audit.class_count:
        raise ValueError("Development panel does not cover every class")
    if not panel_counts.eq(panel_groups_per_class).all():
        raise ValueError("Development panel group count differs by class")
    return audit


def audit_assignments(
    assignments: pd.DataFrame,
    *,
    release_id: str,
    source_sha: str,
) -> ReleaseAudit:
    """Calculate cardinalities and enforce train/dev separation."""

    validate_assignment_columns(assignments)
    train = assignments[assignments["split"] == ReleaseSubset.SFT_TRAIN.value]
    dev = assignments[assignments["split"] == ReleaseSubset.SFT_DEV.value]
    panel = assignments[assignments["is_dev_panel"].astype(bool)]
    overlap = set(train["leakage_group_id"]) & set(dev["leakage_group_id"])
    if overlap:
        raise ValueError(f"Train/dev leakage detected for {len(overlap)} groups")
    if not panel["sample_id"].isin(dev["sample_id"]).all():
        raise ValueError("Development panel contains a non-development sample")
    if panel["leakage_group_id"].nunique() != len(panel):
        raise ValueError("Development panel must contain one image per group")
    return ReleaseAudit(
        release_id=release_id,
        source_image_count=len(assignments),
        source_group_count=int(assignments["leakage_group_id"].nunique()),
        class_count=int(assignments["label"].nunique()),
        source_count=int(assignments["source"].nunique()),
        train_image_count=len(train),
        train_group_count=int(train["leakage_group_id"].nunique()),
        dev_image_count=len(dev),
        dev_group_count=int(dev["leakage_group_id"].nunique()),
        dev_panel_image_count=len(panel),
        dev_panel_group_count=int(panel["leakage_group_id"].nunique()),
        group_overlap_count=0,
        assignment_sha256=assignment_digest(assignments),
        source_release_sha256=source_sha,
    )


def assignment_digest(frame: pd.DataFrame) -> str:
    """Hash the semantic assignment content independent of Parquet metadata."""

    ordered = frame[list(ASSIGNMENT_COLUMNS)].sort_values(
        "sample_id", ignore_index=True
    )
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_assignment_columns(frame: pd.DataFrame) -> None:
    """Validate the exact image-free assignment schema."""

    if tuple(frame.columns) != ASSIGNMENT_COLUMNS:
        raise ValueError("Assignment manifest has an unexpected schema")
    if frame["sample_id"].isna().any() or not frame["sample_id"].is_unique:
        raise ValueError("Assignment sample IDs must be non-null and unique")
    if not set(frame["split"]).issubset(
        {ReleaseSubset.SFT_TRAIN.value, ReleaseSubset.SFT_DEV.value}
    ):
        raise ValueError("Assignment manifest contains an unknown split")


def load_json_mapping(path: Path, name: str) -> Mapping[object, object]:
    """Load a JSON object with a contextual validation error."""

    try:
        document: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load {name} {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ValueError(f"{name} must be an object")
    return document


def child_mapping(
    document: Mapping[object, object], key: str
) -> Mapping[object, object]:
    """Return a required nested JSON object."""

    value = document.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def required_string(document: Mapping[object, object], key: str) -> str:
    """Return a required non-empty JSON string."""

    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def sha256_file(path: Path) -> str:
    """Hash a file in bounded-memory chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_subset_manifests(root: Path, assignments: pd.DataFrame) -> None:
    filters = {
        "sft_train.parquet": assignments["split"] == ReleaseSubset.SFT_TRAIN.value,
        "sft_dev.parquet": assignments["split"] == ReleaseSubset.SFT_DEV.value,
        "dev_panel.parquet": assignments["is_dev_panel"].astype(bool),
    }
    for filename, mask in filters.items():
        scoped = pq.read_table(  # type: ignore[no-untyped-call]
            root / filename
        ).to_pandas()
        validate_assignment_columns(scoped)
        expected_ids = set(assignments.loc[mask, "sample_id"].astype(str))
        if set(scoped["sample_id"].astype(str)) != expected_ids:
            raise ValueError(f"{filename} does not match assignments.parquet")


def _verify_declared_audit(
    release: Mapping[object, object], audit: ReleaseAudit
) -> None:
    declared = child_mapping(release, "audit")
    for key, value in asdict(audit).items():
        if declared.get(key) != value:
            raise ValueError(f"Declared audit field {key} does not match data")


def _verify_artifact_hashes(root: Path, artifacts: Mapping[object, object]) -> None:
    for relative, expected in artifacts.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("Artifact checksums must map paths to SHA-256 strings")
        if sha256_file(root / relative) != expected:
            raise ValueError(f"Artifact checksum mismatch: {relative}")


def required_int(document: Mapping[object, object], key: str) -> int:
    """Return a required JSON integer while rejecting booleans."""

    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value
