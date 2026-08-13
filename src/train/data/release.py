"""Create deterministic, immutable, image-free E1 split releases."""

from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Protocol, cast

import pandas as pd  # type: ignore[import-untyped]

from src.train.config import TrainingConfig
from src.train.data.integrity import (
    ASSIGNMENT_COLUMNS,
    assignment_digest,
    audit_assignments,
    child_mapping,
    inspect_data_release,
    load_json_mapping,
)
from src.train.data.source import (
    load_source_frame,
    source_release_sha256,
    validate_source_pool,
    validate_source_shards,
)
from src.train.data.taxonomy import load_taxonomy
from src.train.data.writer import write_release_files
from src.train.domain import PreparedRelease, ReleaseAudit, ReleaseSubset, Taxonomy


class _GroupAssigner(Protocol):
    def __call__(
        self,
        frame: pd.DataFrame,
        *,
        ratios: dict[str, float],
        seed: int,
        secondary_feature_weight: float,
    ) -> dict[str, str]: ...


def prepare_data_release(config: TrainingConfig) -> PreparedRelease:
    """Build a deterministic group-safe release without duplicating images.

    Existing releases are accepted only when source, data configuration, and
    semantic assignment hashes match exactly.  Otherwise the function fails
    instead of silently changing a frozen release identifier.
    """

    taxonomy = load_taxonomy(config)
    validate_source_shards(config)
    source = load_source_frame(config)
    validate_source_pool(config, source, taxonomy)
    assignments = _build_assignments(config, source, taxonomy)
    source_sha = source_release_sha256(config)
    assignments_sha = assignment_digest(assignments)
    data_config_sha = hashlib.sha256(
        config.dataset.model_dump_json().encode("utf-8")
    ).hexdigest()
    target = config.resolve_path(config.dataset.release_directory)
    if target.exists():
        return _reuse_existing(
            target,
            source_sha=source_sha,
            assignment_sha=assignments_sha,
            data_config_sha=data_config_sha,
        )

    audit = audit_assignments(
        assignments,
        release_id=config.dataset.release_id,
        source_sha=source_sha,
    )
    _validate_expected_splits(config, audit)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        write_release_files(
            temporary,
            config=config,
            assignments=assignments,
            taxonomy=taxonomy,
            audit=audit,
            data_config_sha=data_config_sha,
        )
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _prepared_release(target, inspect_data_release(target))


def _build_assignments(
    config: TrainingConfig,
    source: pd.DataFrame,
    taxonomy: Taxonomy,
) -> pd.DataFrame:
    split_input = source.copy()
    split_input["dataset_id"] = split_input["source"]
    for column in (
        "age_group_standardized",
        "race_ethnicity",
        "skin_tone_system",
        "skin_tone",
        "sex_or_gender_system",
        "sex_or_gender",
    ):
        split_input[column] = None
    split = config.dataset.split
    group_assignments = _group_assigner()(
        split_input,
        ratios={
            ReleaseSubset.SFT_TRAIN.value: split.train_ratio,
            ReleaseSubset.SFT_DEV.value: split.dev_ratio,
        },
        seed=split.seed,
        secondary_feature_weight=split.secondary_feature_weight,
    )
    result = source[
        ["sample_id", "leakage_group_id", "disease_id", "label", "source"]
    ].copy()
    result["split"] = result["leakage_group_id"].map(group_assignments)
    if result["split"].isna().any():
        raise ValueError("At least one row did not receive a split assignment")
    panel_ids = _select_dev_panel(
        result[result["split"] == ReleaseSubset.SFT_DEV.value],
        taxonomy=taxonomy,
        groups_per_class=split.panel_groups_per_class,
        seed=split.panel_seed,
    )
    result["is_dev_panel"] = result["sample_id"].isin(panel_ids)
    return result[list(ASSIGNMENT_COLUMNS)].sort_values("sample_id", ignore_index=True)


def _select_dev_panel(
    dev: pd.DataFrame,
    *,
    taxonomy: Taxonomy,
    groups_per_class: int,
    seed: int,
) -> frozenset[str]:
    group_support = dev.groupby("disease_id")["leakage_group_id"].nunique()
    ordered_classes = sorted(
        taxonomy.classes,
        key=lambda item: (int(group_support.get(item.disease_id, 0)), item.disease_id),
    )
    used_groups: set[str] = set()
    selected_samples: set[str] = set()
    for taxonomy_class in ordered_classes:
        scoped = dev[dev["disease_id"] == taxonomy_class.disease_id]
        representatives: list[tuple[str, str]] = []
        for group_id, group in scoped.groupby("leakage_group_id", sort=True):
            sample_id = min(
                group["sample_id"].astype(str),
                key=lambda value: _seeded_digest(seed, f"sample:{value}"),
            )
            representatives.append((str(group_id), sample_id))
        representatives.sort(
            key=lambda pair: _seeded_digest(
                seed, f"{taxonomy_class.disease_id}:{pair[0]}:{pair[1]}"
            )
        )
        available = [pair for pair in representatives if pair[0] not in used_groups]
        if len(available) < groups_per_class:
            raise ValueError(
                f"Class {taxonomy_class.label} has only {len(available)} "
                "unused development groups"
            )
        for group_id, sample_id in available[:groups_per_class]:
            used_groups.add(group_id)
            selected_samples.add(sample_id)
    return frozenset(selected_samples)


def _validate_expected_splits(config: TrainingConfig, audit: ReleaseAudit) -> None:
    expected = config.dataset.expected
    for field in (
        "train_image_count",
        "train_group_count",
        "dev_image_count",
        "dev_group_count",
    ):
        observed = int(getattr(audit, field))
        expected_value = int(getattr(expected, field))
        if observed != expected_value:
            raise ValueError(
                f"{field} mismatch: expected {expected_value}, found {observed}"
            )


def _reuse_existing(
    root: Path,
    *,
    source_sha: str,
    assignment_sha: str,
    data_config_sha: str,
) -> PreparedRelease:
    release = child_mapping(
        load_json_mapping(root / "release.json", "release manifest"), "release"
    )
    identity = child_mapping(release, "identity")
    expected = {
        "source_release_sha256": source_sha,
        "assignment_sha256": assignment_sha,
        "data_config_sha256": data_config_sha,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(
                f"Frozen release {root} already exists with different {key}"
            )
    return _prepared_release(root, inspect_data_release(root))


def _prepared_release(root: Path, audit: ReleaseAudit) -> PreparedRelease:
    return PreparedRelease(
        root=root,
        release_manifest_path=root / "release.json",
        assignments_path=root / "assignments.parquet",
        train_manifest_path=root / "sft_train.parquet",
        dev_manifest_path=root / "sft_dev.parquet",
        dev_panel_manifest_path=root / "dev_panel.parquet",
        audit=audit,
    )


def _seeded_digest(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _group_assigner() -> _GroupAssigner:
    """Load the legacy split function behind a typed boundary."""

    module = importlib.import_module("src.data_pipeline.splitting")
    candidate: object = module.assign_groups
    if not callable(candidate):
        raise TypeError("src.data_pipeline.splitting.assign_groups is not callable")
    return cast(_GroupAssigner, candidate)
