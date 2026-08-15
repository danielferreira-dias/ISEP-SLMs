"""Materialize the authorized SkinCAP caption configuration for E2.

The builder is additive: it never rewrites the frozen v0.3 diagnosis or
morphology shards. It creates filtered observation-only caption shards and a
separate v0.4.1 release manifest referencing all three configurations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image as PILImage

from src.data_pipeline.isep_distill_dataset import ShardInfo, ShardWriter
from src.data_pipeline.isep_distill_schema import (
    CAPTION_SCHEMA_VERSION,
    caption_features,
    caption_prompt,
    messages,
)
from src.data_pipeline.splitting import assign_groups
from src.train.e2.skincap.audit import (
    DEFAULT_AUDIT_PATHS,
    audit_skincap_observations,
    load_skincap_candidate_frame,
)
from src.train.e2.skincap.domain import SkinCapTransformPolicy
from src.train.e2.skincap.transform import transform_caption

DEFAULT_OUTPUT = Path("data/training/ISEPDistillDataset")
BASE_MANIFEST = Path("metadata/release.json")
BASE_MANIFEST_SHA256 = (
    "6d7b4b5ea8b0041e443aab5aff9491d660dddc9d60f33345fa9a186c94b79b3b"
)
E1_ASSIGNMENTS = Path(
    "data/training/ISEPDermData/releases/e1_label_v1/assignments.parquet"
)
MORPHOLOGY_ASSIGNMENTS = Path(
    "data/training/ISEPDistillDataset/metadata/morphology_assignments.parquet"
)
SKINCAP_IMAGE_ROOT = Path("configs/datasets/skincap/data/skincap")
SKINCAP_REVISION = "4119044b3e14085d7439f88016d93376d433da5f"
RELEASE_ID = "isep_distill_dataset_v0.4.1"
RELEASE_DIRECTORY = Path("releases") / RELEASE_ID
CAPTION_DATA_DIRECTORY = "caption_v0_4_1"
EXPECTED_CAPTION_ROWS = 3_250
CAPTION_SHARD_SIZE = 128
PERMISSION_ATTESTATION_DATE = "2026-08-15"


def build_caption_release(
    root: Path,
    *,
    authorization_attested: bool,
    replace_existing: bool = False,
) -> dict[str, object]:
    """Build the private v0.4.1 release after an explicit permission gate.

    Args:
        root: Local ISEPDistillDataset repository directory.
        authorization_attested: Explicit user attestation of written permission.
        replace_existing: Permit replacement of generated v0.4.1 artifacts.

    Returns:
        The validated JSON-compatible v0.4.1 release manifest.
    """

    if not authorization_attested:
        raise PermissionError(
            "Written SkinCAP derivative permission must be explicitly attested"
        )
    base_path = root / BASE_MANIFEST
    if _sha256_file(base_path) != BASE_MANIFEST_SHA256:
        raise ValueError("Frozen ISEPDistillDataset v0.3 manifest has drifted")
    base = _object(_read_json(base_path), "base release")
    if base.get("release_id") != "isep_distill_dataset_v0.3.0":
        raise ValueError("Caption release must extend ISEPDistillDataset v0.3.0")
    caption_target = root / "data" / CAPTION_DATA_DIRECTORY
    release_target = root / RELEASE_DIRECTORY
    if (caption_target.exists() or release_target.exists()) and not replace_existing:
        raise ValueError("Caption v0.4.1 artifacts already exist; pass --replace")

    temporary = Path(tempfile.mkdtemp(prefix=".isep-caption-", dir=root.parent))
    try:
        (temporary / "data").mkdir(parents=True)
        (temporary / "release").mkdir(parents=True)
        shards, assignments, transform_audit = _write_caption(temporary)
        release = _release_manifest(base, shards, transform_audit)
        assignments.to_parquet(
            temporary / "release/caption_assignments.parquet", index=False
        )
        transform_audit.to_parquet(
            temporary / "release/caption_transform_audit.parquet", index=False
        )
        _write_json(temporary / "release/release.json", release)
        _write_json(
            temporary / "release/quality_summary.json",
            _quality_summary(release, assignments),
        )
        _validate_generated(temporary, release, assignments)
        _install(
            root,
            temporary,
            replace_existing=replace_existing,
        )
        return release
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _write_caption(
    root: Path,
) -> tuple[tuple[ShardInfo, ...], pd.DataFrame, pd.DataFrame]:
    candidates, _ = load_skincap_candidate_frame(DEFAULT_AUDIT_PATHS)
    policy = SkinCapTransformPolicy()
    admitted: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for row in candidates.sort_values("id").to_dict("records"):
        result = transform_caption(str(row["caption_en"]), str(row["disease"]), policy)
        sample_id = f"SKINCAP_CAPTION_{int(row['id']):04d}"
        audit_rows.append(
            {
                "sample_id": sample_id,
                "source_caption_sha256": result.source_sha256,
                "accepted": result.accepted,
                "boundary_kind": result.boundary_kind.value,
                "boundary_offset": result.boundary_offset,
                "word_count": result.word_count,
                "character_count": result.character_count,
                "rejection_reasons": ",".join(
                    reason.value for reason in result.rejection_reasons
                ),
            }
        )
        if result.accepted:
            admitted.append({**row, "sample_id": sample_id, "result": result})
    if len(admitted) != EXPECTED_CAPTION_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_CAPTION_ROWS} admitted captions, found {len(admitted)}"
        )
    frame = _assign_splits(pd.DataFrame(admitted))
    prompt = caption_prompt()
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    writer = ShardWriter(
        root=root,
        config=CAPTION_DATA_DIRECTORY,
        features=caption_features(),
        shard_size=CAPTION_SHARD_SIZE,
    )
    for row in frame.sort_values("sample_id").to_dict("records"):
        image_path = SKINCAP_IMAGE_ROOT / str(row["skincap_file_path"])
        encoded = image_path.read_bytes()
        _verify_image(encoded, str(row["sample_id"]))
        image_sha = hashlib.sha256(encoded).hexdigest()
        result = row["result"]
        target = str(result.observation_text)
        source = str(row["source"])
        writer.add(
            str(row["split"]),
            {
                "image": {"bytes": encoded, "path": str(row["skincap_file_path"])},
                "sample_id": str(row["sample_id"]),
                "case_id": str(row["leakage_group_id"]),
                "task_id": "skincap_observation_caption_v1",
                "image_asset_id": f"skincap:{int(row['id'])}",
                "view_type": "clinical",
                "leakage_group_id": str(row["leakage_group_id"]),
                "source_dataset": source,
                "source_sample_id": str(row["ori_file_path"]),
                "license_id": _license_id(source),
                "split": str(row["split"]),
                "split_inherited_from_e1": bool(row["split_inherited_from_e1"]),
                "split_source": str(row["split_source"]),
                "image_sha256": image_sha,
                "source_caption_sha256": str(result.source_sha256),
                "caption_source_revision": SKINCAP_REVISION,
                "caption_variant": policy.caption_variant,
                "transform_version": policy.version,
                "boundary_kind": result.boundary_kind.value,
                "target_variant": "observation_only_single_sentence_v1",
                "target_source": "human_caption_gold_conditioned_filtered",
                "prompt": prompt,
                "prompt_sha256": prompt_sha,
                "target_text": target,
                "schema_version": CAPTION_SCHEMA_VERSION,
                "quality_status": "accepted",
                "messages": messages(prompt, target),
            },
        )
    assignments = frame[
        [
            "sample_id",
            "leakage_group_id",
            "source",
            "ori_file_path",
            "split",
            "split_inherited_from_e1",
            "split_source",
        ]
    ].rename(columns={"source": "source_dataset", "ori_file_path": "source_sample_id"})
    shards = tuple(replace(shard, config="caption") for shard in writer.finish())
    return shards, assignments, pd.DataFrame(audit_rows)


def _assign_splits(frame: pd.DataFrame) -> pd.DataFrame:
    inherited, split_sources = _frozen_split_contract()
    frame = frame.copy()
    frame["split"] = frame["leakage_group_id"].map(inherited)
    frame["split_source"] = frame["leakage_group_id"].map(split_sources)
    frame["split_inherited_from_e1"] = frame["split_source"].eq("e1_label_v1")
    new = frame.loc[frame["split"].isna()].copy()
    new["dataset_id"] = new["source"].astype(str)
    new["disease_id"] = "UPSTREAM::" + new["disease"].astype(str).str.lower()
    for column in (
        "age_group_standardized",
        "race_ethnicity",
        "skin_tone_system",
        "skin_tone",
        "sex_or_gender_system",
        "sex_or_gender",
    ):
        new[column] = None
    assignments = assign_groups(
        new,
        ratios={"sft_train": 0.85, "sft_dev": 0.15},
        seed=42,
        secondary_feature_weight=0.0,
    )
    missing = frame["split"].isna()
    frame.loc[missing, "split"] = frame.loc[missing, "leakage_group_id"].map(
        assignments
    )
    frame.loc[missing, "split_source"] = "caption_group_split_seed_42"
    if frame["split"].isna().any():
        raise ValueError("At least one SkinCAP caption row has no split")
    train = set(frame.loc[frame["split"].eq("sft_train"), "leakage_group_id"])
    dev = set(frame.loc[frame["split"].eq("sft_dev"), "leakage_group_id"])
    if train & dev:
        raise ValueError("SkinCAP caption train/dev leakage detected")
    return frame


def _frozen_split_contract() -> tuple[dict[str, str], dict[str, str]]:
    """Return one cross-task split for every diagnosis/morphology group."""

    contract: dict[str, str] = {}
    sources: dict[str, str] = {}
    for path, source in (
        (E1_ASSIGNMENTS, "e1_label_v1"),
        (MORPHOLOGY_ASSIGNMENTS, "morphology_v0.3.0"),
    ):
        frame = pd.read_parquet(path, columns=["leakage_group_id", "split"])
        for group_id, split in frame.itertuples(index=False, name=None):
            key = str(group_id)
            value = str(split)
            previous = contract.get(key)
            if previous is not None and previous != value:
                raise ValueError(
                    f"Frozen E2 split conflict for leakage group {key}: "
                    f"{previous} versus {value}"
                )
            contract[key] = value
            sources.setdefault(key, source)
    return contract, sources


def _release_manifest(
    base: dict[str, object],
    shards: tuple[ShardInfo, ...],
    audit_rows: pd.DataFrame,
) -> dict[str, object]:
    report = replace(audit_skincap_observations(), derivatives_materialized=True)
    release = json.loads(json.dumps(base))
    release["release_id"] = RELEASE_ID
    release["schema_version"] = CAPTION_SCHEMA_VERSION
    configs = _object(release.get("configs"), "base configs")
    configs["caption"] = {
        "rows": sum(item.rows for item in shards),
        "sft_train": sum(item.rows for item in shards if item.split == "sft_train"),
        "sft_dev": sum(item.rows for item in shards if item.split == "sft_dev"),
    }
    release["config_schema_versions"] = {
        "diagnosis": "0.3.0",
        "morphology": "0.3.0",
        "caption": CAPTION_SCHEMA_VERSION,
    }
    release["base_release"] = {
        "release_id": "isep_distill_dataset_v0.3.0",
        "manifest_sha256": BASE_MANIFEST_SHA256,
    }
    release["caption_audit"] = report.as_record()
    release["authorization"] = {
        "status": "written_permission_attested_by_dataset_owner",
        "attested_on": PERMISSION_ATTESTATION_DATE,
        "document_stored_in_repository": False,
        "scope": "private_thesis_derivative_dataset_and_model_training",
    }
    sources = _object(release.get("source_revisions"), "source revisions")
    sources["skincap_hub_revision"] = SKINCAP_REVISION
    sources["skincap_metadata_sha256"] = _sha256_file(DEFAULT_AUDIT_PATHS.metadata)
    sources["skincap_transform_version"] = SkinCapTransformPolicy().version
    existing = release.get("shards")
    if not isinstance(existing, list):
        raise ValueError("Base release shard inventory is invalid")
    release["shards"] = [*existing, *(asdict(item) for item in shards)]
    warnings = release.get("warnings")
    if not isinstance(warnings, list):
        raise ValueError("Base release warnings are invalid")
    warnings.extend(
        [
            (
                "SkinCAP captions are gold-conditioned source text filtered to an "
                "observation-only prefix; they are not claimed to be answer-blind "
                "human descriptions."
            ),
            (
                "SkinCAP derivatives remain private and subject to written "
                "permission plus Fitzpatrick17k/DDI upstream terms."
            ),
            (
                "Raw captions and removed suffixes are never exposed in "
                "trainer-visible caption rows."
            ),
        ]
    )
    release["caption_transform_audit_rows"] = len(audit_rows)
    return release


def _quality_summary(
    release: dict[str, object], assignments: pd.DataFrame
) -> dict[str, object]:
    return {
        "release_status": "materialized_and_validated",
        "release_id": RELEASE_ID,
        "schema_version": CAPTION_SCHEMA_VERSION,
        "counts": release["configs"],
        "checks": {
            "base_v0_3_shards_reused_without_rewrite": True,
            "caption_internal_overlap_rows": 0,
            "caption_train_dev_group_overlap": 0,
            "cross_task_train_dev_group_overlap": 0,
            "caption_images_decoded": True,
            "caption_image_sha256_recorded": True,
            "raw_caption_exposed_to_trainer": False,
            "accepted_caption_rows": len(assignments),
        },
    }


def _validate_generated(
    root: Path,
    release: dict[str, object],
    assignments: pd.DataFrame,
) -> None:
    if len(assignments) != EXPECTED_CAPTION_ROWS:
        raise ValueError("Caption assignment count differs from the frozen expectation")
    paths = sorted((root / "data" / CAPTION_DATA_DIRECTORY).glob("*.parquet"))
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
    if rows != EXPECTED_CAPTION_ROWS:
        raise ValueError("Caption shard count differs from the frozen expectation")
    schemas = {str(pq.ParquetFile(path).schema_arrow) for path in paths}
    if len(schemas) != 1:
        raise ValueError("Caption shards do not share one Arrow schema")
    declared = release.get("shards")
    if not isinstance(declared, list):
        raise ValueError("v0.4.1 release shard inventory is invalid")
    caption_entries = [item for item in declared if item.get("config") == "caption"]
    for item in caption_entries:
        path = root / str(item["path"])
        if _sha256_file(path) != item["sha256"]:
            raise ValueError(f"Caption shard checksum mismatch: {path}")


def _install(root: Path, temporary: Path, *, replace_existing: bool) -> None:
    caption_target = root / "data" / CAPTION_DATA_DIRECTORY
    release_target = root / RELEASE_DIRECTORY
    if replace_existing:
        shutil.rmtree(caption_target, ignore_errors=True)
        shutil.rmtree(release_target, ignore_errors=True)
    caption_target.parent.mkdir(parents=True, exist_ok=True)
    release_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(
        str(temporary / "data" / CAPTION_DATA_DIRECTORY),
        caption_target,
    )
    shutil.move(str(temporary / "release"), release_target)


def _license_id(source: str) -> str:
    if source == "fitzpatrick17k":
        return "CC_BY_NC_SA_3_0__SKINCAP_WRITTEN_PERMISSION"
    if source == "ddi":
        return "DDI_RUA__SKINCAP_WRITTEN_PERMISSION"
    raise ValueError(f"Unexpected SkinCAP source: {source}")


def _verify_image(encoded: bytes, sample_id: str) -> None:
    from io import BytesIO

    try:
        with PILImage.open(BytesIO(encoded)) as image:
            image.load()
    except Exception as exc:
        raise ValueError(f"SkinCAP image decode failed for {sample_id}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a string-keyed object")
    return value


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Parse the explicit authorization gate and build the private release."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--authorization-attested", action="store_true")
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args()
    release = build_caption_release(
        arguments.output,
        authorization_attested=arguments.authorization_attested,
        replace_existing=arguments.replace,
    )
    print(json.dumps(release["configs"], sort_keys=True))


if __name__ == "__main__":
    main()
