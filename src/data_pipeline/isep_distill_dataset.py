"""Build the diagnosis and morphology configs of ISEPDistillDataset.

The builder deliberately materializes only tasks with real supervision.  It
reuses the frozen E1 diagnosis assignments, excludes every ISEPDermaBench
Validation/Internal group from SKINCON, and embeds encoded source images in
sharded Parquet files for the private Hugging Face Dataset Viewer.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pyarrow.parquet as pq
import yaml
from datasets import Dataset, Features
from PIL import Image as PILImage

from src.data_pipeline.isep_distill_schema import (
    SCHEMA_VERSION,
    SKINCON_ONTOLOGY_VERSION,
    TAXONOMY_VERSION,
    diagnosis_features,
    diagnosis_prompt,
    messages,
    morphology_features,
    morphology_prompt,
    morphology_target,
)
from src.data_pipeline.splitting import assign_groups

DEFAULT_OUTPUT = Path("data/training/ISEPDistillDataset")
SOURCE_DATA = Path("data/training/ISEPDermData")
E1_ASSIGNMENTS = SOURCE_DATA / "releases/e1_label_v1/assignments.parquet"
FITZ_ANNOTATIONS = Path(
    "configs/datasets/skincon/data/annotations/annotations_fitzpatrick17k.csv"
)
DDI_ANNOTATIONS = Path("configs/datasets/skincon/data/annotations/annotations_ddi.csv")
FITZ_METADATA = Path("configs/datasets/fitzpatrick17k/data/metadata/fitzpatrick17k.csv")
DDI_METADATA = Path("configs/datasets/ddi/data/ddi_metadata.csv")
FITZ_MANIFEST = Path("data/manifests/fitzpatrick17k_c_v3.parquet")
FITZ_ARCHIVE = Path("configs/datasets/fitzpatrick17k/data/raw/fitzpatrick17k.zip")
DDI_IMAGE_ROOT = Path("configs/datasets/ddi/data/images")
BENCHMARK_REFERENCES = Path("data/benchmarks/ISEPDermaBench/references")
TAXONOMY = Path("configs/taxonomies/diseases.yaml")
SOURCE_MAPPINGS = Path("configs/taxonomies/source_disease_mappings.yaml")
ONTOLOGY = DEFAULT_OUTPUT / "metadata/skincon_ontology.json"

EXPECTED_DIAGNOSIS_ROWS = 7_541
EXPECTED_MORPHOLOGY_ROWS = 3_595
EXPECTED_RESERVED_SKINCON_ROWS = 271
DIAGNOSIS_SHARD_SIZE = 64
MORPHOLOGY_SHARD_SIZE = 256


@dataclass(frozen=True, slots=True)
class ShardInfo:
    """Immutable metadata for one generated Parquet shard."""

    config: str
    split: str
    path: str
    rows: int
    bytes: int
    sha256: str


class ShardWriter:
    """Write bounded image-containing batches with an explicit schema."""

    def __init__(
        self,
        *,
        root: Path,
        config: str,
        features: Features,
        shard_size: int,
    ) -> None:
        self._root = root
        self._config = config
        self._features = features
        self._shard_size = shard_size
        self._buffers: dict[str, list[dict[str, object]]] = {
            "sft_train": [],
            "sft_dev": [],
        }
        self._indices: Counter[str] = Counter()
        self.shards: list[ShardInfo] = []

    def add(self, split: str, record: dict[str, object]) -> None:
        """Append a record and flush its split when the shard is full."""

        if split not in self._buffers:
            raise ValueError(f"Unsupported release split: {split}")
        self._buffers[split].append(record)
        if len(self._buffers[split]) >= self._shard_size:
            self._flush(split)

    def finish(self) -> tuple[ShardInfo, ...]:
        """Flush remaining rows and return generated shard metadata."""

        for split in self._buffers:
            self._flush(split)
        return tuple(self.shards)

    def _flush(self, split: str) -> None:
        records = self._buffers[split]
        if not records:
            return
        output = self._root / "data" / self._config
        output.mkdir(parents=True, exist_ok=True)
        index = self._indices[split]
        path = output / f"{split}-{index:05d}.parquet"
        dataset = Dataset.from_list(records, features=self._features)
        dataset.to_parquet(path, compression="zstd")
        self.shards.append(
            ShardInfo(
                config=self._config,
                split=split,
                path=str(path.relative_to(self._root)),
                rows=len(records),
                bytes=path.stat().st_size,
                sha256=_file_sha256(path),
            )
        )
        self._indices[split] += 1
        records.clear()


def build_dataset(root: Path, *, replace: bool = False) -> dict[str, object]:
    """Materialize and validate the first real two-config release.

    Args:
        root: Existing dataset repository scaffold.
        replace: Whether an existing generated data directory may be replaced.

    Returns:
        JSON-serializable release manifest.
    """

    _validate_scaffold(root)
    temporary = Path(tempfile.mkdtemp(prefix=".isep-distill-build-", dir=root.parent))
    try:
        (temporary / "data").mkdir(parents=True)
        (temporary / "metadata").mkdir(parents=True)
        diagnosis_shards = _write_diagnosis(temporary)
        morphology_shards, morphology_assignments, morphology_audit = _write_morphology(
            temporary,
            ontology_path=root / "metadata/skincon_ontology.json",
        )
        morphology_assignments.to_parquet(
            temporary / "metadata/morphology_assignments.parquet", index=False
        )
        all_shards = (*diagnosis_shards, *morphology_shards)
        release = _release_manifest(all_shards, morphology_audit)
        _write_json(temporary / "metadata/release.json", release)
        _write_json(
            temporary / "metadata/quality_summary.json",
            _quality_summary(release, morphology_audit),
        )
        _validate_generated(temporary, release)
        _install_generated(root, temporary, replace=replace)
        return release
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _write_diagnosis(root: Path) -> tuple[ShardInfo, ...]:
    taxonomy = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    labels = tuple(item["canonical_name"] for item in taxonomy["diseases"])
    prompt = diagnosis_prompt(labels)
    prompt_hash = sha256(prompt.encode("utf-8")).hexdigest()
    assignments = pd.read_parquet(E1_ASSIGNMENTS).set_index("sample_id")
    writer = ShardWriter(
        root=root,
        config="diagnosis",
        features=diagnosis_features(),
        shard_size=DIAGNOSIS_SHARD_SIZE,
    )
    observed: set[str] = set()
    for path in sorted((SOURCE_DATA / "data").glob("train-*.parquet")):
        for row in pq.read_table(path).to_pylist():
            sample_id = str(row["sample_id"])
            assigned = assignments.loc[sample_id]
            if str(assigned["leakage_group_id"]) != str(row["leakage_group_id"]):
                raise ValueError(f"Diagnosis group mismatch for {sample_id}")
            encoded = bytes(row["image"]["bytes"])
            _verify_image(encoded, sample_id)
            actual_hash = sha256(encoded).hexdigest()
            if actual_hash != str(row["image_sha256"]):
                raise ValueError(f"Diagnosis image hash mismatch for {sample_id}")
            target = str(row["label"])
            split = str(assigned["split"])
            record: dict[str, object] = {
                "image": {"bytes": encoded, "path": str(row["image"]["path"])},
                "sample_id": sample_id,
                "case_id": str(row["leakage_group_id"]),
                "task_id": "diagnosis_label_only_v1",
                "image_asset_id": sample_id,
                "view_type": "unknown",
                "leakage_group_id": str(row["leakage_group_id"]),
                "source_dataset": str(row["source"]),
                "source_sample_id": str(row["source_image_id"]),
                "license_id": str(row["license_id"]),
                "split": split,
                "is_dev_panel": bool(assigned["is_dev_panel"]),
                "disease_id": str(row["disease_id"]),
                "gold_diagnosis": target,
                "source_label": str(row["source_label"]),
                "gold_provenance": _gold_provenance(str(row["diagnosis_basis"])),
                "taxonomy_version": TAXONOMY_VERSION,
                "image_sha256": actual_hash,
                "target_variant": "canonical_label_v1",
                "target_source": "human_source_join",
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
                "target_text": target,
                "schema_version": SCHEMA_VERSION,
                "quality_status": "accepted",
                "messages": messages(prompt, target),
            }
            writer.add(split, record)
            observed.add(sample_id)
    if len(observed) != EXPECTED_DIAGNOSIS_ROWS or observed != set(assignments.index):
        raise ValueError("Diagnosis rows differ from the frozen E1 release")
    return writer.finish()


def _write_morphology(
    root: Path,
    *,
    ontology_path: Path,
) -> tuple[tuple[ShardInfo, ...], pd.DataFrame, dict[str, object]]:
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    concepts = tuple(str(value) for value in ontology["concepts"])
    prompt = morphology_prompt(concepts)
    prompt_hash = sha256(prompt.encode("utf-8")).hexdigest()
    candidates, audit = _morphology_candidates(concepts)
    writer = ShardWriter(
        root=root,
        config="morphology",
        features=morphology_features(),
        shard_size=MORPHOLOGY_SHARD_SIZE,
    )
    with ZipFile(FITZ_ARCHIVE) as archive:
        for row in candidates.sort_values("sample_id").to_dict("records"):
            source = str(row["source_dataset"])
            source_image_id = str(row["source_sample_id"])
            if source == "fitzpatrick17k":
                encoded = archive.read(f"data/finalfitz17k/{source_image_id}")
            else:
                encoded = (DDI_IMAGE_ROOT / source_image_id).read_bytes()
            sample_id = str(row["sample_id"])
            _verify_image(encoded, sample_id)
            actual_hash = sha256(encoded).hexdigest()
            expected_hash = row.get("expected_image_sha256")
            if (
                isinstance(expected_hash, str)
                and expected_hash
                and actual_hash != expected_hash
            ):
                raise ValueError(f"Morphology image hash mismatch for {sample_id}")
            positive = tuple(str(value) for value in row["positive_concepts"])
            target = morphology_target(positive)
            disease_id = row.get("disease_id")
            record: dict[str, object] = {
                "image": {"bytes": encoded, "path": source_image_id},
                "sample_id": sample_id,
                "case_id": str(row["leakage_group_id"]),
                "task_id": "skincon_morphology_v1",
                "image_asset_id": f"{source}:{Path(source_image_id).stem}",
                "view_type": "unknown",
                "leakage_group_id": str(row["leakage_group_id"]),
                "source_dataset": source,
                "source_sample_id": source_image_id,
                "license_id": str(row["license_id"]),
                "split": str(row["split"]),
                "split_inherited_from_e1": bool(row["split_inherited_from_e1"]),
                "disease_id": str(disease_id) if isinstance(disease_id, str) else None,
                "gold_diagnosis": str(row["gold_diagnosis"]),
                "gold_provenance": str(row["gold_provenance"]),
                "taxonomy_version": TAXONOMY_VERSION,
                "taxonomy_mapping_status": str(row["taxonomy_mapping_status"]),
                "image_sha256": actual_hash,
                "skincon": {
                    "ontology_version": SKINCON_ONTOLOGY_VERSION,
                    "annotation_source": "SKINCON",
                    "source_subset": str(row["source_subset"]),
                    "source_image_id": source_image_id,
                    "positive_concepts": list(positive),
                    "all_concepts_annotated": True,
                },
                "target_variant": "skincon_positive_concepts_v1",
                "target_source": "human_annotated",
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
                "target_text": target,
                "schema_version": SCHEMA_VERSION,
                "quality_status": "accepted",
                "messages": messages(prompt, target),
            }
            writer.add(str(row["split"]), record)
    shards = writer.finish()
    assignments = candidates[
        [
            "sample_id",
            "leakage_group_id",
            "source_dataset",
            "source_sample_id",
            "split",
            "split_inherited_from_e1",
            "disease_id",
            "gold_diagnosis",
        ]
    ].sort_values("sample_id", ignore_index=True)
    return shards, assignments, audit


def _morphology_candidates(
    concepts: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, object]]:
    fitz = _fitz_candidates(concepts)
    ddi = _ddi_candidates(concepts)
    candidates = pd.concat([fitz, ddi], ignore_index=True)
    reserved = _reserved_fitz_groups()
    reserved_mask = candidates["leakage_group_id"].isin(reserved)
    excluded = int(reserved_mask.sum())
    if excluded != EXPECTED_RESERVED_SKINCON_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_RESERVED_SKINCON_ROWS} reserved SKINCON rows, "
            f"found {excluded}"
        )
    candidates = candidates.loc[~reserved_mask].copy()
    if len(candidates) != EXPECTED_MORPHOLOGY_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_MORPHOLOGY_ROWS} morphology rows, found "
            f"{len(candidates)}"
        )
    candidates = _assign_morphology_splits(candidates)
    train_groups = set(
        candidates.loc[candidates["split"].eq("sft_train"), "leakage_group_id"]
    )
    dev_groups = set(
        candidates.loc[candidates["split"].eq("sft_dev"), "leakage_group_id"]
    )
    if train_groups & dev_groups:
        raise ValueError("Morphology train/dev leakage detected")
    audit: dict[str, object] = {
        "raw_usable_rows": len(fitz) + len(ddi),
        "reserved_rows_excluded": excluded,
        "released_rows": len(candidates),
        "released_groups": int(candidates["leakage_group_id"].nunique()),
        "split_rows": {
            key: int(value)
            for key, value in candidates["split"].value_counts().sort_index().items()
        },
        "split_groups": {
            split: int(
                candidates.loc[
                    candidates["split"].eq(split), "leakage_group_id"
                ].nunique()
            )
            for split in ("sft_train", "sft_dev")
        },
        "source_rows": {
            key: int(value)
            for key, value in (
                candidates["source_dataset"].value_counts().sort_index().items()
            )
        },
        "mapped_21_class_rows": int(candidates["disease_id"].notna().sum()),
        "outside_21_class_rows": int(candidates["disease_id"].isna().sum()),
        "inherited_e1_rows": int(candidates["split_inherited_from_e1"].sum()),
    }
    return candidates, audit


def _fitz_candidates(concepts: tuple[str, ...]) -> pd.DataFrame:
    annotations = pd.read_csv(FITZ_ANNOTATIONS)
    annotations = annotations.loc[
        annotations["Do not consider this image"].eq(0)
    ].copy()
    metadata = pd.read_csv(FITZ_METADATA)
    metadata["ImageID"] = metadata["md5hash"].astype(str) + ".jpg"
    joined = annotations.merge(
        metadata[["ImageID", "md5hash", "label"]],
        on="ImageID",
        validate="one_to_one",
    )
    manifest = pq.read_table(
        FITZ_MANIFEST,
        columns=["original_image_id", "leakage_group_id", "image_sha256"],
    ).to_pandas()
    joined = joined.merge(
        manifest,
        left_on="md5hash",
        right_on="original_image_id",
        how="left",
        validate="one_to_one",
    )
    joined["leakage_group_id"] = joined["leakage_group_id"].fillna(
        "FITZPATRICK17K_IMAGE_" + joined["md5hash"].astype(str)
    )
    return _standardize_morphology_source(
        joined,
        concepts=concepts,
        dataset_id="fitzpatrick17k_c",
        source_dataset="fitzpatrick17k",
        source_subset="fitzpatrick17k",
        image_column="ImageID",
        diagnosis_column="label",
        license_id="CC_BY_NC_SA_3_0",
        gold_provenance="atlas_label",
        sample_prefix="SKINCON_FITZ",
    )


def _ddi_candidates(concepts: tuple[str, ...]) -> pd.DataFrame:
    annotations = pd.read_csv(DDI_ANNOTATIONS)
    annotations = annotations.loc[
        annotations["Do not consider this image"].eq(0)
    ].copy()
    metadata = pd.read_csv(DDI_METADATA).rename(columns={"DDI_file": "ImageID"})
    joined = annotations.merge(
        metadata[["ImageID", "DDI_ID", "disease"]],
        on="ImageID",
        validate="one_to_one",
    )
    joined["leakage_group_id"] = "DDI_IMAGE_" + joined["DDI_ID"].astype(str)
    joined["image_sha256"] = None
    return _standardize_morphology_source(
        joined,
        concepts=concepts,
        dataset_id="ddi",
        source_dataset="ddi",
        source_subset="ddi",
        image_column="ImageID",
        diagnosis_column="disease",
        license_id="DDI_RESEARCH_USE_AGREEMENT",
        gold_provenance="histopathology_confirmed",
        sample_prefix="SKINCON_DDI",
    )


def _standardize_morphology_source(
    joined: pd.DataFrame,
    *,
    concepts: tuple[str, ...],
    dataset_id: str,
    source_dataset: str,
    source_subset: str,
    image_column: str,
    diagnosis_column: str,
    license_id: str,
    gold_provenance: str,
    sample_prefix: str,
) -> pd.DataFrame:
    mapping = _disease_mapping(dataset_id)
    result = pd.DataFrame(
        {
            "sample_id": [
                f"{sample_prefix}_{Path(str(value)).stem}"
                for value in joined[image_column]
            ],
            "leakage_group_id": joined["leakage_group_id"].astype(str),
            "source_dataset": source_dataset,
            "source_subset": source_subset,
            "source_sample_id": joined[image_column].astype(str),
            "license_id": license_id,
            "gold_diagnosis": joined[diagnosis_column].astype(str),
            "gold_provenance": gold_provenance,
            "expected_image_sha256": joined["image_sha256"],
        }
    )
    result["disease_id"] = result["gold_diagnosis"].map(
        lambda value: mapping.get(_normalize_label(value))
    )
    result["taxonomy_mapping_status"] = result["disease_id"].map(
        lambda value: (
            "mapped_21_class" if isinstance(value, str) else "outside_21_class"
        )
    )
    result["positive_concepts"] = [
        tuple(concept for concept in concepts if int(row[concept]) == 1)
        for row in joined.to_dict("records")
    ]
    if any(not values for values in result["positive_concepts"]):
        raise ValueError(f"{source_dataset} contains a usable row with no concepts")
    return result


def _assign_morphology_splits(candidates: pd.DataFrame) -> pd.DataFrame:
    e1 = pd.read_parquet(E1_ASSIGNMENTS)
    inherited = dict(zip(e1["leakage_group_id"], e1["split"], strict=True))
    candidates["split"] = candidates["leakage_group_id"].map(inherited)
    candidates["split_inherited_from_e1"] = candidates["split"].notna()
    new = candidates.loc[candidates["split"].isna()].copy()
    new["dataset_id"] = new["source_dataset"]
    new["disease_id"] = new.apply(
        lambda row: (
            row["disease_id"]
            if isinstance(row["disease_id"], str)
            else f"UPSTREAM::{_normalize_label(str(row['gold_diagnosis']))}"
        ),
        axis=1,
    )
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
    missing_mask = candidates["split"].isna()
    candidates.loc[missing_mask, "split"] = candidates.loc[
        missing_mask, "leakage_group_id"
    ].map(assignments)
    if candidates["split"].isna().any():
        raise ValueError("At least one morphology row has no split")
    return candidates


def _reserved_fitz_groups() -> set[str]:
    groups: set[str] = set()
    for path in sorted(BENCHMARK_REFERENCES.rglob("*.parquet")):
        if not any(
            scope in path.name for scope in ("validation", "internal_benchmark")
        ):
            continue
        names = set(pq.ParquetFile(path).schema_arrow.names)
        if not {"source", "leakage_group_id"}.issubset(names):
            continue
        frame = pq.read_table(path, columns=["source", "leakage_group_id"]).to_pandas()
        groups.update(
            frame.loc[
                frame["source"].eq("fitzpatrick17k_c"), "leakage_group_id"
            ].astype(str)
        )
    if not groups:
        raise ValueError("No frozen Fitzpatrick benchmark groups were found")
    return groups


def _disease_mapping(dataset_id: str) -> dict[str, str]:
    taxonomy = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    mappings = yaml.safe_load(SOURCE_MAPPINGS.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for disease in taxonomy["diseases"]:
        names: Iterable[str] = (
            disease["canonical_name"],
            disease["display_name"],
            *disease.get("aliases", []),
        )
        result.update({_normalize_label(name): disease["id"] for name in names})
    result.update(
        {
            _normalize_label(name): disease_id
            for name, disease_id in mappings["global"].items()
        }
    )
    result.update(
        {
            _normalize_label(name): disease_id
            for name, disease_id in mappings["dataset_specific"]
            .get(dataset_id, {})
            .items()
        }
    )
    return result


def _release_manifest(
    shards: tuple[ShardInfo, ...],
    morphology_audit: dict[str, object],
) -> dict[str, object]:
    counts: dict[str, dict[str, int]] = {}
    for config in ("diagnosis", "morphology"):
        scoped = [item for item in shards if item.config == config]
        counts[config] = {
            "rows": sum(item.rows for item in scoped),
            "sft_train": sum(item.rows for item in scoped if item.split == "sft_train"),
            "sft_dev": sum(item.rows for item in scoped if item.split == "sft_dev"),
        }
    return {
        "release_id": "isep_distill_dataset_v0.3.0",
        "schema_version": SCHEMA_VERSION,
        "release_status": "private_research_release",
        "configs": counts,
        "morphology_audit": morphology_audit,
        "source_revisions": {
            "isep_derm_data": "1.3.0",
            "isep_derm_data_hub_revision": "f7403f817376de0dea0048bd3c490e294a0ccaca",
            "e1_assignment_sha256": _file_sha256(E1_ASSIGNMENTS),
            "fitzpatrick17k_metadata_sha256": _file_sha256(FITZ_METADATA),
            "skincon_fitz_sha256": _file_sha256(FITZ_ANNOTATIONS),
            "skincon_ddi_sha256": _file_sha256(DDI_ANNOTATIONS),
            "ddi_metadata_sha256": _file_sha256(DDI_METADATA),
        },
        "shards": [asdict(item) for item in shards],
        "warnings": [
            (
                "DDI is included in private morphology training and is no "
                "longer an independent external benchmark."
            ),
            (
                "DDI has no released patient identifier; its split uses "
                "image-level groups supplied by the source release."
            ),
            (
                "Morphology disease_id is null outside the frozen 21-class "
                "taxonomy; the upstream label is retained only for audit."
            ),
            (
                "PAD_UFES_20_PAT_1460_1598_746 has a non-canonical PNG IDAT "
                "CRC but decodes fully with Pillow load(), matching the E1 "
                "training path; its original bytes and SHA-256 are retained."
            ),
        ],
    }


def _quality_summary(
    release: dict[str, object], morphology_audit: dict[str, object]
) -> dict[str, object]:
    return {
        "release_status": "materialized_and_validated",
        "release_id": release["release_id"],
        "schema_version": SCHEMA_VERSION,
        "counts": release["configs"],
        "morphology": morphology_audit,
        "checks": {
            "diagnosis_matches_frozen_e1": True,
            "morphology_internal_overlap_rows": 0,
            "train_dev_group_overlap": 0,
            "all_images_decoded": True,
            "all_image_sha256_verified_when_available": True,
            "explicit_huggingface_configs_required": True,
        },
    }


def _validate_generated(root: Path, release: dict[str, object]) -> None:
    expected = {
        "diagnosis": EXPECTED_DIAGNOSIS_ROWS,
        "morphology": EXPECTED_MORPHOLOGY_ROWS,
    }
    for config, row_count in expected.items():
        paths = sorted((root / "data" / config).glob("*.parquet"))
        observed = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
        if observed != row_count:
            raise ValueError(f"{config} row count mismatch: {observed} != {row_count}")
        schemas = {str(pq.ParquetFile(path).schema_arrow) for path in paths}
        if len(schemas) != 1:
            raise ValueError(f"{config} shards do not share one Arrow schema")
        for split in ("sft_train", "sft_dev"):
            if not any(path.name.startswith(split) for path in paths):
                raise ValueError(f"{config} is missing split {split}")
    declared = release["shards"]
    if not isinstance(declared, list):
        raise ValueError("Release shard manifest is invalid")
    for item in declared:
        if not isinstance(item, dict):
            raise ValueError("Release shard entry is invalid")
        path = root / str(item["path"])
        if _file_sha256(path) != item["sha256"]:
            raise ValueError(f"Generated shard checksum mismatch: {path}")


def _install_generated(root: Path, temporary: Path, *, replace: bool) -> None:
    generated = root / "data"
    has_parquet = generated.exists() and any(generated.rglob("*.parquet"))
    if has_parquet and not replace:
        raise ValueError("Generated data already exists; pass --replace to rebuild")
    if generated.exists():
        shutil.rmtree(generated)
    shutil.move(str(temporary / "data"), generated)
    for name in (
        "morphology_assignments.parquet",
        "release.json",
        "quality_summary.json",
    ):
        target = root / "metadata" / name
        target.unlink(missing_ok=True)
        shutil.move(str(temporary / "metadata" / name), target)


def _gold_provenance(basis: str) -> str:
    return {
        "atlas_label": "atlas_label",
        "pathology": "histopathology_confirmed",
        "clinical_consensus": "expert_consensus",
        "dermatologist_differential": "dermatologist_differential",
        "source_diagnosis_unspecified": "unknown_provenance",
    }.get(basis, "unknown_provenance")


def _verify_image(encoded: bytes, sample_id: str) -> None:
    try:
        from io import BytesIO

        with PILImage.open(BytesIO(encoded)) as image:
            # Match the actual training decode path. ``verify()`` rejects a
            # PAD-UFES-20 PNG whose IDAT CRC is non-canonical even though
            # Pillow can decode every pixel successfully with ``load()``.
            image.load()
    except Exception as error:
        raise ValueError(f"Image decode failed for {sample_id}: {error}") from error


def _validate_scaffold(root: Path) -> None:
    for relative in (
        "README.md",
        "schemas/canonical_record.schema.json",
        "metadata/skincon_ontology.json",
    ):
        if not (root / relative).is_file():
            raise ValueError(f"Dataset scaffold is missing {relative}")


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the command-line builder."""

    args = _parse_args()
    release = build_dataset(args.output, replace=bool(args.replace))
    print(json.dumps(release["configs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
