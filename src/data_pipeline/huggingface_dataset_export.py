"""Export the 21-class dermatology training pool for Hugging Face.

The export embeds the original encoded image bytes in sharded Parquet files.
It deliberately excludes free-text captions, raw source metadata, local image
URIs, and demographic fields. The source training manifest remains the single
source of truth for selection and leakage grouping.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import tempfile
from typing import Any

from datasets import Dataset, Features, Image as HFImage, Value
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq
import yaml

from src.data_pipeline.deduplication import ImageResolver


EXPORT_SCHEMA_VERSION = "1.0.0"
EXPORT_RELEASE_VERSION = "1.2.0"
DEFAULT_INPUT = Path(
    "data/training/dermatology_multimodal_v1/train_images.parquet"
)
DEFAULT_OUTPUT = Path("data/training/ISEPDermData")
DEFAULT_SHARD_SIZE = 512
EXPECTED_IMAGE_COUNT = 6_858
EXPECTED_GROUP_COUNT = 5_254
EXPECTED_CLASS_COUNT = 21
EXPECTED_PROMOTED_IMAGE_COUNT = 123
EXPECTED_PROMOTED_GROUP_COUNT = 63

PROMOTION_SOURCE_MANIFESTS = (
    Path("data/manifests/fitzpatrick17k_c_v3.parquet"),
    Path("data/manifests/pad_ufes_20_v3.parquet"),
    Path("data/manifests/scin_v3.parquet"),
)
PROMOTION_PROTECTED_TASKS = (
    Path(
        "data/benchmarks/ISEPDermaBench/tasks/visual_top_k/"
        "validation-*.parquet"
    ),
    Path(
        "data/benchmarks/ISEPDermaBench/tasks/visual_top_k/"
        "internal_benchmark-*.parquet"
    ),
)

# Derm1M is deliberately excluded from the canonical Hugging Face release.
# A label-quality audit found source-derived entity-linking errors and images
# whose diagnosis depended on article context rather than visible evidence.
EXCLUDED_SOURCES = {"derm1m"}

SOURCE_DISPLAY_NAMES = {
    "fitzpatrick17k_c": "Fitzpatrick17k-C",
    "hiba": "HIBA",
    "pad_ufes_20": "PAD-UFES-20",
    "scin": "SCIN",
}

EXPORT_COLUMNS = [
    "image",
    "source",
    "label",
    "disease_id",
    "sample_id",
    "source_image_id",
    "source_label",
    "leakage_group_id",
    "diagnosis_basis",
    "image_sha256",
    "license_id",
]

EXPORT_FEATURES = Features(
    {
        "image": HFImage(decode=True),
        "source": Value("string"),
        "label": Value("string"),
        "disease_id": Value("string"),
        "sample_id": Value("string"),
        "source_image_id": Value("string"),
        "source_label": Value("string"),
        "leakage_group_id": Value("string"),
        "diagnosis_basis": Value("string"),
        "image_sha256": Value("string"),
        "license_id": Value("string"),
    }
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_names(root: Path) -> dict[str, str]:
    taxonomy = yaml.safe_load(
        (root / "configs/taxonomies/diseases.yaml").read_text(encoding="utf-8")
    )
    return {
        disease["id"]: disease["canonical_name"]
        for disease in taxonomy["diseases"]
    }


def _image_filename(row: dict[str, Any], encoded: bytes) -> str:
    original = str(row["original_image_id"] or "")
    suffix = Path(original).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
        with Image.open(BytesIO(encoded)) as image:
            image_format = (image.format or "JPEG").lower()
        suffix = ".jpg" if image_format in {"jpeg", "jpg"} else f".{image_format}"
    safe_sample_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["sample_id"]))
    return f"{safe_sample_id}{suffix}"


def _validate_encoded_image(encoded: bytes, *, sample_id: str) -> None:
    try:
        with Image.open(BytesIO(encoded)) as image:
            image.verify()
    except Exception as error:
        raise ValueError(f"Image decode failed for {sample_id}: {error}") from error


def _export_record(
    row: dict[str, Any],
    *,
    resolver: ImageResolver,
    disease_names: dict[str, str],
    verify_images: bool,
) -> dict[str, Any]:
    encoded = resolver.read_bytes(str(row["image_uri"]))
    actual_sha256 = sha256(encoded).hexdigest()
    expected_sha256 = str(row["image_sha256"])
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Image checksum mismatch for {row['sample_id']}: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    if verify_images:
        _validate_encoded_image(encoded, sample_id=str(row["sample_id"]))

    disease_id = str(row["disease_id"])
    return {
        "image": {
            "bytes": encoded,
            "path": _image_filename(row, encoded),
        },
        "source": str(row["dataset_id"]),
        "label": disease_names[disease_id],
        "disease_id": disease_id,
        "sample_id": str(row["sample_id"]),
        "source_image_id": str(row["original_image_id"]),
        "source_label": str(row["disease_original"]),
        "leakage_group_id": str(row["leakage_group_id"]),
        "diagnosis_basis": str(row["diagnosis_basis"]),
        "image_sha256": actual_sha256,
        "license_id": str(row["license_id"]),
    }


def _dataset_card(
    *,
    image_count: int,
    group_count: int,
    class_count: int,
    source_counts: dict[str, int],
    class_counts: list[dict[str, Any]],
) -> str:
    source_rows = "\n".join(
        f"| {SOURCE_DISPLAY_NAMES.get(source, source)} | {count:,} |"
        for source, count in source_counts.items()
    )
    class_rows = "\n".join(
        f"| {row['disease_id']} | {row['label']} | {row['image_count']:,} | "
        f"{row['group_count']:,} |"
        for row in class_counts
    )
    size_category = "1K<n<10K" if image_count < 10_000 else "10K<n<100K"
    return f"""---
pretty_name: ISEPDermData
language:
- en
license: other
task_categories:
- image-classification
tags:
- dermatology
- medical
- image
- multimodal
- private-research-dataset
size_categories:
- {size_category}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# ISEPDermData

ISEPDermData is a private, research-only collection of {image_count:,}
clinical dermatology images mapped to the 21-class ISEP thesis taxonomy.
The release is intended as the source pool for teacher annotation and later
multimodal student fine-tuning.

## Dataset summary

| Statistic | Value |
| --- | ---: |
| Images | {image_count:,} |
| Leakage-safe groups | {group_count:,} |
| Active disease classes | {class_count:,} |
| Split | `train` (unsplit source pool) |

This release is deliberately an unsplit pool. A later version will
materialize `sft_train` and `sft_dev` by `leakage_group_id`; related images
must never cross those splits.

## Schema

The first three columns are the primary training view:

```text
image | source | label
```

Audit columns preserve stable IDs, original source labels, grouping,
diagnostic provenance, checksums, and per-row source licences:

```text
image
source
label
disease_id
sample_id
source_image_id
source_label
leakage_group_id
diagnosis_basis
image_sha256
license_id
```

Free-text captions, raw source metadata, local paths, and demographic fields
are intentionally excluded from this release.

## Source distribution

| Source | Images |
| --- | ---: |
{source_rows}

## Class distribution

| Disease ID | Label | Images | Groups |
| --- | --- | ---: | ---: |
{class_rows}

## Provenance and licences

This is a mixed-source research dataset. Every row retains `source` and
`license_id`; no single licence should be interpreted as replacing the terms
of the original source. The current sources include Fitzpatrick17k-C,
PAD-UFES-20, SCIN, and HIBA. Derm1M was removed in release 1.1.0 after a
label-quality audit identified source-derived entity-linking errors and
context-dependent images unsuitable as direct image-classification targets.
Release 1.2.0 promotes 123 images from 63 previously unrepresented internal
reserve groups into Train. Images from groups represented by Validation or the
sealed Internal Benchmark remain excluded from training.
See `metadata/source_licenses.json` and the upstream dataset documentation
before any redistribution, commercial use, or publication of derived
artifacts.

## Intended use

- dermatology image classification within the fixed 21-class taxonomy;
- teacher-generated visual findings, differential diagnoses, and short
  evidence-grounded rationales;
- research on small multimodal language models;
- group-safe supervised fine-tuning after a separate split is released.

## Limitations

- source labels have heterogeneous diagnostic certainty;
- class and source distributions are imbalanced;
- some groups contain multiple related images;
- the dataset is not a medical device and must not be used as a substitute
  for clinical diagnosis;
- this release does not include out-of-domain or description-only records.

## Reproducibility

The release is generated from the local thesis repository with:

```bash
python -m src.data_pipeline.huggingface_dataset_export
```

Checksums and exact counts are recorded in `release.json`.
"""


def _source_licenses() -> dict[str, Any]:
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "sources": {
            "fitzpatrick17k_c": {
                "display_name": "Fitzpatrick17k-C",
                "license_id": "CC_BY_NC_SA_3_0",
                "notes": "Corrected labels over the upstream Fitzpatrick17k images.",
            },
            "pad_ufes_20": {
                "display_name": "PAD-UFES-20",
                "license_id": "CC_BY_4_0",
            },
            "scin": {
                "display_name": "SCIN",
                "license_id": "SCIN_DATA_USE_LICENSE",
            },
            "hiba": {
                "display_name": "HIBA",
                "license_id": "CC_BY_4_0",
            },
        },
    }


def _promoted_internal_reserve(
    root: Path,
    *,
    base_selected: pd.DataFrame,
) -> pd.DataFrame:
    """Promote only groups absent from train and published evaluations.

    The full unpublished source remainder contains companion images from
    groups represented by the sealed internal benchmark. Those rows are
    deliberately excluded. Only the 63 wholly unrepresented groups are
    reassigned to training.
    """

    source_frames = [
        pd.read_parquet(root / path)
        for path in PROMOTION_SOURCE_MANIFESTS
    ]
    candidates = pd.concat(source_frames, ignore_index=True, sort=False)
    candidates = candidates[
        candidates["include"].fillna(False)
        & candidates["disease_id"].notna()
        & candidates["image_sha256"].notna()
    ].copy()

    protected_groups: set[str] = set()
    protected_paths: list[Path] = []
    for pattern in PROMOTION_PROTECTED_TASKS:
        matches = sorted(root.glob(pattern.as_posix()))
        if not matches:
            raise FileNotFoundError(
                f"No protected ISEPDermaBench tasks match {pattern}"
            )
        protected_paths.extend(matches)
        for path in matches:
            frame = pd.read_parquet(path, columns=["leakage_group_id"])
            protected_groups.update(
                frame["leakage_group_id"].dropna().astype(str)
            )

    base_groups = set(
        base_selected["leakage_group_id"].dropna().astype(str)
    )
    forbidden_groups = base_groups | protected_groups
    promoted = candidates[
        ~candidates["leakage_group_id"].astype(str).isin(forbidden_groups)
    ].copy()
    promoted = promoted.sort_values("sample_id", kind="stable").reset_index(
        drop=True
    )
    actual = {
        "images": len(promoted),
        "groups": promoted["leakage_group_id"].nunique(),
    }
    expected = {
        "images": EXPECTED_PROMOTED_IMAGE_COUNT,
        "groups": EXPECTED_PROMOTED_GROUP_COUNT,
    }
    if actual != expected:
        raise ValueError(
            f"Unexpected internal-reserve promotion: {actual} != {expected}"
        )
    if set(promoted["leakage_group_id"].astype(str)) & protected_groups:
        raise ValueError("Promoted groups overlap a published evaluation")
    if set(promoted["leakage_group_id"].astype(str)) & base_groups:
        raise ValueError("Promoted groups overlap the existing training pool")

    promoted["training_role"] = "in_domain_diagnosis"
    promoted["promotion_policy"] = (
        "internal_test_reserve_unrepresented_group_to_train_v1"
    )
    promoted.attrs["protected_task_paths"] = [
        path.relative_to(root).as_posix() for path in protected_paths
    ]
    return promoted


def build_huggingface_export(
    root: Path,
    *,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    shard_size: int = DEFAULT_SHARD_SIZE,
    limit: int | None = None,
    verify_images: bool = True,
) -> dict[str, Any]:
    """Materialize the private ISEPDermData Parquet release."""

    root = root.resolve()
    source_path = root / input_path
    output = root / output_path
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Move it aside before rebuilding."
        )
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")

    source = pd.read_parquet(source_path)
    base_selected = source[
        (source["training_role"] == "in_domain_diagnosis")
        & ~source["dataset_id"].isin(EXCLUDED_SOURCES)
    ].copy()
    promoted = _promoted_internal_reserve(
        root,
        base_selected=base_selected,
    )
    selected = pd.concat(
        [base_selected, promoted],
        ignore_index=True,
        sort=False,
    )
    selected = selected.sort_values("sample_id", kind="stable").reset_index(drop=True)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected.head(limit).copy()

    if selected["sample_id"].duplicated().any():
        raise ValueError("The selected pool contains duplicate sample_id values")
    if selected["disease_id"].isna().any():
        raise ValueError("Every selected row must have a disease_id")
    if limit is None:
        actual = {
            "images": len(selected),
            "groups": selected["leakage_group_id"].nunique(),
            "classes": selected["disease_id"].nunique(),
        }
        expected = {
            "images": EXPECTED_IMAGE_COUNT,
            "groups": EXPECTED_GROUP_COUNT,
            "classes": EXPECTED_CLASS_COUNT,
        }
        if actual != expected:
            raise ValueError(f"Unexpected diagnosis pool counts: {actual} != {expected}")

    disease_names = _canonical_names(root)
    unknown_ids = sorted(set(selected["disease_id"]) - set(disease_names))
    if unknown_ids:
        raise ValueError(f"Unknown active disease IDs: {unknown_ids}")

    class_frame = (
        selected.groupby("disease_id", as_index=False)
        .agg(
            image_count=("sample_id", "size"),
            group_count=("leakage_group_id", "nunique"),
        )
        .sort_values("disease_id")
    )
    class_frame.insert(
        1,
        "label",
        class_frame["disease_id"].map(disease_names),
    )
    source_frame = (
        selected.groupby("dataset_id", as_index=False)
        .agg(
            image_count=("sample_id", "size"),
            group_count=("leakage_group_id", "nunique"),
            class_count=("disease_id", "nunique"),
        )
        .sort_values("dataset_id")
        .rename(columns={"dataset_id": "source"})
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
    )
    data_directory = temporary / "data"
    metadata_directory = temporary / "metadata"
    data_directory.mkdir()
    metadata_directory.mkdir()

    shard_paths: list[Path] = []
    source_counts: Counter[str] = Counter()
    total_encoded_bytes = 0
    try:
        with ImageResolver(root) as resolver:
            for shard_start in range(0, len(selected), shard_size):
                frame = selected.iloc[shard_start : shard_start + shard_size]
                records = []
                for row in frame.to_dict(orient="records"):
                    record = _export_record(
                        row,
                        resolver=resolver,
                        disease_names=disease_names,
                        verify_images=verify_images,
                    )
                    records.append(record)
                    source_counts[record["source"]] += 1
                    total_encoded_bytes += len(record["image"]["bytes"])

                shard_index = len(shard_paths)
                shard_count = (len(selected) + shard_size - 1) // shard_size
                shard_path = data_directory / (
                    f"train-{shard_index:05d}-of-{shard_count:05d}.parquet"
                )
                dataset = Dataset.from_list(records, features=EXPORT_FEATURES)
                dataset.to_parquet(shard_path)
                shard_paths.append(shard_path)
                print(
                    f"Wrote shard {shard_index + 1}/{shard_count}: "
                    f"{len(records)} images"
                )

        class_frame.to_csv(metadata_directory / "class_distribution.csv", index=False)
        source_frame.to_csv(
            metadata_directory / "source_distribution.csv",
            index=False,
        )
        (
            promoted.groupby(
                ["dataset_id", "leakage_group_id"],
                as_index=False,
            )
            .agg(
                image_count=("sample_id", "size"),
                class_count=("disease_id", "nunique"),
            )
            .sort_values(["dataset_id", "leakage_group_id"])
            .to_csv(
                metadata_directory / "promoted_reserve_groups.csv",
                index=False,
            )
        )
        (metadata_directory / "taxonomy.json").write_text(
            json.dumps(
                {
                    "taxonomy_id": "dermatology_diseases",
                    "active_class_count": len(disease_names),
                    "classes": [
                        {"disease_id": disease_id, "label": label}
                        for disease_id, label in disease_names.items()
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (metadata_directory / "source_licenses.json").write_text(
            json.dumps(_source_licenses(), indent=2) + "\n",
            encoding="utf-8",
        )

        shard_metadata = [
            {
                "path": path.relative_to(temporary).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
                "rows": pq.ParquetFile(path).metadata.num_rows,
            }
            for path in shard_paths
        ]
        release = {
            "release": {
                "id": "ISEPDermData",
                "version": EXPORT_RELEASE_VERSION,
                "schema_version": EXPORT_SCHEMA_VERSION,
                "visibility": "private",
                "split": "train",
                "image_count": len(selected),
                "leakage_group_count": selected["leakage_group_id"].nunique(),
                "class_count": selected["disease_id"].nunique(),
                "source_count": selected["dataset_id"].nunique(),
                "embedded_image_bytes": total_encoded_bytes,
                "source_manifest": input_path.as_posix(),
                "source_manifest_sha256": _file_sha256(source_path),
                "selection": (
                    "training_role == 'in_domain_diagnosis' and "
                    "dataset_id not in {'derm1m'}, plus 63 wholly "
                    "unrepresented internal-reserve groups"
                ),
                "internal_reserve_promotion": {
                    "policy": (
                        "internal_test_reserve_unrepresented_group_to_train_v1"
                    ),
                    "image_count": len(promoted),
                    "group_count": promoted[
                        "leakage_group_id"
                    ].nunique(),
                    "source_counts": {
                        str(key): int(value)
                        for key, value in promoted["dataset_id"]
                        .value_counts()
                        .sort_index()
                        .items()
                    },
                    "protected_evaluation_tasks": promoted.attrs.get(
                        "protected_task_paths",
                        [],
                    ),
                    "audit_file": (
                        "metadata/promoted_reserve_groups.csv"
                    ),
                },
                "excluded_sources": {
                    "derm1m": (
                        "Removed after label-quality audit: source-derived "
                        "entity-linking errors and context-dependent images."
                    )
                },
                "columns": EXPORT_COLUMNS,
                "shard_size": shard_size,
                "shards": shard_metadata,
            }
        }
        (temporary / "release.json").write_text(
            json.dumps(release, indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            _dataset_card(
                image_count=len(selected),
                group_count=selected["leakage_group_id"].nunique(),
                class_count=selected["disease_id"].nunique(),
                source_counts=dict(sorted(source_counts.items())),
                class_counts=class_frame.to_dict(orient="records"),
            ),
            encoding="utf-8",
        )

        temporary.rename(output)
    except Exception:
        print(f"Incomplete export retained for inspection at: {temporary}")
        raise

    return release["release"]


def validate_huggingface_export(
    root: Path,
    *,
    output_path: Path = DEFAULT_OUTPUT,
    expect_full_release: bool = True,
) -> dict[str, int]:
    """Validate row identities, schema metadata, counts, and shard checksums."""

    output = root.resolve() / output_path
    release = json.loads((output / "release.json").read_text(encoding="utf-8"))[
        "release"
    ]
    shards = sorted((output / "data").glob("train-*.parquet"))
    if not shards:
        raise ValueError("No Parquet shards were found")
    expected_shards = [output / item["path"] for item in release["shards"]]
    if shards != expected_shards:
        raise ValueError("Materialized shards differ from release.json")

    sample_ids: set[str] = set()
    groups: set[str] = set()
    disease_ids: set[str] = set()
    sources: set[str] = set()
    total_rows = 0
    for shard, metadata in zip(shards, release["shards"], strict=True):
        if _file_sha256(shard) != metadata["sha256"]:
            raise ValueError(f"Shard checksum mismatch: {shard}")
        parquet = pq.ParquetFile(shard)
        if parquet.metadata.num_rows != metadata["rows"]:
            raise ValueError(f"Shard row count mismatch: {shard}")
        table = pq.read_table(
            shard,
            columns=[
                "sample_id",
                "leakage_group_id",
                "disease_id",
                "source",
            ],
        ).to_pandas()
        overlap = sample_ids.intersection(table["sample_id"])
        if overlap:
            raise ValueError(f"Duplicate sample IDs across shards: {sorted(overlap)[:3]}")
        sample_ids.update(table["sample_id"])
        groups.update(table["leakage_group_id"])
        disease_ids.update(table["disease_id"])
        sources.update(table["source"])
        total_rows += len(table)

        huggingface_metadata = parquet.schema_arrow.metadata or {}
        if b"huggingface" not in huggingface_metadata:
            raise ValueError(f"Missing Hugging Face schema metadata: {shard}")

    first_image = pq.read_table(shards[0], columns=["image"])[0][0].as_py()
    if not first_image or not first_image.get("bytes") or not first_image.get("path"):
        raise ValueError("The first embedded image is incomplete")
    _validate_encoded_image(first_image["bytes"], sample_id="first_exported_image")

    actual = {
        "images": total_rows,
        "groups": len(groups),
        "classes": len(disease_ids),
        "sources": len(sources),
        "shards": len(shards),
    }
    release_counts = {
        "images": int(release["image_count"]),
        "groups": int(release["leakage_group_count"]),
        "classes": int(release["class_count"]),
        "sources": int(release["source_count"]),
        "shards": len(release["shards"]),
    }
    if actual != release_counts:
        raise ValueError(f"Export counts differ from release: {actual} != {release_counts}")
    if expect_full_release:
        expected = {
            "images": EXPECTED_IMAGE_COUNT,
            "groups": EXPECTED_GROUP_COUNT,
            "classes": EXPECTED_CLASS_COUNT,
        }
        comparable = {key: actual[key] for key in expected}
        if comparable != expected:
            raise ValueError(f"Unexpected full release counts: {comparable} != {expected}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or validate the private ISEPDermData release."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-image-decode-check", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--allow-partial-validation",
        action="store_true",
        help="Do not require the fixed full-release counts.",
    )
    args = parser.parse_args()

    if args.validate_only:
        result = validate_huggingface_export(
            args.project_root,
            output_path=args.output,
            expect_full_release=not args.allow_partial_validation,
        )
    else:
        result = build_huggingface_export(
            args.project_root,
            input_path=args.input,
            output_path=args.output,
            shard_size=args.shard_size,
            limit=args.limit,
            verify_images=not args.skip_image_decode_check,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
