"""Aggregate-only audit of SkinCAP observation candidates."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pyarrow.parquet as pq

from src.train.e2.skincap.domain import SkinCapAuditReport, SkinCapTransformPolicy
from src.train.e2.skincap.transform import transform_caption


@dataclass(frozen=True, slots=True)
class SkinCapAuditPaths:
    """Pinned local inputs needed to reproduce the aggregate audit."""

    metadata: Path = Path("configs/datasets/skincap/data/skincap_v240715.xlsx")
    fitz_manifest: Path = Path("data/manifests/fitzpatrick17k_c_v3.parquet")
    ddi_metadata: Path = Path("configs/datasets/ddi/data/ddi_metadata.csv")
    benchmark_references: Path = Path("data/benchmarks/ISEPDermaBench/references")


DEFAULT_AUDIT_PATHS = SkinCapAuditPaths()
DEFAULT_TRANSFORM_POLICY = SkinCapTransformPolicy()


def audit_skincap_observations(
    paths: SkinCapAuditPaths = DEFAULT_AUDIT_PATHS,
    policy: SkinCapTransformPolicy = DEFAULT_TRANSFORM_POLICY,
) -> SkinCapAuditReport:
    """Audit every real caption without persisting clinical or derived text."""

    candidates, eligibility = load_skincap_candidate_frame(paths)
    results = tuple(
        transform_caption(str(row.caption_en), str(row.disease), policy)
        for row in candidates.itertuples(index=False)
    )
    accepted_mask = tuple(result.accepted for result in results)
    accepted = candidates.loc[list(accepted_mask)]
    words = sorted(result.word_count for result in results if result.accepted)
    if not words:
        raise ValueError("SkinCAP transform accepted no observation candidates")
    boundary_counts = Counter(result.boundary_kind.value for result in results)
    rejection_counts = Counter(
        reason.value for result in results for reason in result.rejection_reasons
    )
    return SkinCapAuditReport(
        transform_version=policy.version,
        caption_variant=policy.caption_variant,
        metadata_sha256=_sha256(paths.metadata),
        downloaded_rows=eligibility.downloaded_rows,
        author_excluded_rows=eligibility.author_excluded_rows,
        usable_before_leakage_rows=eligibility.usable_before_leakage_rows,
        frozen_validation_overlap_rows=eligibility.frozen_validation_overlap_rows,
        frozen_internal_overlap_rows=eligibility.frozen_internal_overlap_rows,
        technical_candidate_rows=len(candidates),
        technical_candidate_groups=int(candidates["leakage_group_id"].nunique()),
        accepted_observation_rows=len(accepted),
        rejected_observation_rows=len(candidates) - len(accepted),
        accepted_by_source=tuple(
            (str(key), int(value))
            for key, value in accepted["source"].value_counts().sort_index().items()
        ),
        boundary_counts=tuple(sorted(boundary_counts.items())),
        rejection_counts=tuple(sorted(rejection_counts.items())),
        observation_word_min=words[0],
        observation_word_median=float(statistics.median(words)),
        observation_word_p95=_percentile(words, 0.95),
        observation_word_max=words[-1],
    )


@dataclass(frozen=True, slots=True)
class SkinCapEligibilityCounts:
    """Aggregate counts associated with the reusable candidate frame."""

    downloaded_rows: int
    author_excluded_rows: int
    usable_before_leakage_rows: int
    frozen_validation_overlap_rows: int
    frozen_internal_overlap_rows: int


def load_skincap_candidate_frame(
    paths: SkinCapAuditPaths = DEFAULT_AUDIT_PATHS,
) -> tuple[pd.DataFrame, SkinCapEligibilityCounts]:
    """Load technical candidates after source and frozen leakage exclusions.

    This function deliberately returns the gated clinical text only to local
    pipeline code. Callers must not log the frame or persist raw captions in
    trainer-visible artifacts.
    """

    _require_inputs(paths)
    metadata = pd.read_excel(paths.metadata, header=1)
    _validate_metadata(metadata)
    joined = _attach_leakage_groups(metadata, paths)
    validation_groups = _reserved_groups(paths.benchmark_references, "validation")
    internal_groups = _reserved_groups(paths.benchmark_references, "internal_benchmark")
    overlap = validation_groups & internal_groups
    if overlap:
        raise ValueError("Frozen Validation and Internal groups overlap")

    usable = joined.loc[joined["Do not consider this image"].eq(0)].copy()
    validation_mask = usable["leakage_group_id"].isin(validation_groups)
    internal_mask = usable["leakage_group_id"].isin(internal_groups)
    candidates = usable.loc[~validation_mask & ~internal_mask].copy()
    counts = SkinCapEligibilityCounts(
        downloaded_rows=len(joined),
        author_excluded_rows=int(joined["Do not consider this image"].ne(0).sum()),
        usable_before_leakage_rows=len(usable),
        frozen_validation_overlap_rows=int(validation_mask.sum()),
        frozen_internal_overlap_rows=int(internal_mask.sum()),
    )
    return candidates, counts


def write_audit_report(report: SkinCapAuditReport, output: Path) -> None:
    """Atomically write aggregate metrics while excluding all clinical text."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    payload = json.dumps(
        report.as_record(),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    temporary.write_text(payload + "\n", encoding="utf-8")
    os.replace(temporary, output)


def _attach_leakage_groups(
    metadata: pd.DataFrame, paths: SkinCapAuditPaths
) -> pd.DataFrame:
    manifest = pq.read_table(  # type: ignore[no-untyped-call]
        paths.fitz_manifest,
        columns=["original_image_id", "leakage_group_id"],
    ).to_pandas()
    manifest["ori_file_path"] = manifest["original_image_id"].astype(str) + ".jpg"
    fitz = metadata.loc[metadata["source"].eq("fitzpatrick17k")].merge(
        manifest[["ori_file_path", "leakage_group_id"]],
        on="ori_file_path",
        how="left",
        validate="one_to_one",
    )
    fitz["leakage_group_id"] = fitz["leakage_group_id"].fillna(
        "FITZPATRICK17K_IMAGE_"
        + fitz["ori_file_path"].astype(str).str.removesuffix(".jpg")
    )
    ddi_metadata = pd.read_csv(paths.ddi_metadata)
    ddi = metadata.loc[metadata["source"].eq("ddi")].merge(
        ddi_metadata[["DDI_file", "DDI_ID"]],
        left_on="ori_file_path",
        right_on="DDI_file",
        validate="one_to_one",
    )
    ddi["leakage_group_id"] = "DDI_IMAGE_" + ddi["DDI_ID"].astype(str)
    joined = pd.concat((fitz, ddi), ignore_index=True)
    if len(joined) != len(metadata) or joined["leakage_group_id"].isna().any():
        raise ValueError("SkinCAP upstream join lost rows or leakage groups")
    return joined


def _reserved_groups(root: Path, scope: str) -> set[str]:
    groups: set[str] = set()
    for path in sorted(root.rglob(f"*{scope}*.parquet")):
        names = set(
            pq.ParquetFile(path).schema_arrow.names  # type: ignore[no-untyped-call]
        )
        if not {"source", "leakage_group_id"}.issubset(names):
            continue
        frame = pq.read_table(  # type: ignore[no-untyped-call]
            path, columns=["source", "leakage_group_id"]
        ).to_pandas()
        groups.update(
            frame.loc[
                frame["source"].eq("fitzpatrick17k_c"), "leakage_group_id"
            ].astype(str)
        )
    if not groups:
        raise ValueError(f"No frozen Fitzpatrick groups found for {scope}")
    return groups


def _validate_metadata(frame: pd.DataFrame) -> None:
    required = {
        "id",
        "ori_file_path",
        "disease",
        "caption_en",
        "source",
        "Do not consider this image",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"SkinCAP metadata columns are missing: {sorted(missing)}")
    if len(frame) != 4_000 or frame["id"].nunique() != 4_000:
        raise ValueError("SkinCAP metadata must contain exactly 4,000 unique rows")
    if frame["caption_en"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("SkinCAP contains an empty English caption")
    if set(frame["source"].astype(str)) != {"fitzpatrick17k", "ddi"}:
        raise ValueError("SkinCAP contains an unexpected source dataset")


def _require_inputs(paths: SkinCapAuditPaths) -> None:
    for path in (
        paths.metadata,
        paths.fitz_manifest,
        paths.ddi_metadata,
        paths.benchmark_references,
    ):
        if not path.exists():
            raise FileNotFoundError(f"SkinCAP audit input is missing: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[int], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile without values")
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction
