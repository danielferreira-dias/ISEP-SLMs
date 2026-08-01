"""Command-line pipeline for normalized manifests and disease coverage."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as pq

from src.data_pipeline.adapters import (
    build_ddi,
    build_fitzpatrick17k_c,
    build_pad_ufes_20,
    build_scin,
    build_skindisnet,
)
from src.data_pipeline.common import (
    MANIFEST_ARROW_SCHEMA,
    MANIFEST_SCHEMA_VERSION,
    DiseaseMapper,
    load_yaml,
    write_manifest,
)
from src.data_pipeline.deduplication import deduplicate_manifests
from src.data_pipeline.confusion_sets import (
    build_confusion_set_release,
    validate_confusion_set_release,
)
from src.data_pipeline.reporting import build_reports, write_combined_pool
from src.data_pipeline.splitting import (
    build_benchmark_release,
    validate_benchmark_release,
)


SUPPORTED_BUILDERS: dict[
    str,
    Callable[[Path, dict[str, Any], DiseaseMapper, Callable[[str], None]], list[dict[str, Any]]],
] = {
    "fitzpatrick17k_c": build_fitzpatrick17k_c,
    "pad_ufes_20": build_pad_ufes_20,
    "scin": build_scin,
    "skindisnet": build_skindisnet,
    "ddi": build_ddi,
}


def build_pipeline(root: Path, selected_ids: list[str] | None = None) -> dict[str, Any]:
    """Build selected manifests and all reports when contributors are available."""

    catalog_document = load_yaml(root / "configs/datasets/catalog.yaml")
    policy_document = load_yaml(root / "configs/datasets/disease_inclusion.yaml")
    policy = policy_document["disease_inclusion"] | {
        key: value
        for key, value in policy_document.items()
        if key != "disease_inclusion"
    }
    mapper = DiseaseMapper(
        root / "configs/taxonomies/diseases.yaml",
        root / "configs/taxonomies/source_disease_mappings.yaml",
    )

    catalog_entries = {
        entry["id"]: entry
        for entry in catalog_document["datasets"]
        if entry["id"] in SUPPORTED_BUILDERS
    }
    requested = selected_ids or list(SUPPORTED_BUILDERS)
    unknown = sorted(set(requested) - set(catalog_entries))
    if unknown:
        raise ValueError(f"Unsupported dataset IDs: {', '.join(unknown)}")

    manifest_paths: dict[str, Path] = {}
    row_counts: dict[str, int] = {}
    for dataset_id in requested:
        entry = catalog_entries[dataset_id]
        config = load_yaml(root / entry["config"])
        output_path = root / config["manifest"]["output"]
        print(f"[{dataset_id}] Building {output_path.relative_to(root)}", flush=True)
        rows = SUPPORTED_BUILDERS[dataset_id](root, config, mapper, _progress)
        row_counts[dataset_id] = write_manifest(rows, output_path)
        manifest_paths[dataset_id] = output_path
        print(f"[{dataset_id}] Wrote {row_counts[dataset_id]} rows", flush=True)

    contributor_ids = policy["dataset_roles"]["taxonomy_contributors"]
    deduplication = deduplicate_manifests(
        manifest_paths=manifest_paths,
        root=root,
        policy=policy,
        progress=_progress,
    )
    print(
        "[deduplication] "
        f"{deduplication['pair_count']} candidate pairs, "
        f"{deduplication['group_count']} duplicate groups, "
        f"{deduplication['exact_exclusion_count']} newly excluded rows",
        flush=True,
    )
    for name, path in deduplication["paths"].items():
        print(f"[deduplication] {name}: {path.relative_to(root)}", flush=True)

    reports: dict[str, Path] = {}
    benchmark_release: dict[str, Any] | None = None
    confusion_set_release: dict[str, Any] | None = None
    combined_path = root / "data/combined/visual_top_k_development_pool_v3.parquet"
    combined_rows: int | None = None
    if all(dataset_id in manifest_paths for dataset_id in contributor_ids):
        combined_rows = write_combined_pool(
            manifest_paths,
            contributor_ids,
            combined_path,
        )
        reports = build_reports(
            manifest_paths=manifest_paths,
            contributor_ids=contributor_ids,
            mapper=mapper,
            policy=policy,
            root=root,
        )
        print(
            f"[combined] Wrote {combined_rows} rows to {combined_path.relative_to(root)}",
            flush=True,
        )
        for name, path in reports.items():
            print(f"[report] {name}: {path.relative_to(root)}", flush=True)
        split_document = load_yaml(
            root / "configs/datasets/visual_top_k_split.yaml"
        )
        required_release_ids = set(
            split_document["split"]["internal"]["source_datasets"]
        ) | set(split_document["split"]["external"]["datasets"])
        if required_release_ids.issubset(manifest_paths):
            benchmark_release = build_benchmark_release(
                root=root,
                manifest_paths=manifest_paths,
            )
            print(
                "[benchmark-release] "
                f"{len(benchmark_release['assignments'])} internal leakage "
                "groups assigned",
                flush=True,
            )
            for name, path in benchmark_release["paths"].items():
                print(
                    f"[benchmark-release] {name}: "
                    f"{path.relative_to(root)}",
                    flush=True,
                )
            confusion_set_release = build_confusion_set_release(root)
            print(
                "[confusion-set-release] "
                f"{confusion_set_release['integrity']['pair_count']} pairs, "
                f"{confusion_set_release['integrity']['task_count']} tasks",
                flush=True,
            )
            for name, path in confusion_set_release["paths"].items():
                print(
                    f"[confusion-set-release] {name}: "
                    f"{path.relative_to(root)}",
                    flush=True,
                )
        else:
            missing_release_ids = sorted(
                required_release_ids - set(manifest_paths)
            )
            print(
                "[benchmark-release] Skipped because manifests were not "
                "rebuilt: "
                + ", ".join(missing_release_ids),
                flush=True,
            )
    else:
        missing = sorted(set(contributor_ids) - set(manifest_paths))
        print(
            "[reports] Skipped because contributor manifests were not all rebuilt: "
            + ", ".join(missing),
            flush=True,
        )

    return {
        "manifest_paths": manifest_paths,
        "row_counts": row_counts,
        "deduplication": deduplication,
        "combined_path": combined_path if combined_rows is not None else None,
        "combined_rows": combined_rows,
        "reports": reports,
        "benchmark_release": benchmark_release,
        "confusion_set_release": confusion_set_release,
    }


def validate_outputs(root: Path) -> None:
    """Validate the generated manifest invariants that do not require image decoding."""

    catalog = load_yaml(root / "configs/datasets/catalog.yaml")
    paths: list[Path] = []
    for entry in catalog["datasets"]:
        if entry["id"] not in SUPPORTED_BUILDERS:
            continue
        config = load_yaml(root / entry["config"])
        path = root / config["manifest"]["output"]
        if path.exists():
            paths.append(path)

    sample_ids: set[str] = set()
    duplicate_to_leakage_group: dict[str, str] = {}
    for path in paths:
        table = pq.read_table(path)
        frame = table.select(
            [
                "schema_version",
                "sample_id",
                "group_id",
                "leakage_group_id",
                "canonical_source_label",
                "disease_id",
                "reference_diagnoses",
                "mapping_status",
                "image_sha256",
                "perceptual_hash",
                "perceptual_hash_algorithm",
                "duplicate_group_id",
                "deduplication_status",
                "include",
                "exclusion_reason",
            ]
        ).to_pandas()
        if table.schema.names != MANIFEST_ARROW_SCHEMA.names:
            raise ValueError(f"Manifest schema-column mismatch in {path}")
        if not frame["schema_version"].eq(MANIFEST_SCHEMA_VERSION).all():
            raise ValueError(f"Manifest schema-version mismatch in {path}")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate sample_id values in {path}")
        overlap = sample_ids.intersection(frame["sample_id"])
        if overlap:
            raise ValueError(f"Cross-manifest sample_id collision: {sorted(overlap)[:3]}")
        sample_ids.update(frame["sample_id"])
        if frame["group_id"].isna().any():
            raise ValueError(f"Null group_id values in {path}")
        if frame["leakage_group_id"].isna().any():
            raise ValueError(f"Null leakage_group_id values in {path}")
        inconsistent_source_groups = (
            frame.groupby("group_id")["leakage_group_id"].nunique() > 1
        )
        if inconsistent_source_groups.any():
            raise ValueError(
                f"Source group mapped to multiple leakage groups in {path}"
            )
        duplicate_rows = frame[frame["duplicate_group_id"].notna()]
        inconsistent_duplicate_groups = (
            duplicate_rows.groupby("duplicate_group_id")[
                "leakage_group_id"
            ].nunique()
            > 1
        )
        if inconsistent_duplicate_groups.any():
            raise ValueError(
                f"Duplicate group mapped to multiple leakage groups in {path}"
            )
        for duplicate_group_id, leakage_group_id in (
            duplicate_rows[
                ["duplicate_group_id", "leakage_group_id"]
            ]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ):
            previous = duplicate_to_leakage_group.setdefault(
                duplicate_group_id,
                leakage_group_id,
            )
            if previous != leakage_group_id:
                raise ValueError(
                    "Cross-manifest duplicate group mapped to multiple "
                    f"leakage groups: {duplicate_group_id}"
                )
        if frame["image_sha256"].str.fullmatch(r"[0-9a-f]{64}").ne(True).any():
            raise ValueError(f"Invalid or missing SHA-256 values in {path}")
        if frame["perceptual_hash"].str.fullmatch(r"[0-9a-f]{16}").ne(True).any():
            raise ValueError(f"Invalid or missing perceptual hashes in {path}")
        if frame["perceptual_hash_algorithm"].isna().any():
            raise ValueError(f"Missing perceptual-hash algorithm in {path}")
        exact_exclusions = frame["deduplication_status"].isin(
            ["redundant_exact", "exact_label_conflict"]
        )
        if frame.loc[exact_exclusions, "include"].any():
            raise ValueError(f"Included exact duplicate exclusion in {path}")
        if frame.loc[exact_exclusions, "exclusion_reason"].isna().any():
            raise ValueError(f"Exact duplicate exclusion without reason in {path}")
        invalid_include = frame["include"] & frame["disease_id"].isna()
        if invalid_include.any():
            raise ValueError(f"Included rows without disease_id in {path}")
        for record in frame.to_dict(orient="records"):
            references = record["reference_diagnoses"]
            if len(references) > 0:
                first_id = references[0]["disease_id"]
                record_id = record["disease_id"]
                if record_id != record_id:
                    record_id = None
                if record_id != first_id:
                    raise ValueError(f"Primary-reference mismatch in {path}")
                if (
                    record["canonical_source_label"]
                    != references[0]["canonical_source_label"]
                ):
                    raise ValueError(f"Canonical-label mismatch in {path}")
                if record["mapping_status"] != references[0]["mapping_status"]:
                    raise ValueError(f"Mapping-status mismatch in {path}")
                ranks = [int(item["rank"]) for item in references]
                if ranks != list(range(1, len(ranks) + 1)):
                    raise ValueError(f"Non-consecutive reference ranks in {path}")
        print(f"[validate] {path.relative_to(root)}: {len(frame)} rows", flush=True)

    print(f"[validate] {len(sample_ids)} globally unique samples", flush=True)
    split_document = load_yaml(
        root / "configs/datasets/visual_top_k_split.yaml"
    )
    release_path = (
        root
        / split_document["split"]["outputs"]["directory"]
        / split_document["split"]["outputs"]["release_manifest"]
    )
    if release_path.exists():
        release = validate_benchmark_release(root)
        print(
            f"[validate] benchmark release {release['id']} "
            f"version {release['version']}",
            flush=True,
        )
    confusion_benchmark = load_yaml(
        root / "configs/benchmarks/derma_isep/visual_confusion_sets.yaml"
    )
    confusion_release_path = (
        root / confusion_benchmark["dataset"]["release_manifest"]
    )
    if confusion_release_path.exists():
        confusion_release = validate_confusion_set_release(root)
        print(
            f"[validate] confusion-set release {confusion_release['id']} "
            f"version {confusion_release['version']}",
            flush=True,
        )


def _progress(message: str) -> None:
    print(f"  {message}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized dermatology manifests and coverage reports."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(SUPPORTED_BUILDERS),
        help="Optional subset to rebuild. Reports require all taxonomy contributors.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing manifests without rebuilding them.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    if not args.validate_only:
        build_pipeline(root, args.datasets)
    validate_outputs(root)


if __name__ == "__main__":
    main()
