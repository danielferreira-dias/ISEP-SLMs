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
)
from src.data_pipeline.common import DiseaseMapper, load_yaml, write_manifest
from src.data_pipeline.reporting import build_reports, write_combined_pool


SUPPORTED_BUILDERS: dict[
    str,
    Callable[[Path, dict[str, Any], DiseaseMapper, Callable[[str], None]], list[dict[str, Any]]],
] = {
    "fitzpatrick17k_c": build_fitzpatrick17k_c,
    "pad_ufes_20": build_pad_ufes_20,
    "scin": build_scin,
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
    reports: dict[str, Path] = {}
    combined_path = root / "data/combined/visual_top_k_development_pool_v2.parquet"
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
        "combined_path": combined_path if combined_rows is not None else None,
        "combined_rows": combined_rows,
        "reports": reports,
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
    for path in paths:
        table = pq.read_table(path)
        frame = table.select(
            [
                "sample_id",
                "group_id",
                "canonical_source_label",
                "disease_id",
                "reference_diagnoses",
                "mapping_status",
                "include",
            ]
        ).to_pandas()
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate sample_id values in {path}")
        overlap = sample_ids.intersection(frame["sample_id"])
        if overlap:
            raise ValueError(f"Cross-manifest sample_id collision: {sorted(overlap)[:3]}")
        sample_ids.update(frame["sample_id"])
        if frame["group_id"].isna().any():
            raise ValueError(f"Null group_id values in {path}")
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
