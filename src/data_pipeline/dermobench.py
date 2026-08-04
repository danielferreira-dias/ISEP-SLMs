"""Prepare and validate the local DermoBench release.

The upstream annotations keep their original relative image paths.  In the
released image archive, a subset of the Derm1M-EDU filenames was normalized
differently.  This module extracts the archive and creates an explicit index
from every annotation path to its extracted image, without modifying the
frozen annotation files.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import unicodedata
import zipfile


DEFAULT_RELEASE_ROOT = Path("data/benchmarks/DermoBench/release")
ARCHIVE_NAME = "dermobench_release_imgs.zip"
IMAGE_DIRECTORY_NAME = "images"
INDEX_NAME = "image_index.json"


@dataclass(frozen=True, slots=True)
class AnnotationFile:
    """One immutable annotation file and its referenced image paths."""

    path: Path
    rows: int
    image_paths: frozenset[str]


def prepare_dermobench(
    *,
    release_root: Path,
    extract: bool,
) -> dict[str, object]:
    """Validate the release and optionally extract it for model evaluation."""

    release_root = release_root.resolve()
    archive_path = release_root / ARCHIVE_NAME
    if not archive_path.is_file():
        raise FileNotFoundError(f"DermoBench image archive is missing: {archive_path}")

    annotation_files = _load_annotation_files(release_root)
    annotation_paths = set().union(
        *(file.image_paths for file in annotation_files)
    )
    with zipfile.ZipFile(archive_path) as archive:
        archive_paths = _archive_image_paths(archive)
        image_index, resolution_counts = _build_image_index(
            annotation_paths=annotation_paths,
            archive_paths=archive_paths,
        )
        if extract:
            _extract_images(archive=archive, output_root=release_root / IMAGE_DIRECTORY_NAME)

    image_root = release_root / IMAGE_DIRECTORY_NAME
    images_available = image_root.is_dir()
    if images_available:
        _validate_extracted_images(
            image_root=image_root,
            image_index=image_index,
        )

    index_path = release_root / INDEX_NAME
    manifest: dict[str, object] = {
        "schema_version": 1,
        "benchmark": "DermoBench",
        "archive": {
            "path": ARCHIVE_NAME,
            "sha256": _file_sha256(archive_path),
            "image_root": IMAGE_DIRECTORY_NAME,
            "image_count": len(archive_paths),
        },
        "annotations": {
            "file_count": len(annotation_files),
            "row_count": sum(file.rows for file in annotation_files),
            "unique_image_references": len(annotation_paths),
            "files": [
                {
                    "path": str(file.path.relative_to(release_root)),
                    "rows": file.rows,
                    "unique_image_references": len(file.image_paths),
                    "sha256": _file_sha256(file.path),
                }
                for file in annotation_files
            ],
        },
        "resolution": {
            "exact": resolution_counts["exact"],
            "normalized_filename": resolution_counts["normalized_filename"],
            "suffix_and_similarity": resolution_counts["suffix_and_similarity"],
        },
        "images_available": images_available,
        "image_paths": dict(sorted(image_index.items())),
    }
    index_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_annotation_files(release_root: Path) -> tuple[AnnotationFile, ...]:
    """Read every checked-in JSON/JSONL annotation file."""

    paths = sorted(
        path
        for path in release_root.rglob("*")
        if path.suffix in {".json", ".jsonl"} and path.name != INDEX_NAME
    )
    if not paths:
        raise FileNotFoundError(f"No DermoBench annotations found in {release_root}")
    files: list[AnnotationFile] = []
    for path in paths:
        if path.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Annotation file must contain a list of rows: {path}")
        try:
            image_paths = frozenset(str(row["image"]) for row in rows)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid DermoBench annotation row in {path}") from exc
        if not image_paths or any(not value for value in image_paths):
            raise ValueError(f"Annotation file has an empty image path: {path}")
        files.append(AnnotationFile(path=path, rows=len(rows), image_paths=image_paths))
    return tuple(files)


def _archive_image_paths(archive: zipfile.ZipFile) -> set[str]:
    """Return safe relative image paths stored below the archive's ``imgs/`` root."""

    paths: set[str] = set()
    for member in archive.infolist():
        name = member.filename
        if not name.startswith("imgs/") or name.endswith("/"):
            continue
        relative = name.removeprefix("imgs/")
        _validate_relative_path(relative)
        paths.add(relative)
    if not paths:
        raise ValueError("DermoBench archive contains no image files below imgs/")
    return paths


def _build_image_index(
    *,
    annotation_paths: set[str],
    archive_paths: set[str],
) -> tuple[dict[str, str], Counter[str]]:
    """Resolve annotation image names to archive names deterministically."""

    normalized_index: dict[str, list[str]] = defaultdict(list)
    suffix_index: dict[str, list[str]] = defaultdict(list)
    for archive_path in archive_paths:
        normalized_index[_normalize_path(archive_path)].append(archive_path)
        suffix = _page_suffix(archive_path)
        if suffix is not None:
            suffix_index[suffix].append(archive_path)

    mapping: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for annotation_path in sorted(annotation_paths):
        _validate_relative_path(annotation_path)
        if annotation_path in archive_paths:
            mapping[annotation_path] = annotation_path
            counts["exact"] += 1
            continue
        normalized_matches = normalized_index[_normalize_path(annotation_path)]
        if len(normalized_matches) == 1:
            mapping[annotation_path] = normalized_matches[0]
            counts["normalized_filename"] += 1
            continue
        suffix = _page_suffix(annotation_path)
        suffix_matches = suffix_index[suffix] if suffix is not None else []
        if not suffix_matches:
            raise ValueError(f"No archive image matches annotation path: {annotation_path}")
        # Page suffixes identify every remaining path; two coincident page
        # numbers occur in this release.  Similarity over normalized names
        # resolves those deterministically and is checked for a tie.
        scored = sorted(
            (
                SequenceMatcher(
                    None,
                    _normalize_path(annotation_path),
                    _normalize_path(candidate),
                ).ratio(),
                candidate,
            )
            for candidate in suffix_matches
        )
        best_score, best_candidate = scored[-1]
        if len(scored) > 1 and scored[-2][0] == best_score:
            raise ValueError(f"Ambiguous archive image match: {annotation_path}")
        mapping[annotation_path] = best_candidate
        counts["suffix_and_similarity"] += 1
    if len(mapping) != len(annotation_paths):
        raise ValueError("DermoBench image index is incomplete")
    return mapping, counts


def _extract_images(*, archive: zipfile.ZipFile, output_root: Path) -> None:
    """Extract archive files below ``images/`` while rejecting unsafe paths."""

    output_root.mkdir(parents=True, exist_ok=True)
    for member in archive.infolist():
        if not member.filename.startswith("imgs/") or member.is_dir():
            continue
        relative = member.filename.removeprefix("imgs/")
        _validate_relative_path(relative)
        output_path = output_root / relative
        if output_path.exists() and output_path.stat().st_size == member.file_size:
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, output_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)


def _validate_extracted_images(*, image_root: Path, image_index: dict[str, str]) -> None:
    """Ensure every annotation reference now resolves to a regular file."""

    missing = [
        annotation_path
        for annotation_path, archive_path in image_index.items()
        if not (image_root / archive_path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Extracted DermoBench images are incomplete; first missing path: "
            f"{missing[0]}"
        )


def _normalize_path(value: str) -> str:
    """Canonicalize filename punctuation differences without discarding letters."""

    normalized = unicodedata.normalize("NFKD", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _page_suffix(value: str) -> str | None:
    """Use the source page/image identifier retained across filename variants."""

    name = PurePosixPath(value).name
    match = re.search(r"(_\d{5}_\d{5}(?:_\d+)?\.[A-Za-z0-9]+)$", name)
    return match.group(1).casefold() if match else None


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise ValueError(f"Unsafe relative image path: {value!r}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and extract the local DermoBench release."
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        default=DEFAULT_RELEASE_ROOT,
        help=f"DermoBench release directory (default: {DEFAULT_RELEASE_ROOT})",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract images into <release-root>/images before validating them.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_dermobench(
        release_root=args.release_root,
        extract=args.extract,
    )
    resolution = manifest["resolution"]
    annotations = manifest["annotations"]
    assert isinstance(resolution, dict)
    assert isinstance(annotations, dict)
    print(
        "DermoBench ready: "
        f"{annotations['row_count']} tasks, "
        f"{annotations['unique_image_references']} referenced images, "
        f"{resolution['exact']} exact paths, "
        f"{resolution['normalized_filename']} normalized paths, "
        f"{resolution['suffix_and_similarity']} suffix-resolved paths."
    )
    print(f"Image index: {(args.release_root / INDEX_NAME).resolve()}")
    if not bool(manifest["images_available"]):
        print("Images were not extracted; rerun with --extract before inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
