"""Smoke-load ISEPDistillDataset from the Hub and print one row.

Reads ``HF_TOKEN`` from the process environment or the repo ``.env``.

Run from the repository root:

    uv run python project/dataset/load.py
    uv run python project/dataset/load.py --config diagnosis --split sft_train
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


def _ensure_repo_on_path() -> None:
    """Put the repository root on ``sys.path`` so ``project.dataset`` imports."""

    repo_root = Path(__file__).resolve().parents[2]
    root = str(repo_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse config/split overrides for the smoke load."""

    parser = argparse.ArgumentParser(
        description="Load ISEPDistillDataset and print the first row.",
    )
    parser.add_argument(
        "--config",
        default="diagnosis",
        help="Hub config name. Default: diagnosis.",
    )
    parser.add_argument(
        "--split",
        default="sft_dev",
        help="Hub split name. Default: sft_dev.",
    )
    return parser.parse_args(argv)


def _preview(table: object) -> None:
    """Print size, columns, and identity fields from the first row."""

    columns = getattr(table, "column_names", None)
    print(f"rows={len(table)}")  # type: ignore[arg-type]
    if isinstance(columns, list):
        print(f"columns={columns}")

    row = table[0]  # type: ignore[index]
    if not isinstance(row, Mapping):
        raise TypeError("Loaded table did not return a mapping row")

    sample_id = row.get("sample_id")
    split = row.get("split")
    gold = row.get("gold_diagnosis")
    target = row.get("target_text")
    image = row.get("image")
    print(f"sample_id={sample_id!r}")
    print(f"split={split!r}")
    print(f"gold_diagnosis={gold!r}")
    print(f"image_type={type(image).__name__}")
    if isinstance(target, str):
        preview = target.replace("\n", " ")[:80]
        print(f"target_preview={preview!r}")


def main(argv: Sequence[str] | None = None) -> None:
    """Load one config/split and print a short preview of the first example."""

    _ensure_repo_on_path()
    from project.dataset import DistillDataset

    args = parse_args(argv)
    dataset = DistillDataset.load(config=args.config, split=args.split)
    table = dataset.get(args.config, args.split)

    print(f"configs={list(dataset.configs())}")
    print(f"splits={list(dataset.splits(args.config))}")
    _preview(table)


if __name__ == "__main__":
    main()
