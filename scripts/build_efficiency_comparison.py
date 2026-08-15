#!/usr/bin/env python3
"""Build the final five-model same-hardware efficiency comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.benchmark.efficiency_comparison import build_efficiency_comparison

DEFAULT_MODELS = (
    "qwen_3_8_27b",
    "qwen_3_6_27b",
    "qwen_3_5_4b",
    "qwen_3_5_4b_e1_frozen_vision",
    "qwen_3_5_4b_e1_vision_lora",
)


def main() -> int:
    """Parse paths and materialize the controlled thesis comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/efficiency_cohort_v1")
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    result = build_efficiency_comparison(
        args.root.resolve(),
        model_ids=tuple(args.models),
        output_directory=(
            args.output_directory.resolve() if args.output_directory else None
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
