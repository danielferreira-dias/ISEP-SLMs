#!/usr/bin/env python3
"""Run the four ISEPDermaBench Internal Benchmark tasks through one endpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark.cli import main as benchmark_main  # noqa: E402


BENCHMARKS = (
    "visual_top_k_closed_set",
    "visual_disease_confusion_sets",
    "evidence_grounded_diagnosis",
    "open_ended_diagnosis",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--case-limit",
        type=int,
        help=(
            "Optional target number of cases per task. Confusion cases are "
            "selected as pairs, so an odd target is rounded up."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.case_limit is not None and args.case_limit < 1:
        raise ValueError("--case-limit must be positive")

    for benchmark in BENCHMARKS:
        command = [
            "run",
            "--model",
            args.model,
            "--benchmark",
            benchmark,
            "--evaluation-set",
            "internal_benchmark",
            "--base-url",
            args.base_url,
            "--thinking-mode",
            "disabled",
            "--temperature",
            str(args.temperature),
            "--output-root",
            str(args.output_root),
        ]
        if args.case_limit is not None:
            limit = args.case_limit
            if benchmark == "visual_disease_confusion_sets":
                limit = (limit + 1) // 2
            command.extend(("--limit", str(limit)))
        if args.dry_run:
            command.append("--dry-run")

        status = benchmark_main(command, root=ROOT)
        if status != 0:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
