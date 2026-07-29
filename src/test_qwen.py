"""Run a one-case pipeline smoke test with Qwen3.5-2B.

By default, the script asks the benchmark pipeline to manage a local vLLM
server. This mode requires a compatible Linux accelerator environment. Pass
``--base-url`` to test against an already running OpenAI-compatible vLLM
server, or ``--dry-run`` to validate the complete pipeline without inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark.cli import (
    REASONING_CAPTURE_CHOICES,
    main as benchmark_main,
)


MODEL_CONFIG = ROOT / "configs/models/smoke/qwen_3_5_2b.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/smoke/qwen_3_5_2b"


def build_parser() -> argparse.ArgumentParser:
    """Build the focused Qwen smoke-test command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Run one or more real dermatology benchmark cases through "
            "Qwen/Qwen3.5-2B and the project benchmark pipeline."
        )
    )
    parser.add_argument(
        "--benchmark",
        default="visual_top_k_closed_set",
        help="Benchmark ID or YAML path (default: visual_top_k_closed_set).",
    )
    parser.add_argument(
        "--evaluation-set",
        help="Optional named evaluation set from the benchmark config.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_integer,
        default=1,
        help="Number of benchmark cases to run (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic sample-selection seed (default: 42).",
    )
    parser.add_argument(
        "--base-url",
        help=(
            "Use an existing vLLM endpoint, for example "
            "http://127.0.0.1:8000/v1. If omitted, vLLM is managed by "
            "the pipeline."
        ),
    )
    parser.add_argument(
        "--transformers",
        action="store_true",
        help=(
            "Load the model directly with Transformers, preserving "
            "enable_thinking on Apple MPS without a vLLM server."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Managed vLLM host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=8000,
        help="Managed vLLM port (default: 8000).",
    )
    parser.add_argument(
        "--startup-timeout",
        type=_positive_float,
        default=900.0,
        help="Managed vLLM startup timeout in seconds (default: 900).",
    )
    parser.add_argument(
        "--reasoning-capture",
        choices=REASONING_CAPTURE_CHOICES,
        default="available",
        help="Reasoning channel retained by the pipeline (default: available).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Result directory (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs, data, images, prompts, and schemas only.",
    )
    return parser


def pipeline_arguments(args: argparse.Namespace) -> list[str]:
    """Translate smoke-test options into the public benchmark CLI contract."""

    values = [
        "run",
        "--model",
        str(MODEL_CONFIG),
        "--benchmark",
        args.benchmark,
        "--limit",
        str(args.limit),
        "--seed",
        str(args.seed),
        "--reasoning-capture",
        args.reasoning_capture,
        "--batch-size",
        "1",
        "--output-root",
        str(args.output_root),
    ]
    if args.evaluation_set:
        values.extend(["--evaluation-set", args.evaluation_set])
    if args.transformers:
        if args.base_url:
            raise ValueError(
                "--transformers cannot be combined with --base-url"
            )
        values.extend(
            [
                "--backend-profile",
                "transformers_mps",
                "--server-mode",
                "endpoint",
            ]
        )
    elif args.base_url:
        values.extend(
            [
                "--server-mode",
                "endpoint",
                "--base-url",
                args.base_url,
            ]
        )
    else:
        values.extend(
            [
                "--server-mode",
                "managed",
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--startup-timeout",
                str(args.startup_timeout),
            ]
        )
    if args.dry_run:
        values.append("--dry-run")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Qwen smoke test through the production benchmark pipeline."""

    args = build_parser().parse_args(argv)
    return benchmark_main(pipeline_arguments(args), root=ROOT)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
