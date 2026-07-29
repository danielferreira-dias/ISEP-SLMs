"""Repository command entry point."""

from __future__ import annotations

from collections.abc import Sequence

from src.benchmark.cli import main as benchmark_main


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate to the benchmark command-line interface."""

    return benchmark_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
