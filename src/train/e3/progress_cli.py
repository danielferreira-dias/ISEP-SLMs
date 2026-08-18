"""Read-only terminal watcher for E3 teacher-generation progress."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from src.train.e3.progress import (
    E3CampaignState,
    E3ProgressPaths,
    E3ProgressStore,
    render_terminal,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isep-e3-progress",
        description="Watch an E3 teacher-generation campaign without mutating it.",
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--no-clear", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Wait for campaign creation, then poll until it becomes terminal."""

    arguments = _parser().parse_args(argv)
    if arguments.interval <= 0:
        print("isep-e3-progress: --interval must be greater than zero", file=sys.stderr)
        return 2
    try:
        store = _open_campaign(
            arguments.output_directory,
            interval=arguments.interval,
            wait=not arguments.once,
        )
        while True:
            snapshot = store.read_snapshot()
            if not arguments.no_clear and not arguments.once and sys.stdout.isatty():
                sys.stdout.write("\033[2J\033[H")
            if arguments.as_json:
                print(json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False))
            else:
                print(render_terminal(snapshot), flush=True)
            terminal = snapshot.get("status") in {
                E3CampaignState.COMPLETED.value,
                E3CampaignState.INTERRUPTED.value,
                E3CampaignState.FAILED.value,
            }
            if arguments.once or terminal:
                return 0
            time.sleep(arguments.interval)
    except (OSError, ValueError) as exc:
        print(f"isep-e3-progress: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nisep-e3-progress: stopped by user", file=sys.stderr)
        return 130


def _open_campaign(
    output_directory: Path,
    *,
    interval: float,
    wait: bool,
) -> E3ProgressStore:
    """Open a campaign, optionally waiting without creating or mutating it."""

    paths = E3ProgressPaths.below(output_directory)
    announced = False
    while not paths.manifest.exists():
        if not wait:
            return E3ProgressStore.open(output_directory)
        if not announced:
            print(
                "isep-e3-progress: waiting for campaign to start at "
                f"{paths.root}",
                file=sys.stderr,
                flush=True,
            )
            announced = True
        time.sleep(interval)
    return E3ProgressStore.open(output_directory)


if __name__ == "__main__":
    raise SystemExit(main())
