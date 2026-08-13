#!/usr/bin/env python3
"""Resume only the validated Vision arm of the E1 t=0.6 campaign."""

from __future__ import annotations

from pathlib import Path

from run_e1_t06_campaign import (
    CONDITIONS,
    EXPECTED_METRICS,
    run_gate,
    run_suite,
    start_server,
    stop_server,
    write_status,
)


def _metric_count(path: Path) -> int:
    """Return the number of completed metric documents below ``path``."""

    return len(tuple(path.glob("**/metrics.json"))) if path.exists() else 0


def main() -> int:
    """Validate the frozen arm and execute the previously unstarted Vision arm."""

    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/e1_epoch3_historical_t06_benchmarks"
    frozen_output = output / "frozen"
    vision_output = output / "vision"
    vision = next(condition for condition in CONDITIONS if condition.key == "vision")

    frozen_metrics = _metric_count(frozen_output)
    vision_metrics = _metric_count(vision_output)
    if frozen_metrics != EXPECTED_METRICS:
        raise RuntimeError(
            f"Frozen arm has {frozen_metrics} metrics; expected {EXPECTED_METRICS}"
        )
    if vision_output.exists() or vision_metrics != 0:
        raise RuntimeError("Refusing to mix with an existing Vision output directory")
    if not (root / vision.model_path).is_dir():
        raise RuntimeError(f"Vision model is missing: {vision.model_path}")

    write_status(root, status="running", detail="Resuming validated vision condition")
    try:
        start_server(root, vision)
        run_gate(root, vision, limit=1)
        run_gate(root, vision, limit=10)
        run_suite(root, vision)
        stop_server(root)
        write_status(
            root,
            status="completed",
            detail="Frozen and resumed vision t=0.6 suites completed",
        )
    except Exception as exc:
        write_status(root, status="failed", detail=f"{type(exc).__name__}: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
