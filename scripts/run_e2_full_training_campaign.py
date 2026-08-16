#!/usr/bin/env python3
"""Run the paired confirmatory E2 conditions sequentially on one GPU."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.train.config import load_training_config
from src.train.scientific import validate_controlled_pair


@dataclass(frozen=True, slots=True)
class Condition:
    """Describe one immutable E2 training invocation."""

    name: str
    config: Path
    experiment: str
    run_id: str

    @property
    def run_directory(self) -> Path:
        """Return the output directory expected for this condition."""

        return Path("outputs/training") / self.experiment / self.run_id


CONDITIONS = (
    Condition(
        name="frozen",
        config=Path("configs/training/e2_skincon_skincap_frozen_vision.yaml"),
        experiment="e2_skincon_skincap_frozen_vision",
        run_id="full-l40s-e2-skincap-frozen-seed42-20260816",
    ),
    Condition(
        name="vision",
        config=Path("configs/training/e2_skincon_skincap_unsloth_all.yaml"),
        experiment="e2_skincon_skincap_unsloth_all",
        run_id="full-l40s-e2-skincap-vision-seed42-20260816",
    ),
)
STATUS = Path("outputs/training/e2_full_campaign_status.json")
LOG_ROOT = Path("outputs/training/_launch_logs")


def main() -> int:
    """Validate the paired protocol and run Frozen followed by Vision LoRA."""

    _preflight()
    completed: list[str] = []
    _write_status("running", current=None, completed=completed)
    for condition in CONDITIONS:
        _write_status("running", current=condition.name, completed=completed)
        return_code = _run(condition)
        if return_code != 0:
            _write_status(
                "failed",
                current=condition.name,
                completed=completed,
                return_code=return_code,
            )
            return return_code
        _require_completed(condition.run_directory)
        completed.append(condition.name)
    _write_status("completed", current=None, completed=completed)
    return 0


def _preflight() -> None:
    frozen, vision = CONDITIONS
    for condition in CONDITIONS:
        if not condition.config.is_file():
            raise FileNotFoundError(condition.config)
        if condition.run_directory.exists():
            raise FileExistsError(condition.run_directory)
    validate_controlled_pair(
        load_training_config(frozen.config),
        load_training_config(vision.config),
    )
    if STATUS.exists():
        raise FileExistsError(STATUS)
    free_bytes = shutil.disk_usage("/workspace").free
    if free_bytes < 40 * 1024**3:
        raise RuntimeError("RunPod workspace has less than 40 GiB free")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if query.stdout.strip():
        raise RuntimeError("GPU already has an active compute process")


def _run(condition: Condition) -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{condition.run_id}.log"
    command = (
        sys.executable,
        "-m",
        "src.train.cli",
        "run",
        "--config",
        str(condition.config),
        "--run-id",
        condition.run_id,
    )
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return completed.returncode


def _require_completed(run_directory: Path) -> None:
    status_path = run_directory / "manifests/status.json"
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "completed":
        raise RuntimeError(f"Training run did not complete: {run_directory}")


def _write_status(
    status: str,
    *,
    current: str | None,
    completed: list[str],
    return_code: int | None = None,
) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_conditions": completed,
        "current_condition": current,
        "return_code": return_code,
        "status": status,
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(STATUS)


if __name__ == "__main__":
    raise SystemExit(main())
