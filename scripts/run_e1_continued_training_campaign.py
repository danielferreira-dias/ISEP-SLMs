#!/usr/bin/env python3
"""Run the two E1 epoch-3-to-5 continuations sequentially on one GPU."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Condition:
    """Describe one immutable continuation invocation."""

    name: str
    config: Path
    experiment: str
    run_id: str

    @property
    def run_directory(self) -> Path:
        """Return the pipeline output directory expected for this condition."""

        return Path("outputs/training") / self.experiment / self.run_id

    @property
    def parent_checkpoint(self) -> Path:
        """Return the staged epoch-three checkpoint used for continuation."""

        return self.run_directory / "checkpoints" / "checkpoint-2367"


CONDITIONS = (
    Condition(
        name="frozen",
        config=Path("configs/training/e1_label_frozen_vision_continued.yaml"),
        experiment="e1_label_frozen_vision_continued",
        run_id="continued-l40s-frozen-seed3407-epoch5-20260814",
    ),
    Condition(
        name="vision",
        config=Path("configs/training/e1_label_unsloth_all_continued.yaml"),
        experiment="e1_label_unsloth_all_continued",
        run_id="continued-l40s-vision-seed3407-epoch5-20260814",
    ),
)
STATUS = Path("outputs/training/e1_continued_campaign_status.json")
LOG_ROOT = Path("outputs/training/_launch_logs")


def main() -> int:
    """Validate the idle GPU and execute Frozen followed by Vision LoRA."""

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
    for condition in CONDITIONS:
        if not condition.config.is_file():
            raise FileNotFoundError(condition.config)
        if condition.run_directory.exists():
            status_path = condition.run_directory / "manifests" / "run_status.json"
            pipeline_status = condition.run_directory / "manifests" / "status.json"
            selected_status = status_path if status_path.is_file() else pipeline_status
            if not selected_status.is_file():
                raise FileExistsError(condition.run_directory)
            payload = json.loads(selected_status.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("status") != "failed":
                raise FileExistsError(condition.run_directory)
            if not condition.parent_checkpoint.is_dir():
                raise FileNotFoundError(condition.parent_checkpoint)
    free_bytes = shutil.disk_usage("/workspace").free
    if free_bytes < 20 * 1024**3:
        raise RuntimeError("RunPod workspace has less than 20 GiB free")
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
    resume = condition.run_directory.exists()
    suffix = "-resume" if resume else ""
    log_path = _next_log_path(condition.run_id, suffix=suffix)
    command: tuple[str, ...] = (
        sys.executable,
        "-m",
        "src.train.cli",
        "run",
        "--config",
        str(condition.config),
        "--run-id",
        condition.run_id,
    )
    if resume:
        command = (*command, "--resume-from", str(condition.parent_checkpoint))
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return completed.returncode


def _next_log_path(run_id: str, *, suffix: str) -> Path:
    """Return a new launch-log path without overwriting earlier attempts."""

    candidate = LOG_ROOT / f"{run_id}{suffix}.log"
    attempt = 2
    while candidate.exists():
        candidate = LOG_ROOT / f"{run_id}{suffix}-{attempt}.log"
        attempt += 1
    return candidate


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
