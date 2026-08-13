"""Environment provenance collected without exposing credentials."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from src.train.domain import JsonValue

_PACKAGES = (
    "datasets",
    "matplotlib",
    "nvidia-ml-py",
    "pydantic",
    "tensorboard",
    "torch",
    "transformers",
    "trl",
    "unsloth",
)


def collect_environment(project_root: Path) -> dict[str, JsonValue]:
    """Collect reproducibility metadata without reading secret values.

    Args:
        project_root: Repository whose Git state belongs to the experiment.

    Returns:
        JSON-compatible environment and package metadata.
    """

    commit = _git_value(project_root, ("rev-parse", "HEAD"))
    dirty_output = _git_value(project_root, ("status", "--porcelain"))
    packages: dict[str, JsonValue] = {}
    for package in _PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": commit,
        "git_dirty": bool(dirty_output),
        "packages": packages,
        "nvidia_smi": _nvidia_smi(),
    }


def _git_value(project_root: Path, arguments: tuple[str, ...]) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _nvidia_smi() -> list[JsonValue] | None:
    """Return non-secret GPU and driver metadata when NVIDIA SMI exists."""

    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    devices: list[JsonValue] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        index, name, uuid, driver, memory_mib = fields
        try:
            memory_value: JsonValue = int(memory_mib)
        except ValueError:
            memory_value = memory_mib
        devices.append(
            {
                "index": index,
                "name": name,
                "uuid": uuid,
                "driver_version": driver,
                "memory_total_mib": memory_value,
            }
        )
    # Round-trip through JSON to assert that subprocess text did not create an
    # accidental non-serializable value before it reaches a run manifest.
    json.dumps(devices, allow_nan=False)
    return devices
