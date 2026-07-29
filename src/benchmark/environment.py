"""Safe environment metadata for reproducible benchmark runs."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable


TRACKED_PACKAGES = (
    "jsonschema",
    "openai",
    "pandas",
    "pyarrow",
    "pyyaml",
    "vllm",
)


def collect_environment(
    *,
    root: Path,
    credential_env_names: Iterable[str] = (),
) -> dict[str, Any]:
    """Collect versions and credential presence without reading secret values."""

    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            package: _package_version(package)
            for package in TRACKED_PACKAGES
        },
        "git": _git_metadata(root),
        "gpu": _gpu_metadata(),
        "credentials": {
            name: {"configured": bool(os.environ.get(name))}
            for name in sorted(set(credential_env_names))
        },
    }


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_metadata(root: Path) -> dict[str, Any]:
    commit = _run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
    )
    status = _run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
    )
    return {
        "commit": commit or None,
        "dirty": bool(status),
    }


def _gpu_metadata() -> dict[str, Any]:
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return {"available": False, "devices": []}
    devices = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            devices.append(
                {
                    "name": parts[0],
                    "memory_total_mib": int(parts[1]),
                    "driver_version": parts[2],
                }
            )
    return {"available": bool(devices), "devices": devices}


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
