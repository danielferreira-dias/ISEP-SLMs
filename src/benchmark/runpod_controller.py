"""Run a benchmark on RunPod while durably mirroring outputs to this machine.

The benchmark process and image files remain on the pod, next to the private
loopback vLLM endpoints.  Only result artifacts are copied over SSH.  This
avoids uploading every benchmark image from the workstation while ensuring
that an interrupted or deleted pod cannot erase an entire evaluation run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import threading
import time
from typing import Callable, Sequence, TextIO


DEFAULT_DERMOBENCH_OUTPUT = PurePosixPath(
    "outputs/dermobench_full_v1/temp_0_6_thinking_off"
)
DEFAULT_CONTEXT_OUTPUT = PurePosixPath(
    "outputs/clinical_context_ablation_v1/temp_0_6_thinking_off"
)
DEFAULT_LOCAL_LOG = Path(
    "runs/benchmarks/dermobench_then_context_controller.log"
)


@dataclass(frozen=True, slots=True)
class RunPodConnection:
    """SSH connection details for one explicitly identified RunPod."""

    host: str
    port: int
    user: str
    identity_file: Path
    known_hosts_file: Path | None = None

    @property
    def destination(self) -> str:
        return f"{self.user}@{self.host}"

    def ssh_transport(self, *, keepalive: bool = False) -> list[str]:
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=12",
        ]
        if keepalive:
            command.extend(
                (
                    "-o",
                    "ServerAliveInterval=30",
                    "-o",
                    "ServerAliveCountMax=4",
                )
            )
        if self.known_hosts_file is not None:
            command.extend(
                ("-o", f"UserKnownHostsFile={self.known_hosts_file}")
            )
        command.extend(
            (
                "-p",
                str(self.port),
                "-i",
                str(self.identity_file),
            )
        )
        return command


@dataclass(frozen=True, slots=True)
class MirrorTarget:
    """One project-relative directory mirrored between pod and workstation."""

    relative_path: PurePosixPath


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    """Fully resolved controller settings."""

    connection: RunPodConnection
    local_project_root: Path
    remote_project_root: PurePosixPath
    targets: tuple[MirrorTarget, ...]
    local_log: Path
    temperature: float
    batch_size: int
    sync_interval_seconds: float
    skip_dermobench_tasks: tuple[str, ...] = ()


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def build_ssh_command(
    connection: RunPodConnection,
    remote_command: str,
    *,
    keepalive: bool = False,
) -> list[str]:
    """Return a non-interactive SSH command without exposing an endpoint."""

    return [
        *connection.ssh_transport(keepalive=keepalive),
        connection.destination,
        remote_command,
    ]


def _rsync_shell(connection: RunPodConnection) -> str:
    return shlex.join(connection.ssh_transport())


def build_rsync_pull_command(
    config: ControllerConfig,
    target: MirrorTarget,
) -> list[str]:
    """Build a safe incremental pull that never deletes local artifacts."""

    remote = config.remote_project_root / target.relative_path
    local = config.local_project_root / Path(str(target.relative_path))
    return [
        "rsync",
        "-az",
        "--partial",
        "--update",
        "--no-owner",
        "--no-group",
        "-e",
        _rsync_shell(config.connection),
        f"{config.connection.destination}:{remote}/",
        f"{local}/",
    ]


def build_rsync_push_command(
    config: ControllerConfig,
    target: MirrorTarget,
) -> list[str]:
    """Build an initial recovery push for a missing remote checkpoint."""

    remote = config.remote_project_root / target.relative_path
    local = config.local_project_root / Path(str(target.relative_path))
    return [
        "rsync",
        "-az",
        "--partial",
        "--update",
        "--no-owner",
        "--no-group",
        "-e",
        _rsync_shell(config.connection),
        f"{local}/",
        f"{config.connection.destination}:{remote}/",
    ]


def build_remote_runner_command(config: ControllerConfig) -> str:
    """Build the foreground command whose lifetime is tied to this controller."""

    root = str(config.remote_project_root)
    python = str(config.remote_project_root / ".venv/bin/python")
    runner = str(
        config.remote_project_root / "scripts/run_dermobench_and_context.py"
    )
    argv = [
        python,
        runner,
        "--project-root",
        root,
        "--temperature",
        str(config.temperature),
        "--batch-size",
        str(config.batch_size),
    ]
    for task in config.skip_dermobench_tasks:
        argv.extend(("--skip-dermobench-task", task))
    return f"cd {shlex.quote(root)} && exec {shlex.join(argv)}"


def remote_path_exists(
    config: ControllerConfig,
    relative_path: PurePosixPath,
    *,
    run: CommandRunner = subprocess.run,
) -> bool:
    """Return whether a project-relative path exists on the pod."""

    remote = config.remote_project_root / relative_path
    result = run(
        build_ssh_command(
            config.connection,
            f"test -e {shlex.quote(str(remote))}",
        ),
        check=False,
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def endpoints_ready(
    config: ControllerConfig,
    *,
    run: CommandRunner = subprocess.run,
) -> bool:
    """Require both private loopback vLLM endpoints before launching a run."""

    checks = " && ".join(
        f"curl -fsS --max-time 5 http://127.0.0.1:{port}/health >/dev/null"
        for port in (8000, 8002)
    )
    result = run(
        build_ssh_command(config.connection, checks),
        check=False,
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def prepare_checkpoints(
    config: ControllerConfig,
    *,
    run: CommandRunner = subprocess.run,
) -> None:
    """Recover local checkpoints into an empty pod, otherwise pull remote state.

    A local directory is pushed only when the corresponding remote directory
    is absent.  This prevents an older local checkpoint from overwriting a pod
    that has already made more progress.
    """

    for target in config.targets:
        local = config.local_project_root / Path(str(target.relative_path))
        remote_exists = remote_path_exists(config, target.relative_path, run=run)
        if remote_exists:
            mirror_target(config, target, run=run)
            continue
        if not local.is_dir():
            continue
        remote = config.remote_project_root / target.relative_path
        mkdir = run(
            build_ssh_command(
                config.connection,
                f"mkdir -p {shlex.quote(str(remote))}",
            ),
            check=False,
            text=True,
            capture_output=True,
        )
        if mkdir.returncode != 0:
            raise RuntimeError(
                f"Could not create remote checkpoint directory: {mkdir.stderr}"
            )
        pushed = run(
            build_rsync_push_command(config, target),
            check=False,
            text=True,
            capture_output=True,
        )
        if pushed.returncode != 0:
            raise RuntimeError(
                f"Could not restore {target.relative_path}: {pushed.stderr}"
            )
        print(f"Restored local checkpoint: {target.relative_path}", flush=True)


def mirror_target(
    config: ControllerConfig,
    target: MirrorTarget,
    *,
    run: CommandRunner = subprocess.run,
) -> bool:
    """Pull one existing remote result directory into the local repository."""

    if not remote_path_exists(config, target.relative_path, run=run):
        return False
    local = config.local_project_root / Path(str(target.relative_path))
    local.mkdir(parents=True, exist_ok=True)
    result = run(
        build_rsync_pull_command(config, target),
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not mirror {target.relative_path}: {result.stderr}"
        )
    return True


def mirror_once(
    config: ControllerConfig,
    *,
    run: CommandRunner = subprocess.run,
) -> int:
    """Perform one complete local checkpoint cycle."""

    return sum(
        mirror_target(config, target, run=run) for target in config.targets
    )


def prediction_summary(config: ControllerConfig) -> tuple[int, int]:
    """Count locally durable prediction files and complete JSONL records."""

    files = 0
    records = 0
    for target in config.targets:
        local = config.local_project_root / Path(str(target.relative_path))
        if not local.is_dir():
            continue
        for path in local.rglob("predictions.jsonl"):
            files += 1
            with path.open("rb") as handle:
                records += sum(1 for line in handle if line.endswith(b"\n"))
    return files, records


def _tee_process_output(stream: TextIO, local_log: Path) -> None:
    """Copy the remote runner stream to the terminal and a durable local log."""

    local_log.parent.mkdir(parents=True, exist_ok=True)
    with local_log.open("a", encoding="utf-8", buffering=1) as handle:
        for line in stream:
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)
            handle.flush()


def run_controller(config: ControllerConfig) -> int:
    """Run the remote benchmark and checkpoint it locally until completion."""

    if not endpoints_ready(config):
        raise RuntimeError("RunPod endpoints 8000 and 8002 are not both ready")

    prepare_checkpoints(config)
    remote_runner = subprocess.Popen(
        build_ssh_command(
            config.connection,
            build_remote_runner_command(config),
            keepalive=True,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if remote_runner.stdout is None:  # pragma: no cover - defensive Popen guard
        raise RuntimeError("Could not capture the remote runner output")
    output_thread = threading.Thread(
        target=_tee_process_output,
        args=(remote_runner.stdout, config.local_log),
        name="runpod-runner-log",
        daemon=True,
    )
    output_thread.start()
    print(f"Started remote benchmark through SSH PID {remote_runner.pid}.", flush=True)

    sync_failed = False
    try:
        while remote_runner.poll() is None:
            try:
                mirrored = mirror_once(config)
                files, records = prediction_summary(config)
                print(
                    "Local checkpoint: "
                    f"{mirrored} targets, {files} prediction files, "
                    f"{records} complete records.",
                    flush=True,
                )
            except Exception as exc:
                sync_failed = True
                print(
                    f"WARNING: local checkpoint failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(config.sync_interval_seconds)
    except KeyboardInterrupt:
        print("Interrupted; closing the controlling SSH session.", file=sys.stderr)
        remote_runner.terminate()
    finally:
        return_code = remote_runner.wait()
        output_thread.join(timeout=10)

    final_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            mirror_once(config)
            final_error = None
            break
        except Exception as exc:
            final_error = exc
            print(
                f"WARNING: final checkpoint attempt {attempt}/3 failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(min(config.sync_interval_seconds, 10.0))
    if final_error is not None:
        raise RuntimeError("Final local checkpoint failed") from final_error

    files, records = prediction_summary(config)
    print(
        f"Final local checkpoint: {files} prediction files, {records} records.",
        flush=True,
    )
    if return_code != 0:
        return return_code
    if sync_failed:
        print(
            "The run completed after one or more transient checkpoint failures; "
            "the final checkpoint succeeded.",
            flush=True,
        )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--ssh-port", type=int, required=True)
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--known-hosts-file", type=Path)
    parser.add_argument("--local-project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--remote-project-root",
        type=PurePosixPath,
        default=PurePosixPath("/workspace/ISEP"),
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sync-interval-seconds", type=float, default=15.0)
    parser.add_argument(
        "--skip-dermobench-task",
        action="append",
        default=[],
        help=(
            "Forward one task exclusion to the remote DermoBench orchestrator. "
            "Repeat to skip multiple tasks."
        ),
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ControllerConfig:
    if not 1 <= args.ssh_port <= 65535:
        raise ValueError("--ssh-port must be between 1 and 65535")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.sync_interval_seconds <= 0:
        raise ValueError("--sync-interval-seconds must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")

    local_root = args.local_project_root.resolve()
    connection = RunPodConnection(
        host=args.host,
        port=args.ssh_port,
        user=args.ssh_user,
        identity_file=args.identity_file.expanduser().resolve(),
        known_hosts_file=(
            args.known_hosts_file.expanduser().resolve()
            if args.known_hosts_file is not None
            else None
        ),
    )
    return ControllerConfig(
        connection=connection,
        local_project_root=local_root,
        remote_project_root=args.remote_project_root,
        targets=(
            MirrorTarget(DEFAULT_DERMOBENCH_OUTPUT),
            MirrorTarget(DEFAULT_CONTEXT_OUTPUT),
        ),
        local_log=local_root / DEFAULT_LOCAL_LOG,
        temperature=args.temperature,
        batch_size=args.batch_size,
        sync_interval_seconds=args.sync_interval_seconds,
        skip_dermobench_tasks=tuple(args.skip_dermobench_task),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_controller(config_from_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
