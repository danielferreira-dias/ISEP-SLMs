#!/usr/bin/env python3
"""Start one detached vLLM server from an ISEP model YAML and verify it."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_config", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--thinking", choices=("config", "on", "off"), default="config")
    parser.add_argument("--gdn-prefill-backend", choices=("auto", "triton", "flashinfer", "cutedsl"), default="auto")
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/vllm"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_server_config(args: argparse.Namespace):
    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    from src.config import load_model_config
    from src.inference.vllm import server_config_from_model

    model_path = args.model_config
    if not model_path.is_absolute():
        model_path = root / model_path
    model = load_model_config(model_path, root=root)
    config = server_config_from_model(model, host=args.host, port=args.port)
    if args.gpu_memory_utilization is not None:
        if not 0 < args.gpu_memory_utilization < 1:
            raise ValueError("--gpu-memory-utilization must be between 0 and 1")
        config = replace(config, gpu_memory_utilization=args.gpu_memory_utilization)

    extra = list(config.additional_args)
    extra.extend(["--max-num-seqs", str(args.max_num_seqs)])

    enabled = model.reasoning.chat_template_kwargs.enable_thinking
    if args.thinking != "config":
        enabled = args.thinking == "on"
    extra.extend([
        "--default-chat-template-kwargs",
        json.dumps({"enable_thinking": enabled}),
    ])

    repo_id = model.source.repo_id.lower()
    backend = args.gdn_prefill_backend
    if backend == "auto" and ("qwen3.5" in repo_id or "qwen3.6" in repo_id):
        backend = "triton"
    if backend != "auto":
        extra.extend(["--gdn-prefill-backend", backend])

    return model, replace(
        config,
        served_model_name=model.source.repo_id,
        additional_args=tuple(extra),
        startup_timeout_seconds=float(args.startup_timeout),
    )


def endpoint_ready(port: int, expected_model: str) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            if not 200 <= response.status < 300:
                return False
        with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, ValueError):
        return False
    return expected_model in {item.get("id") for item in payload.get("data", [])}


def main() -> int:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    if args.max_num_seqs < 1 or args.startup_timeout < 1:
        raise ValueError("sequence count and startup timeout must be positive")

    model, config = load_server_config(args)
    command = list(config.command())
    print("Command:", json.dumps(command))
    if args.dry_run:
        return 0

    if endpoint_ready(args.port, model.source.repo_id):
        print(f"Endpoint already ready on port {args.port}.")
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    slug = model.model.id.replace("/", "_")
    log_path = args.log_dir / f"{slug}-{args.port}.log"
    pid_path = args.log_dir / f"{slug}-{args.port}.pid"
    if pid_path.exists():
        old_pid = pid_path.read_text(encoding="utf-8").strip()
        if old_pid.isdigit():
            try:
                os.kill(int(old_pid), 0)
            except ProcessLookupError:
                pid_path.unlink()
            else:
                raise RuntimeError(f"PID file points to a live process: {old_pid}")

    environment = os.environ.copy()
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=args.project_root.resolve(),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    print(f"Started PID {process.pid}; log: {log_path}; waiting for readiness.")

    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"vLLM exited with code {code}; inspect {log_path}")
        if endpoint_ready(args.port, model.source.repo_id):
            print(f"Ready: http://127.0.0.1:{args.port}/v1 ({model.source.repo_id})")
            return 0
        time.sleep(5)

    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
    raise TimeoutError(f"vLLM was not ready within {args.startup_timeout}s; inspect {log_path}")


if __name__ == "__main__":
    raise SystemExit(main())
