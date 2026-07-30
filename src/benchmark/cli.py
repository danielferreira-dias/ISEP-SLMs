"""Command-line orchestration for reproducible multimodal benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import yaml

from src.benchmark.datasets import (
    LoadedBenchmarkDataset,
    load_benchmark_dataset,
)
from src.benchmark.environment import collect_environment
from src.benchmark.executor import (
    BenchmarkExecutor,
    ExecutionConfig,
)
from src.benchmark.images import prepare_benchmark_image
from src.benchmark.results import (
    RunPaths,
    RunWriter,
    canonical_hash,
    count_statuses,
    create_run_directory,
    file_sha256,
    read_jsonl,
)
from src.benchmark.task_adapters import build_task_adapter
from src.config import (
    BenchmarkConfig,
    ModelConfig,
    list_benchmark_configs,
    list_model_configs,
    load_benchmark_config,
    load_model_config,
)
from src.data_pipeline.deduplication import ImageResolver
from src.inference.factory import create_backend
from src.inference.local import LocalBackend
from src.inference.vllm import (
    ManagedVllmServer,
    server_config_from_model,
)


REASONING_CAPTURE_CHOICES = (
    "available",
    "full",
    "summary",
    "tokens_only",
    "none",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the public benchmark command-line interface."""

    parser = argparse.ArgumentParser(
        prog="python -m src.benchmark.cli",
        description=(
            "Run reproducible multimodal dermatology benchmarks from YAML "
            "model and benchmark configurations."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "list-models",
        help="List validated model configurations.",
    )
    subparsers.add_parser(
        "list-benchmarks",
        help="List validated benchmark configurations and evaluation sets.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Validate or execute one model/benchmark combination.",
    )
    run_parser.add_argument(
        "--model",
        required=True,
        help="Model ID, YAML filename, or model YAML path.",
    )
    run_parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark ID, YAML filename, or benchmark YAML path.",
    )
    run_parser.add_argument(
        "--evaluation-set",
        help="Named evaluation set; defaults to the benchmark protocol.",
    )
    run_parser.add_argument(
        "--limit",
        type=_positive_integer,
        help=(
            "Maximum selected cases. For the confusion benchmark this is "
            "the number of image pairs and therefore produces twice as many "
            "tasks."
        ),
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Model-independent subset seed (default: 42).",
    )
    run_parser.add_argument(
        "--backend-profile",
        help="Named backend profile from the selected model YAML.",
    )
    run_parser.add_argument(
        "--server-mode",
        choices=("endpoint", "managed"),
        default="endpoint",
        help=(
            "Use an existing endpoint or start a local vLLM server "
            "(default: endpoint)."
        ),
    )
    run_parser.add_argument(
        "--base-url",
        help=(
            "Explicit base URL for a local vLLM endpoint. When omitted, "
            "VLLM_BASE_URL or http://127.0.0.1:8000/v1 is used."
        ),
    )
    run_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for a managed vLLM server (default: 127.0.0.1).",
    )
    run_parser.add_argument(
        "--port",
        type=_port,
        default=8000,
        help="Port for a managed vLLM server (default: 8000).",
    )
    run_parser.add_argument(
        "--vllm-executable",
        default="vllm",
        help="vLLM executable for managed mode (default: vllm).",
    )
    run_parser.add_argument(
        "--startup-timeout",
        type=_positive_float,
        default=900.0,
        help="Managed-server startup timeout in seconds (default: 900).",
    )
    run_parser.add_argument(
        "--reasoning-capture",
        choices=REASONING_CAPTURE_CHOICES,
        default="available",
        help=(
            "Retain provider-available reasoning separately from the scored "
            "final JSON (default: available)."
        ),
    )
    run_parser.add_argument(
        "--batch-size",
        type=_positive_integer,
        help="Override only the runtime request concurrency.",
    )
    run_parser.add_argument(
        "--output-root",
        type=Path,
        help="Override the benchmark output directory.",
    )
    run_parser.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIRECTORY",
        help="Resume an interrupted or failed run after identity validation.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate configs, selection, schema, and selected image access "
            "without loading a model, starting vLLM, or calling an API."
        ),
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path | None = None,
) -> int:
    """Run the CLI and return a process-compatible exit status."""

    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = (root or _project_root()).resolve()
    try:
        if args.command == "list-models":
            _print_models(project_root)
            return 0
        if args.command == "list-benchmarks":
            _print_benchmarks(project_root)
            return 0
        if args.command == "run":
            return _run_command(args, root=project_root)
        parser.error(f"Unsupported command: {args.command}")
    except KeyboardInterrupt:
        print("Benchmark interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Benchmark error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 2


def _run_command(args: argparse.Namespace, *, root: Path) -> int:
    model = load_model_config(
        args.model,
        root=root,
        backend_profile=args.backend_profile,
    )
    benchmark = load_benchmark_config(args.benchmark, root=root)
    raw_model = _load_yaml(model.config_path)
    raw_benchmark = _load_yaml(benchmark.config_path)
    prompt = _load_yaml(benchmark.prompt_path)
    schema = _load_json(benchmark.schema_path)
    disease_taxonomy = _load_yaml(benchmark.taxonomy.disease_path)
    disease_items = disease_taxonomy.get("diseases")
    if not isinstance(disease_items, list):
        raise ValueError(
            "Disease taxonomy must contain a top-level 'diseases' list"
        )

    _validate_compatibility(model=model, benchmark=benchmark)
    _validate_runtime_options(args=args, model=model)
    adapter = build_task_adapter(
        benchmark_config=raw_benchmark,
        prompt_config=prompt,
        schema=schema,
        disease_taxonomy_items=disease_items,
    )
    dataset = load_benchmark_dataset(
        root=root,
        config=raw_benchmark,
        evaluation_set=args.evaluation_set,
        limit=args.limit,
        seed=args.seed,
    )
    if not dataset.samples:
        raise ValueError("The selected benchmark dataset contains no tasks")

    credential_names = _credential_environment_names(model)
    environment = collect_environment(
        root=root,
        credential_env_names=credential_names,
    )
    dry_run_summary = _dry_run_summary(
        model=model,
        benchmark=benchmark,
        dataset=dataset,
        environment=environment,
        server_mode=args.server_mode,
        reasoning_capture=args.reasoning_capture,
    )
    with ImageResolver(root) as resolver:
        def image_loader(image_uri: str) -> bytes:
            return prepare_benchmark_image(
                resolver.read_bytes(image_uri),
                benchmark.image_preprocessing,
            )

        if args.dry_run:
            _validate_selected_images(dataset, image_loader)
            print(json.dumps(dry_run_summary, indent=2, sort_keys=True))
            return 0

        identity = _run_identity(
            model=model,
            benchmark=benchmark,
            dataset=dataset,
            args=args,
        )
        paths = _resolve_run_paths(
            args=args,
            benchmark=benchmark,
            model=model,
            identity=identity,
            root=root,
        )
        writer = RunWriter(
            paths,
            identity=identity,
            resume=args.resume is not None,
        )
        writer.initialize(
            manifest=_run_manifest(
                model=model,
                benchmark=benchmark,
                dataset=dataset,
                args=args,
            ),
            config_snapshot={
                "model": raw_model,
                "benchmark": raw_benchmark,
                "prompt": prompt,
                "output_schema": schema,
                "disease_taxonomy": disease_taxonomy,
            },
            selection=dataset.selection,
            environment=environment,
        )

        server: ManagedVllmServer | None = None
        try:
            backend, server = _create_runtime_backend(
                model=model,
                args=args,
                paths=paths,
            )
            execution = ExecutionConfig(
                batch_size=(
                    args.batch_size
                    if args.batch_size is not None
                    else benchmark.execution.batch_size
                ),
                max_output_tokens=benchmark.execution.max_output_tokens,
                run_seed=args.seed,
                save_rendered_prompts=(
                    benchmark.execution.save_rendered_prompts
                ),
            )
            summary = BenchmarkExecutor(
                backend=backend,
                adapter=adapter,
                image_loader=image_loader,
                writer=writer,
                execution=execution,
                generation=model.generation,
            ).run(dataset.samples)
        except BaseException as exc:
            _finalize_failed_run(writer, exc)
            raise
        finally:
            if server is not None:
                server.stop()

    print(
        json.dumps(
            {
                "status": "completed",
                "run_directory": str(paths.directory),
                "counts": summary.counts,
                "metrics_path": str(paths.metrics),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _create_runtime_backend(
    *,
    model: ModelConfig,
    args: argparse.Namespace,
    paths: RunPaths,
) -> tuple[Any, ManagedVllmServer | None]:
    profile = model.backend.active_profile
    if args.server_mode == "managed":
        if profile.type != "local" or profile.engine != "vllm":
            raise ValueError(
                "--server-mode managed is available only for local vLLM "
                "profiles"
            )
        server_config = server_config_from_model(
            model,
            host=args.host,
            port=args.port,
            executable=args.vllm_executable,
        )
        server_config = replace(
            server_config,
            startup_timeout_seconds=args.startup_timeout,
        )
        server = ManagedVllmServer(
            server_config,
            log_path=paths.server_log,
        )
        server.start()
        try:
            backend = LocalBackend(
                model,
                base_url=server_config.base_url,
                reasoning_capture=args.reasoning_capture,
            )
            backend.require_ready()
            return backend, server
        except BaseException:
            server.stop()
            raise

    if args.base_url is not None:
        if profile.type != "local" or profile.engine != "vllm":
            raise ValueError(
                "--base-url can be used only with a local vLLM profile"
            )
        backend = LocalBackend(
            model,
            base_url=args.base_url,
            reasoning_capture=args.reasoning_capture,
        )
    else:
        backend = create_backend(
            model,
            reasoning_capture=args.reasoning_capture,
        )
    if profile.type == "local" and hasattr(backend, "require_ready"):
        backend.require_ready()
    return backend, None


def _validate_compatibility(
    *,
    model: ModelConfig,
    benchmark: BenchmarkConfig,
) -> None:
    if not model.usage.benchmark:
        raise ValueError(f"Model {model.model.id!r} is not enabled for benchmarks")
    if "image" not in model.capabilities.modalities:
        raise ValueError(f"Model {model.model.id!r} does not support images")
    if (
        benchmark.structured_output.mode
        not in model.capabilities.structured_output_modes
    ):
        raise ValueError(
            f"Model {model.model.id!r} does not support structured-output "
            f"mode {benchmark.structured_output.mode!r}"
        )
    cap = benchmark.execution.max_output_tokens
    maximum_output = model.capabilities.maximum_output_tokens
    if maximum_output is not None and cap > maximum_output:
        raise ValueError(
            f"Benchmark output cap {cap} exceeds model maximum "
            f"{maximum_output}"
        )
    context_length = (
        model.backend.active_profile.max_model_len
        or model.capabilities.context_length_tokens
    )
    if context_length is not None and cap >= context_length:
        raise ValueError(
            f"Benchmark output cap {cap} must be smaller than runtime "
            f"context length {context_length}"
        )


def _validate_runtime_options(
    *,
    args: argparse.Namespace,
    model: ModelConfig,
) -> None:
    profile = model.backend.active_profile
    if args.server_mode == "managed" and (
        profile.type != "local" or profile.engine != "vllm"
    ):
        raise ValueError(
            "--server-mode managed is available only for local vLLM profiles"
        )
    if args.base_url is not None and (
        profile.type != "local" or profile.engine != "vllm"
    ):
        raise ValueError(
            "--base-url can be used only with a local vLLM profile"
        )
    if args.server_mode == "managed" and args.base_url is not None:
        raise ValueError(
            "--base-url cannot be combined with --server-mode managed"
        )
    if args.dry_run and args.resume is not None:
        raise ValueError("--dry-run cannot be combined with --resume")
    if args.resume is not None and args.output_root is not None:
        raise ValueError("--resume cannot be combined with --output-root")


def _validate_selected_images(
    dataset: LoadedBenchmarkDataset,
    image_loader: Callable[[str], bytes],
) -> None:
    """Prove that every selected image can be prepared before execution."""

    seen: set[str] = set()
    for sample in dataset.samples:
        if sample.image_uri in seen:
            continue
        image_bytes = image_loader(sample.image_uri)
        if not image_bytes:
            raise ValueError(
                f"Selected image is empty: {sample.image_uri!r}"
            )
        seen.add(sample.image_uri)


def _run_identity(
    *,
    model: ModelConfig,
    benchmark: BenchmarkConfig,
    dataset: LoadedBenchmarkDataset,
    args: argparse.Namespace,
) -> dict[str, str]:
    """Build the immutable identity checked before every resume."""

    related_taxonomy_hashes = {
        "disease": file_sha256(benchmark.taxonomy.disease_path),
    }
    if benchmark.taxonomy.confusion_sets_path is not None:
        related_taxonomy_hashes["confusion_sets"] = file_sha256(
            benchmark.taxonomy.confusion_sets_path
        )
    profile = model.backend.active_profile
    runtime_binding = {
        "base_url_argument": args.base_url,
        "endpoint": _environment_binding(profile.endpoint_env),
        "base_url": _environment_binding(profile.base_url_env),
        "deployment": _environment_binding(profile.deployment_env),
        "request_model": _environment_binding(profile.model_env),
        "api_version": _environment_binding(profile.api_version_env),
        "managed_host": args.host if args.server_mode == "managed" else None,
        "managed_port": args.port if args.server_mode == "managed" else None,
        "vllm_executable": (
            args.vllm_executable
            if args.server_mode == "managed"
            else None
        ),
    }
    effective_batch_size = (
        args.batch_size
        if args.batch_size is not None
        else benchmark.execution.batch_size
    )
    return {
        "model_id": model.model.id,
        "model_config_sha256": file_sha256(model.config_path),
        "backend_profile": model.backend.active_profile.name,
        "benchmark_id": benchmark.benchmark.id,
        "benchmark_version": benchmark.benchmark.version,
        "benchmark_config_sha256": file_sha256(benchmark.config_path),
        "prompt_sha256": file_sha256(benchmark.prompt_path),
        "schema_sha256": file_sha256(benchmark.schema_path),
        "taxonomy_sha256": canonical_hash(related_taxonomy_hashes),
        "evaluation_set": dataset.evaluation_set,
        "manifest_sha256": dataset.manifest_sha256,
        "release_sha256": dataset.release_sha256,
        "selection_sha256": str(dataset.selection["selection_hash"]),
        "seed": str(args.seed),
        "server_mode": args.server_mode,
        "runtime_binding_sha256": canonical_hash(runtime_binding),
        "reasoning_capture": args.reasoning_capture,
        "batch_size": str(effective_batch_size),
        "max_output_tokens": str(benchmark.execution.max_output_tokens),
        "structured_output_mode": benchmark.structured_output.mode,
    }


def _environment_binding(name: str | None) -> dict[str, Any] | None:
    """Hash non-secret endpoint/deployment values used to resume a run."""

    if not name:
        return None
    value = os.environ.get(name)
    return {
        "name": name,
        "configured": value is not None,
        "value_sha256": (
            canonical_hash(value)
            if value is not None
            else None
        ),
    }


def _resolve_run_paths(
    *,
    args: argparse.Namespace,
    benchmark: BenchmarkConfig,
    model: ModelConfig,
    identity: dict[str, str],
    root: Path,
) -> RunPaths:
    if args.resume is not None:
        directory = args.resume
        if not directory.is_absolute():
            directory = root / directory
        return RunPaths.from_directory(directory.resolve())
    output_root = args.output_root or benchmark.output_directory
    if not output_root.is_absolute():
        output_root = root / output_root
    return create_run_directory(
        output_root=output_root.resolve(),
        benchmark_id=benchmark.benchmark.id,
        model_id=model.model.id,
        identity_hash=canonical_hash(identity),
    )


def _run_manifest(
    *,
    model: ModelConfig,
    benchmark: BenchmarkConfig,
    dataset: LoadedBenchmarkDataset,
    args: argparse.Namespace,
) -> dict[str, Any]:
    profile = model.backend.active_profile
    return {
        "schema_version": 1,
        "model": {
            "id": model.model.id,
            "source": model.source.repo_id or model.source.model_name,
        },
        "benchmark": {
            "id": benchmark.benchmark.id,
            "version": benchmark.benchmark.version,
            "task": benchmark.benchmark.task,
        },
        "backend": {
            "profile": profile.name,
            "type": profile.type,
            "engine": profile.engine,
            "api_style": profile.api_style,
            "server_mode": args.server_mode,
        },
        "evaluation": {
            "evaluation_set": dataset.evaluation_set,
            "selected_units": dataset.selection["selected_unit_count"],
            "selected_tasks": dataset.selection["selected_task_count"],
            "seed": args.seed,
            "max_output_tokens": benchmark.execution.max_output_tokens,
            "reasoning_capture": args.reasoning_capture,
            "structured_output": benchmark.structured_output.mode,
        },
    }


def _dry_run_summary(
    *,
    model: ModelConfig,
    benchmark: BenchmarkConfig,
    dataset: LoadedBenchmarkDataset,
    environment: dict[str, Any],
    server_mode: str,
    reasoning_capture: str,
) -> dict[str, Any]:
    profile = model.backend.active_profile
    return {
        "status": "dry_run_valid",
        "model": model.model.id,
        "benchmark": benchmark.benchmark.id,
        "benchmark_version": benchmark.benchmark.version,
        "evaluation_set": dataset.evaluation_set,
        "selected_units": dataset.selection["selected_unit_count"],
        "selected_tasks": dataset.selection["selected_task_count"],
        "manifest_sha256": dataset.manifest_sha256,
        "selection_sha256": dataset.selection["selection_hash"],
        "backend": {
            "profile": profile.name,
            "type": profile.type,
            "engine": profile.engine,
            "server_mode": server_mode,
        },
        "reasoning_capture": reasoning_capture,
        "max_output_tokens": benchmark.execution.max_output_tokens,
        "gpu_available": environment["gpu"]["available"],
        "credential_environment": environment["credentials"],
        "network_or_model_called": False,
    }


def _credential_environment_names(model: ModelConfig) -> tuple[str, ...]:
    profile = model.backend.active_profile
    values = (
        profile.endpoint_env,
        profile.base_url_env,
        profile.api_key_env,
        profile.deployment_env,
        profile.model_env,
        profile.api_version_env,
    )
    names = {value for value in values if value}
    if model.source.access == "gated":
        names.add("HF_TOKEN")
    if profile.type == "local":
        names.add("VLLM_BASE_URL")
    return tuple(sorted(names))


def _finalize_failed_run(
    writer: RunWriter,
    exc: BaseException,
) -> None:
    """Mark only a still-running manifest as failed."""

    try:
        document = _load_yaml(writer.paths.manifest)
        if document.get("status") != "running":
            return
        records = read_jsonl(writer.paths.predictions)
        writer.finalize(
            status="failed",
            counts=count_statuses(records),
            error=f"{type(exc).__name__}: {exc}",
        )
    except Exception:
        # Preserve the original execution failure.
        return


def _print_models(root: Path) -> None:
    rows = [
        (
            config.model.id,
            config.model.display_name,
            config.backend.active_profile.name,
            config.backend.active_profile.engine,
            config.source.repo_id or config.source.model_name or "-",
        )
        for config in list_model_configs(root=root)
    ]
    _print_table(
        ("MODEL ID", "DISPLAY NAME", "PROFILE", "ENGINE", "SOURCE"),
        rows,
    )


def _print_benchmarks(root: Path) -> None:
    rows = []
    for config in list_benchmark_configs(root=root):
        evaluation_sets = ",".join(
            item.id for item in config.dataset.evaluation_sets
        )
        rows.append(
            (
                config.benchmark.id,
                config.benchmark.version,
                config.benchmark.task,
                config.dataset.default_evaluation_set,
                evaluation_sets,
            )
        )
    _print_table(
        (
            "BENCHMARK ID",
            "VERSION",
            "TASK",
            "DEFAULT SET",
            "EVALUATION SETS",
        ),
        rows,
    )


def _print_table(
    headers: tuple[str, ...],
    rows: Sequence[tuple[str, ...]],
) -> None:
    widths = [
        max(
            len(headers[index]),
            *(len(str(row[index])) for row in rows),
        )
        for index in range(len(headers))
    ]
    print(
        "  ".join(
            header.ljust(widths[index])
            for index, header in enumerate(headers)
        )
    )
    print(
        "  ".join("-" * width for width in widths)
    )
    for row in rows:
        print(
            "  ".join(
                str(value).ljust(widths[index])
                for index, value in enumerate(row)
            )
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())
