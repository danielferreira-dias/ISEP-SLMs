"""Thin command-line interface for the reproducible E1 training pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from src.train.config import TrainingConfig, load_training_config
from src.train.data import inspect_data_release, prepare_data_release
from src.train.domain import VisionTuningProfile
from src.train.scientific import (
    config_hash,
    controlled_training_hash,
    label_contract_hash,
    prompt_hash,
    replicate_training_document,
    validate_controlled_pair,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isep-train",
        description="Reproducible Unsloth LoRA training for ISEP E1_label.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-config")
    validate.add_argument("--config", type=Path, required=True)

    prepare = commands.add_parser("prepare-data")
    prepare.add_argument("--config", type=Path, required=True)

    inspect = commands.add_parser("inspect-data")
    inspect.add_argument("--release", type=Path, required=True)

    smoke = commands.add_parser("smoke-test")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--run-id")
    smoke.add_argument("--seed", type=int, choices=(42, 3407, 2026))

    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--resume-from", type=Path)
    run.add_argument("--run-id")
    run.add_argument("--seed", type=int, choices=(42, 3407, 2026))

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--checkpoints", choices=("all",), default="all")

    report = commands.add_parser("report")
    report.add_argument("--run-dir", type=Path, required=True)

    compare = commands.add_parser("compare")
    compare.add_argument("--runs", type=Path, nargs="+", required=True)
    compare.add_argument("--output-dir", type=Path)
    compare.add_argument("--bootstrap-iterations", type=int, default=10_000)
    compare.add_argument("--bootstrap-seed", type=int, default=3407)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate arguments, dispatch one command, and return a process code."""

    arguments = _parser().parse_args(argv)
    try:
        payload = _dispatch(arguments)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"isep-train: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
            allow_nan=False,
        )
    )
    return 0


def _dispatch(arguments: argparse.Namespace) -> dict[str, object]:
    command = str(arguments.command)
    if command == "validate-config":
        config = load_training_config(arguments.config)
        paired = _validate_known_pair(arguments.config)
        return {
            "valid": True,
            "experiment_id": config.experiment.id,
            "config_sha256": config_hash(config),
            "known_pair_validated": paired,
        }
    if command == "prepare-data":
        config = load_training_config(arguments.config)
        release = prepare_data_release(config)
        return {
            "release": str(release.root),
            "audit": asdict(release.audit),
        }
    if command == "inspect-data":
        return {"audit": asdict(inspect_data_release(arguments.release))}
    if command in {"smoke-test", "run"}:
        config = load_training_config(arguments.config)
        if arguments.seed is not None:
            trainer = config.trainer.model_copy(update={"seed": arguments.seed})
            config = config.model_copy(update={"trainer": trainer})
        from src.train.pipeline import run_training

        result = run_training(
            config,
            resume_from=(arguments.resume_from if command == "run" else None),
            smoke=command == "smoke-test",
            run_id=arguments.run_id,
        )
        return {
            "run_directory": str(result.run_directory),
            "best_checkpoint": str(result.best_checkpoint),
            "macro_f1": result.best_metrics.macro_f1,
            "top1_accuracy": result.best_metrics.top1_accuracy,
        }
    if command == "evaluate":
        from src.train.finalize import evaluate_run

        result = evaluate_run(arguments.run_dir.resolve())
        return {
            "run_directory": str(result.run_directory),
            "best_checkpoint": str(result.best_checkpoint),
            "macro_f1": result.best_metrics.macro_f1,
        }
    if command == "report":
        from src.train.reporting import build_run_report

        directory = arguments.run_dir.resolve()
        build_run_report(directory)
        report_path = directory / "report" / "report.html"
        if not report_path.is_file():
            raise RuntimeError(
                "Report was not generated because final classification metrics "
                "are missing"
            )
        return {
            "run_directory": str(directory),
            "report": str(report_path),
        }
    if command == "compare":
        from src.train.artifacts import compare_runs

        ordered_runs = _validate_comparison_protocol(tuple(arguments.runs))
        output = arguments.output_dir or _comparison_output()
        artefacts = compare_runs(
            ordered_runs,
            output.resolve(),
            bootstrap_iterations=arguments.bootstrap_iterations,
            bootstrap_seed=arguments.bootstrap_seed,
        )
        return {
            "output_directory": str(output.resolve()),
            "comparison": str(artefacts.json_path),
            "report": str(artefacts.html_path),
        }
    raise ValueError(f"Unknown command: {command}")


def _validate_known_pair(config_path: Path) -> bool:
    root = config_path.resolve().parent
    pairs = (
        (
            root / "e1_label_frozen_vision.yaml",
            root / "e1_label_unsloth_all.yaml",
        ),
        (
            root / "e1_label_frozen_vision_continued.yaml",
            root / "e1_label_unsloth_all_continued.yaml",
        ),
    )
    for frozen, visual in pairs:
        if config_path.resolve() in {frozen, visual}:
            validate_controlled_pair(
                load_training_config(frozen),
                load_training_config(visual),
            )
            return True
    return False


def _comparison_output() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("outputs") / "training" / "comparisons" / timestamp


def _validate_comparison_protocol(
    run_directories: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Require the two E1 conditions and all three confirmation seeds."""

    from src.train.run_io import load_run_config

    if len(run_directories) != 6:
        raise ValueError("Confirmatory E1 comparison requires exactly six runs")
    entries = tuple(
        (load_run_config(path.resolve()), path.resolve()) for path in run_directories
    )
    for config, path in entries:
        _validate_confirmation_run(config, path)
    configs = tuple(config for config, _ in entries)
    grouped: dict[
        VisionTuningProfile,
        dict[int, tuple[TrainingConfig, Path]],
    ] = {
        VisionTuningProfile.FROZEN_VISION: {},
        VisionTuningProfile.UNSLOTH_ALL: {},
    }
    for config, path in entries:
        profile = config.experiment.vision_profile
        seed = config.trainer.seed
        if seed in grouped[profile]:
            raise ValueError(f"Duplicate seed {seed} for profile {profile.value}")
        grouped[profile][seed] = (config, path)
    required = {42, 3407, 2026}
    for profile, by_seed in grouped.items():
        if set(by_seed) != required:
            raise ValueError(
                f"Profile {profile.value} must contain seeds 42, 3407, and 2026"
            )
    frozen = grouped[VisionTuningProfile.FROZEN_VISION]
    visual = grouped[VisionTuningProfile.UNSLOTH_ALL]
    for seed in sorted(required):
        left = frozen[seed][0]
        right = visual[seed][0]
        validate_controlled_pair(left, right)
    replicate_documents = {
        json.dumps(
            replicate_training_document(config),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for config in configs
    }
    if len(replicate_documents) != 1:
        raise ValueError("Confirmation runs differ outside condition and seed")
    return tuple(
        grouped[profile][seed][1]
        for profile in (
            VisionTuningProfile.FROZEN_VISION,
            VisionTuningProfile.UNSLOTH_ALL,
        )
        for seed in sorted(required)
    )


def _validate_confirmation_run(config: TrainingConfig, run_directory: Path) -> None:
    """Reject smoke, incomplete, or internally inconsistent run snapshots."""

    from src.train.artifacts import load_comparable_run
    from src.train.data import load_taxonomy
    from src.train.evaluation import RunContract, evaluate_predictions
    from src.train.execution.io import read_json_object
    from src.train.phases.label_only import LabelOnlyPhase
    from src.train.run_io import load_execution_profile

    if load_execution_profile(run_directory) != "full":
        raise ValueError(f"Confirmation run must use the full profile: {run_directory}")
    for filename in ("status.json", "run_status.json"):
        status = read_json_object(run_directory / "manifests" / filename).get("status")
        if status != "completed":
            raise ValueError(
                f"Confirmation run is not completed in {filename}: {run_directory}"
            )

    snapshot = load_comparable_run(run_directory)
    expected_count = 1229
    if config.dataset.expected.dev_image_count != expected_count:
        raise ValueError("Confirmation config must declare 1,229 development images")
    if (
        snapshot.metrics.sample_count != expected_count
        or len(snapshot.predictions) != expected_count
    ):
        raise ValueError(
            f"Confirmation snapshot must contain 1,229 predictions: {run_directory}"
        )
    sample_ids = {prediction.sample_id for prediction in snapshot.predictions}
    if len(sample_ids) != expected_count:
        raise ValueError(
            f"Confirmation snapshot has duplicate sample IDs: {run_directory}"
        )
    recomputed = evaluate_predictions(
        snapshot.predictions,
        snapshot.metrics.labels,
    )
    if recomputed != snapshot.metrics:
        raise ValueError(
            f"Confirmation metrics do not match stored predictions: {run_directory}"
        )
    if (
        snapshot.experiment_id != config.experiment.id
        or snapshot.run_id != run_directory.name
        or snapshot.seed != config.trainer.seed
    ):
        raise ValueError(
            f"Confirmation snapshot identity differs from its config: {run_directory}"
        )

    dataset_audit = read_json_object(run_directory / "manifests" / "dataset_audit.json")
    split_hash = dataset_audit.get("assignment_sha256")
    if not isinstance(split_hash, str) or not split_hash:
        raise ValueError(f"Run dataset audit has no assignment hash: {run_directory}")
    taxonomy = load_taxonomy(config)
    phase = LabelOnlyPhase(taxonomy)
    expected_contract = RunContract(
        dataset_revision=config.dataset.hub_revision,
        split_hash=split_hash,
        prompt_hash=prompt_hash(phase.prompt),
        model_revision=config.model.revision,
        label_contract_hash=label_contract_hash(taxonomy),
        training_contract_hash=controlled_training_hash(config),
    )
    if snapshot.contract != expected_contract:
        raise ValueError(
            f"Confirmation snapshot contract differs from its config: {run_directory}"
        )

    best = read_json_object(run_directory / "manifests" / "best_checkpoint.json")
    best_id = best.get("checkpoint_id")
    prediction_checkpoints = {
        prediction.checkpoint_id for prediction in snapshot.predictions
    }
    if not isinstance(best_id, str) or prediction_checkpoints != {best_id}:
        raise ValueError(
            "Final predictions do not belong to the selected checkpoint: "
            f"{run_directory}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
