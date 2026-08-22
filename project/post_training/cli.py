"""Command-line boundary for runnable and planned post-training stages."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from project.post_training._availability import StageAvailability
from project.post_training.grpo import GRPO_STAGE
from project.post_training.opd import OPD_STAGE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isep-post-train",
        description="ISEP student post-training (E3 SFT; E4/E5 status contracts).",
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    sft = stages.add_parser("sft", help="E3 multitask supervised fine-tuning")
    actions = sft.add_subparsers(dest="action", required=True)
    validate = actions.add_parser("validate-config")
    validate.add_argument("--config", type=Path, required=True)

    smoke = actions.add_parser("smoke-test")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--run-id")

    run = actions.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--run-id")
    run.add_argument("--resume-from", type=Path)

    stages.add_parser(
        "status",
        help="Show which post-training stages have real implementations",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one post-training command with machine-readable output."""

    arguments = _parser().parse_args(argv)
    try:
        payload = _dispatch(arguments)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"isep-post-train: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False))
    return 0


def _dispatch(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.stage == "status":
        return {
            "stages": [
                {
                    "stage_id": "e3_sft",
                    "experiment_id": "E3",
                    "method": "SFT",
                    "implemented": True,
                    "entry_point": "isep-post-train sft run",
                },
                _availability_document(OPD_STAGE),
                _availability_document(GRPO_STAGE),
            ]
        }
    if arguments.stage != "sft":
        raise ValueError(f"Unknown post-training stage: {arguments.stage}")

    from project.post_training.sft.runner import (
        audit_sft_configuration,
        run_sft,
    )

    if arguments.action == "validate-config":
        audit = audit_sft_configuration(arguments.config)
        return {
            "valid": True,
            "stage_id": audit.config.stage.id,
            "experiment": audit.config.stage.experiment,
            "initialization": audit.config.stage.initialization,
            "model_id": audit.student.student.model.id,
            "model_revision": audit.student.student.model.revision,
            "config_sha256": audit.config_sha256,
            "dataset_contract_sha256": audit.dataset_contract_sha256,
            "train_rows": audit.config.datasets.train.expected_rows,
            "dev_rows": audit.config.datasets.dev.expected_rows,
            "checkpoint_selection_split": audit.config.datasets.dev.split,
            "external_benchmark_selection": False,
        }
    if arguments.action in {"smoke-test", "run"}:
        result = run_sft(
            arguments.config,
            smoke=arguments.action == "smoke-test",
            run_id=arguments.run_id,
            resume_from_checkpoint=(
                arguments.resume_from if arguments.action == "run" else None
            ),
        )
        return {
            "run_directory": str(result.run_directory),
            "global_step": result.backend_result.global_step,
            "checkpoints": [
                str(item.path) for item in result.backend_result.checkpoints
            ],
            "final_adapter": str(result.backend_result.final_adapter_dir),
            "checkpoint_selection_status": result.checkpoint_selection_status,
            "config_sha256": result.config_sha256,
            "dataset_contract_sha256": result.dataset_contract_sha256,
        }
    raise ValueError(f"Unknown SFT action: {arguments.action}")


def _availability_document(stage: StageAvailability) -> dict[str, object]:
    return {
        "stage_id": stage.stage_id,
        "experiment_id": stage.experiment_id,
        "method": stage.method,
        "implemented": stage.implemented,
        "required_parent_stages": list(stage.required_parent_stages),
        "optional_parent_stages": list(stage.optional_parent_stages),
        "planned_framework": stage.planned_framework,
    }


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
