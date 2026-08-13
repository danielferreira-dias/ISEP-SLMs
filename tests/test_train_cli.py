"""Tests for CLI-level safeguards that span multiple training runs."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from src.train.artifacts import ArtifactStore, write_comparable_run_snapshot
from src.train.cli import (
    _validate_comparison_protocol,
    _validate_confirmation_run,
    main,
)
from src.train.config import TrainingConfig
from src.train.data import load_taxonomy
from src.train.domain import VisionTuningProfile
from src.train.evaluation import (
    ComparableRun,
    PredictionRecord,
    RunContract,
    evaluate_predictions,
)
from src.train.phases.label_only import LabelOnlyPhase
from src.train.scientific import (
    controlled_training_hash,
    label_contract_hash,
    prompt_hash,
    resolved_config_document,
)
from tests.test_train_data import _toy_config, _write_toy_source


def _condition(
    base: TrainingConfig,
    *,
    visual: bool,
    seed: int,
) -> TrainingConfig:
    experiment = base.experiment.model_copy(
        update={
            "id": "e1_visual" if visual else "e1_frozen",
            "vision_profile": (
                VisionTuningProfile.UNSLOTH_ALL
                if visual
                else VisionTuningProfile.FROZEN_VISION
            ),
        }
    )
    lora = base.lora.model_copy(update={"finetune_vision_layers": visual})
    trainer = base.trainer.model_copy(update={"seed": seed})
    return TrainingConfig.model_validate(
        {
            **base.model_dump(exclude={"project_root", "source_config_path"}),
            "experiment": experiment.model_dump(),
            "lora": lora.model_dump(),
            "trainer": trainer.model_dump(),
            "project_root": base.project_root,
        },
        strict=True,
    )


def _write_run_config(
    path: Path,
    config: TrainingConfig,
    *,
    execution_profile: str = "full",
) -> None:
    manifests = path / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "config.resolved.json").write_text(
        json.dumps(resolved_config_document(config)), encoding="utf-8"
    )
    (manifests / "execution_context.json").write_text(
        json.dumps(
            {
                "project_root": str(config.project_root),
                "source_config_path": None,
                "execution_profile": execution_profile,
            }
        ),
        encoding="utf-8",
    )


def _confirmation_config(config: TrainingConfig) -> TrainingConfig:
    expected = config.dataset.expected.model_copy(update={"dev_image_count": 1229})
    dataset = config.dataset.model_copy(update={"expected": expected})
    return config.model_copy(update={"dataset": dataset})


def _write_confirmation_snapshot(
    path: Path,
    config: TrainingConfig,
    *,
    sample_count: int = 1229,
) -> None:
    store = ArtifactStore.at(path, create=True)
    taxonomy = load_taxonomy(config)
    labels = taxonomy.labels
    checkpoint_id = "checkpoint-30"
    predictions = tuple(
        PredictionRecord(
            sample_id=f"sample_{index:04d}",
            leakage_group_id=f"group_{index:04d}",
            true_label=labels[index % len(labels)],
            raw_output=labels[index % len(labels)],
            predicted_label=labels[index % len(labels)],
            is_valid=True,
            checkpoint_id=checkpoint_id,
            seed=config.trainer.seed,
        )
        for index in range(sample_count)
    )
    phase = LabelOnlyPhase(taxonomy)
    write_comparable_run_snapshot(
        store,
        ComparableRun(
            experiment_id=config.experiment.id,
            run_id=path.name,
            seed=config.trainer.seed,
            contract=RunContract(
                dataset_revision=config.dataset.hub_revision,
                split_hash="split-hash",
                prompt_hash=prompt_hash(phase.prompt),
                model_revision=config.model.revision,
                label_contract_hash=label_contract_hash(taxonomy),
                training_contract_hash=controlled_training_hash(config),
            ),
            predictions=predictions,
            metrics=evaluate_predictions(predictions, labels),
        ),
    )


def _write_complete_confirmation_run(path: Path, config: TrainingConfig) -> None:
    store = ArtifactStore.at(path, create=True)
    _write_run_config(path, config)
    store.write_status("completed")
    store.write_json("manifests", "run_status.json", {"status": "completed"})
    store.write_json(
        "manifests",
        "dataset_audit.json",
        {"assignment_sha256": "split-hash"},
    )
    store.write_json(
        "manifests",
        "best_checkpoint.json",
        {"checkpoint_id": "checkpoint-30"},
    )
    _write_confirmation_snapshot(path, config)


class TrainingCliTests(unittest.TestCase):
    def test_confirmatory_compare_requires_two_conditions_and_three_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            _write_toy_source(source)
            base = _toy_config(root, source, root / "release")
            paths: list[Path] = []
            for visual in (False, True):
                for seed in (42, 3407, 2026):
                    config = _condition(base, visual=visual, seed=seed)
                    run = root / f"{'visual' if visual else 'frozen'}-{seed}"
                    _write_run_config(run, config)
                    paths.append(run)

            with patch("src.train.cli._validate_confirmation_run") as validate_run:
                _validate_comparison_protocol(tuple(paths))
            self.assertEqual(validate_run.call_count, 6)
            with self.assertRaisesRegex(ValueError, "exactly six"):
                _validate_comparison_protocol(tuple(paths[:-1]))

    def test_confirmation_run_requires_full_completed_consistent_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            _write_toy_source(source)
            base = _toy_config(root, source, root / "release")
            config = _confirmation_config(_condition(base, visual=False, seed=42))
            run = root / "confirmation-run"
            _write_complete_confirmation_run(run, config)
            store = ArtifactStore.at(run)

            _validate_confirmation_run(config, run)

            _write_run_config(run, config, execution_profile="smoke")
            with self.assertRaisesRegex(ValueError, "full profile"):
                _validate_confirmation_run(config, run)
            _write_run_config(run, config)

            store.write_status("failed")
            with self.assertRaisesRegex(ValueError, "not completed"):
                _validate_confirmation_run(config, run)
            store.write_status("completed")

            _write_confirmation_snapshot(run, config, sample_count=1228)
            with self.assertRaisesRegex(ValueError, "1,229 predictions"):
                _validate_confirmation_run(config, run)
            _write_confirmation_snapshot(run, config)

            metrics_path = run / "metrics" / "classification.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["top1_accuracy"] = 0.0
            store.write_json("metrics", "classification.json", metrics)
            with self.assertRaisesRegex(ValueError, "do not match"):
                _validate_confirmation_run(config, run)
            _write_confirmation_snapshot(run, config)

            contract_path = run / "manifests" / "run_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["training_contract_hash"] = "0" * 64
            store.write_json("manifests", "run_contract.json", contract)
            with self.assertRaisesRegex(ValueError, "contract differs"):
                _validate_confirmation_run(config, run)

    def test_report_command_fails_when_final_metrics_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = ArtifactStore.create(Path(temporary), "experiment", "run")
            errors = io.StringIO()
            with (
                patch("src.train.reporting.build_run_report", return_value=None),
                redirect_stderr(errors),
            ):
                code = main(["report", "--run-dir", str(run.layout.run_directory)])

        self.assertEqual(code, 1)
        self.assertIn("classification metrics are missing", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
