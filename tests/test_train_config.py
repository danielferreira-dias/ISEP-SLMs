"""Tests for the strict immutable training YAML contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.train.config import TrainingConfig, TrainingConfigError, load_training_config
from src.train.domain import TrainingPhaseName, VisionTuningProfile
from src.train.scientific import validate_controlled_pair


class TrainingConfigTests(unittest.TestCase):
    def test_yaml_loads_defaults_and_resolves_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.yaml"
            path.write_text(yaml.safe_dump(_minimal_document()), encoding="utf-8")
            config = load_training_config(path)

        self.assertEqual(config.experiment.phase, TrainingPhaseName.E1_LABEL)
        self.assertEqual(
            config.experiment.vision_profile,
            VisionTuningProfile.FROZEN_VISION,
        )
        self.assertEqual(config.trainer.effective_batch_size, 8)
        self.assertEqual(config.dataset.expected.image_count, 7541)
        self.assertEqual(config.dataset.expected.dev_image_count, 1229)
        self.assertEqual(config.dataset.split.panel_groups_per_class, 10)
        self.assertFalse(config.artifacts.checkpoint_hub.enabled)

    def test_production_configs_enable_the_pinned_private_checkpoint_repo(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        for filename in (
            "e1_label_frozen_vision.yaml",
            "e1_label_unsloth_all.yaml",
        ):
            with self.subTest(filename=filename):
                config = load_training_config(
                    project_root / "configs" / "training" / filename
                )
                hub = config.artifacts.checkpoint_hub
                self.assertTrue(hub.enabled)
                self.assertTrue(hub.private)
                self.assertFalse(hub.upload_smoke)
                self.assertEqual(
                    hub.repo_id,
                    "danielfdias98/ISEP-training-checkpoints",
                )

    def test_e2_config_is_human_only_and_starts_from_base(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_training_config(
            project_root / "configs" / "training" / "e2_skincon_unsloth_all.yaml"
        )

        self.assertEqual(config.experiment.phase, TrainingPhaseName.E2_SKINCON)
        self.assertIsNone(config.continuation)
        self.assertIsNotNone(config.e2)
        assert config.e2 is not None
        self.assertEqual(config.e2.expected.morphology_train, 3068)
        self.assertEqual(config.e2.expected.morphology_concepts, 48)
        self.assertTrue(config.e2.verify_all_shards)

    def test_e2_skincap_ablation_pins_the_additive_release(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_training_config(
            project_root
            / "configs"
            / "training"
            / "e2_skincon_skincap_unsloth_all.yaml"
        )

        self.assertEqual(config.experiment.phase, TrainingPhaseName.E2_SKINCON)
        self.assertIsNotNone(config.e2)
        assert config.e2 is not None
        self.assertEqual(config.e2.release_id, "isep_distill_dataset_v0.4.1")
        self.assertEqual(config.e2.expected.caption_train, 2767)
        self.assertEqual(config.e2.expected.caption_dev, 483)
        self.assertEqual(
            config.e2.hub_revision,
            "b215f0474e4931b5951da768e79a0d579d26919d",
        )

    def test_e2_confirmatory_pair_differs_only_in_visual_lora(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        frozen = load_training_config(
            project_root
            / "configs"
            / "training"
            / "e2_skincon_skincap_frozen_vision.yaml"
        )
        visual = load_training_config(
            project_root
            / "configs"
            / "training"
            / "e2_skincon_skincap_unsloth_all.yaml"
        )

        self.assertEqual(frozen.trainer.learning_rate, 1e-4)
        self.assertEqual(visual.trainer.learning_rate, 1e-4)
        self.assertEqual(frozen.trainer.seed, 42)
        self.assertEqual(visual.trainer.seed, 42)
        validate_controlled_pair(frozen, visual)

    def test_e2_lr1e4_pilot_is_valid_without_weakening_e1(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        visual = load_training_config(
            project_root
            / "configs"
            / "training"
            / "e2_skincon_skincap_unsloth_all_lr1e4_pilot.yaml"
        )
        frozen = load_training_config(
            project_root
            / "configs"
            / "training"
            / "e2_skincon_skincap_frozen_vision_lr1e4_pilot.yaml"
        )

        self.assertEqual(visual.experiment.phase, TrainingPhaseName.E2_SKINCON)
        self.assertEqual(visual.trainer.learning_rate, 1e-4)
        self.assertEqual(frozen.trainer.learning_rate, 1e-4)
        validate_controlled_pair(frozen, visual)

        e1_document = _minimal_document()
        e1_document["trainer"] = {"learning_rate": 1e-4}
        with self.assertRaisesRegex(ValidationError, "E1 fixes learning_rate"):
            TrainingConfig.model_validate(e1_document, strict=True)

    def test_unknown_yaml_field_is_rejected(self) -> None:
        document = _minimal_document()
        document["unexpected"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.yaml"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            with self.assertRaises(TrainingConfigError):
                load_training_config(path)

    def test_vision_condition_must_match_lora_flag(self) -> None:
        document = _minimal_document()
        document["lora"] = {"finetune_vision_layers": True}
        with self.assertRaises(ValidationError):
            TrainingConfig.model_validate(document, strict=True)

    def test_unknown_phase_is_rejected(self) -> None:
        document = _minimal_document()
        experiment = dict(document["experiment"])
        experiment["phase"] = "structured"
        document["experiment"] = experiment
        with self.assertRaises(ValidationError):
            TrainingConfig.model_validate(document, strict=True)

    def test_configuration_is_frozen(self) -> None:
        config = TrainingConfig.model_validate(_minimal_document(), strict=True)
        with self.assertRaises(ValidationError):
            config.trainer.seed = 7  # type: ignore[misc]

    def test_fixed_model_lora_and_optimizer_recipe_cannot_drift(self) -> None:
        mutations: tuple[tuple[str, dict[str, object]], ...] = (
            ("model", {"repo_id": "other/model"}),
            (
                "lora",
                {"finetune_vision_layers": False, "rank": 8},
            ),
            ("trainer", {"learning_rate": 1e-4}),
        )
        for section, replacement in mutations:
            with self.subTest(section=section):
                document = _minimal_document()
                document[section] = replacement
                with self.assertRaises(ValidationError):
                    TrainingConfig.model_validate(document, strict=True)

    def test_controlled_pair_rejects_any_nonvision_difference(self) -> None:
        frozen_document = _minimal_document()
        visual_document = _minimal_document()
        visual_document["experiment"] = {
            "id": "e1_label_unsloth_all",
            "phase": "e1_label",
            "vision_profile": "unsloth_all",
        }
        visual_document["lora"] = {"finetune_vision_layers": True}
        frozen = TrainingConfig.model_validate(frozen_document, strict=True)
        visual = TrainingConfig.model_validate(visual_document, strict=True)

        validate_controlled_pair(frozen, visual)
        changed_dataset = visual.dataset.model_copy(update={"hub_revision": "b" * 40})
        changed_visual = visual.model_copy(update={"dataset": changed_dataset})
        with self.assertRaisesRegex(ValueError, "differ outside"):
            validate_controlled_pair(frozen, changed_visual)


def _minimal_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": {
            "id": "e1_label_frozen_vision",
            "phase": "e1_label",
            "vision_profile": "frozen_vision",
        },
        "dataset": {
            "source_directory": "data/training/ISEPDermData",
            "release_directory": ("data/training/ISEPDermData/releases/e1_label_v1"),
        },
        "model": {},
        "lora": {"finetune_vision_layers": False},
        "trainer": {},
    }


if __name__ == "__main__":
    unittest.main()
