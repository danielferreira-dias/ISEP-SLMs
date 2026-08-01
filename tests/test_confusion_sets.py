"""Tests for the paired visual disease confusion-set release."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import pandas as pd
import yaml

from src.data_pipeline.confusion_sets import (
    build_confusion_tasks,
    validate_confusion_set_release,
    validate_confusion_validation_release,
    validate_confusion_task_frame,
)


ROOT = Path(__file__).resolve().parents[1]


class ConfusionSetReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.definition = yaml.safe_load(
            (
                ROOT
                / "configs/taxonomies/disease_confusion_sets.yaml"
            ).read_text(encoding="utf-8")
        )
        cls.disease_taxonomy = yaml.safe_load(
            (
                ROOT / "configs/taxonomies/diseases.yaml"
            ).read_text(encoding="utf-8")
        )
        cls.active_ids = {
            str(item["id"])
            for item in cls.disease_taxonomy["diseases"]
        }
        targets = {
            "melanocytic_lookalike_lesions": 30,
            "keratinocytic_lesions": 53,
            "eczematous_dermatitis": 17,
            "acneiform_follicular": 15,
            "reactive_eruptions": 23,
        }
        rows = []
        for set_config in cls.definition["high_confusability_sets"]:
            set_id = str(set_config["id"])
            for disease_id in set_config["disease_ids"]:
                for index in range(targets[set_id]):
                    sample_id = f"{disease_id}_{index:03d}"
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "image_uri": f"images/{sample_id}.jpg",
                            "disease_id": str(disease_id),
                            "leakage_group_id": f"GROUP_{sample_id}",
                            "dataset_id": "synthetic_test",
                        }
                    )
        cls.source = pd.DataFrame(rows)

    def test_task_generation_is_deterministic_and_paired(self) -> None:
        first, first_selection = build_confusion_tasks(
            source=self.source,
            definition=self.definition,
            active_disease_ids=self.active_ids,
        )
        second, second_selection = build_confusion_tasks(
            source=self.source,
            definition=self.definition,
            active_disease_ids=self.active_ids,
        )

        self.assertEqual(first_selection, second_selection)
        self.assertEqual(len(first), 828)
        self.assertEqual(first["sample_id"].nunique(), 414)
        self.assertEqual(first["pair_id"].nunique(), 414)
        self.assertEqual(first["task_id"].tolist(), second["task_id"].tolist())
        self.assertEqual(
            first["candidate_disease_ids"].tolist(),
            second["candidate_disease_ids"].tolist(),
        )
        self.assertTrue((first.groupby("pair_id").size() == 2).all())

    def test_generated_tasks_pass_all_integrity_checks(self) -> None:
        tasks, _ = build_confusion_tasks(
            source=self.source,
            definition=self.definition,
            active_disease_ids=self.active_ids,
        )
        integrity = validate_confusion_task_frame(
            tasks=tasks,
            source=self.source,
            definition=self.definition,
            active_disease_ids=self.active_ids,
        )

        self.assertTrue(integrity["passed"])
        self.assertTrue(all(integrity["candidate_checks"].values()))
        self.assertTrue(all(integrity["paired_checks"].values()))
        self.assertTrue(all(integrity["count_checks"].values()))
        self.assertTrue(all(integrity["balanced_within_set"].values()))

    def test_release_checksums_and_schema_are_valid(self) -> None:
        benchmark = yaml.safe_load(
            (
                ROOT
                / "data/benchmarks/ISEPDermaBench/artifacts/configs/"
                "visual_confusion_sets.yaml"
            ).read_text(encoding="utf-8")
        )
        required_paths = [
            ROOT / benchmark["dataset"]["source_manifest"],
            ROOT / benchmark["dataset"]["task_manifest"],
            ROOT / benchmark["dataset"]["release_manifest"],
        ]
        if not all(path.exists() for path in required_paths):
            self.skipTest("Local generated benchmark artifacts are unavailable")
        release = validate_confusion_set_release(ROOT)
        self.assertEqual(release["id"], "visual_confusion_sets_dataset_v1")
        self.assertTrue(release["integrity_passed"])

    def test_validation_release_is_separate_and_balanced(self) -> None:
        release = validate_confusion_validation_release(ROOT)
        self.assertEqual(
            release["evaluation_origin"],
            "development_validation",
        )
        self.assertEqual(release["counts"]["selected_images"], 417)
        self.assertEqual(release["counts"]["tasks"], 834)

        benchmark = yaml.safe_load(
            (
                ROOT
                / "configs/benchmarks/derma_isep/visual_confusion_sets.yaml"
            ).read_text(encoding="utf-8")
        )
        evaluation = benchmark["dataset"]["evaluation_sets"][
            "validation_paired_confusion_tasks"
        ]
        tasks = pd.read_parquet(ROOT / evaluation["manifest"])
        self.assertEqual(len(tasks), 834)
        self.assertEqual(tasks["sample_id"].nunique(), 417)
        self.assertEqual(tasks["leakage_group_id"].nunique(), 417)
        self.assertEqual(tasks["disease_id"].nunique(), 15)

    def test_config_metrics_have_english_descriptions(self) -> None:
        benchmark = yaml.safe_load(
            (
                ROOT
                / "configs/benchmarks/derma_isep/visual_confusion_sets.yaml"
            ).read_text(encoding="utf-8")
        )
        metric_ids = {
            str(metric["id"])
            for metric in benchmark["metrics"]
        }
        self.assertIn("confusability_accuracy_gap", metric_ids)
        self.assertIn("high_confusability_accuracy", metric_ids)
        for metric in benchmark["metrics"]:
            self.assertTrue(str(metric["description"]).strip())

        schema = json.loads(
            (
                ROOT
                / "data/benchmarks/ISEPDermaBench/artifacts/schemas/"
                "visual_confusion_sets.schema.json"
            ).read_text(encoding="utf-8")
        )
        predictions = schema["properties"]["predictions"]
        self.assertEqual(predictions["minItems"], 3)
        self.assertEqual(predictions["maxItems"], 3)


if __name__ == "__main__":
    unittest.main()
