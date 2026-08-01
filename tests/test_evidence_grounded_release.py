"""Tests for the materialized evidence-grounded benchmark release."""

from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data/benchmarks/ISEPDermaBench"


def _split(kind: str, split: str) -> pd.DataFrame:
    paths = sorted(
        (RELEASE / kind / "evidence_grounded_diagnosis").glob(
            f"{split}-*.parquet"
        )
    )
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


class EvidenceGroundedReleaseTests(unittest.TestCase):
    def test_materialized_release_has_expected_cohorts(self) -> None:
        tasks = _split("tasks", "external_ddi")
        refs = _split("references", "external_ddi")
        self.assertEqual(len(tasks), 636)
        self.assertEqual(tasks["leakage_group_id"].nunique(), 632)
        self.assertEqual(
            {
                "morphology": int(refs["score_morphology"].sum()),
                "description": int(refs["score_description"].sum()),
                "diagnosis": int(refs["score_diagnosis"].sum()),
            },
            {
                "morphology": 636,
                "description": 635,
                "diagnosis": 294,
            },
        )

    def test_config_points_to_schema_stable_external_manifest(self) -> None:
        config = yaml.safe_load(
            (
                ROOT
                / "data/benchmarks/ISEPDermaBench/artifacts/configs/"
                "evidence_grounded_diagnosis.yaml"
            ).read_text()
        )
        self.assertEqual(config["benchmark"]["id"], "evidence_grounded_diagnosis")
        self.assertEqual(len(_split("tasks", "external_ddi")), 636)

    def test_internal_validation_release_has_independent_references(self) -> None:
        tasks = _split("tasks", "validation")
        refs = _split("references", "validation")
        self.assertEqual(len(tasks), 137)
        self.assertEqual(tasks["leakage_group_id"].nunique(), 137)
        self.assertEqual(
            {
                "morphology": int(refs["score_morphology"].sum()),
                "description": int(refs["score_description"].sum()),
                "diagnosis": int(refs["score_diagnosis"].sum()),
            },
            {
                "morphology": 137,
                "description": 124,
                "diagnosis": 137,
            },
        )

        self.assertEqual(refs["reference_disease_id"].nunique(), 19)

    def test_internal_benchmark_has_independent_references(self) -> None:
        tasks = _split("tasks", "internal_benchmark")
        refs = _split("references", "internal_benchmark")
        self.assertEqual(len(tasks), 134)
        self.assertEqual(tasks["leakage_group_id"].nunique(), 134)
        self.assertEqual(
            {
                "morphology": int(refs["score_morphology"].sum()),
                "description": int(refs["score_description"].sum()),
                "diagnosis": int(refs["score_diagnosis"].sum()),
            },
            {
                "morphology": 134,
                "description": 119,
                "diagnosis": 134,
            },
        )

        self.assertEqual(refs["reference_disease_id"].nunique(), 19)

    def test_external_experiment_references_benchmark_and_all_models(
        self,
    ) -> None:
        experiment = yaml.safe_load(
            (
                ROOT
                / "configs/experiments/"
                "zero_shot_evidence_grounded_v1.yaml"
            ).read_text()
        )
        self.assertEqual(
            experiment["benchmark"]["config"],
            "evidence_grounded_diagnosis",
        )
        self.assertFalse(
            experiment["selection_policy"]["teacher_selection_allowed"]
        )
        self.assertEqual(len(experiment["models"]), 9)
        for model in experiment["models"]:
            self.assertTrue((ROOT / model["config"]).is_file())
        self.assertEqual(
            experiment["cohorts"]["morphology"]["expected_samples"],
            636,
        )
        self.assertEqual(
            experiment["cohorts"]["description"]["expected_samples"],
            635,
        )
        self.assertEqual(
            experiment["cohorts"]["diagnosis"]["expected_samples"],
            294,
        )


if __name__ == "__main__":
    unittest.main()
