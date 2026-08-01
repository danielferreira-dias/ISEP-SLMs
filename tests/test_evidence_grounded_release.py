"""Tests for the materialized evidence-grounded benchmark release."""

from __future__ import annotations

from pathlib import Path
import unittest

import pyarrow.parquet as pq
import yaml

from src.data_pipeline.evidence_grounded import (
    TASK_ARROW_SCHEMA,
    validate_evidence_grounded_internal_release,
    validate_evidence_grounded_release,
    validate_evidence_grounded_validation_release,
)


ROOT = Path(__file__).resolve().parents[1]


class EvidenceGroundedReleaseTests(unittest.TestCase):
    def test_materialized_release_has_expected_cohorts(self) -> None:
        result = validate_evidence_grounded_release(ROOT)
        self.assertTrue(result["passed"])
        self.assertEqual(result["checksum_errors"], [])
        self.assertEqual(result["sample_count"], 636)
        self.assertEqual(result["unique_group_count"], 632)
        self.assertEqual(
            result["cohorts"],
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
                / "configs/benchmarks/derma_isep/evidence_grounded_diagnosis.yaml"
            ).read_text()
        )
        dataset = config["dataset"]
        self.assertEqual(dataset["evaluation_origin"], "external")
        path = ROOT / dataset["manifest"]
        table = pq.read_table(path)
        self.assertEqual(table.schema, TASK_ARROW_SCHEMA)
        self.assertEqual(table.num_rows, 636)

    def test_internal_validation_release_has_independent_references(self) -> None:
        result = validate_evidence_grounded_validation_release(ROOT)
        self.assertTrue(result["passed"])
        self.assertEqual(result["checksum_errors"], [])
        self.assertEqual(result["sample_count"], 137)
        self.assertEqual(result["unique_group_count"], 137)
        self.assertEqual(
            result["cohorts"],
            {
                "morphology": 137,
                "description": 124,
                "diagnosis": 137,
            },
        )

        config = yaml.safe_load(
            (
                ROOT
                / "configs/benchmarks/derma_isep/"
                "evidence_grounded_diagnosis.yaml"
            ).read_text()
        )
        evaluation = config["dataset"]["evaluation_sets"][
            "validation_fitzpatrick_evidence"
        ]
        table = pq.read_table(ROOT / evaluation["manifest"])
        self.assertEqual(table.schema, TASK_ARROW_SCHEMA)
        frame = table.to_pandas()
        self.assertTrue(
            frame["evaluation_origin"]
            .eq("development_validation")
            .all()
        )
        self.assertEqual(frame["disease_id"].nunique(), 19)

    def test_internal_benchmark_has_independent_references(self) -> None:
        result = validate_evidence_grounded_internal_release(ROOT)
        self.assertTrue(result["passed"])
        self.assertEqual(result["checksum_errors"], [])
        self.assertEqual(result["sample_count"], 134)
        self.assertEqual(result["unique_group_count"], 134)
        self.assertEqual(
            result["cohorts"],
            {
                "morphology": 134,
                "description": 119,
                "diagnosis": 134,
            },
        )

        config = yaml.safe_load(
            (
                ROOT
                / "configs/benchmarks/derma_isep/"
                "evidence_grounded_diagnosis.yaml"
            ).read_text()
        )
        evaluation = config["dataset"]["evaluation_sets"][
            "internal_benchmark_evidence"
        ]
        table = pq.read_table(ROOT / evaluation["manifest"])
        self.assertEqual(table.schema, TASK_ARROW_SCHEMA)
        frame = table.to_pandas()
        self.assertTrue(
            frame["evaluation_origin"].eq("internal_benchmark").all()
        )
        self.assertEqual(frame["disease_id"].nunique(), 19)

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
            "configs/benchmarks/derma_isep/evidence_grounded_diagnosis.yaml",
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
