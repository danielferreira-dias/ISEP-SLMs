"""Tests for the Hugging Face-ready ISEPDermaBench release."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import pyarrow.parquet as pq

from src.data_pipeline.huggingface_benchmark_export import (
    validate_huggingface_benchmark_export,
)


ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "data/benchmarks/ISEPDermaBench"


class HuggingFaceBenchmarkExportTests(unittest.TestCase):
    def test_materialized_release_is_valid(self) -> None:
        result = validate_huggingface_benchmark_export(ROOT)
        self.assertEqual(result["split_count"], 14)
        self.assertEqual(result["task_count"], 7_184)
        self.assertGreater(result["embedded_image_bytes"], 0)

    def test_expected_task_counts_are_frozen(self) -> None:
        release = json.loads(
            (EXPORT / "release.json").read_text(encoding="utf-8")
        )["release"]
        observed = {
            (item["benchmark"], item["split"]): item["task_count"]
            for item in release["splits"]
        }
        self.assertEqual(
            observed,
            {
                ("visual_top_k", "validation"): 1_000,
                ("visual_top_k", "internal_benchmark"): 1_000,
                ("visual_top_k", "external_ddi"): 300,
                ("visual_top_k", "external_skindisnet"): 1_365,
                ("visual_confusion_sets", "validation"): 834,
                ("visual_confusion_sets", "internal_benchmark"): 828,
                ("evidence_grounded_diagnosis", "validation"): 137,
                (
                    "evidence_grounded_diagnosis",
                    "internal_benchmark",
                ): 134,
                ("evidence_grounded_diagnosis", "external_ddi"): 636,
                ("open_ended_diagnosis", "validation"): 100,
                ("open_ended_diagnosis", "internal_benchmark"): 300,
                ("visual_grounding_no_image", "validation"): 50,
                (
                    "general_visual_hallucination_audit",
                    "validation",
                ): 300,
                (
                    "dermatology_counterfactual_hallucination",
                    "validation",
                ): 200,
            },
        )

    def test_task_inputs_do_not_contain_scoring_references(self) -> None:
        task_path = next(
            (EXPORT / "tasks/visual_top_k").glob("validation-*.parquet")
        )
        reference_path = next(
            (EXPORT / "references/visual_top_k").glob(
                "validation-*.parquet"
            )
        )
        task_columns = set(pq.read_schema(task_path).names)
        reference_columns = set(pq.read_schema(reference_path).names)
        self.assertIn("image", task_columns)
        self.assertIn("system_prompt", task_columns)
        self.assertIn("user_prompt", task_columns)
        self.assertNotIn("reference_disease_id", task_columns)
        self.assertIn("reference_disease_id", reference_columns)
        self.assertNotIn("image", reference_columns)


if __name__ == "__main__":
    unittest.main()
