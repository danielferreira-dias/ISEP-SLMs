"""Focused tests for benchmark task-adapter selection and preparation."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import yaml

from src.benchmark.runner import BenchmarkSample
from src.benchmark.task_adapters import (
    ClinicalContextAblationTaskAdapter,
    ConfusionSetTaskAdapter,
    EvidenceGroundedTaskAdapter,
    VisualGroundingNoImageTaskAdapter,
    VisualTopKTaskAdapter,
    build_task_adapter,
)


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTaskAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        taxonomy = _yaml("configs/taxonomies/diseases.yaml")
        cls.diseases = taxonomy["diseases"]

    def test_factory_prepares_existing_visual_top_k_task(self) -> None:
        adapter = _adapter(
            "visual_top_k",
            self.diseases,
        )
        self.assertIsInstance(adapter, VisualTopKTaskAdapter)
        prepared = adapter.prepare(_sample())
        disease_enum = prepared.schema["properties"]["predictions"][
            "items"
        ]["properties"]["disease_id"]["enum"]

        self.assertEqual(prepared.sample_id, "S1")
        self.assertIn("- D001: Melanoma", prepared.user_prompt)
        self.assertEqual(disease_enum, [item["id"] for item in self.diseases])

    def test_confusion_adapter_requires_and_narrows_candidates(self) -> None:
        adapter = _adapter(
            "visual_confusion_sets",
            self.diseases,
        )
        self.assertIsInstance(adapter, ConfusionSetTaskAdapter)
        sample = _sample(
            candidates=("D001", "D002", "D003"),
        )
        prepared = adapter.prepare(sample)
        response = adapter.parse_response(
            "test",
            json.dumps(
                {
                    "predictions": [
                        {"rank": 1, "disease_id": "D001"},
                        {"rank": 2, "disease_id": "D002"},
                        {"rank": 3, "disease_id": "D004"},
                    ]
                }
            ),
            prepared_task=prepared,
        )

        self.assertEqual(
            prepared.allowed_disease_ids,
            ("D001", "D002", "D003"),
        )
        self.assertFalse(response.schema_valid)
        self.assertIn(
            "prediction_2_disease_id_unknown",
            response.validation_errors,
        )

    def test_evidence_adapter_renders_both_taxonomies(self) -> None:
        config_path = "evidence_grounded_diagnosis"
        adapter = _adapter(config_path, self.diseases)
        self.assertIsInstance(adapter, EvidenceGroundedTaskAdapter)
        prepared = adapter.prepare(_sample())
        concept_enum = prepared.schema["properties"]["findings"]["items"][
            "properties"
        ]["concept_id"]["enum"]

        self.assertIn("- plaque: Plaque", prepared.user_prompt)
        self.assertIn("- D001: Melanoma", prepared.user_prompt)
        self.assertEqual(len(concept_enum), 48)
        self.assertEqual(
            prepared.schema["properties"]["differential"]["minItems"],
            6,
        )

    def test_no_image_adapter_accepts_explicit_abstention(self) -> None:
        adapter = _adapter("visual_grounding_no_image", self.diseases)
        self.assertIsInstance(adapter, VisualGroundingNoImageTaskAdapter)
        prepared = adapter.prepare(_sample())
        response = adapter.parse_response(
            "test",
            json.dumps(
                {
                    "image_status": "not_evaluable",
                    "visual_findings": [],
                    "predictions": [],
                    "confidence": "low",
                }
            ),
            prepared_task=prepared,
        )

        self.assertIn("may or may not", prepared.system_prompt)
        self.assertTrue(response.schema_valid)
        self.assertTrue(response.metadata["semantic_valid"])

    def test_context_ablation_renders_explicit_context(self) -> None:
        adapter = _adapter("clinical_context_ablation", self.diseases)
        self.assertIsInstance(adapter, ClinicalContextAblationTaskAdapter)
        sample = _sample(
            metadata={
                "clinical_context": "- Reported lesion symptoms: itching."
            }
        )
        prepared = adapter.prepare(sample)

        self.assertIn("Reported lesion symptoms: itching", prepared.user_prompt)
        self.assertEqual(len(prepared.allowed_disease_ids), 21)


def _adapter(
    key: str,
    diseases: list[dict],
):
    release = ROOT / "data/benchmarks/ISEPDermaBench/artifacts"
    benchmark = _yaml(
        f"data/benchmarks/ISEPDermaBench/artifacts/configs/{key}.yaml"
    )
    prompt_name = {
        "visual_top_k": "top_k",
        "visual_confusion_sets": "confusion_sets",
        "evidence_grounded_diagnosis": "evidence_grounded_diagnosis",
        "visual_grounding_no_image": "visual_grounding_no_image",
        "clinical_context_ablation": "clinical_context_ablation",
    }[key]
    prompt = _yaml(
        f"data/benchmarks/ISEPDermaBench/artifacts/prompts/{prompt_name}.yaml"
    )
    schema = json.loads(
        (release / "schemas" / f"{key}.schema.json").read_text(
            encoding="utf-8"
        )
    )
    return build_task_adapter(
        benchmark_config=benchmark,
        prompt_config=prompt,
        schema=schema,
        disease_taxonomy_items=diseases,
    )


def _sample(
    *,
    candidates: tuple[str, ...] | None = None,
    metadata: dict | None = None,
) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="S1",
        task_id="T1",
        image_uri="image.jpg",
        disease_id="D001",
        candidate_disease_ids=candidates,
        metadata=metadata or {},
    )


def _yaml(relative_path: str) -> dict:
    return yaml.safe_load(
        (ROOT / relative_path).read_text(encoding="utf-8")
    )


if __name__ == "__main__":
    unittest.main()
