"""Tests for the frozen ISEPDermaBench runtime loader."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import yaml

from src.benchmark.isep_dermabench import (
    FrozenISEPDermaBenchAdapter,
    list_isep_dermabench_configs,
    load_isep_dermabench_config,
    load_isep_dermabench_dataset,
)
from src.benchmark.task_adapters import build_task_adapter


ROOT = Path(__file__).resolve().parents[1]


class ISEPDermaBenchLoaderTests(unittest.TestCase):
    def test_lists_all_frozen_protocols(self) -> None:
        configs = list_isep_dermabench_configs(root=ROOT)
        self.assertEqual(
            {config.benchmark.id for config in configs},
            {
                "visual_top_k_closed_set",
                "visual_disease_confusion_sets",
                "evidence_grounded_diagnosis",
                "open_ended_diagnosis",
                "visual_grounding_no_image",
                "general_visual_hallucination_audit",
                "dermatology_counterfactual_hallucination",
                "clinical_context_ablation",
            },
        )

    def test_hallucination_audits_have_frozen_development_cohorts(self) -> None:
        expected = {
            "general_visual_hallucination_audit": 300,
            "dermatology_counterfactual_hallucination": 200,
        }
        for benchmark_id, count in expected.items():
            with self.subTest(benchmark_id=benchmark_id):
                config = load_isep_dermabench_config(benchmark_id, root=ROOT)
                loaded = load_isep_dermabench_dataset(
                    root=ROOT,
                    benchmark=config,
                    evaluation_set="validation",
                    limit=None,
                    seed=42,
                    source="local",
                )
                self.assertEqual(len(loaded.samples), count)
                self.assertNotIn(
                    "reference_disease_id", loaded.frame.columns
                )

    def test_no_image_control_is_validation_only_and_reference_isolated(
        self,
    ) -> None:
        config = load_isep_dermabench_config(
            "visual_grounding_no_image",
            root=ROOT,
        )
        loaded = load_isep_dermabench_dataset(
            root=ROOT,
            benchmark=config,
            evaluation_set="validation",
            limit=None,
            seed=42,
            source="local",
        )

        self.assertEqual(len(loaded.samples), 50)
        self.assertEqual(
            len(
                {
                    sample.metadata["leakage_group_id"]
                    for sample in loaded.samples
                }
            ),
            50,
        )
        for sample in loaded.samples:
            self.assertEqual(
                sample.metadata["condition"],
                "uniform_gray_no_image",
            )
            self.assertEqual(len(sample.candidate_disease_ids or ()), 21)
        self.assertNotIn("reference_disease_id", loaded.frame.columns)

    def test_open_ended_inputs_do_not_expose_scoring_references(self) -> None:
        config = load_isep_dermabench_config(
            "open_ended_diagnosis",
            root=ROOT,
        )
        loaded = load_isep_dermabench_dataset(
            root=ROOT,
            benchmark=config,
            evaluation_set="validation",
            limit=3,
            seed=42,
            source="local",
        )

        self.assertEqual(len(loaded.samples), 3)
        for sample in loaded.samples:
            self.assertEqual(sample.candidate_disease_ids, ())
            self.assertEqual(sample.response_schema, {})
            self.assertEqual(sample.metadata["output_mode"], "free_text")
            self.assertNotIn(sample.disease_id, sample.user_prompt or "")

    def test_local_task_and_reference_views_are_joined_for_scoring(self) -> None:
        config = load_isep_dermabench_config(
            "visual_top_k_closed_set",
            root=ROOT,
        )
        loaded = load_isep_dermabench_dataset(
            root=ROOT,
            benchmark=config,
            evaluation_set="internal_benchmark",
            limit=3,
            seed=42,
            source="local",
        )
        self.assertEqual(len(loaded.samples), 3)
        for sample in loaded.samples:
            self.assertTrue(sample.image_bytes)
            self.assertTrue(sample.system_prompt)
            self.assertTrue(sample.user_prompt)
            self.assertIsInstance(sample.response_schema, dict)
            self.assertEqual(len(sample.candidate_disease_ids or ()), 21)
            self.assertTrue(sample.disease_id.startswith("D"))
            self.assertIn("reference_disease_id", sample.metadata)
            self.assertNotIn("image", sample.metadata)

    def test_confusion_limit_selects_complete_pairs(self) -> None:
        config = load_isep_dermabench_config(
            "visual_disease_confusion_sets",
            root=ROOT,
        )
        loaded = load_isep_dermabench_dataset(
            root=ROOT,
            benchmark=config,
            evaluation_set="internal_benchmark",
            limit=2,
            seed=42,
            source="local",
        )
        self.assertEqual(len(loaded.samples), 4)
        pair_ids = [sample.metadata["pair_id"] for sample in loaded.samples]
        self.assertEqual(len(set(pair_ids)), 2)
        self.assertTrue(
            all(pair_ids.count(pair_id) == 2 for pair_id in set(pair_ids))
        )

    def test_context_limit_selects_complete_identical_image_pairs(self) -> None:
        config = load_isep_dermabench_config(
            "clinical_context_ablation",
            root=ROOT,
        )
        loaded = load_isep_dermabench_dataset(
            root=ROOT,
            benchmark=config,
            evaluation_set="internal_benchmark",
            limit=2,
            seed=42,
            source="local",
        )
        self.assertEqual(len(loaded.samples), 4)
        pairs: dict[str, list] = {}
        for sample in loaded.samples:
            pairs.setdefault(sample.metadata["pair_id"], []).append(sample)
        self.assertEqual(len(pairs), 2)
        for pair in pairs.values():
            self.assertEqual(
                {sample.metadata["condition"] for sample in pair},
                {"image_only", "image_plus_context"},
            )
            self.assertEqual(len({sample.image_bytes for sample in pair}), 1)
            context_sample = next(
                sample
                for sample in pair
                if sample.metadata["condition"] == "image_plus_context"
            )
            self.assertIn(
                "Patient-reported context:",
                context_sample.user_prompt or "",
            )
        self.assertNotIn("reference_disease_id", loaded.frame.columns)

    def test_adapter_uses_row_frozen_prompt_and_schema(self) -> None:
        config = load_isep_dermabench_config(
            "evidence_grounded_diagnosis",
            root=ROOT,
        )
        raw = yaml.safe_load(config.config_path.read_text(encoding="utf-8"))
        prompt = yaml.safe_load(config.prompt_path.read_text(encoding="utf-8"))
        schema = json.loads(config.schema_path.read_text(encoding="utf-8"))
        taxonomy = yaml.safe_load(
            config.taxonomy.disease_path.read_text(encoding="utf-8")
        )
        delegate = build_task_adapter(
            benchmark_config=raw,
            prompt_config=prompt,
            schema=schema,
            disease_taxonomy_items=taxonomy["diseases"],
        )
        adapter = FrozenISEPDermaBenchAdapter(delegate)
        loaded = load_isep_dermabench_dataset(
            root=ROOT,
            benchmark=config,
            evaluation_set="internal_benchmark",
            limit=1,
            seed=42,
            source="local",
        )
        sample = loaded.samples[0]
        prepared = adapter.prepare(sample)
        self.assertEqual(prepared.system_prompt, sample.system_prompt)
        self.assertEqual(prepared.user_prompt, sample.user_prompt)
        self.assertEqual(prepared.schema, sample.response_schema)

    def test_general_hallucination_accepts_no_disease_candidates(self) -> None:
        config = load_isep_dermabench_config(
            "general_visual_hallucination_audit", root=ROOT
        )
        raw = yaml.safe_load(config.config_path.read_text(encoding="utf-8"))
        prompt = yaml.safe_load(config.prompt_path.read_text(encoding="utf-8"))
        schema = json.loads(config.schema_path.read_text(encoding="utf-8"))
        taxonomy = yaml.safe_load(
            config.taxonomy.disease_path.read_text(encoding="utf-8")
        )
        delegate = build_task_adapter(
            benchmark_config=raw,
            prompt_config=prompt,
            schema=schema,
            disease_taxonomy_items=taxonomy["diseases"],
        )
        loaded = load_isep_dermabench_dataset(
            root=ROOT,
            benchmark=config,
            evaluation_set="validation",
            limit=1,
            seed=42,
            source="local",
        )
        prepared = FrozenISEPDermaBenchAdapter(delegate).prepare(
            loaded.samples[0]
        )
        self.assertEqual(prepared.allowed_disease_ids, ())
        self.assertTrue(prepared.schema)


if __name__ == "__main__":
    unittest.main()
