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
    def test_lists_the_three_frozen_protocols(self) -> None:
        configs = list_isep_dermabench_configs(root=ROOT)
        self.assertEqual(
            {config.benchmark.id for config in configs},
            {
                "visual_top_k_closed_set",
                "visual_disease_confusion_sets",
                "evidence_grounded_diagnosis",
            },
        )

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


if __name__ == "__main__":
    unittest.main()
