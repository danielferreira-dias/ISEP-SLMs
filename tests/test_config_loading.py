"""Contract tests for typed model and benchmark configuration loaders."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest

import yaml

from src.config import (
    BenchmarkConfigError,
    ModelConfigError,
    list_benchmark_configs,
    load_benchmark_config,
    load_model_config,
)


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkConfigLoaderTests(unittest.TestCase):
    def test_all_benchmarks_have_defaults_budgets_and_prompt_only(self) -> None:
        configs = list_benchmark_configs(root=ROOT)

        self.assertEqual(len(configs), 3)
        by_id = {item.benchmark.id: item for item in configs}
        self.assertEqual(
            by_id[
                "visual_top_k_closed_set"
            ].dataset.default_evaluation_set,
            "internal_benchmark_1000",
        )
        self.assertEqual(
            by_id[
                "visual_disease_confusion_sets"
            ].dataset.default_evaluation_set,
            "paired_confusion_tasks",
        )
        self.assertEqual(
            by_id[
                "evidence_grounded_diagnosis"
            ].dataset.default_evaluation_set,
            "external_ddi_evidence",
        )
        self.assertEqual(
            by_id["visual_top_k_closed_set"].max_output_tokens,
            8192,
        )
        self.assertEqual(
            by_id[
                "visual_disease_confusion_sets"
            ].max_output_tokens,
            8192,
        )
        self.assertEqual(
            by_id["evidence_grounded_diagnosis"].max_output_tokens,
            8192,
        )
        for config in configs:
            self.assertEqual(config.structured_output.mode, "prompt_only")
            self.assertEqual(
                config.image_preprocessing.profile,
                "dermatology_api_safe_rgb_jpeg_v1",
            )
            self.assertEqual(
                config.image_preprocessing.max_encoded_bytes,
                45_000,
            )
            self.assertTrue(config.dataset.default.manifest.is_file())
            self.assertEqual(
                config.output_directory,
                ROOT / "outputs/benchmark_runs",
            )

    def test_benchmark_loads_by_id_and_relative_path(self) -> None:
        by_id = load_benchmark_config(
            "visual_top_k_closed_set", root=ROOT
        )
        by_path = load_benchmark_config(
            "configs/benchmarks/visual_top_k.yaml",
            root=ROOT,
        )
        self.assertEqual(by_id, by_path)

    def test_unknown_benchmark_and_evaluation_set_are_clear(self) -> None:
        with self.assertRaisesRegex(
            BenchmarkConfigError, "Unknown benchmark ID"
        ):
            load_benchmark_config("does_not_exist", root=ROOT)
        config = load_benchmark_config(
            "visual_top_k_closed_set", root=ROOT
        )
        with self.assertRaisesRegex(
            BenchmarkConfigError, "available sets"
        ):
            config.dataset.evaluation_set("does_not_exist")


class StrictModelConfigLoaderTests(unittest.TestCase):
    def test_loaded_dataclasses_are_frozen(self) -> None:
        config = load_model_config("qwen_3_5_4b", root=ROOT)
        with self.assertRaises(FrozenInstanceError):
            config.model.id = "changed"

    def test_unknown_model_and_profile_are_clear(self) -> None:
        with self.assertRaisesRegex(ModelConfigError, "Unknown model ID"):
            load_model_config("does_not_exist", root=ROOT)
        with self.assertRaisesRegex(
            ModelConfigError, "available profiles"
        ):
            load_model_config(
                "kimi_k2_6",
                root=ROOT,
                backend_profile="does_not_exist",
            )

    def test_unknown_runtime_field_is_rejected_with_its_path(self) -> None:
        source_path = ROOT / "configs/models/qwen_small_4b.yaml"
        document = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        document["generation"]["temperatur"] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ModelConfigError,
                r"generation contains unknown field.*temperatur",
            ):
                load_model_config(path, root=ROOT)


class ExperimentConfigContractTests(unittest.TestCase):
    def test_experiments_defer_to_benchmark_protocol_settings(self) -> None:
        for path in sorted(
            (ROOT / "configs/experiments").glob("*.yaml")
        ):
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertNotIn("overrides", document)
            benchmark = load_benchmark_config(
                document["benchmark"]["config"],
                root=ROOT,
            )
            self.assertEqual(
                document["benchmark"]["evaluation_set"],
                benchmark.dataset.default_evaluation_set,
            )


if __name__ == "__main__":
    unittest.main()
