"""Contract tests for typed model and benchmark configuration loaders."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest

import yaml

from src.config import (
    ModelConfigError,
    load_model_config,
)
from src.benchmark.isep_dermabench import (
    list_isep_dermabench_configs,
    load_isep_dermabench_config,
)


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkConfigLoaderTests(unittest.TestCase):
    def test_all_benchmarks_have_defaults_budgets_and_prompt_only(self) -> None:
        configs = list_isep_dermabench_configs(root=ROOT)

        self.assertEqual(len(configs), 4)
        by_id = {item.benchmark.id: item for item in configs}
        self.assertEqual(
            by_id[
                "visual_top_k_closed_set"
            ].dataset.default_evaluation_set,
            "internal_benchmark",
        )
        self.assertEqual(
            by_id[
                "visual_disease_confusion_sets"
            ].dataset.default_evaluation_set,
            "internal_benchmark",
        )
        self.assertEqual(
            by_id[
                "evidence_grounded_diagnosis"
            ].dataset.default_evaluation_set,
            "internal_benchmark",
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
            self.assertTrue(config.dataset.default.manifest.is_dir())
            self.assertEqual(
                config.output_directory,
                ROOT / "outputs/benchmark_runs",
            )

    def test_benchmark_loads_by_id_and_relative_path(self) -> None:
        by_id = load_isep_dermabench_config(
            "visual_top_k_closed_set", root=ROOT
        )
        by_path = load_isep_dermabench_config(
            "visual_top_k.yaml",
            root=ROOT,
        )
        self.assertEqual(by_id, by_path)

    def test_unknown_benchmark_and_evaluation_set_are_clear(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Unknown ISEPDermaBench benchmark"
        ):
            load_isep_dermabench_config("does_not_exist", root=ROOT)
        config = load_isep_dermabench_config(
            "visual_top_k_closed_set", root=ROOT
        )
        with self.assertRaisesRegex(
            ValueError, "Unknown evaluation set"
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
                "qwen_3_5_4b",
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

    def test_reasoning_budget_must_be_positive(self) -> None:
        source_path = ROOT / "configs/models/qwen_3_6_27b.yaml"
        document = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        document["generation"]["reasoning_max_tokens"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ModelConfigError,
                r"generation.reasoning_max_tokens must be positive",
            ):
                load_model_config(path, root=ROOT)

    def test_reasoning_budget_and_effort_are_mutually_exclusive(self) -> None:
        source_path = (
            ROOT / "configs/models/qwen_3_7_flash_openrouter.yaml"
        )
        document = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        document["generation"]["reasoning_effort"] = "medium"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ModelConfigError,
                r"reasoning_effort and .*reasoning_max_tokens.*mutually exclusive",
            ):
                load_model_config(path, root=ROOT)

class ExperimentConfigContractTests(unittest.TestCase):
    def test_experiments_defer_to_benchmark_protocol_settings(self) -> None:
        for path in sorted(
            (ROOT / "configs/experiments").glob("*.yaml")
        ):
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertNotIn("overrides", document)
            benchmark_entries = document.get("benchmarks")
            if benchmark_entries is None:
                benchmark_entries = [document["benchmark"]]
            for entry in benchmark_entries:
                benchmark = load_isep_dermabench_config(
                    entry["config"],
                    root=ROOT,
                )
                evaluation_set = benchmark.dataset.evaluation_set(
                    entry["evaluation_set"]
                )
                self.assertTrue(evaluation_set.manifest.is_dir())

    def test_teacher_selection_is_validation_only(self) -> None:
        path = (
            ROOT
            / "configs/experiments/"
            "teacher_selection_visual_validation_v1.yaml"
        )
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        benchmark = load_isep_dermabench_config(
            document["benchmark"]["config"],
            root=ROOT,
        )
        evaluation_set = benchmark.dataset.evaluation_set(
            document["benchmark"]["evaluation_set"]
        )

        self.assertEqual(
            document["experiment"]["type"],
            "teacher_selection",
        )
        self.assertEqual(
            evaluation_set.role,
            "development_validation",
        )
        self.assertTrue(
            document["selection_policy"]["teacher_selection_allowed"]
        )
        self.assertEqual(
            document["selection_policy"]["structured_output_mode"],
            benchmark.structured_output.mode,
        )
        self.assertFalse(
            document["selection_policy"]["aggregate_score_allowed"]
        )
        self.assertEqual(
            document["repetitions"]["initial_screen"],
            {"count": 1, "seeds": [42]},
        )
        self.assertEqual(
            document["repetitions"]["finalists"]["count"],
            3,
        )
        self.assertEqual(
            document["repetitions"]["finalists"]["seeds"],
            [42, 43, 44],
        )

    def test_final_visual_experiment_forbids_selection(self) -> None:
        path = (
            ROOT / "configs/experiments/zero_shot_visual_v1.yaml"
        )
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        policy = document["selection_policy"]

        self.assertEqual(
            document["benchmark"]["evaluation_set"],
            "internal_benchmark",
        )
        for key in (
            "teacher_selection_allowed",
            "prompt_selection_allowed",
            "checkpoint_selection_allowed",
            "threshold_selection_allowed",
            "parser_selection_allowed",
            "generation_setting_selection_allowed",
        ):
            self.assertFalse(policy[key])

    def test_validation_screening_uses_fixed_paired_thinking_cohorts(
        self,
    ) -> None:
        document = yaml.safe_load(
            (
                ROOT
                / "configs/experiments/"
                "teacher_selection_validation_screening_v1.yaml"
            ).read_text(encoding="utf-8")
        )
        protocol = document["thinking_protocol"]
        self.assertEqual(protocol["phase_1"]["cli_override"], "disabled")
        self.assertEqual(protocol["phase_2"]["cli_override"], "enabled")
        self.assertTrue(protocol["phase_2"]["paired_with_phase_1"])
        self.assertEqual(
            protocol["phase_2"]["max_output_tokens_override"],
            14_336,
        )
        self.assertEqual(
            protocol["phase_2"]["reasoning_max_tokens"],
            10_240,
        )
        self.assertIsNone(protocol["fixed_exception"])
        self.assertEqual(len(document["benchmarks"]), 4)
        eligibility = {
            Path(entry["config"]).stem: entry
            for entry in document["models"]
        }
        self.assertTrue(eligibility["minimax_m3_openrouter"]["enabled"])
        self.assertTrue(eligibility["mimo_v2_5_openrouter"]["enabled"])
        self.assertTrue(eligibility["qwen_3_6_27b"]["enabled"])
        self.assertTrue(eligibility["qwen_small_4b"]["enabled"])
        self.assertEqual(len(eligibility), 5)
        self.assertNotIn("mimo_v2_5_pro_openrouter", eligibility)
        self.assertNotIn("laguna_s_2_1_openrouter", eligibility)


if __name__ == "__main__":
    unittest.main()
