"""Contract tests for benchmark model configurations."""

from __future__ import annotations

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class QwenModelConfigTests(unittest.TestCase):
    def test_qwen_3_5_general_thinking_profile(self) -> None:
        for filename, expected_id in [
            ("qwen_small_4b.yaml", "qwen_3_5_4b"),
            ("qwen_small_9b.yaml", "qwen_3_5_9b"),
        ]:
            config = _load_model(filename)
            self.assertEqual(config["model"]["id"], expected_id)
            self.assertEqual(
                config["source"]["repo_id"],
                f"Qwen/Qwen3.5-{expected_id.rsplit('_', 1)[-1].upper()}",
            )
            self.assertTrue(config["reasoning"]["enabled"])
            self.assertTrue(
                config["reasoning"]["chat_template_kwargs"][
                    "enable_thinking"
                ]
            )
            self.assertEqual(
                config["generation"],
                {
                    "profile": "thinking_general_tasks",
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0.0,
                    "presence_penalty": 1.5,
                    "repetition_penalty": 1.0,
                },
            )

    def test_qwen_3_6_uses_its_current_presence_penalty(self) -> None:
        config = _load_model("qwen_3_6_27b.yaml")
        self.assertEqual(config["generation"]["presence_penalty"], 0.0)
        self.assertEqual(config["generation"]["repetition_penalty"], 1.0)

    def test_zero_shot_experiment_replaces_2b_with_9b(self) -> None:
        experiment = yaml.safe_load(
            (
                ROOT / "configs/experiments/zero_shot_visual_v1.yaml"
            ).read_text()
        )
        paths = {
            entry["config"] for entry in experiment["models"]
        }
        self.assertIn("configs/models/qwen_small_9b.yaml", paths)
        self.assertNotIn("configs/models/qwen_small_2b.yaml", paths)


class OtherMultimodalModelConfigTests(unittest.TestCase):
    def test_gemma_4_models_use_recommended_sampling(self) -> None:
        for filename in [
            "gemma_4_e4b_it.yaml",
            "gemma_4_31b_it.yaml",
        ]:
            config = _load_model(filename)
            self.assertEqual(
                config["generation"],
                {
                    "profile": "gemma_4_recommended",
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 64,
                    "repetition_penalty": 1.0,
                },
            )
            self.assertTrue(config["reasoning"]["enabled"])
            self.assertTrue(
                config["reasoning"]["exclude_from_structured_output"]
            )

    def test_minicpm_uses_native_transformers_recipe(self) -> None:
        config = _load_model("minicpm_v_4_6.yaml")
        self.assertEqual(
            config["source"]["repo_id"],
            "openbmb/MiniCPM-V-4.6",
        )
        self.assertEqual(
            config["backend"]["model_class"],
            "AutoModelForImageTextToText",
        )
        self.assertEqual(
            config["backend"]["minimum_transformers_version"],
            "5.7.0",
        )
        self.assertFalse(config["security"]["trust_remote_code"])
        self.assertEqual(config["processor"]["image"]["downsample_mode"], "4x")
        self.assertEqual(
            config["generation"],
            {
                "profile": "official_default",
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
            },
        )

    def test_medgemma_keeps_its_official_greedy_decoding(self) -> None:
        config = _load_model("medgemma_1_5_4b.yaml")
        self.assertEqual(config["model"]["family"], "medgemma")
        self.assertEqual(config["model"]["domain"], "medical")
        self.assertEqual(
            config["backend"]["model_class"],
            "AutoModelForImageTextToText",
        )
        self.assertEqual(
            config["generation"],
            {
                "profile": "official_greedy",
                "do_sample": False,
            },
        )

    def test_zero_shot_experiment_includes_remaining_models(self) -> None:
        experiment = yaml.safe_load(
            (
                ROOT / "configs/experiments/zero_shot_visual_v1.yaml"
            ).read_text()
        )
        paths = {entry["config"] for entry in experiment["models"]}
        self.assertTrue(
            {
                "configs/models/gemma_4_31b_it.yaml",
                "configs/models/gemma_4_e4b_it.yaml",
                "configs/models/minicpm_v_4_6.yaml",
                "configs/models/medgemma_1_5_4b.yaml",
            }.issubset(paths)
        )
        self.assertIn(
            "gemma_4_31b_it",
            experiment["teacher_selection"]["candidate_model_ids"],
        )

    def test_gemma_e4b_uses_official_transformers_model(self) -> None:
        config = _load_model("gemma_4_e4b_it.yaml")
        self.assertEqual(config["model"]["id"], "gemma_4_e4b_it")
        self.assertEqual(
            config["source"]["repo_id"],
            "google/gemma-4-E4B-it",
        )
        self.assertEqual(config["backend"]["engine"], "transformers")
        self.assertEqual(
            config["backend"]["model_class"],
            "AutoModelForMultimodalLM",
        )
        self.assertNotIn("artifact", config)
        self.assertTrue(config["usage"]["fine_tuning"])


def _load_model(filename: str) -> dict:
    return yaml.safe_load(
        (ROOT / "configs/models" / filename).read_text()
    )


if __name__ == "__main__":
    unittest.main()
