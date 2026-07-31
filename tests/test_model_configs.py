"""Contract tests for benchmark model configurations."""

from __future__ import annotations

from pathlib import Path
import unittest

import yaml

from src.config.models import (
    AzureModelConfig,
    LocalModelConfig,
    list_model_configs,
    load_model_config,
)


ROOT = Path(__file__).resolve().parents[1]


class QwenModelConfigTests(unittest.TestCase):
    def test_qwen_3_5_disables_thinking_but_keeps_general_sampling(
        self,
    ) -> None:
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
            self.assertFalse(config["reasoning"]["enabled"])
            self.assertFalse(
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

    def test_qwen_3_6_disables_thinking_but_keeps_general_sampling(
        self,
    ) -> None:
        config = _load_model("qwen_3_6_27b.yaml")
        self.assertFalse(config["reasoning"]["enabled"])
        self.assertFalse(
            config["reasoning"]["chat_template_kwargs"][
                "enable_thinking"
            ]
        )
        self.assertEqual(config["generation"]["presence_penalty"], 0.0)
        self.assertEqual(config["generation"]["repetition_penalty"], 1.0)
        self.assertEqual(
            config["backend"]["profiles"]["vllm"][
                "request_timeout_seconds"
            ],
            1200,
        )

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
            if filename == "gemma_4_31b_it.yaml":
                self.assertEqual(
                    config["backend"]["profiles"]["vllm"][
                        "request_timeout_seconds"
                    ],
                    1200,
                )
            self.assertFalse(config["reasoning"]["enabled"])
            self.assertFalse(
                config["reasoning"]["chat_template_kwargs"][
                    "enable_thinking"
                ]
            )
            self.assertTrue(
                config["reasoning"]["exclude_from_structured_output"]
            )

    def test_minicpm_uses_managed_vllm_profile(self) -> None:
        config = _load_model("minicpm_v_4_6.yaml")
        self.assertEqual(
            config["source"]["repo_id"],
            "openbmb/MiniCPM-V-4.6",
        )
        profile = config["backend"]["profiles"]["vllm"]
        self.assertEqual(config["backend"]["default_profile"], "vllm")
        self.assertEqual(profile["engine"], "vllm")
        self.assertEqual(profile["max_model_len"], 16384)
        self.assertEqual(profile["limit_images_per_prompt"], 1)
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
            config["capabilities"]["structured_output_modes"],
            ["prompt_only"],
        )
        self.assertEqual(
            config["backend"]["profiles"]["vllm"]["engine"],
            "vllm",
        )
        self.assertEqual(
            config["reasoning"]["content_parser"],
            "medgemma_special_tokens",
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

    def test_gemma_e4b_uses_official_hugging_face_model(self) -> None:
        config = _load_model("gemma_4_e4b_it.yaml")
        self.assertEqual(config["model"]["id"], "gemma_4_e4b_it")
        self.assertEqual(
            config["source"]["repo_id"],
            "google/gemma-4-E4B-it",
        )
        self.assertEqual(
            config["backend"]["profiles"]["vllm"]["engine"],
            "vllm",
        )
        self.assertNotIn("artifact", config)
        self.assertTrue(config["usage"]["fine_tuning"])


class TypedModelConfigTests(unittest.TestCase):
    def test_all_nine_models_load_into_frozen_typed_configs(self) -> None:
        configs = list_model_configs(root=ROOT)

        self.assertEqual(len(configs), 9)
        self.assertEqual(len({item.model.id for item in configs}), 9)
        local = [
            item
            for item in configs
            if isinstance(item, LocalModelConfig)
        ]
        api = [
            item
            for item in configs
            if isinstance(item, AzureModelConfig)
        ]
        self.assertEqual(len(local), 7)
        self.assertEqual(len(api), 2)
        for config in local:
            profile = config.backend.active_profile
            self.assertEqual(profile.engine, "vllm")
        self.assertTrue(profile.managed)
        self.assertEqual(profile.max_model_len, 16384)
        self.assertEqual(profile.limit_images_per_prompt, 1)

    def test_small_modal_models_support_vllm_json_schema_mode(self) -> None:
        for config_id in (
            "minicpm_v_4_6",
            "gemma_4_e4b_it",
            "qwen_3_5_4b",
        ):
            with self.subTest(config_id=config_id):
                model = load_model_config(config_id, root=ROOT)
                self.assertIn(
                    "json_schema",
                    model.capabilities.structured_output_modes,
                )

    def test_kimi_profiles_and_azure_credentials_are_independent(self) -> None:
        kimi = load_model_config("kimi_k2_6", root=ROOT)
        gpt = load_model_config("gpt_5_6_luna", root=ROOT)

        self.assertEqual(
            kimi.backend.active_profile.api_style,
            "chat_completions",
        )
        self.assertEqual(
            gpt.backend.active_profile.api_style,
            "responses",
        )
        self.assertNotEqual(kimi.endpoint_env, gpt.endpoint_env)
        self.assertNotEqual(kimi.api_key_env, gpt.api_key_env)
        self.assertIsNone(kimi.backend.active_profile.api_version_env)
        self.assertIsNone(gpt.backend.active_profile.api_version_env)
        self.assertIsNone(kimi.generation.reasoning_effort)
        self.assertEqual(kimi.generation.thinking_mode, "disabled")
        self.assertEqual(
            kimi.backend.profile("azure").thinking_control,
            "reasoning_effort",
        )
        self.assertEqual(kimi.generation.profile, "kimi_instant")
        self.assertEqual(kimi.generation.temperature, 0.6)
        self.assertEqual(kimi.generation.top_p, 0.95)
        self.assertEqual(gpt.generation.reasoning_effort, "high")
        self.assertEqual(gpt.generation.profile, "provider_default")
        for config in (kimi, gpt):
            self.assertIsNone(config.generation.do_sample)
            self.assertIsNone(config.generation.seed)
        self.assertIsNone(gpt.generation.temperature)
        self.assertIsNone(gpt.generation.top_p)
        endpoint = load_model_config(
            "kimi_k2_6",
            root=ROOT,
            backend_profile="vllm_endpoint",
        )
        self.assertEqual(
            endpoint.backend.active_profile.engine,
            "vllm_endpoint",
        )
        self.assertEqual(
            endpoint.backend.active_profile.api_style,
            "chat_completions",
        )

    def test_model_can_be_loaded_by_repo_relative_path(self) -> None:
        config = load_model_config(
            "configs/models/qwen_small_4b.yaml",
            root=ROOT,
        )
        self.assertEqual(config.model.id, "qwen_3_5_4b")

    def test_qwen_2b_smoke_config_is_multimodal_and_not_shortlisted(
        self,
    ) -> None:
        config = load_model_config(
            "configs/models/smoke/qwen_3_5_2b.yaml",
            root=ROOT,
        )

        self.assertEqual(config.model.id, "qwen_3_5_2b_smoke")
        self.assertEqual(config.source.repo_id, "Qwen/Qwen3.5-2B")
        self.assertIn("image", config.capabilities.modalities)
        self.assertEqual(config.reasoning.parser, "qwen3")
        self.assertFalse(
            config.reasoning.chat_template_kwargs.enable_thinking
        )
        self.assertNotIn(
            config.model.id,
            {
                item.model.id
                for item in list_model_configs(root=ROOT)
            },
        )


def _load_model(filename: str) -> dict:
    return yaml.safe_load(
        (ROOT / "configs/models" / filename).read_text()
    )


if __name__ == "__main__":
    unittest.main()
