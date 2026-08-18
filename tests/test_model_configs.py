"""Contract tests for benchmark model configurations."""

from __future__ import annotations

import unittest
from pathlib import Path

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
        config = _load_model("qwen_small_4b.yaml")
        self.assertEqual(config["model"]["id"], "qwen_3_5_4b")
        self.assertEqual(
            config["source"]["repo_id"],
            "Qwen/Qwen3.5-4B",
        )
        self.assertFalse(config["reasoning"]["enabled"])
        self.assertFalse(config["reasoning"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(
            config["generation"],
            {
                "profile": "thinking_general_tasks",
                "reasoning_max_tokens": 10_240,
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
        self.assertFalse(config["reasoning"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(config["generation"]["presence_penalty"], 0.0)
        self.assertEqual(config["generation"]["repetition_penalty"], 1.0)
        self.assertEqual(
            config["generation"]["reasoning_max_tokens"],
            10_240,
        )
        self.assertEqual(config["generation"]["thinking_mode"], "disabled")
        self.assertEqual(
            config["backend"]["profiles"]["vllm"]["request_timeout_seconds"],
            1200,
        )


class TypedModelConfigTests(unittest.TestCase):
    def test_all_models_load_into_frozen_typed_configs(self) -> None:
        configs = list_model_configs(root=ROOT)

        expected_ids = {
            "gemini_3_5_flash_lite_openrouter",
            "gemini_3_7_flash_medium_openrouter",
            "gpt_5_6_luna",
            "gpt_5_6_sol",
            "mimo_v2_5_openrouter",
            "minimax_m3_openrouter",
            "qwen_3_5_4b",
            "qwen_3_5_4b_e1_frozen_vision",
            "qwen_3_5_4b_e1_frozen_vision_t06",
            "qwen_3_5_4b_e1_vision_lora",
            "qwen_3_5_4b_e1_vision_lora_attribution",
            "qwen_3_5_4b_e1_vision_lora_t06",
            "qwen_3_5_4b_e2_frozen_vision_t06",
            "qwen_3_5_4b_e2_vision_lora_t06",
            "qwen_3_6_27b",
            "qwen_3_7_flash_openrouter",
            "qwen_3_8_27b",
            "qwen_3_8_max_openrouter",
        }
        self.assertEqual({item.model.id for item in configs}, expected_ids)
        local = [item for item in configs if isinstance(item, LocalModelConfig)]
        api = [item for item in configs if isinstance(item, AzureModelConfig)]
        self.assertEqual(len(local), 10)
        self.assertEqual(len(api), 8)
        for config in local:
            profile = config.backend.active_profile
            self.assertIn(profile.engine, {"vllm", "transformers"})
            self.assertEqual(profile.max_model_len, 32768)
            self.assertEqual(profile.limit_images_per_prompt, 1)

    def test_official_student_supports_vllm_json_schema_mode(self) -> None:
        model = load_model_config("qwen_3_5_4b", root=ROOT)
        self.assertIn(
            "json_schema",
            model.capabilities.structured_output_modes,
        )

    def test_sol_azure_profile_is_multimodal_responses_api(self) -> None:
        sol = load_model_config("gpt_5_6_sol", root=ROOT)
        self.assertEqual(sol.model.display_name, "GPT-5.6 Sol (Azure)")
        self.assertEqual(sol.source.model_name, "gpt-5.6-sol")
        self.assertEqual(sol.backend.active_profile.engine, "azure_openai")
        self.assertEqual(sol.backend.active_profile.api_style, "responses")
        self.assertEqual(
            sol.backend.active_profile.endpoint_env,
            "GPT_5_6_SOL_AZURE_ENDPOINT",
        )
        self.assertEqual(
            sol.backend.active_profile.deployment_env,
            "GPT_5_6_SOL_AZURE_DEPLOYMENT",
        )
        self.assertEqual(
            sol.backend.active_profile.api_key_env,
            "GPT_5_6_SOL_AZURE_API_KEY",
        )
        self.assertEqual(sol.capabilities.modalities, ("text", "image"))
        self.assertEqual(sol.generation.profile, "e3_teacher_medium")
        self.assertEqual(sol.generation.reasoning_effort, "medium")

    def test_openrouter_profiles_use_explicit_provider_model_slugs(self) -> None:
        expected = {
            "gpt_5_6_luna": "openai/gpt-5.6-luna-pro",
        }
        for model_id, request_model in expected.items():
            with self.subTest(model_id=model_id):
                config = load_model_config(
                    model_id,
                    root=ROOT,
                    backend_profile="openrouter",
                )
                profile = config.backend.active_profile
                self.assertEqual(profile.engine, "openrouter")
                self.assertEqual(profile.request_model, request_model)
                self.assertEqual(
                    profile.thinking_control,
                    "openrouter_reasoning",
                )
                self.assertEqual(profile.api_key_env, "OPENROUTER_API_KEY")

        judge = load_model_config(
            "qwen_3_7_flash_openrouter",
            root=ROOT,
        )
        self.assertEqual(
            judge.backend.active_profile.request_model,
            "qwen/qwen3.7-flash",
        )
        self.assertIn(
            "json_schema",
            judge.capabilities.structured_output_modes,
        )
        self.assertEqual(judge.generation.reasoning_max_tokens, 10_240)
        self.assertEqual(judge.generation.thinking_mode, "disabled")
        self.assertIsNone(judge.generation.reasoning_effort)
        self.assertIsNone(judge.generation.temperature)
        self.assertEqual(judge.generation.top_p, 0.95)
        self.assertEqual(judge.generation.presence_penalty, 0.0)
        self.assertEqual(
            judge.backend.active_profile.provider.only,
            ("alibaba",),
        )
        self.assertFalse(judge.backend.active_profile.provider.allow_fallbacks)
        self.assertTrue(judge.backend.active_profile.provider.require_parameters)

        max_model = load_model_config(
            "qwen_3_8_max_openrouter",
            root=ROOT,
        )
        self.assertEqual(
            max_model.backend.active_profile.request_model,
            "qwen/qwen3.8-max",
        )
        self.assertEqual(
            max_model.backend.active_profile.provider.only,
            ("alibaba",),
        )
        self.assertFalse(max_model.backend.active_profile.provider.allow_fallbacks)
        self.assertTrue(max_model.backend.active_profile.provider.require_parameters)
        self.assertIn("image", max_model.capabilities.modalities)
        self.assertIn(
            "json_schema",
            max_model.capabilities.structured_output_modes,
        )
        self.assertTrue(max_model.reasoning.enabled)
        self.assertEqual(max_model.generation.thinking_mode, "enabled")
        self.assertEqual(max_model.generation.reasoning_effort, "low")
        self.assertIsNone(max_model.generation.reasoning_max_tokens)

    def test_new_openrouter_reasoning_candidates_match_capabilities(self) -> None:
        minimax = load_model_config(
            "minimax_m3_openrouter",
            root=ROOT,
        )
        self.assertTrue(minimax.usage.benchmark)
        self.assertIn("image", minimax.capabilities.modalities)
        self.assertEqual(
            minimax.backend.active_profile.request_model,
            "minimax/minimax-m3",
        )
        self.assertEqual(minimax.generation.temperature, 1.0)
        self.assertEqual(minimax.generation.top_p, 0.95)
        self.assertEqual(minimax.generation.reasoning_max_tokens, 10_240)
        self.assertEqual(
            minimax.backend.active_profile.provider.only,
            ("minimax",),
        )
        self.assertFalse(minimax.backend.active_profile.provider.allow_fallbacks)
        self.assertFalse(minimax.backend.active_profile.supports_seed)

        mimo = load_model_config("mimo_v2_5_openrouter", root=ROOT)
        self.assertTrue(mimo.usage.benchmark)
        self.assertIn("image", mimo.capabilities.modalities)
        self.assertEqual(
            mimo.backend.active_profile.request_model,
            "xiaomi/mimo-v2.5",
        )
        self.assertIn(
            "json_schema",
            mimo.capabilities.structured_output_modes,
        )
        self.assertEqual(mimo.generation.temperature, 1.0)
        self.assertEqual(mimo.generation.top_p, 0.95)
        self.assertEqual(mimo.generation.reasoning_max_tokens, 10_240)
        self.assertEqual(
            mimo.backend.active_profile.provider.only,
            ("xiaomi",),
        )
        self.assertFalse(mimo.backend.active_profile.provider.allow_fallbacks)
        self.assertFalse(mimo.backend.active_profile.supports_seed)

        gemini = load_model_config(
            "gemini_3_5_flash_lite_openrouter",
            root=ROOT,
        )
        self.assertTrue(gemini.usage.benchmark)
        self.assertIn("image", gemini.capabilities.modalities)
        self.assertEqual(
            gemini.backend.active_profile.request_model,
            "google/gemini-3.5-flash-lite",
        )
        self.assertEqual(
            gemini.backend.active_profile.provider.only,
            ("google-ai-studio",),
        )
        self.assertFalse(gemini.backend.active_profile.provider.allow_fallbacks)
        self.assertTrue(gemini.backend.active_profile.provider.require_parameters)
        self.assertTrue(gemini.reasoning.enabled)
        self.assertEqual(gemini.generation.thinking_mode, "enabled")
        self.assertEqual(gemini.generation.reasoning_effort, "minimal")
        self.assertIsNone(gemini.generation.reasoning_max_tokens)
        self.assertIsNone(gemini.generation.temperature)
        self.assertIsNone(gemini.generation.top_p)
        self.assertIsNone(gemini.generation.seed)

    def test_model_can_be_loaded_by_repo_relative_path(self) -> None:
        config = load_model_config(
            "configs/models/qwen_small_4b.yaml",
            root=ROOT,
        )
        self.assertEqual(config.model.id, "qwen_3_5_4b")


def _load_model(filename: str) -> dict:
    return yaml.safe_load((ROOT / "configs/models" / filename).read_text())


if __name__ == "__main__":
    unittest.main()
