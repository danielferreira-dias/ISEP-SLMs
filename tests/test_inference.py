"""Focused unit tests for inference transports and reasoning capture."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from src.inference.base import (
    InferenceConfigurationError,
    InferenceRequest,
    InferenceTransportError,
)
from src.inference.azure import AzureBackend
from src.inference.factory import create_backend
from src.inference.local import LocalBackend
from src.inference.openai_compatible import (
    OpenAICompatibleChatBackend,
)
from src.inference.responses import AzureResponsesBackend
from src.inference.vllm import (
    ManagedVllmServer,
    VllmBackend,
    VllmServerConfig,
    server_config_from_model,
)


class _CreateEndpoint:
    def __init__(
        self,
        response=None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.payload = None

    def create(self, **kwargs):
        self.payload = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    def __init__(
        self,
        *,
        chat_response=None,
        responses_response=None,
        error: Exception | None = None,
        model_ids: tuple[str, ...] = (),
    ) -> None:
        self.chat_create = _CreateEndpoint(chat_response, error)
        self.responses_create = _CreateEndpoint(
            responses_response,
            error,
        )
        self.chat = SimpleNamespace(
            completions=self.chat_create,
        )
        self.responses = self.responses_create
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=[
                    SimpleNamespace(id=model_id)
                    for model_id in model_ids
                ]
            )
        )


class OpenAICompatibleBackendTests(unittest.TestCase):
    def test_chat_transport_separates_final_text_and_reasoning(self) -> None:
        response = SimpleNamespace(
            id="chat_123",
            model="served-model",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content='{"predictions":[]}',
                        reasoning="Inspecting morphology.",
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=20,
                completion_tokens=12,
                total_tokens=32,
                completion_tokens_details=SimpleNamespace(
                    reasoning_tokens=7
                ),
            ),
        )
        client = _FakeClient(chat_response=response)
        backend = OpenAICompatibleChatBackend(
            model_id="benchmark-model",
            request_model="served-model",
            client=client,
            generation={
                "reasoning_effort": "high",
                "temperature": 1.0,
                "max_new_tokens": 100,
            },
            reasoning_capture="full",
        )

        result = backend.generate_result(
            system_prompt="System",
            user_prompt="User",
            image_bytes=b"\xff\xd8\xffimage",
            schema={"type": "object"},
            generation={"max_output_tokens": 50, "seed": 9},
            request_id="sample_1",
        )

        self.assertEqual(result.final_text, '{"predictions":[]}')
        self.assertEqual(result.reasoning.text, "Inspecting morphology.")
        self.assertEqual(result.reasoning.source_field, "reasoning")
        self.assertEqual(result.reasoning.token_count, 7)
        self.assertEqual(result.request_id, "sample_1")
        payload = client.chat_create.payload
        self.assertEqual(payload["model"], "served-model")
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(payload["max_tokens"], 50)
        self.assertEqual(payload["seed"], 9)
        self.assertNotIn("response_format", payload)
        user_content = payload["messages"][1]["content"]
        self.assertEqual(user_content[0]["type"], "image_url")
        self.assertTrue(
            user_content[0]["image_url"]["url"].startswith(
                "data:image/jpeg;base64,"
            )
        )
        self.assertEqual(user_content[1]["type"], "text")

    def test_chat_transport_accepts_legacy_reasoning_content(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="{}",
                        reasoning_content="Legacy trace",
                    ),
                )
            ],
            usage=None,
        )
        backend = OpenAICompatibleChatBackend(
            model_id="kimi",
            client=_FakeClient(chat_response=response),
            reasoning_capture="full",
        )

        result = backend.generate_result(
            system_prompt="System",
            user_prompt="User",
            image_bytes=b"\x89PNG\r\n\x1a\nimage",
            schema={},
        )

        self.assertEqual(result.reasoning.text, "Legacy trace")
        self.assertEqual(
            result.reasoning.source_field,
            "reasoning_content",
        )

    def test_chat_transport_can_prepend_system_text_for_medgemma(
        self,
    ) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="{}"),
                )
            ],
            usage=None,
        )
        client = _FakeClient(chat_response=response)
        backend = OpenAICompatibleChatBackend(
            model_id="medgemma",
            client=client,
            supports_system_role=False,
        )

        backend.generate_result(
            system_prompt="Clinical system instructions.",
            user_prompt="Assess the image.",
            image_bytes=b"image",
            schema={},
        )

        messages = client.chat_create.payload["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(
            content[1]["text"],
            "Clinical system instructions.\n\nAssess the image.",
        )

    def test_summary_mode_never_relabels_raw_reasoning_as_summary(
        self,
    ) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="{}",
                        reasoning="Raw reasoning",
                    ),
                )
            ],
            usage=SimpleNamespace(
                completion_tokens_details=SimpleNamespace(
                    reasoning_tokens=4
                )
            ),
        )
        backend = OpenAICompatibleChatBackend(
            model_id="model",
            client=_FakeClient(chat_response=response),
            reasoning_capture="summary",
        )

        result = backend.generate_result(
            system_prompt="System",
            user_prompt="User",
            image_bytes=b"image",
            schema={},
        )

        self.assertIsNone(result.reasoning.text)
        self.assertEqual(result.reasoning.token_count, 4)

    def test_transport_failure_does_not_expose_provider_secret(self) -> None:
        secret = "super-secret-api-key"
        backend = OpenAICompatibleChatBackend(
            model_id="model",
            client=_FakeClient(error=RuntimeError(secret)),
        )

        with self.assertRaises(InferenceTransportError) as context:
            backend.generate_result(
                system_prompt="System",
                user_prompt="User",
                image_bytes=b"image",
                schema={},
            )

        self.assertNotIn(secret, str(context.exception))

    def test_default_batch_api_preserves_request_order(self) -> None:
        responses = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=str(index)),
                    )
                ],
                usage=None,
            )
            for index in range(2)
        ]

        class SequencedEndpoint:
            def create(self, **kwargs):
                return responses.pop(0)

        client = _FakeClient()
        client.chat.completions = SequencedEndpoint()
        backend = OpenAICompatibleChatBackend(
            model_id="model",
            client=client,
        )
        requests = [
            InferenceRequest(
                system_prompt="System",
                user_prompt=f"User {index}",
                image_bytes=b"image",
                schema={},
            )
            for index in range(2)
        ]

        results = backend.generate_batch(requests)

        self.assertEqual(
            [result.final_text for result in results],
            ["0", "1"],
        )


class ResponsesBackendTests(unittest.TestCase):
    def test_responses_requests_official_summary_and_marks_truncation(
        self,
    ) -> None:
        response = SimpleNamespace(
            id="resp_123",
            model="deployment",
            status="incomplete",
            incomplete_details=SimpleNamespace(
                reason="max_output_tokens"
            ),
            output_text=None,
            output=[
                SimpleNamespace(
                    type="reasoning",
                    # Raw content must not be retained by this backend.
                    content=[
                        SimpleNamespace(
                            type="reasoning_text",
                            text="Raw chain of thought",
                        )
                    ],
                    summary=[
                        SimpleNamespace(
                            type="summary_text",
                            text="Compared visible findings.",
                        )
                    ],
                ),
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text=json.dumps({"predictions": []}),
                        )
                    ],
                ),
            ],
            usage=SimpleNamespace(
                input_tokens=18,
                output_tokens=30,
                total_tokens=48,
                output_tokens_details=SimpleNamespace(
                    reasoning_tokens=21
                ),
            ),
        )
        client = _FakeClient(responses_response=response)
        backend = AzureResponsesBackend(
            model_id="gpt",
            deployment_name="deployment",
            client=client,
            generation={"reasoning_effort": "high"},
            reasoning_capture="full",
        )

        result = backend.generate_result(
            system_prompt="System",
            user_prompt="User",
            image_bytes=b"image",
            schema={"type": "object"},
            generation={"max_output_tokens": 30},
        )

        self.assertEqual(
            result.reasoning.text,
            "Compared visible findings.",
        )
        self.assertNotIn("Raw chain", result.reasoning.text)
        self.assertEqual(result.reasoning.token_count, 21)
        self.assertEqual(result.finish_reason, "length")
        self.assertTrue(result.metadata["truncated"])
        self.assertEqual(
            result.metadata["incomplete_reason"],
            "max_output_tokens",
        )
        payload = client.responses_create.payload
        self.assertEqual(
            payload["reasoning"],
            {"effort": "high", "summary": "auto"},
        )
        self.assertEqual(payload["max_output_tokens"], 30)
        self.assertNotIn("text", payload)

    def test_responses_json_schema_is_explicit_opt_in(self) -> None:
        response = SimpleNamespace(
            output_text="{}",
            output=[],
            usage=None,
            status="completed",
        )
        client = _FakeClient(responses_response=response)
        backend = AzureResponsesBackend(
            model_id="gpt",
            client=client,
            use_json_schema=True,
        )

        backend.generate_result(
            system_prompt="System",
            user_prompt="User",
            image_bytes=b"image",
            schema={"type": "object"},
        )

        text_format = client.responses_create.payload["text"]["format"]
        self.assertEqual(text_format["type"], "json_schema")
        self.assertTrue(text_format["strict"])


class VllmBackendTests(unittest.TestCase):
    def test_preflight_checks_health_and_exact_served_model(self) -> None:
        backend = VllmBackend(
            model_id="qwen",
            request_model="Qwen/Qwen3.5-4B",
            client=_FakeClient(
                model_ids=("Qwen/Qwen3.5-4B",)
            ),
            health_probe=lambda url, timeout: (
                url == "http://localhost:8000/health"
            ),
            base_url="http://localhost:8000/v1",
        )

        result = backend.preflight()

        self.assertTrue(result.ok)
        self.assertEqual(
            result.checks,
            ("health_endpoint", "model_available"),
        )

    def test_managed_server_starts_waits_and_terminates(self) -> None:
        captured = {}

        class FakeProcess:
            pid = 123

            def __init__(self) -> None:
                self.exit_code = None
                self.terminated = False

            def poll(self):
                return self.exit_code

            def terminate(self):
                self.terminated = True
                self.exit_code = 0

            def wait(self, timeout):
                return self.exit_code

            def kill(self):
                self.exit_code = -9

        process = FakeProcess()

        def process_factory(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return process

        config = VllmServerConfig(
            model="Qwen/Qwen3.5-4B",
            reasoning_parser="qwen3",
            max_model_len=4096,
        )
        server = ManagedVllmServer(
            config,
            environment={"HF_TOKEN": "private-token"},
            process_factory=process_factory,
            ready_check=lambda: True,
        )

        server.start()
        self.assertTrue(server.is_running)
        self.assertIn("--reasoning-parser", captured["command"])
        self.assertNotIn("private-token", captured["command"])
        server.stop()

        self.assertTrue(process.terminated)
        self.assertFalse(server.is_running)

    def test_managed_server_rejects_secret_cli_arguments(self) -> None:
        with self.assertRaises(InferenceConfigurationError):
            VllmServerConfig(
                model="model",
                additional_args=("--api-key=secret",),
            )

    def test_managed_server_writes_to_optional_log_file(self) -> None:
        captured = {}

        class FakeProcess:
            pid = 456

            def __init__(self) -> None:
                self.exit_code = None

            def poll(self):
                return self.exit_code

            def terminate(self):
                self.exit_code = 0

            def wait(self, timeout):
                return self.exit_code

            def kill(self):
                self.exit_code = -9

        def process_factory(command, **kwargs):
            captured.update(kwargs)
            kwargs["stdout"].write(b"server output\n")
            return FakeProcess()

        with TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "logs" / "vllm.log"
            server = ManagedVllmServer(
                VllmServerConfig(model="Qwen/Qwen3.5-4B"),
                process_factory=process_factory,
                ready_check=lambda: True,
                log_path=log_path,
            )

            server.start()
            self.assertIs(captured["stdout"], captured["stderr"])
            log_handle = captured["stdout"]
            server.stop()

            self.assertTrue(log_handle.closed)
            self.assertEqual(
                log_path.read_text(encoding="utf-8"),
                "server output\n",
            )

    def test_server_config_is_built_from_normalized_model_config(
        self,
    ) -> None:
        model_config = SimpleNamespace(
            source=SimpleNamespace(repo_id="Qwen/Qwen3.5-9B"),
            processor=None,
            backend=SimpleNamespace(
                active_profile=SimpleNamespace(
                    engine="vllm",
                    dtype="bfloat16",
                    managed=True,
                    managed_allowed=True,
                    tensor_parallel_size=2,
                    max_model_len=8192,
                    gpu_memory_utilization=0.85,
                    limit_images_per_prompt=1,
                )
            ),
            reasoning=SimpleNamespace(parser="qwen3"),
        )

        server_config = server_config_from_model(
            model_config,
            port=8100,
        )

        self.assertEqual(server_config.model, "Qwen/Qwen3.5-9B")
        self.assertEqual(server_config.port, 8100)
        self.assertEqual(server_config.dtype, "bfloat16")
        self.assertEqual(server_config.tensor_parallel_size, 2)
        self.assertEqual(server_config.max_model_len, 8192)
        self.assertEqual(server_config.gpu_memory_utilization, 0.85)
        self.assertEqual(server_config.reasoning_parser, "qwen3")
        self.assertIn(
            "--limit-mm-per-prompt",
            server_config.command(),
        )
        self.assertIn('{"image": 1}', server_config.command())

    def test_server_config_includes_image_processor_kwargs(self) -> None:
        model_config = SimpleNamespace(
            source=SimpleNamespace(repo_id="openbmb/MiniCPM-V-4.6"),
            processor=SimpleNamespace(
                image=SimpleNamespace(
                    downsample_mode="4x",
                    max_slice_nums=36,
                )
            ),
            backend=SimpleNamespace(
                active_profile=SimpleNamespace(
                    engine="vllm",
                    managed=True,
                    managed_allowed=True,
                    limit_images_per_prompt=1,
                )
            ),
            reasoning=SimpleNamespace(parser=None),
        )

        command = server_config_from_model(model_config).command()

        processor_flag_index = command.index(
            "--mm-processor-kwargs"
        )
        self.assertEqual(
            json.loads(command[processor_flag_index + 1]),
            {
                "downsample_mode": "4x",
                "max_slice_nums": 36,
            },
        )


class BackendFactoryTests(unittest.TestCase):
    def test_factory_dispatches_normalized_backend_profiles(self) -> None:
        generation = SimpleNamespace(
            temperature=0.0,
            max_new_tokens=32,
        )
        local_config = SimpleNamespace(
            model_id="qwen",
            backend=SimpleNamespace(
                active_profile=SimpleNamespace(
                    engine="vllm",
                    base_url_env=None,
                    api_key_env=None,
                    model_env=None,
                )
            ),
            source=SimpleNamespace(repo_id="Qwen/Qwen3.5-4B"),
            reasoning=SimpleNamespace(
                capture_mode="none",
                chat_template_kwargs={"enable_thinking": False},
            ),
            generation=generation,
        )
        azure_config = SimpleNamespace(
            model_id="gpt",
            backend=SimpleNamespace(
                active_profile=SimpleNamespace(
                    engine="azure_openai",
                    api_style="responses",
                    endpoint_env="TEST_ENDPOINT",
                    api_key_env="TEST_KEY",
                    deployment_env="TEST_DEPLOYMENT",
                    model_env=None,
                    api_version_env=None,
                    base_url_env=None,
                )
            ),
            source=SimpleNamespace(model_name="gpt-deployment"),
            reasoning=SimpleNamespace(capture_mode="summary"),
            generation=generation,
        )

        local = create_backend(
            local_config,
            client=_FakeClient(),
        )
        azure = create_backend(
            azure_config,
            client=_FakeClient(),
        )

        self.assertIsInstance(local, LocalBackend)
        self.assertIsInstance(azure, AzureBackend)


if __name__ == "__main__":
    unittest.main()
