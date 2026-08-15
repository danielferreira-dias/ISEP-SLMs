"""Focused unit tests for inference transports and reasoning capture."""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from src.inference.azure import AzureBackend
from src.inference.base import (
    InferenceConfigurationError,
    InferenceRequest,
    InferenceSafetyRefusal,
    InferenceTransportError,
)
from src.inference.factory import create_backend
from src.inference.local import LocalBackend
from src.inference.openai_compatible import (
    OpenAICompatibleChatBackend,
)
from src.inference.reasoning_parsing import separate_embedded_reasoning
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
                data=[SimpleNamespace(id=model_id) for model_id in model_ids]
            )
        )


class _AsyncCreateEndpoint:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.payload = None

    async def create(self, **kwargs):
        self.payload = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class _FakeAsyncClient:
    def __init__(self, *, chat_response=None, error=None) -> None:
        self.chat_create = _AsyncCreateEndpoint(chat_response, error)
        self.chat = SimpleNamespace(completions=self.chat_create)


class _AsyncStream:
    def __init__(self, chunks) -> None:
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def close(self) -> None:
        self.closed = True


class OpenAICompatibleBackendTests(unittest.TestCase):
    def test_async_chat_transport_uses_native_async_client(self) -> None:
        response = SimpleNamespace(
            id="chat_async",
            model="served-model",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content='{"result":"ok"}'),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )
        async_client = _FakeAsyncClient(chat_response=response)
        backend = OpenAICompatibleChatBackend(
            model_id="model",
            request_model="served-model",
            async_client=async_client,
        )
        request = InferenceRequest(
            system_prompt="System",
            user_prompt="User",
            image_bytes=b"image",
            schema={},
        )

        result = asyncio.run(backend.acomplete(request))

        self.assertEqual(result.final_text, '{"result":"ok"}')
        self.assertEqual(result.usage.total_tokens, 15)
        self.assertEqual(
            async_client.chat_create.payload["model"],
            "served-model",
        )

    def test_async_stream_preserves_content_reasoning_and_usage(self) -> None:
        stream = _AsyncStream(
            [
                SimpleNamespace(
                    id="stream_async",
                    model="served-model",
                    choices=[
                        SimpleNamespace(
                            finish_reason=None,
                            delta=SimpleNamespace(
                                content=None,
                                reasoning_content="Inspect ",
                            ),
                        )
                    ],
                    usage=None,
                ),
                SimpleNamespace(
                    id="stream_async",
                    model="served-model",
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            delta=SimpleNamespace(
                                content='{"ok":true}',
                                reasoning_content="image.",
                            ),
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=7,
                        total_tokens=17,
                        completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
                    ),
                ),
            ]
        )
        backend = OpenAICompatibleChatBackend(
            model_id="model",
            async_client=_FakeAsyncClient(chat_response=stream),
            reasoning_capture="full",
            stream_responses=True,
        )

        result = asyncio.run(
            backend.acomplete(
                InferenceRequest(
                    system_prompt="System",
                    user_prompt="User",
                    image_bytes=b"image",
                    schema={},
                )
            )
        )

        self.assertEqual(result.final_text, '{"ok":true}')
        self.assertEqual(result.reasoning.text, "Inspect image.")
        self.assertEqual(result.usage.reasoning_tokens, 2)
        self.assertTrue(result.metadata["async_transport"])
        self.assertTrue(stream.closed)

    def test_openrouter_uses_unified_reasoning_and_text_first(self) -> None:
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
            model_id="luna",
            request_model="openai/gpt-5.6-luna-pro",
            client=client,
            generation={"reasoning_effort": "high"},
            thinking_control="openrouter_reasoning",
            provider_routing={
                "only": ["alibaba"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
            image_first=False,
            include_extended_sampling=False,
        )

        backend.generate_result(
            system_prompt="System",
            user_prompt="User",
            image_bytes=b"image",
            schema={},
        )

        payload = client.chat_create.payload
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(
            payload["extra_body"]["reasoning"],
            {"effort": "high", "exclude": False},
        )
        self.assertEqual(
            payload["extra_body"]["provider"],
            {
                "only": ["alibaba"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        )
        self.assertEqual(
            payload["messages"][-1]["content"][0]["type"],
            "text",
        )
        self.assertNotIn("top_k", payload["extra_body"])

    def test_vllm_thinking_control_uses_chat_template_kwargs(self) -> None:
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
            model_id="chat-model",
            client=client,
            generation={"thinking_mode": "disabled"},
            thinking_control="chat_template",
        )

        backend.generate_result(
            system_prompt="System",
            user_prompt="User",
            image_bytes=b"image",
            schema={},
        )

        self.assertFalse(
            client.chat_create.payload["extra_body"]["chat_template_kwargs"]["thinking"]
        )

    def test_vllm_sends_numeric_thinking_budget_when_enabled(self) -> None:
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
            model_id="qwen",
            client=client,
            generation={
                "thinking_mode": "enabled",
                "reasoning_max_tokens": 8192,
            },
            thinking_control="chat_template",
        )

        backend.generate_result(
            system_prompt="System",
            user_prompt="User",
            image_bytes=b"image",
            schema={},
        )

        self.assertEqual(
            client.chat_create.payload["extra_body"]["thinking_token_budget"],
            8192,
        )

    def test_openrouter_sends_numeric_reasoning_budget_when_enabled(self) -> None:
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
            model_id="qwen-flash",
            client=client,
            generation={
                "thinking_mode": "enabled",
                "reasoning_max_tokens": 8192,
            },
            thinking_control="openrouter_reasoning",
            include_extended_sampling=False,
        )

        backend.generate_result(
            system_prompt="System",
            user_prompt="User",
            image_bytes=b"image",
            schema={},
        )

        self.assertEqual(
            client.chat_create.payload["extra_body"]["reasoning"],
            {
                "max_tokens": 8192,
                "enabled": True,
                "exclude": False,
            },
        )

    def test_openrouter_explicitly_disables_reasoning_effort(self) -> None:
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
            model_id="reasoning-model",
            client=client,
            generation={
                "thinking_mode": "disabled",
                "reasoning_max_tokens": 8192,
            },
            thinking_control="openrouter_reasoning",
            include_extended_sampling=False,
        )

        backend.generate_result(
            system_prompt="System",
            user_prompt="User",
            image_bytes=b"image",
            schema={},
        )

        self.assertEqual(
            client.chat_create.payload["extra_body"]["reasoning"],
            {
                "effort": "none",
                "enabled": False,
                "exclude": False,
            },
        )

    def test_provider_can_omit_unsupported_seed(self) -> None:
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
            model_id="provider-without-seed",
            client=client,
            generation={"seed": 42, "temperature": 1.0},
            include_seed=False,
        )

        backend.generate_result(
            system_prompt="System",
            user_prompt="User",
            image_bytes=b"image",
            schema={},
        )

        payload = client.chat_create.payload
        self.assertNotIn("seed", payload)
        self.assertEqual(payload["temperature"], 1.0)

    def test_azure_fallback_maps_disabled_thinking_to_no_effort(self) -> None:
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
            model_id="chat-model",
            client=client,
            generation={"thinking_mode": "disabled"},
            thinking_control="reasoning_effort",
        )

        backend.generate_result(
            system_prompt="System",
            user_prompt="User",
            image_bytes=b"image",
            schema={},
        )

        self.assertEqual(
            client.chat_create.payload["reasoning_effort"],
            "none",
        )

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
                completion_tokens_details=SimpleNamespace(reasoning_tokens=7),
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
            user_content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
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
            model_id="chat-model",
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
                completion_tokens_details=SimpleNamespace(reasoning_tokens=4)
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
    def test_content_policy_error_becomes_structured_safety_refusal(
        self,
    ) -> None:
        class ProviderError(RuntimeError):
            status_code = 400
            body = {
                "error": {
                    "code": "content_policy_violation",
                    "message": "Image processing blocked.",
                    "innererror": {
                        "content_filter_result": {
                            "violence": {
                                "filtered": True,
                                "severity": "medium",
                            }
                        }
                    },
                }
            }

        backend = AzureResponsesBackend(
            model_id="gpt",
            client=_FakeClient(error=ProviderError("secret")),
        )

        with self.assertRaises(InferenceSafetyRefusal) as context:
            backend.generate_result(
                system_prompt="System",
                user_prompt="User",
                image_bytes=b"image",
                schema={},
            )

        self.assertEqual(
            context.exception.details["code"],
            "content_policy_violation",
        )
        self.assertTrue(
            context.exception.details["content_filter"]["innererror"][
                "content_filter_result"
            ]["violence"]["filtered"]
        )

    def test_responses_requests_official_summary_and_marks_truncation(
        self,
    ) -> None:
        response = SimpleNamespace(
            id="resp_123",
            model="deployment",
            reasoning=SimpleNamespace(
                effort="high",
                summary="detailed",
            ),
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
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
                output_tokens_details=SimpleNamespace(reasoning_tokens=21),
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
        self.assertEqual(
            result.reasoning.source_field,
            "output.reasoning.summary",
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
            schema={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "uniqueItems": True,
                    }
                },
            },
        )

        text_format = client.responses_create.payload["text"]["format"]
        self.assertEqual(text_format["type"], "json_schema")
        self.assertTrue(text_format["strict"])
        self.assertNotIn(
            "uniqueItems",
            text_format["schema"]["properties"]["items"],
        )

    def test_responses_transport_exposes_only_structured_error_detail(
        self,
    ) -> None:
        class ProviderError(RuntimeError):
            status_code = 400
            body = {
                "error": {
                    "code": "invalid_value",
                    "message": "Unsupported image format.",
                }
            }

        secret = "super-secret-api-key"
        backend = AzureResponsesBackend(
            model_id="gpt",
            client=_FakeClient(error=ProviderError(secret)),
        )

        with self.assertRaises(InferenceTransportError) as context:
            backend.generate_result(
                system_prompt="System",
                user_prompt="User",
                image_bytes=b"image",
                schema={},
            )

        message = str(context.exception)
        self.assertIn("type=ProviderError", message)
        self.assertIn("status=400", message)
        self.assertIn("code=invalid_value", message)
        self.assertIn("Unsupported image format.", message)
        self.assertNotIn(secret, message)


class VllmBackendTests(unittest.TestCase):
    def test_medgemma_reasoning_without_final_answer_stays_empty(self) -> None:
        separated = separate_embedded_reasoning(
            "<unused94>thought\nRank D003 first.<unused95>",
            parser="medgemma_special_tokens",
        )

        self.assertEqual(separated.reasoning_text, "Rank D003 first.")
        self.assertEqual(separated.final_text, "")
        self.assertTrue(separated.complete_block)

    def test_medgemma_stream_separates_embedded_reasoning(self) -> None:
        chunks = [
            SimpleNamespace(
                id="medgemma_stream",
                model="google/medgemma-1.5-4b-it",
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            content="<unused94>thought\nInspect nails.",
                        ),
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                id="medgemma_stream",
                model="google/medgemma-1.5-4b-it",
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        delta=SimpleNamespace(
                            content=('<unused95>{"predictions":["D003"]}'),
                        ),
                    )
                ],
                usage=None,
            ),
        ]

        class StreamingEndpoint:
            def create(self, **kwargs):
                return iter(chunks)

        client = _FakeClient()
        client.chat.completions = StreamingEndpoint()
        backend = VllmBackend(
            model_id="medgemma",
            client=client,
            reasoning_capture="full",
            embedded_reasoning_parser="medgemma_special_tokens",
        )

        result = backend.generate_result(
            system_prompt="System",
            user_prompt="User",
            image_bytes=b"image",
            schema={},
        )

        self.assertEqual(
            result.final_text,
            '{"predictions":["D003"]}',
        )
        self.assertEqual(result.reasoning.text, "Inspect nails.")
        self.assertEqual(
            result.reasoning.source_field,
            "content.medgemma_special_tokens",
        )
        self.assertTrue(result.metadata["embedded_reasoning_block_complete"])

    def test_completion_streams_content_reasoning_and_usage(self) -> None:
        chunks = [
            SimpleNamespace(
                id="chat_stream",
                model="Qwen/Qwen3.6-27B",
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            content=None,
                            reasoning_content="Inspect ",
                        ),
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                id="chat_stream",
                model="Qwen/Qwen3.6-27B",
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        delta=SimpleNamespace(
                            content='{"predictions":[]}',
                            reasoning_content="morphology.",
                        ),
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                id="chat_stream",
                model="Qwen/Qwen3.6-27B",
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=20,
                    completion_tokens=12,
                    total_tokens=32,
                    completion_tokens_details=SimpleNamespace(
                        reasoning_tokens=7,
                    ),
                ),
            ),
        ]

        class StreamingEndpoint:
            def __init__(self) -> None:
                self.payload = None

            def create(self, **kwargs):
                self.payload = kwargs
                return iter(chunks)

        client = _FakeClient()
        endpoint = StreamingEndpoint()
        client.chat.completions = endpoint
        backend = VllmBackend(
            model_id="qwen",
            request_model="Qwen/Qwen3.6-27B",
            client=client,
            reasoning_capture="full",
        )

        result = backend.generate_result(
            system_prompt="System",
            user_prompt="User",
            image_bytes=b"image",
            schema={},
        )

        self.assertTrue(endpoint.payload["stream"])
        self.assertEqual(
            endpoint.payload["stream_options"],
            {"include_usage": True},
        )
        self.assertEqual(result.final_text, '{"predictions":[]}')
        self.assertEqual(result.reasoning.text, "Inspect morphology.")
        self.assertEqual(result.reasoning.token_count, 7)
        self.assertEqual(result.finish_reason, "stop")
        self.assertTrue(result.metadata["streamed"])

    def test_preflight_checks_health_and_exact_served_model(self) -> None:
        backend = VllmBackend(
            model_id="qwen",
            request_model="Qwen/Qwen3.5-4B",
            client=_FakeClient(model_ids=("Qwen/Qwen3.5-4B",)),
            health_probe=lambda url, timeout: url == "http://localhost:8000/health",
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
            source=SimpleNamespace(repo_id="Qwen/Qwen3.5-4B"),
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

        self.assertEqual(server_config.model, "Qwen/Qwen3.5-4B")
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
            source=SimpleNamespace(repo_id="example/multimodal-model"),
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

        processor_flag_index = command.index("--mm-processor-kwargs")
        self.assertEqual(
            json.loads(command[processor_flag_index + 1]),
            {
                "downsample_mode": "4x",
                "max_slice_nums": 36,
            },
        )

    def test_server_config_allows_explicit_controlled_unmanaged_override(self) -> None:
        """The cohort starter may opt into an endpoint-only model profile."""

        model_config = SimpleNamespace(
            source=SimpleNamespace(repo_id="example/private-merged-model"),
            processor=None,
            backend=SimpleNamespace(
                active_profile=SimpleNamespace(
                    engine="vllm",
                    managed=False,
                    managed_allowed=False,
                )
            ),
            reasoning=SimpleNamespace(parser=None),
        )

        with self.assertRaises(InferenceConfigurationError):
            server_config_from_model(model_config)
        server_config = server_config_from_model(
            model_config,
            allow_unmanaged=True,
        )
        self.assertEqual(server_config.model, "example/private-merged-model")


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
