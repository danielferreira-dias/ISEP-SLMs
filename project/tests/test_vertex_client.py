"""Vertex teacher message conversion and generateContent mapping."""

from types import SimpleNamespace

import pytest
from google.genai.errors import ClientError

from project.teacher.client import (
    TeacherCompletionError,
    create_teacher_client,
)
from project.teacher.teacher import TeacherModel, TeacherProvider
from project.teacher.vertex import (
    VertexTeacherClient,
    jpeg_bytes_from_data_url,
    response_from_vertex,
    split_stage_messages,
)

_JPEG_URL = "data:image/jpeg;base64,YQ=="


def _load_vertex() -> TeacherModel:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return TeacherModel.from_yaml(
        root / "configs" / "teacher_configs" / "gemini_3_7_flash_vertex.yaml"
    )


def _stage_messages() -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "Describe only what is visible."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Examine the supplied skin image."},
                {
                    "type": "image_url",
                    "image_url": {"url": _JPEG_URL, "detail": "auto"},
                },
            ],
        },
    ]


class _FakeVertex:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            text='{"ok": true}',
            candidates=[SimpleNamespace(finish_reason="STOP")],
            usage_metadata=SimpleNamespace(
                prompt_token_count=4,
                candidates_token_count=5,
                thoughts_token_count=2,
                total_token_count=11,
            ),
            prompt_feedback=None,
        )


class _FlakyVertex(_FakeVertex):
    def __init__(self, *, status_code: int, failures: int) -> None:
        super().__init__()
        self.status_code = status_code
        self.failures = failures

    def _generate(self, **kwargs: object) -> object:
        if len(self.calls) < self.failures:
            self.calls.append(kwargs)
            raise ClientError(
                self.status_code,
                {"error": {"message": "transient test failure"}},
            )
        return super()._generate(**kwargs)


def test_jpeg_bytes_from_data_url() -> None:
    assert jpeg_bytes_from_data_url(_JPEG_URL) == b"a"


def test_split_stage_messages_extracts_system_user_and_image() -> None:
    system, user_text, image = split_stage_messages(_stage_messages())
    assert system.startswith("Describe only")
    assert "Examine" in user_text
    assert image == b"a"


def test_create_teacher_client_selects_vertex() -> None:
    teacher = _load_vertex()
    fake = _FakeVertex()
    client = create_teacher_client(teacher, client=fake)
    assert isinstance(client, VertexTeacherClient)
    assert teacher.provider is TeacherProvider.VERTEX


def test_complete_stage_sends_schema_and_thinking() -> None:
    teacher = _load_vertex()
    fake = _FakeVertex()
    client = VertexTeacherClient(teacher, client=fake)
    response = client.complete_stage("A", _stage_messages())

    assert response.content_json == {"ok": True}
    assert response.usage is not None
    assert response.usage.prompt_tokens == 4
    assert response.usage.completion_tokens == 5
    assert response.usage.thoughts_tokens == 2
    assert response.usage.request_attempts == 1
    assert response.usage.cost == pytest.approx(0.00002925)
    assert response.usage.cost_currency == "USD"
    assert response.usage.cost_basis == "estimated_list_price"
    kwargs = fake.calls[0]
    assert kwargs["model"] == "gemini-3.7-flash"
    config = kwargs["config"]
    assert config.system_instruction.startswith("Describe only")
    assert config.max_output_tokens == 16384
    assert config.seed == 42
    assert config.response_mime_type == "application/json"
    assert config.automatic_function_calling.disable is True
    assert config.thinking_config.thinking_level == "MEDIUM"
    assert config.thinking_config.include_thoughts is False


def test_transient_429_retries_then_succeeds() -> None:
    teacher = _load_vertex()
    fake = _FlakyVertex(status_code=429, failures=2)
    client = VertexTeacherClient(teacher, client=fake, sleep=lambda _delay: None)

    response = client.complete_stage("A", _stage_messages())

    assert len(fake.calls) == 3
    assert response.usage is not None
    assert response.usage.request_attempts == 3


def test_transient_429_stops_after_configured_attempts() -> None:
    teacher = _load_vertex()
    assert teacher.retry is not None
    fake = _FlakyVertex(status_code=429, failures=teacher.retry.max_attempts)
    client = VertexTeacherClient(teacher, client=fake, sleep=lambda _delay: None)

    with pytest.raises(
        TeacherCompletionError,
        match="provider_http_error:429",
    ) as caught:
        client.complete_stage("A", _stage_messages())

    assert len(fake.calls) == teacher.retry.max_attempts
    assert caught.value.usage is not None
    assert caught.value.usage.request_attempts == teacher.retry.max_attempts


def test_non_transient_400_is_not_retried() -> None:
    teacher = _load_vertex()
    fake = _FlakyVertex(status_code=400, failures=1)
    client = VertexTeacherClient(teacher, client=fake, sleep=lambda _delay: None)

    with pytest.raises(
        TeacherCompletionError,
        match="provider_http_error:400",
    ) as caught:
        client.complete_stage("A", _stage_messages())

    assert len(fake.calls) == 1
    assert caught.value.usage is not None
    assert caught.value.usage.request_attempts == 1


def test_max_tokens_finish_reason_raises() -> None:
    completion = SimpleNamespace(
        text='{"a": 1}',
        candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
        prompt_feedback=None,
        usage_metadata=None,
    )
    with pytest.raises(TeacherCompletionError, match="length"):
        response_from_vertex(completion)


def test_safety_block_raises() -> None:
    completion = SimpleNamespace(
        text="",
        candidates=[],
        prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
        usage_metadata=None,
    )
    with pytest.raises(TeacherCompletionError, match="provider_safety_refusal"):
        response_from_vertex(completion)
