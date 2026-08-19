"""TeacherClient extra_body split and completion error mapping."""

from types import SimpleNamespace

import pytest

from project.teacher.client import (
    TeacherClient,
    TeacherCompletionError,
    _response_from_completion,
)
from project.teacher.teacher import TeacherModel


class _FakeOpenAI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    native_finish_reason="stop",
                    message=SimpleNamespace(content='{"ok": true}'),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=2,
                total_tokens=3,
                cost=0.01,
            ),
        )


def test_complete_stage_sends_provider_in_extra_body() -> None:
    teacher = TeacherModel.from_yaml()
    fake = _FakeOpenAI()
    client = TeacherClient(teacher, client=fake)
    response = client.complete_stage(
        "A",
        [{"role": "user", "content": "hello"}],
    )
    assert response.content_json == {"ok": True}
    kwargs = fake.calls[0]
    assert "provider" not in kwargs
    extra = kwargs["extra_body"]
    assert extra["provider"]["only"] == ["google-vertex"]
    assert extra["reasoning"]["exclude"] is True
    assert "temperature" not in kwargs


def test_length_finish_reason_raises() -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="length",
                native_finish_reason="length",
                message=SimpleNamespace(content='{"a": 1}'),
            )
        ]
    )
    with pytest.raises(TeacherCompletionError, match="length"):
        _response_from_completion(completion)
