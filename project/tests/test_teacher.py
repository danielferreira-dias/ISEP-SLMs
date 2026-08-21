"""Unit tests for the teacher YAML loader in ``project/teacher/teacher.py``."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from project.teacher.teacher import (
    TeacherAPI,
    TeacherModel,
    TeacherProvider,
    VertexAPI,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "teacher_configs" / "gemini_3_7_flash.yaml"
VERTEX_CONFIG = (
    PROJECT_ROOT / "configs" / "teacher_configs" / "gemini_3_7_flash_vertex.yaml"
)
PLACEHOLDER_RE = re.compile(r"\{\{[a-zA-Z0-9_]+\}\}")


@pytest.fixture(scope="module")
def teacher() -> TeacherModel:
    return TeacherModel.from_yaml()


def _default_payload() -> dict[str, Any]:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _write_teacher_yaml(tmp_path: Path, payload: dict[str, Any]) -> Path:
    dest = tmp_path / "teacher.yaml"
    dest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return dest


def _load_mutated(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> TeacherModel:
    payload = _default_payload()
    mutate(payload)
    return TeacherModel.from_yaml(
        _write_teacher_yaml(tmp_path, payload),
        project_root=PROJECT_ROOT,
    )


class TestFromYamlDefaultConfig:
    def test_identity_routing_generation_reasoning_and_stages(
        self, teacher: TeacherModel
    ) -> None:
        assert teacher.config_path == DEFAULT_CONFIG
        assert teacher.name == "gemini_3_7_flash"
        assert teacher.provider is TeacherProvider.OPENROUTER
        assert teacher.model.id == "google/gemini-3.7-flash"
        assert teacher.routing.only == ("google-vertex",)
        assert teacher.routing.allow_fallbacks is False
        assert teacher.routing.require_parameters is True
        assert teacher.generation.max_tokens == 16384
        assert teacher.generation.seed == 42
        assert teacher.reasoning.effort == "high"
        assert teacher.reasoning.exclude is True
        assert teacher.retry is None
        assert set(teacher.stages) == {"A", "B"}


class TestStages:
    def test_stage_names_and_prompt_shapes(self, teacher: TeacherModel) -> None:
        stage_a = teacher.stage("A")
        stage_b = teacher.stage("B")

        assert stage_a.name == "morphology"
        assert stage_b.name == "reasoning"
        assert stage_a.prompt.system.strip()
        assert stage_a.prompt.user.strip()
        assert stage_a.prompt.version == "e3_stage_a_v1"
        assert (
            stage_a.prompt.sha256
            == "c28f6ff4f9a47ba23bc02f2a6d14541ee5afeeaf134bca5cf48936f150121a4f"
        )
        assert stage_b.prompt.version == "e3_stage_b_v1"
        assert (
            stage_b.prompt.sha256
            == "b8239b38c24eac6037c22bcfcbc3573deb37cd5011c97d234e4179f20718125e"
        )
        assert PLACEHOLDER_RE.search(stage_a.prompt.user) is None
        assert "{{gold_diagnosis}}" in stage_b.prompt.user
        assert "{{stage_a_json}}" in stage_b.prompt.user

    def test_stage_b_render_user_substitutes_placeholders(
        self, teacher: TeacherModel
    ) -> None:
        rendered = teacher.stage("B").prompt.render_user(
            gold_diagnosis="melanoma",
            stage_a_json="{}",
        )
        assert "melanoma" in rendered
        assert "{}" in rendered
        assert "{{gold_diagnosis}}" not in rendered
        assert "{{stage_a_json}}" not in rendered

    def test_render_user_without_required_placeholders_raises(
        self, teacher: TeacherModel
    ) -> None:
        with pytest.raises(ValueError, match="missing placeholders"):
            teacher.stage("B").prompt.render_user()

    def test_unknown_stage_raises_key_error(self, teacher: TeacherModel) -> None:
        with pytest.raises(KeyError, match="Unknown teacher stage 'C'"):
            teacher.stage("C")


class TestSchemas:
    def test_stage_a_schema_title_and_required(self, teacher: TeacherModel) -> None:
        schema = teacher.stage("A").json_schema.schema
        assert schema["title"] == "StageAMorphology"
        required = schema["required"]
        assert "image_assessment" in required
        assert "observations" in required
        assert "clinical_caption" in required

    def test_stage_b_schema_required(self, teacher: TeacherModel) -> None:
        required = teacher.stage("B").json_schema.schema["required"]
        assert "anchor_evidence_status" in required
        assert "diagnostic_confidence" in required
        assert "differential_comparisons" in required
        assert "limitations" in required
        assert "response_policy" in required
        assert "diagnosis" in required
        assert "clinical_reasoning" in required


class TestOpenRouterBody:
    def test_stage_a_body_shape(self, teacher: TeacherModel) -> None:
        messages = [{"role": "user", "content": "describe the lesion"}]
        body = teacher.openrouter_body("A", messages)

        assert body["model"] == "google/gemini-3.7-flash"
        assert body["max_tokens"] == 16384
        assert body["seed"] == 42
        assert body["reasoning"]["exclude"] is True
        assert body["provider"]["only"] == ["google-vertex"]
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        schema = body["response_format"]["json_schema"]["schema"]
        assert isinstance(schema, dict)
        assert schema
        assert "temperature" not in body
        assert body["messages"] == messages


class TestVertexConfig:
    def test_loads_project_location_and_thinking(self) -> None:
        teacher = TeacherModel.from_yaml(VERTEX_CONFIG)

        assert teacher.provider is TeacherProvider.VERTEX
        assert isinstance(teacher.api, VertexAPI)
        assert teacher.api.project == "here4beer-472720"
        assert teacher.api.location == "global"
        assert teacher.model.id == "gemini-3.7-flash"
        assert teacher.routing is None
        assert teacher.reasoning.effort == "medium"
        assert teacher.reasoning.as_vertex_thinking() == {
            "thinking_level": "MEDIUM",
            "include_thoughts": False,
        }
        assert teacher.retry is not None
        assert teacher.retry.max_attempts == 6
        assert teacher.retry.initial_delay_seconds == 2.0
        assert teacher.retry.max_delay_seconds == 60.0
        assert teacher.retry.exponential_base == 2.0
        assert teacher.retry.jitter_seconds == 1.0
        assert teacher.retry.retryable_status_codes == (
            408,
            429,
            500,
            502,
            503,
            504,
        )
        assert teacher.pricing is not None
        assert teacher.pricing.input_per_million_tokens_usd == 0.75
        assert teacher.pricing.output_per_million_tokens_usd == 3.75
        assert teacher.pricing.traffic_type == "standard_global"
        assert teacher.pricing.effective_through == "2026-12-31"

        config = teacher.vertex_generate_config("A")
        assert config["max_output_tokens"] == 16384
        assert config["seed"] == 42
        assert config["response_mime_type"] == "application/json"
        assert config["thinking_config"]["thinking_level"] == "MEDIUM"
        assert config["thinking_config"]["include_thoughts"] is False
        assert isinstance(config["response_json_schema"], dict)

    def test_openrouter_body_is_rejected(self) -> None:
        teacher = TeacherModel.from_yaml(VERTEX_CONFIG)
        with pytest.raises(TypeError, match="openrouter_body"):
            teacher.openrouter_body("A", [])

    def test_default_config_rejects_vertex_generate_config(
        self, teacher: TeacherModel
    ) -> None:
        with pytest.raises(TypeError, match="vertex_generate_config"):
            teacher.vertex_generate_config("A")


class TestInvalidConfigs:
    def test_missing_output_stages_raises(self, tmp_path: Path) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            del payload["teacher"]["output"]["stages"]

        with pytest.raises(TypeError, match=re.escape("teacher.output.stages")):
            _load_mutated(tmp_path, mutate)

    def test_invalid_reasoning_effort_raises(self, tmp_path: Path) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["teacher"]["reasoning"]["effort"] = "extreme"

        with pytest.raises(ValueError, match=re.escape("teacher.reasoning.effort")):
            _load_mutated(tmp_path, mutate)

    def test_invalid_vertex_retry_attempts_raises(self, tmp_path: Path) -> None:
        payload = yaml.safe_load(VERTEX_CONFIG.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        payload["teacher"]["retry"]["max_attempts"] = 0
        path = _write_teacher_yaml(tmp_path, payload)

        with pytest.raises(ValueError, match=re.escape("teacher.retry.max_attempts")):
            TeacherModel.from_yaml(path, project_root=PROJECT_ROOT)

    def test_missing_prompt_ref_raises(self, tmp_path: Path) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["teacher"]["output"]["stages"]["A"]["prompt_ref"] = (
                "configs/teacher_configs/prompts/missing.md"
            )

        with pytest.raises(FileNotFoundError, match="Referenced file does not exist"):
            _load_mutated(tmp_path, mutate)

    def test_missing_schema_ref_raises(self, tmp_path: Path) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["teacher"]["output"]["stages"]["A"]["response_format"][
                "json_schema"
            ]["schema_ref"] = "configs/schemas/missing.json"

        with pytest.raises(FileNotFoundError, match="Referenced file does not exist"):
            _load_mutated(tmp_path, mutate)

    def test_frozen_prompt_hash_mismatch_raises(self, tmp_path: Path) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["teacher"]["output"]["stages"]["A"]["prompt_sha256"] = (
                "0" * 64
            )

        with pytest.raises(ValueError, match="frozen prompt SHA-256 mismatch"):
            _load_mutated(tmp_path, mutate)

    @pytest.mark.parametrize(
        "markdown",
        [
            pytest.param("## user\nDescribe the lesion.\n", id="missing_system"),
            pytest.param("## system\nYou are a dermatologist.\n", id="missing_user"),
        ],
    )
    def test_prompt_missing_required_heading_raises(
        self, tmp_path: Path, markdown: str
    ) -> None:
        prompt_path = tmp_path / "broken_prompt.md"
        prompt_path.write_text(markdown, encoding="utf-8")

        def mutate(payload: dict[str, Any]) -> None:
            payload["teacher"]["output"]["stages"]["A"]["prompt_ref"] = str(prompt_path)

        with pytest.raises(ValueError, match="## system"):
            _load_mutated(tmp_path, mutate)

    def test_unknown_provider_raises(self, tmp_path: Path) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["teacher"]["provider"] = "anthropic"

        with pytest.raises(ValueError, match=re.escape("teacher.provider")):
            _load_mutated(tmp_path, mutate)

    def test_vertex_missing_project_raises(self, tmp_path: Path) -> None:
        payload = yaml.safe_load(VERTEX_CONFIG.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        del payload["teacher"]["api"]["project"]
        path = _write_teacher_yaml(tmp_path, payload)
        with pytest.raises(TypeError, match=re.escape("teacher.api.project")):
            TeacherModel.from_yaml(path, project_root=PROJECT_ROOT)


class TestApiKey:
    def test_api_key_raises_when_unset(
        self, teacher: TeacherModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert isinstance(teacher.api, TeacherAPI)
        with pytest.raises(OSError, match="OPENROUTER_API_KEY"):
            teacher.api.api_key()

    def test_api_key_returns_value_when_set(
        self, teacher: TeacherModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
        assert isinstance(teacher.api, TeacherAPI)
        assert teacher.api.api_key() == "test-openrouter-key"
