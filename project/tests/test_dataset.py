"""Unit tests for the ISEPDistillDataset YAML loader."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from project.dataset.dataset import (
    AUTO,
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    DistillDataset,
    DistillDatasetSpec,
    _card_configs,
    _readme_front_matter,
)

DEFAULT_REPO = "danielfdias98/ISEPDistillDataset"


def _default_payload() -> dict[str, Any]:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _write_dataset_yaml(tmp_path: Path, payload: dict[str, Any]) -> Path:
    dest = tmp_path / "dataset.yaml"
    dest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return dest


def _load_mutated(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> DistillDatasetSpec:
    payload = _default_payload()
    mutate(payload)
    return DistillDatasetSpec.from_yaml(
        _write_dataset_yaml(tmp_path, payload),
        project_root=PROJECT_ROOT,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class FakeHub:
    """In-memory Hub used by load tests. Never talks to the network."""

    catalogs: dict[str, dict[str, object]]
    calls: list[tuple[str, str, str]]
    expected_token: str = "test-token"

    def list_configs(
        self,
        *,
        repo_id: str,
        revision: str,
        token: str,
    ) -> tuple[str, ...]:
        assert repo_id == DEFAULT_REPO
        assert revision == "main"
        assert token == self.expected_token
        return tuple(self.catalogs)

    def list_splits(
        self,
        *,
        repo_id: str,
        config: str,
        revision: str,
        token: str,
    ) -> tuple[str, ...]:
        assert repo_id == DEFAULT_REPO
        return tuple(self.catalogs[config])

    def load_split(
        self,
        *,
        repo_id: str,
        config: str,
        split: str,
        revision: str,
        token: str,
    ) -> object:
        self.calls.append((config, split, token))
        return self.catalogs[config][split]


def _hub() -> FakeHub:
    return FakeHub(
        catalogs={
            "diagnosis": {
                "sft_train": object(),
                "sft_dev": object(),
            },
            "morphology": {
                "sft_train": object(),
                "sft_dev": object(),
            },
            "caption": {
                "sft_train": object(),
                "sft_dev": object(),
            },
        },
        calls=[],
    )


LOCAL_DATASET_CARD = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "training"
    / "ISEPDistillDataset"
    / "README.md"
)


class TestDatasetCard:
    def test_local_readme_configs_and_splits(self) -> None:
        text = LOCAL_DATASET_CARD.read_text(encoding="utf-8")
        configs = _card_configs(_readme_front_matter(text))
        names = tuple(name for name, _splits in configs)

        assert names == ("diagnosis", "morphology", "caption")
        assert configs[0][1] == ("sft_train", "sft_dev")
        assert configs[1][0] == "morphology"
        assert configs[2][1] == ("sft_train", "sft_dev")

    def test_missing_front_matter_is_empty(self) -> None:
        assert _readme_front_matter("# no yaml") == {}
        assert _card_configs({}) == ()


class TestFromYamlDefaultConfig:
    def test_hub_identity_and_auto_loading(self) -> None:
        spec = DistillDatasetSpec.from_yaml()

        assert spec.config_path == DEFAULT_CONFIG
        assert spec.huggingface.repo_id == DEFAULT_REPO
        assert spec.huggingface.revision == "main"
        assert spec.huggingface.token_env == "HF_TOKEN"
        assert spec.loading.config == AUTO
        assert spec.loading.split == AUTO


class TestInvalidConfigs:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "absent.yaml"
        with pytest.raises(FileNotFoundError, match="Dataset config not found"):
            DistillDatasetSpec.from_yaml(missing)

    def test_blank_repo_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match=re.escape("dataset.huggingface.repo_id")):
            _load_mutated(
                tmp_path,
                lambda payload: payload["dataset"]["huggingface"].__setitem__(
                    "repo_id",
                    "   ",
                ),
            )

    def test_missing_loading_mapping_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match=re.escape("dataset.loading")):
            _load_mutated(tmp_path, lambda payload: payload["dataset"].pop("loading"))


class TestLoad:
    def test_auto_loads_every_config_and_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "test-token")
        hub = _hub()
        loaded = DistillDataset.load(hub=hub)

        assert loaded.configs() == ("diagnosis", "morphology", "caption")
        assert loaded.splits("diagnosis") == ("sft_train", "sft_dev")
        assert (
            loaded.get("diagnosis", "sft_train")
            is hub.catalogs["diagnosis"]["sft_train"]
        )
        assert len(loaded.tables) == 6
        mapping = loaded.as_mapping()
        assert set(mapping) == {"diagnosis", "morphology", "caption"}
        assert set(mapping["caption"]) == {"sft_train", "sft_dev"}

    def test_pinned_config_and_split(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "test-token")
        spec = _load_mutated(
            tmp_path,
            lambda payload: payload["dataset"]["loading"].update(
                {"config": "morphology", "split": "sft_dev"}
            ),
        )
        hub = _hub()
        loaded = spec.load(hub=hub)

        assert loaded.configs() == ("morphology",)
        assert loaded.splits("morphology") == ("sft_dev",)
        assert hub.calls == [("morphology", "sft_dev", "test-token")]

    def test_load_overrides_yaml_config_and_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "test-token")
        hub = _hub()
        loaded = DistillDataset.load(
            hub=hub,
            config="diagnosis",
            split="sft_dev",
        )

        assert loaded.configs() == ("diagnosis",)
        assert loaded.splits("diagnosis") == ("sft_dev",)
        assert hub.calls == [("diagnosis", "sft_dev", "test-token")]

    def test_unknown_config_lists_known_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "test-token")
        spec = _load_mutated(
            tmp_path,
            lambda payload: payload["dataset"]["loading"].__setitem__(
                "config",
                "structured",
            ),
        )
        with pytest.raises(
            KeyError,
            match="Known configs: diagnosis, morphology, caption",
        ):
            spec.load(hub=_hub())

    def test_unknown_split_lists_known_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "test-token")
        spec = _load_mutated(
            tmp_path,
            lambda payload: payload["dataset"]["loading"].update(
                {"config": "diagnosis", "split": "train"}
            ),
        )
        with pytest.raises(KeyError, match="Known splits: sft_train, sft_dev"):
            spec.load(hub=_hub())

    def test_missing_token_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with pytest.raises(OSError, match="Missing environment variable HF_TOKEN"):
            DistillDataset.load(hub=_hub(), project_root=tmp_path)

    def test_token_falls_back_to_dotenv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        (tmp_path / ".env").write_text("HF_TOKEN=dotenv-token\n", encoding="utf-8")
        spec = DistillDatasetSpec.from_yaml(DEFAULT_CONFIG, project_root=tmp_path)
        hub = FakeHub(
            catalogs=_hub().catalogs,
            calls=[],
            expected_token="dotenv-token",
        )
        loaded = DistillDataset.from_spec(spec, hub=hub)
        assert loaded.configs() == ("diagnosis", "morphology", "caption")
        assert hub.calls[0] == ("diagnosis", "sft_train", "dotenv-token")

    def test_environment_token_wins_over_dotenv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "env-token")
        (tmp_path / ".env").write_text("HF_TOKEN=dotenv-token\n", encoding="utf-8")
        spec = DistillDatasetSpec.from_yaml(DEFAULT_CONFIG, project_root=tmp_path)
        assert spec.token() == "env-token"

    def test_unknown_loaded_table_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "test-token")
        loaded = DistillDataset.load(hub=_hub())
        with pytest.raises(KeyError, match="Unknown loaded table"):
            loaded.get("diagnosis", "train")
        with pytest.raises(KeyError, match="Unknown loaded config"):
            loaded.splits("structured")
