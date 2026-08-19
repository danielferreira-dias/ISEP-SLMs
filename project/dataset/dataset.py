"""Load ``ISEPDistillDataset`` from YAML into typed dataclasses.

Public entry point: ``DistillDataset.load()``.

The YAML names the Hub repo and whether config/split should be discovered
(``auto``) or pinned. Broken configs fail here, before ``datasets`` is called.
The Hub token is read from ``token_env`` at load time, then from a nearby
``.env`` file, and is never stored on the dataclass. Before Hub calls the
resolved token is exported to ``HF_TOKEN`` so ``huggingface_hub`` sees it.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, cast

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "dataset" / "isep_distill_dataset.yaml"
AUTO = "auto"


class DatasetHub(Protocol):
    """Minimal Hub surface used to discover and load dataset configs."""

    def list_configs(
        self,
        *,
        repo_id: str,
        revision: str,
        token: str,
    ) -> tuple[str, ...]:
        """Return config names published on the Hub revision."""
        ...

    def list_splits(
        self,
        *,
        repo_id: str,
        config: str,
        revision: str,
        token: str,
    ) -> tuple[str, ...]:
        """Return split names for one config on the Hub revision."""
        ...

    def load_split(
        self,
        *,
        repo_id: str,
        config: str,
        split: str,
        revision: str,
        token: str,
    ) -> object:
        """Return one Hugging Face ``Dataset`` for a config/split pair."""
        ...


def _require_mapping(value: object, path: str) -> dict[str, object]:
    """Return ``value`` as a dict, or raise if the YAML node is not an object."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return dict(value)


def _require_str(value: object, path: str) -> str:
    """Return a non-empty string, or raise if the YAML node is missing or blank."""

    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path} must be a non-empty string")
    return value.strip()


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` dotenv file. Does not execute the file."""

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _export_hub_token(token: str, token_env: str) -> None:
    """Copy the resolved token into the env vars Hugging Face libraries read."""

    os.environ[token_env] = token
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def _readme_front_matter(text: str) -> dict[str, object]:
    """Return the YAML mapping between the first two ``---`` markers."""

    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return {}

    rest = stripped.removeprefix("---")
    rest = rest.removeprefix("\n")
    end = rest.find("\n---")
    if end == -1:
        return {}

    raw = yaml.safe_load(rest[:end])
    if not isinstance(raw, Mapping):
        return {}
    return dict(raw)


def _splits_from_data_files(value: object) -> tuple[str, ...]:
    """Read split names from a dataset-card ``data_files`` node."""

    if isinstance(value, Mapping):
        return tuple(str(key) for key in value)

    if not isinstance(value, list):
        return ()

    names: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            split = item.get("split")
            if isinstance(split, str) and split.strip():
                names.append(split.strip())
    return tuple(names)


def _card_configs(
    meta: Mapping[str, object],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return ``(config_name, splits)`` pairs from a dataset card."""

    raw = meta.get("configs")
    if not isinstance(raw, list):
        return ()

    configs: list[tuple[str, tuple[str, ...]]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        name = item.get("config_name")
        if not isinstance(name, str) or not name.strip():
            continue
        configs.append((name.strip(), _splits_from_data_files(item.get("data_files"))))
    return tuple(configs)


def _download_dataset_readme(
    *,
    repo_id: str,
    revision: str,
    token: str,
) -> str:
    """Download the Hub dataset card. Fail if the repo is not reachable."""

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
    except ImportError as exc:
        raise RuntimeError(
            "The training extra with the 'huggingface-hub' package is required "
            "to load ISEPDistillDataset from the Hub"
        ) from exc

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type="dataset",
            revision=revision,
            token=token,
        )
    except (OSError, HfHubHTTPError, RepositoryNotFoundError) as exc:
        raise OSError(
            f"Cannot access dataset {repo_id!r} on the Hub. "
            "Check HF_TOKEN in the environment or the repository .env."
        ) from exc

    return Path(str(path)).read_text(encoding="utf-8")


def _resolve_names(
    requested: str,
    available: Sequence[str],
    *,
    kind: str,
) -> tuple[str, ...]:
    """Resolve ``auto`` to every published name, or require an exact match."""

    names = tuple(available)
    if not names:
        raise ValueError(f"Hub published no {kind} names")

    if requested == AUTO:
        return names

    if requested in names:
        return (requested,)

    known = ", ".join(names)
    raise KeyError(f"Unknown dataset {kind} {requested!r}. Known {kind}s: {known}")


@dataclass(frozen=True, slots=True, kw_only=True)
class HuggingFaceRef:
    """Hub identity for ``ISEPDistillDataset``. The token stays in the environment."""

    repo_id: str
    revision: str
    token_env: str

    def token(self, *, dotenv_paths: Sequence[Path] = ()) -> str:
        """Read the Hub secret named by ``token_env``.

        The process environment wins. If it is blank, nearby ``.env`` files are
        checked in order. The token is returned, never stored.

        Args:
            dotenv_paths: Optional ``.env`` files to read after the environment.

        Returns:
            The non-empty token string.

        Raises:
            OSError: If the variable is missing from the environment and every
                dotenv file.
        """

        key = os.environ.get(self.token_env, "").strip()
        if key:
            return key

        for path in dotenv_paths:
            if not path.is_file():
                continue
            parsed = _parse_dotenv(path).get(self.token_env, "").strip()
            if parsed:
                return parsed

        raise OSError(f"Missing environment variable {self.token_env}")


@dataclass(frozen=True, slots=True, kw_only=True)
class LoadingSpec:
    """Which Hub config and split to load. ``auto`` discovers every published name."""

    config: str
    split: str


@dataclass(frozen=True, slots=True, kw_only=True)
class DistillTable:
    """One loaded Hugging Face table, identified by config and split."""

    config: str
    split: str
    data: object


@dataclass(frozen=True, slots=True, kw_only=True)
class DistillDatasetSpec:
    """Parsed ``isep_distill_dataset.yaml`` after type checks."""

    huggingface: HuggingFaceRef
    loading: LoadingSpec
    config_path: Path
    project_root: Path

    @classmethod
    def from_yaml(
        cls,
        path: str | Path | None = None,
        *,
        project_root: Path | None = None,
    ) -> DistillDatasetSpec:
        """Load the dataset YAML. Defaults to ``isep_distill_dataset.yaml``.

        Args:
            path: YAML path. Relative paths are resolved against ``project_root``.
            project_root: Directory used for relative paths. Defaults to ``project/``.

        Returns:
            A frozen spec. Does not call the Hub.

        Raises:
            FileNotFoundError: If the YAML file is missing.
            TypeError: If a required node is the wrong type or blank.
        """

        root = (project_root or PROJECT_ROOT).resolve()
        config_path = Path(path) if path is not None else DEFAULT_CONFIG
        if not config_path.is_absolute():
            config_path = (root / config_path).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Dataset config not found: {config_path}")

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload = _require_mapping(raw, str(config_path))
        dataset = _require_mapping(payload.get("dataset"), "dataset")
        huggingface = _require_mapping(
            dataset.get("huggingface"),
            "dataset.huggingface",
        )
        loading = _require_mapping(dataset.get("loading"), "dataset.loading")

        return cls(
            huggingface=HuggingFaceRef(
                repo_id=_require_str(
                    huggingface.get("repo_id"),
                    "dataset.huggingface.repo_id",
                ),
                revision=_require_str(
                    huggingface.get("revision"),
                    "dataset.huggingface.revision",
                ),
                token_env=_require_str(
                    huggingface.get("token_env"),
                    "dataset.huggingface.token_env",
                ),
            ),
            loading=LoadingSpec(
                config=_require_str(loading.get("config"), "dataset.loading.config"),
                split=_require_str(loading.get("split"), "dataset.loading.split"),
            ),
            config_path=config_path,
            project_root=root,
        )

    def token(self) -> str:
        """Read the Hub token from the environment, then nearby ``.env`` files."""

        return self.huggingface.token(
            dotenv_paths=(
                self.project_root / ".env",
                self.project_root.parent / ".env",
            )
        )

    def load(self, *, hub: DatasetHub | None = None) -> DistillDataset:
        """Download the configured Hub tables.

        Args:
            hub: Optional test double. Defaults to the ``datasets`` Hub client.

        Returns:
            Loaded tables for every resolved config/split pair.
        """

        return DistillDataset.from_spec(self, hub=hub)


class HuggingFaceDatasetHub:
    """Hub client that reads configs from the dataset card, then loads tables."""

    def __init__(self) -> None:
        self._readme: dict[tuple[str, str], str] = {}

    def list_configs(
        self,
        *,
        repo_id: str,
        revision: str,
        token: str,
    ) -> tuple[str, ...]:
        """Return config names published on the Hub revision."""

        names = tuple(name for name, _splits in self._configs(repo_id, revision, token))
        if not names:
            raise OSError(
                f"Dataset {repo_id!r} README has no Hub configs. "
                "Refusing to use a cached 'default' layout."
            )
        return names

    def list_splits(
        self,
        *,
        repo_id: str,
        config: str,
        revision: str,
        token: str,
    ) -> tuple[str, ...]:
        """Return split names for one config on the Hub revision."""

        configs = self._configs(repo_id, revision, token)
        for name, splits in configs:
            if name == config:
                if not splits:
                    raise ValueError(
                        f"Hub config {config!r} has no splits in the dataset card"
                    )
                return splits

        known = ", ".join(name for name, _splits in configs)
        raise KeyError(f"Unknown dataset config {config!r}. Known configs: {known}")

    def load_split(
        self,
        *,
        repo_id: str,
        config: str,
        split: str,
        revision: str,
        token: str,
    ) -> object:
        """Return one Hugging Face ``Dataset`` for a config/split pair."""

        _export_hub_token(token, "HF_TOKEN")
        load_dataset = _datasets_api("load_dataset")
        return load_dataset(
            repo_id,
            name=config,
            split=split,
            revision=revision,
            token=token,
        )

    def _configs(
        self,
        repo_id: str,
        revision: str,
        token: str,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Parse ``(config, splits)`` from the cached dataset card."""

        _export_hub_token(token, "HF_TOKEN")
        key = (repo_id, revision)
        if key not in self._readme:
            self._readme[key] = _download_dataset_readme(
                repo_id=repo_id,
                revision=revision,
                token=token,
            )
        return _card_configs(_readme_front_matter(self._readme[key]))


@dataclass(frozen=True, slots=True, kw_only=True)
class DistillDataset:
    """Loaded ``ISEPDistillDataset`` tables, keyed by Hub config then split."""

    spec: DistillDatasetSpec
    tables: tuple[DistillTable, ...]

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        hub: DatasetHub | None = None,
        project_root: Path | None = None,
        config: str | None = None,
        split: str | None = None,
    ) -> DistillDataset:
        """Parse the YAML and load matching Hub tables.

        Args:
            path: YAML path. Defaults to ``project/dataset/isep_distill_dataset.yaml``.
            hub: Optional test double. Defaults to the ``datasets`` Hub client.
            project_root: Directory used for relative YAML paths.
            config: Override ``dataset.loading.config``. ``None`` keeps the YAML value.
            split: Override ``dataset.loading.split``. ``None`` keeps the YAML value.

        Returns:
            Frozen tables for every resolved config/split pair.
        """

        spec = DistillDatasetSpec.from_yaml(path, project_root=project_root)
        if config is not None or split is not None:
            spec = replace(
                spec,
                loading=LoadingSpec(
                    config=config if config is not None else spec.loading.config,
                    split=split if split is not None else spec.loading.split,
                ),
            )
        return cls.from_spec(spec, hub=hub)

    @classmethod
    def from_spec(
        cls,
        spec: DistillDatasetSpec,
        *,
        hub: DatasetHub | None = None,
    ) -> DistillDataset:
        """Load Hub tables described by an already-parsed spec."""

        client = hub if hub is not None else HuggingFaceDatasetHub()
        source = spec.huggingface
        token = spec.token()
        configs = _resolve_names(
            spec.loading.config,
            client.list_configs(
                repo_id=source.repo_id,
                revision=source.revision,
                token=token,
            ),
            kind="config",
        )

        tables: list[DistillTable] = []
        for config in configs:
            splits = _resolve_names(
                spec.loading.split,
                client.list_splits(
                    repo_id=source.repo_id,
                    config=config,
                    revision=source.revision,
                    token=token,
                ),
                kind="split",
            )
            for split in splits:
                tables.append(
                    DistillTable(
                        config=config,
                        split=split,
                        data=client.load_split(
                            repo_id=source.repo_id,
                            config=config,
                            split=split,
                            revision=source.revision,
                            token=token,
                        ),
                    )
                )

        if not tables:
            raise ValueError("No dataset tables were loaded")

        return cls(spec=spec, tables=tuple(tables))

    def configs(self) -> tuple[str, ...]:
        """Return loaded config names in first-seen order."""

        return tuple(dict.fromkeys(table.config for table in self.tables))

    def splits(self, config: str) -> tuple[str, ...]:
        """Return loaded split names for one config.

        Raises:
            KeyError: If ``config`` was not loaded.
        """

        names = tuple(table.split for table in self.tables if table.config == config)
        if not names:
            known = ", ".join(self.configs())
            raise KeyError(f"Unknown loaded config {config!r}. Known configs: {known}")
        return names

    def get(self, config: str, split: str) -> object:
        """Return the Hugging Face ``Dataset`` for one config/split pair.

        Raises:
            KeyError: If that pair was not loaded.
        """

        for table in self.tables:
            if table.config == config and table.split == split:
                return table.data

        known = ", ".join(f"{table.config}/{table.split}" for table in self.tables)
        raise KeyError(
            f"Unknown loaded table {config!r}/{split!r}. Known tables: {known}"
        )

    def as_mapping(self) -> dict[str, dict[str, object]]:
        """Return ``{config: {split: dataset}}`` for trainer-style lookup."""

        mapping: dict[str, dict[str, object]] = {}
        for table in self.tables:
            mapping.setdefault(table.config, {})[table.split] = table.data
        return mapping


def _datasets_api(name: str) -> Callable[..., object]:
    """Import one ``datasets`` helper, or explain the missing extra.

    Raises:
        RuntimeError: If the ``datasets`` package is not installed.
        AttributeError: If the installed package is missing ``name``.
    """

    try:
        import datasets  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "The training extra with the 'datasets' package is required "
            "to load ISEPDistillDataset from the Hub"
        ) from exc
    return cast(Callable[..., object], getattr(datasets, name))
