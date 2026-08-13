"""Typed, lazy Hugging Face client for private checkpoint commits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from src.train.config import CheckpointHubConfig


@dataclass(frozen=True, slots=True)
class HubUploadFile:
    """Map one verified local checkpoint file to its Hub path."""

    local_path: Path
    path_in_repo: str


@dataclass(frozen=True, slots=True)
class HubCommit:
    """Record the immutable identity returned by a Hub commit."""

    oid: str
    url: str


class HubCommitClient(Protocol):
    """Minimal Hugging Face operations used by checkpoint mirroring."""

    def ensure_private_repo(self, config: CheckpointHubConfig) -> None:
        """Create or verify the configured private model repository."""

    def commit_files(
        self,
        config: CheckpointHubConfig,
        files: tuple[HubUploadFile, ...],
        *,
        message: str,
        description: str,
    ) -> HubCommit:
        """Commit an exact, already-audited set of local files."""


class HuggingFaceHubClient:
    """Lazy Hugging Face implementation that never reads or logs a token."""

    def ensure_private_repo(self, config: CheckpointHubConfig) -> None:
        """Create the destination if needed and reject public visibility."""

        api = _hub_api()
        api.create_repo(
            repo_id=config.repo_id,
            repo_type=config.repo_type,
            private=True,
            exist_ok=True,
        )
        info = api.repo_info(
            config.repo_id,
            repo_type=config.repo_type,
            revision=config.revision,
        )
        if getattr(info, "private", None) is not True:
            raise RuntimeError(
                f"Checkpoint repository must remain private: {config.repo_id}"
            )

    def commit_files(
        self,
        config: CheckpointHubConfig,
        files: tuple[HubUploadFile, ...],
        *,
        message: str,
        description: str,
    ) -> HubCommit:
        """Upload one epoch as one Hub commit using explicit file operations."""

        from huggingface_hub import CommitOperationAdd

        operations: list[object] = [
            CommitOperationAdd(
                path_in_repo=item.path_in_repo,
                path_or_fileobj=str(item.local_path),
            )
            for item in files
        ]
        result = _hub_api().create_commit(
            repo_id=config.repo_id,
            repo_type=config.repo_type,
            revision=config.revision,
            operations=operations,
            commit_message=message,
            commit_description=description,
            run_as_future=False,
        )
        oid = getattr(result, "oid", None)
        url = getattr(result, "commit_url", None)
        if not isinstance(oid, str) or not oid:
            raise RuntimeError("Hugging Face commit did not return an oid")
        if not isinstance(url, str) or not url:
            raise RuntimeError("Hugging Face commit did not return a URL")
        return HubCommit(oid=oid, url=url)


class _HubApi(Protocol):
    """Structural type for the dynamically imported Hugging Face client."""

    def create_repo(self, **kwargs: object) -> object:
        """Create a repository."""

    def repo_info(self, *args: object, **kwargs: object) -> object:
        """Return repository metadata."""

    def create_commit(self, **kwargs: object) -> object:
        """Create a repository commit."""


def _hub_api() -> _HubApi:
    from huggingface_hub import HfApi

    return cast(_HubApi, HfApi())
