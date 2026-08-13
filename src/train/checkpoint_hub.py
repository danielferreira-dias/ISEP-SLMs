"""Private, auditable Hugging Face mirroring for epoch checkpoints."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from src.train.backends.contracts import CheckpointEvent, CheckpointObserver
from src.train.checkpoint_hub_client import (
    HubCommitClient,
    HubUploadFile,
    HuggingFaceHubClient,
)
from src.train.config import CheckpointHubConfig
from src.train.execution.identity import RunIdentity
from src.train.execution.io import (
    JsonValue,
    atomic_write_json,
    read_json_array,
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_CHECKPOINT_FILES = frozenset(
    {
        "README.md",
        "adapter_config.json",
        "adapter_model.bin",
        "adapter_model.safetensors",
        "added_tokens.json",
        "chat_template.jinja",
        "generation_config.json",
        "isep_checkpoint.json",
        "merges.txt",
        "optimizer.pt",
        "preprocessor_config.json",
        "processor_config.json",
        "rng_state.pth",
        "scheduler.pt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "trainer_state.json",
        "training_args.bin",
        "vocab.json",
    }
)
_REQUIRED_CHECKPOINT_FILES = frozenset(
    {
        "adapter_config.json",
        "isep_checkpoint.json",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json",
    }
)


class HubCheckpointMirror(CheckpointObserver):
    """Mirror full-run epoch checkpoints into a private Hub repository."""

    def __init__(
        self,
        *,
        config: CheckpointHubConfig,
        identity: RunIdentity,
        seed: int,
        audit_path: Path,
        client: HubCommitClient | None = None,
    ) -> None:
        """Retain a fixed destination and local append-only upload audit."""

        if identity.execution_profile != "full":
            raise ValueError("Hub checkpoint mirroring is reserved for full runs")
        if seed not in {42, 3407, 2026}:
            raise ValueError("Checkpoint Hub seed must be an approved E1 seed")
        for field, value in (
            ("experiment_id", identity.experiment_id),
            ("run_id", identity.run_id),
        ):
            if _IDENTIFIER.fullmatch(value) is None:
                raise ValueError(f"Unsafe Hub {field}: {value!r}")
        self._config = config
        self._identity = identity
        self._seed = seed
        self._audit_path = audit_path
        self._client = client or HuggingFaceHubClient()

    def validate_destination(self) -> None:
        """Fail before GPU allocation unless the repository exists privately."""

        self._client.ensure_private_repo(self._config)

    def on_checkpoint(self, event: CheckpointEvent) -> None:
        """Validate and commit one complete integer-epoch checkpoint."""

        epoch = _integer_epoch(event.epoch)
        remote_root = _remote_root(self._identity, self._seed, epoch)
        local_files, tree_hash = _checkpoint_files(event.path, remote_root)
        records = _upload_records(self._audit_path)
        previous = _record_for_path(records, remote_root)
        self.validate_destination()
        if previous is not None:
            previous_hash = previous.get("checkpoint_tree_sha256")
            if previous_hash != tree_hash:
                raise RuntimeError(
                    "Hub checkpoint path already records different bytes: "
                    f"{remote_root}"
                )
            return

        commit = self._client.commit_files(
            self._config,
            local_files,
            message=(
                f"{self._identity.experiment_id} seed {self._seed} epoch {epoch:02d}"
            ),
            description=(
                "Resumable BF16 LoRA checkpoint. Contains model/trainer state "
                "and provenance manifests; contains no clinical images."
            ),
        )
        records.append(
            {
                "checkpoint_tree_sha256": tree_hash,
                "commit_oid": commit.oid,
                "commit_url": commit.url,
                "epoch": epoch,
                "global_step": event.global_step,
                "path_in_repo": remote_root,
                "repo_id": self._config.repo_id,
                "revision": self._config.revision,
                "uploaded_at_utc": datetime.now(UTC).isoformat(),
                "uploaded_files": [item.path_in_repo for item in local_files],
            }
        )
        atomic_write_json(self._audit_path, records)


def create_checkpoint_mirror(
    *,
    config: CheckpointHubConfig,
    identity: RunIdentity,
    seed: int,
    manifests_directory: Path,
    smoke: bool,
    client: HubCommitClient | None = None,
) -> HubCheckpointMirror | None:
    """Build the configured mirror, excluding smoke runs by contract."""

    if not config.enabled or (smoke and not config.upload_smoke):
        return None
    return HubCheckpointMirror(
        config=config,
        identity=identity,
        seed=seed,
        audit_path=manifests_directory / "checkpoint_uploads.json",
        client=client,
    )


def _integer_epoch(value: float | None) -> int:
    if value is None or not math.isfinite(value):
        raise RuntimeError("Hub checkpoint event has no finite epoch")
    rounded = round(value)
    if abs(value - rounded) > 1e-6 or rounded <= 0:
        raise RuntimeError(f"Hub checkpoints require an integer epoch, found {value}")
    return rounded


def _remote_root(identity: RunIdentity, seed: int, epoch: int) -> str:
    path = PurePosixPath(
        identity.experiment_id,
        f"seed-{seed}",
        identity.run_id,
        f"checkpoint-epoch-{epoch:02d}",
    )
    return path.as_posix()


def _checkpoint_files(
    checkpoint: Path,
    remote_root: str,
) -> tuple[tuple[HubUploadFile, ...], str]:
    if not checkpoint.is_dir():
        raise RuntimeError(f"Checkpoint directory does not exist: {checkpoint}")
    entries = tuple(sorted(checkpoint.iterdir(), key=lambda item: item.name))
    invalid = tuple(
        entry.name
        for entry in entries
        if entry.is_symlink()
        or not entry.is_file()
        or entry.name not in _SAFE_CHECKPOINT_FILES
    )
    if invalid:
        raise RuntimeError(
            "Checkpoint contains files outside the private upload allowlist: "
            + ", ".join(invalid)
        )
    names = {entry.name for entry in entries}
    missing = sorted(_REQUIRED_CHECKPOINT_FILES - names)
    weight_names = names & {"adapter_model.bin", "adapter_model.safetensors"}
    if missing or len(weight_names) != 1:
        details = [*missing]
        if len(weight_names) != 1:
            details.append("exactly one adapter weight file")
        raise RuntimeError(
            "Checkpoint is incomplete for Hub upload: " + ", ".join(details)
        )

    hashes = tuple((entry.name, _sha256(entry)) for entry in entries)
    digest = hashlib.sha256()
    for name, value in hashes:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    uploads = tuple(
        HubUploadFile(
            local_path=entry,
            path_in_repo=(PurePosixPath(remote_root) / entry.name).as_posix(),
        )
        for entry in entries
    )
    return uploads, digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _upload_records(path: Path) -> list[JsonValue]:
    if not path.is_file():
        return []
    records = read_json_array(path)
    if not all(isinstance(item, dict) for item in records):
        raise ValueError(f"Checkpoint upload audit contains a non-object: {path}")
    return records


def _record_for_path(
    records: Iterable[JsonValue], path_in_repo: str
) -> dict[str, JsonValue] | None:
    matches = tuple(
        item
        for item in records
        if isinstance(item, dict) and item.get("path_in_repo") == path_in_repo
    )
    if len(matches) > 1:
        raise ValueError(f"Duplicate Hub checkpoint audit path: {path_in_repo}")
    return matches[0] if matches else None
