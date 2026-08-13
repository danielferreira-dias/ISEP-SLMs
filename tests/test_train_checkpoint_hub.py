"""Tests for private, epoch-addressed Hugging Face checkpoint mirroring."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.train.backends.contracts import CheckpointEvent, ModelLoadSpec
from src.train.checkpoint_hub import (
    HubCheckpointMirror,
    create_checkpoint_mirror,
)
from src.train.checkpoint_hub_client import HubCommit, HubUploadFile
from src.train.config import CheckpointHubConfig
from src.train.execution.identity import CheckpointRecorder, RunIdentity
from src.train.execution.io import read_json_array


class _FakeHubClient:
    def __init__(self) -> None:
        self.ensure_calls = 0
        self.commits: list[tuple[tuple[HubUploadFile, ...], str, str]] = []

    def ensure_private_repo(self, config: CheckpointHubConfig) -> None:
        self.ensure_calls += 1
        if config.private is not True:
            raise AssertionError("test destination must be private")

    def commit_files(
        self,
        config: CheckpointHubConfig,
        files: tuple[HubUploadFile, ...],
        *,
        message: str,
        description: str,
    ) -> HubCommit:
        del config
        self.commits.append((files, message, description))
        return HubCommit(
            oid=f"commit-{len(self.commits)}",
            url=f"https://huggingface.co/test/commit/{len(self.commits)}",
        )


def _identity(*, profile: str = "full") -> RunIdentity:
    return RunIdentity(
        experiment_id="e1_label_frozen_vision",
        run_id="run-20260812",
        config_hash="config-hash",
        dataset_hash="dataset-hash",
        model_id="Qwen/Qwen3.5-4B",
        model_revision=ModelLoadSpec().revision,
        execution_profile=profile,
    )


def _checkpoint(directory: Path, *, epoch: int = 1) -> CheckpointEvent:
    path = directory / f"checkpoint-{epoch * 100}"
    path.mkdir(parents=True)
    for filename in (
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
    ):
        (path / filename).write_text(f"{filename}-{epoch}", encoding="utf-8")
    event = CheckpointEvent(path=path, global_step=epoch * 100, epoch=float(epoch))
    CheckpointRecorder(_identity()).on_checkpoint(event)
    return event


class CheckpointHubTests(unittest.TestCase):
    def test_epoch_checkpoint_uses_exact_private_repository_hierarchy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = _FakeHubClient()
            audit_path = root / "manifests" / "checkpoint_uploads.json"
            mirror = HubCheckpointMirror(
                config=CheckpointHubConfig(enabled=True),
                identity=_identity(),
                seed=3407,
                audit_path=audit_path,
                client=client,
            )

            mirror.on_checkpoint(_checkpoint(root))

            self.assertEqual(client.ensure_calls, 1)
            self.assertEqual(len(client.commits), 1)
            files, message, description = client.commits[0]
            expected_prefix = (
                "e1_label_frozen_vision/seed-3407/run-20260812/checkpoint-epoch-01/"
            )
            self.assertTrue(files)
            self.assertTrue(
                all(item.path_in_repo.startswith(expected_prefix) for item in files)
            )
            self.assertIn("epoch 01", message)
            self.assertIn("no clinical images", description)
            self.assertFalse(any(item.path_in_repo.endswith(".jpg") for item in files))
            records = read_json_array(audit_path)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertIsInstance(record, dict)
            assert isinstance(record, dict)
            self.assertEqual(record["commit_oid"], "commit-1")
            self.assertEqual(record["path_in_repo"], expected_prefix.removesuffix("/"))
            self.assertIsInstance(record["checkpoint_tree_sha256"], str)

    def test_repeated_identical_event_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = _FakeHubClient()
            mirror = HubCheckpointMirror(
                config=CheckpointHubConfig(enabled=True),
                identity=_identity(),
                seed=3407,
                audit_path=root / "audit.json",
                client=client,
            )
            event = _checkpoint(root)

            mirror.on_checkpoint(event)
            mirror.on_checkpoint(event)

            self.assertEqual(len(client.commits), 1)
            self.assertEqual(client.ensure_calls, 2)

    def test_existing_remote_path_rejects_different_local_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mirror = HubCheckpointMirror(
                config=CheckpointHubConfig(enabled=True),
                identity=_identity(),
                seed=3407,
                audit_path=root / "audit.json",
                client=_FakeHubClient(),
            )
            event = _checkpoint(root)
            mirror.on_checkpoint(event)
            (event.path / "optimizer.pt").write_text("changed", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "different bytes"):
                mirror.on_checkpoint(event)

    def test_unexpected_clinical_file_is_never_uploaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = _FakeHubClient()
            mirror = HubCheckpointMirror(
                config=CheckpointHubConfig(enabled=True),
                identity=_identity(),
                seed=3407,
                audit_path=root / "audit.json",
                client=client,
            )
            event = _checkpoint(root)
            (event.path / "patient.jpg").write_bytes(b"clinical image")

            with self.assertRaisesRegex(RuntimeError, "outside.*allowlist"):
                mirror.on_checkpoint(event)
            self.assertFalse(client.commits)

    def test_noninteger_epoch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = _checkpoint(root)
            invalid = CheckpointEvent(event.path, event.global_step, 1.5)
            mirror = HubCheckpointMirror(
                config=CheckpointHubConfig(enabled=True),
                identity=_identity(),
                seed=3407,
                audit_path=root / "audit.json",
                client=_FakeHubClient(),
            )

            with self.assertRaisesRegex(RuntimeError, "integer epoch"):
                mirror.on_checkpoint(invalid)

    def test_factory_disables_upload_for_smoke_and_disabled_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifests = Path(temporary)
            self.assertIsNone(
                create_checkpoint_mirror(
                    config=CheckpointHubConfig(enabled=True),
                    identity=_identity(profile="smoke"),
                    seed=3407,
                    manifests_directory=manifests,
                    smoke=True,
                    client=_FakeHubClient(),
                )
            )
            self.assertIsNone(
                create_checkpoint_mirror(
                    config=CheckpointHubConfig(enabled=False),
                    identity=_identity(),
                    seed=3407,
                    manifests_directory=manifests,
                    smoke=False,
                    client=_FakeHubClient(),
                )
            )


if __name__ == "__main__":
    unittest.main()
