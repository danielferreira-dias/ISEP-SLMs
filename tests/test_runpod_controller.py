"""Tests for the durable RunPod benchmark controller."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from dataclasses import replace
import tempfile
import unittest

from src.benchmark.runpod_controller import (
    ControllerConfig,
    MirrorTarget,
    RunPodConnection,
    build_remote_runner_command,
    build_rsync_pull_command,
    build_rsync_push_command,
    prediction_summary,
)


class RunPodControllerTests(unittest.TestCase):
    def _config(self, local_root: Path) -> ControllerConfig:
        return ControllerConfig(
            connection=RunPodConnection(
                host="203.0.113.10",
                port=40055,
                user="root",
                identity_file=Path("/keys/runpod"),
                known_hosts_file=Path("/tmp/runpod-known-hosts"),
            ),
            local_project_root=local_root,
            remote_project_root=PurePosixPath("/workspace/ISEP"),
            targets=(MirrorTarget(PurePosixPath("outputs/test")),),
            local_log=local_root / "runs/benchmarks/test.log",
            temperature=0.6,
            batch_size=8,
            sync_interval_seconds=15.0,
        )

    def test_pull_is_incremental_and_never_deletes_local_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = build_rsync_pull_command(
                self._config(Path(directory)),
                MirrorTarget(PurePosixPath("outputs/test")),
            )

        self.assertEqual(command[:2], ["rsync", "-az"])
        self.assertIn("--partial", command)
        self.assertIn("--update", command)
        self.assertNotIn("--delete", command)
        self.assertEqual(
            command[-2],
            "root@203.0.113.10:/workspace/ISEP/outputs/test/",
        )

    def test_push_only_builds_a_checkpoint_restore_without_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = build_rsync_push_command(
                self._config(Path(directory)),
                MirrorTarget(PurePosixPath("outputs/test")),
            )

        self.assertNotIn("--delete", command)
        self.assertTrue(command[-2].endswith("/outputs/test/"))
        self.assertEqual(
            command[-1],
            "root@203.0.113.10:/workspace/ISEP/outputs/test/",
        )

    def test_remote_runner_preserves_fixed_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = build_remote_runner_command(self._config(Path(directory)))

        self.assertIn("scripts/run_dermobench_and_context.py", command)
        self.assertIn("--temperature 0.6", command)
        self.assertIn("--batch-size 8", command)
        self.assertIn("--project-root /workspace/ISEP", command)

    def test_remote_runner_forwards_explicit_task_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                self._config(Path(directory)),
                skip_dermobench_tasks=(
                    "task_3_1_diagnostic_reasoning_without_morphology",
                    "task_3_2_diagnostic_reasoning_with_morphology",
                ),
            )
            command = build_remote_runner_command(config)

        self.assertIn(
            "--skip-dermobench-task "
            "task_3_1_diagnostic_reasoning_without_morphology",
            command,
        )
        self.assertIn(
            "--skip-dermobench-task "
            "task_3_2_diagnostic_reasoning_with_morphology",
            command,
        )

    def test_prediction_summary_counts_only_complete_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            predictions = root / "outputs/test/task/model/run/predictions.jsonl"
            predictions.parent.mkdir(parents=True)
            predictions.write_bytes(b'{"id":1}\n{"id":2}\n{"partial":')

            files, records = prediction_summary(config)

        self.assertEqual(files, 1)
        self.assertEqual(records, 2)


if __name__ == "__main__":
    unittest.main()
