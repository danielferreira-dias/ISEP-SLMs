"""CPU-only tests for durable training execution and resume identity."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.train.backends.contracts import (
    BackendFitResult,
    BackendPrediction,
    CheckpointEvent,
    CheckpointObserver,
    FineTuneRequest,
    GenerationSpec,
    LoadedCheckpoint,
    LoraSpec,
    MetricEvent,
    MetricSink,
    ModelLoadSpec,
    PredictionSample,
    RuntimeInfo,
    TrainerSpec,
)
from src.train.backends.parameters import build_trainable_parameter_manifest
from src.train.execution.executor import TrainingExecutor
from src.train.execution.identity import (
    CheckpointRecorder,
    RunIdentity,
    validate_resume_checkpoint,
)
from src.train.execution.io import (
    atomic_write_json,
    atomic_write_text,
    read_json_array,
    read_json_object,
)
from src.train.reporting import _resource_points
from src.train.resource_metrics import resource_summary


class _Sink(MetricSink):
    def __init__(self) -> None:
        self.events: list[MetricEvent] = []
        self.closed = False

    def write(self, event: MetricEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        self.closed = True


class _Monitor:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class _CheckpointObserver:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.events: list[CheckpointEvent] = []
        self.manifest_existed_when_called = False

    def on_checkpoint(self, event: CheckpointEvent) -> None:
        self.manifest_existed_when_called = (
            event.path / "isep_checkpoint.json"
        ).is_file()
        self.events.append(event)
        if self.failure is not None:
            raise self.failure


class _Parameter:
    requires_grad = True

    def numel(self) -> int:
        return 16


class _Model:
    def named_parameters(self) -> list[tuple[str, _Parameter]]:
        return [
            ("model.layers.0.self_attn.q_proj.lora_A", _Parameter()),
            ("model.layers.0.mlp.up_proj.lora_A", _Parameter()),
        ]


class _FakeBackend:
    name = "fake"

    def __init__(
        self,
        *,
        failure: Exception | None = None,
        global_step: int = 10,
        metrics: dict[str, float | int] | None = None,
    ) -> None:
        self.failure = failure
        self.global_step = global_step
        self.metrics = metrics or {"train_loss": 0.5}
        self.resume_from_checkpoint: Path | None = None

    def validate_runtime(self) -> RuntimeInfo:
        return _runtime()

    def fit(
        self,
        request: FineTuneRequest,
        *,
        metric_sink: MetricSink,
        checkpoint_observer: CheckpointObserver,
        resume_from_checkpoint: Path | None = None,
    ) -> BackendFitResult:
        self.resume_from_checkpoint = resume_from_checkpoint
        if self.failure is not None:
            raise self.failure
        checkpoint = request.trainer.output_dir / f"checkpoint-{self.global_step}"
        checkpoint.mkdir(parents=True)
        for filename in (
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
            "adapter_config.json",
            "adapter_model.safetensors",
        ):
            (checkpoint / filename).write_text("state", encoding="utf-8")
        event = CheckpointEvent(
            checkpoint,
            global_step=self.global_step,
            epoch=float(self.global_step) / 10.0,
        )
        checkpoint_observer.on_checkpoint(event)
        metric_sink.write(
            MetricEvent("loss", 0.5, step=self.global_step, epoch=event.epoch)
        )
        final_adapter = request.trainer.output_dir / "final_adapter"
        final_adapter.mkdir(exist_ok=True)
        return BackendFitResult(
            global_step=self.global_step,
            training_loss=0.5,
            metrics=self.metrics,
            checkpoints=(event,),
            final_adapter_dir=final_adapter,
            trainable_parameters=build_trainable_parameter_manifest(_Model()),
            runtime=_runtime(),
        )

    def load_base(self, model: ModelLoadSpec) -> LoadedCheckpoint:
        del model
        return LoadedCheckpoint(object(), object(), _runtime(), None)

    def load_checkpoint(
        self,
        *,
        model: ModelLoadSpec,
        checkpoint_path: Path,
    ) -> LoadedCheckpoint:
        del model
        return LoadedCheckpoint(object(), object(), _runtime(), checkpoint_path)

    def predict(
        self,
        loaded: LoadedCheckpoint,
        samples: list[PredictionSample],
        *,
        generation: GenerationSpec | None = None,
    ) -> tuple[BackendPrediction, ...]:
        del loaded, generation
        return tuple(BackendPrediction(item.sample_id, "label") for item in samples)

    def release(self, loaded: LoadedCheckpoint) -> None:
        del loaded


def _runtime() -> RuntimeInfo:
    return RuntimeInfo("test", "12.8", "NVIDIA test", 1, True, 1_000_000)


def _identity() -> RunIdentity:
    return RunIdentity(
        experiment_id="e1",
        run_id="run-1",
        config_hash="config-hash",
        dataset_hash="dataset-hash",
        model_id="Qwen/Qwen3.5-4B",
        model_revision=ModelLoadSpec().revision,
        execution_profile="full",
    )


class TrainExecutionTests(unittest.TestCase):
    def test_downstream_observer_runs_after_resume_manifest_is_durable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            observer = _CheckpointObserver()
            request = FineTuneRequest(
                model=ModelLoadSpec(),
                lora=LoraSpec(finetune_vision_layers=False),
                trainer=TrainerSpec(output_dir=run_dir / "checkpoints"),
                train_dataset=object(),
                eval_dataset=object(),
            )

            TrainingExecutor(
                backend=_FakeBackend(),
                run_dir=run_dir,
                identity=_identity(),
                metric_sink=_Sink(),
                resource_monitor=_Monitor(),
                checkpoint_observer=observer,
            ).execute(request)

            self.assertTrue(observer.manifest_existed_when_called)
            self.assertEqual(len(observer.events), 1)

    def test_remote_checkpoint_failure_preserves_resumable_local_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            request = FineTuneRequest(
                model=ModelLoadSpec(),
                lora=LoraSpec(finetune_vision_layers=False),
                trainer=TrainerSpec(output_dir=run_dir / "checkpoints"),
                train_dataset=object(),
                eval_dataset=object(),
            )
            observer = _CheckpointObserver(failure=RuntimeError("Hub unavailable"))

            with self.assertRaisesRegex(RuntimeError, "Hub unavailable"):
                TrainingExecutor(
                    backend=_FakeBackend(),
                    run_dir=run_dir,
                    identity=_identity(),
                    metric_sink=_Sink(),
                    resource_monitor=_Monitor(),
                    checkpoint_observer=observer,
                ).execute(request)

            checkpoint = run_dir / "checkpoints" / "checkpoint-10"
            validate_resume_checkpoint(checkpoint, _identity())
            status = read_json_object(run_dir / "manifests" / "run_status.json")
            self.assertEqual(status["status"], "failed")

    def test_resume_retries_mirroring_of_existing_checkpoint_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            request = FineTuneRequest(
                model=ModelLoadSpec(),
                lora=LoraSpec(finetune_vision_layers=False),
                trainer=TrainerSpec(output_dir=run_dir / "checkpoints"),
                train_dataset=object(),
                eval_dataset=object(),
            )
            TrainingExecutor(
                backend=_FakeBackend(),
                run_dir=run_dir,
                identity=_identity(),
                metric_sink=_Sink(),
                resource_monitor=_Monitor(),
            ).execute(request)
            status_path = run_dir / "manifests" / "run_status.json"
            status = read_json_object(status_path)
            status["status"] = "interrupted"
            atomic_write_json(status_path, status)
            observer = _CheckpointObserver()

            TrainingExecutor(
                backend=_FakeBackend(global_step=20),
                run_dir=run_dir,
                identity=_identity(),
                metric_sink=_Sink(),
                resource_monitor=_Monitor(),
                checkpoint_observer=observer,
            ).execute(
                request,
                resume_from_checkpoint=run_dir / "checkpoints" / "checkpoint-10",
            )

            self.assertEqual(
                [event.global_step for event in observer.events],
                [10, 20],
            )

    def test_success_writes_status_result_and_resume_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            sink = _Sink()
            monitor = _Monitor()
            request = FineTuneRequest(
                model=ModelLoadSpec(),
                lora=LoraSpec(finetune_vision_layers=False),
                trainer=TrainerSpec(output_dir=run_dir / "checkpoints"),
                train_dataset=object(),
                eval_dataset=object(),
            )
            result = TrainingExecutor(
                backend=_FakeBackend(),
                run_dir=run_dir,
                identity=_identity(),
                metric_sink=sink,
                resource_monitor=monitor,
            ).execute(request)

            status = read_json_object(run_dir / "manifests" / "run_status.json")
            self.assertEqual(status["status"], "completed")
            self.assertEqual(result.global_step, 10)
            self.assertTrue(monitor.started)
            self.assertTrue(monitor.stopped)
            self.assertTrue(sink.closed)
            checkpoint = run_dir / "checkpoints" / "checkpoint-10"
            validate_resume_checkpoint(checkpoint, _identity())
            (checkpoint / "optimizer.pt").unlink()
            with self.assertRaisesRegex(RuntimeError, "not resumable"):
                validate_resume_checkpoint(checkpoint, _identity())
            backend_result = read_json_object(
                run_dir / "manifests" / "backend_result.json"
            )
            self.assertEqual(backend_result["global_step"], 10)
            backend_sessions = read_json_array(
                run_dir / "manifests" / "backend_sessions.json"
            )
            self.assertEqual(len(backend_sessions), 1)

    def test_resume_accumulates_backend_runtime_and_weighted_throughput(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            request = FineTuneRequest(
                model=ModelLoadSpec(),
                lora=LoraSpec(finetune_vision_layers=False),
                trainer=TrainerSpec(output_dir=run_dir / "checkpoints"),
                train_dataset=object(),
                eval_dataset=object(),
            )
            TrainingExecutor(
                backend=_FakeBackend(
                    global_step=10,
                    metrics={
                        "train_runtime": 10.0,
                        "train_samples_per_second": 2.0,
                    },
                ),
                run_dir=run_dir,
                identity=_identity(),
                metric_sink=_Sink(),
                resource_monitor=_Monitor(),
            ).execute(request)

            status_path = run_dir / "manifests" / "run_status.json"
            status = read_json_object(status_path)
            status["status"] = "interrupted"
            atomic_write_json(status_path, status)
            resumed_backend = _FakeBackend(
                global_step=20,
                metrics={
                    "train_runtime": 20.0,
                    "train_samples_per_second": 4.0,
                },
            )
            TrainingExecutor(
                backend=resumed_backend,
                run_dir=run_dir,
                identity=_identity(),
                metric_sink=_Sink(),
                resource_monitor=_Monitor(),
            ).execute(
                request,
                resume_from_checkpoint=run_dir / "checkpoints" / "checkpoint-10",
            )

            latest = read_json_object(run_dir / "manifests" / "backend_result.json")
            sessions = read_json_array(run_dir / "manifests" / "backend_sessions.json")
            summary = resource_summary(run_dir)
            self.assertEqual(latest["global_step"], 20)
            self.assertEqual(
                resumed_backend.resume_from_checkpoint,
                run_dir / "checkpoints" / "checkpoint-10",
            )
            self.assertEqual(len(sessions), 2)
            self.assertEqual(summary.duration_seconds, 30.0)
            self.assertAlmostEqual(summary.gpu_hours or 0.0, 30.0 / 3600.0)
            self.assertAlmostEqual(
                summary.train_samples_per_second or 0.0,
                (10.0 * 2.0 + 20.0 * 4.0) / 30.0,
            )

    def test_resource_summary_falls_back_to_legacy_latest_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            atomic_write_json(
                run_dir / "manifests" / "backend_result.json",
                {
                    "metrics": {
                        "train_runtime": 12.5,
                        "train_samples_per_second": 3.25,
                        "train_tokens_per_second": 96.0,
                        "train_steps_per_second": 0.5,
                    },
                    "trainable_parameters": {"total": 123456},
                },
            )
            atomic_write_text(
                run_dir / "logs" / "resources.jsonl",
                "\n".join(
                    (
                        '{"elapsed_seconds":0,"gpu_memory_used_bytes":1073741824,'
                        '"process_rss_bytes":2147483648,"gpu_power_watts":100,'
                        '"gpu_utilization_percent":50,"gpu_temperature_celsius":60}',
                        '{"elapsed_seconds":3600,"gpu_memory_used_bytes":2147483648,'
                        '"process_rss_bytes":4294967296,"gpu_power_watts":100,'
                        '"gpu_utilization_percent":70,"gpu_temperature_celsius":70}',
                    )
                )
                + "\n",
            )
            checkpoint = run_dir / "checkpoints" / "checkpoint-10"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter.bin").write_bytes(b"x" * 1024)
            atomic_write_json(
                run_dir / "manifests" / "best_checkpoint.json",
                {
                    "checkpoint_id": "checkpoint-10",
                    "path": "/remote/path/that/is/not/local",
                },
            )

            summary = resource_summary(run_dir)

            self.assertEqual(summary.duration_seconds, 12.5)
            self.assertEqual(summary.train_samples_per_second, 3.25)
            self.assertEqual(summary.train_tokens_per_second, 96.0)
            self.assertEqual(summary.average_step_seconds, 2.0)
            self.assertEqual(summary.peak_vram_gib, 2.0)
            self.assertEqual(summary.average_vram_gib, 1.5)
            self.assertEqual(summary.peak_ram_gib, 4.0)
            self.assertEqual(summary.average_ram_gib, 3.0)
            self.assertEqual(summary.average_gpu_utilization_percent, 60.0)
            self.assertEqual(summary.average_power_watts, 100.0)
            self.assertEqual(summary.energy_wh, 100.0)
            self.assertEqual(summary.maximum_temperature_celsius, 70.0)
            self.assertEqual(summary.trainable_parameters, 123456)
            self.assertAlmostEqual(summary.checkpoint_size_gib or 0.0, 1024 / 1024**3)

    def test_resource_points_normalize_elapsed_time_after_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            samples = (
                '{"elapsed_seconds":1.0,"gpu_memory_used_bytes":1024}',
                '{"elapsed_seconds":6.0,"gpu_memory_used_bytes":2048}',
                '{"elapsed_seconds":1.0,"gpu_memory_used_bytes":3072}',
                '{"elapsed_seconds":5.0,"gpu_memory_used_bytes":4096}',
            )
            atomic_write_text(
                run_dir / "logs" / "resources.jsonl",
                "\n".join(samples) + "\n",
            )

            points = _resource_points(run_dir)

            self.assertEqual(
                [point.elapsed_seconds for point in points],
                [1.0, 6.0, 7.0, 11.0],
            )

    def test_failure_is_durable_and_never_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            backend = _FakeBackend(failure=RuntimeError("out of memory"))
            request = FineTuneRequest(
                model=ModelLoadSpec(),
                lora=LoraSpec(finetune_vision_layers=False),
                trainer=TrainerSpec(output_dir=run_dir / "checkpoints"),
                train_dataset=object(),
                eval_dataset=object(),
            )
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                TrainingExecutor(
                    backend=backend,
                    run_dir=run_dir,
                    identity=_identity(),
                    metric_sink=_Sink(),
                    resource_monitor=_Monitor(),
                ).execute(request)

            status = read_json_object(run_dir / "manifests" / "run_status.json")
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["error_message"], "out of memory")

    def test_checkpoint_missing_optimizer_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint-1"
            checkpoint.mkdir()
            recorder = CheckpointRecorder(_identity())

            with self.assertRaisesRegex(RuntimeError, "not resumable"):
                recorder.on_checkpoint(
                    CheckpointEvent(checkpoint, global_step=1, epoch=1.0)
                )

    def test_resume_rejects_dataset_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint-1"
            checkpoint.mkdir()
            for filename in (
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "trainer_state.json",
                "adapter_config.json",
                "adapter_model.safetensors",
            ):
                (checkpoint / filename).write_text("state", encoding="utf-8")
            CheckpointRecorder(_identity()).on_checkpoint(
                CheckpointEvent(checkpoint, global_step=1, epoch=1.0)
            )
            mismatched = RunIdentity(
                experiment_id="e1",
                run_id="run-1",
                config_hash="config-hash",
                dataset_hash="different",
                model_id="Qwen/Qwen3.5-4B",
                model_revision=ModelLoadSpec().revision,
                execution_profile="full",
            )

            with self.assertRaisesRegex(RuntimeError, "does not match"):
                validate_resume_checkpoint(checkpoint, mismatched)


if __name__ == "__main__":
    unittest.main()
