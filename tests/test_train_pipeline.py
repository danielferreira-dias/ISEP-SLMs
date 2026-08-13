"""CPU integration test for the complete E1 orchestration contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

from src.train.artifacts import ArtifactStore
from src.train.backends import (
    BackendFitResult,
    BackendPrediction,
    CheckpointEvent,
    CheckpointObserver,
    FineTuneRequest,
    GenerationSpec,
    LoadedCheckpoint,
    MetricEvent,
    MetricSink,
    ModelLoadSpec,
    PredictionSample,
    RuntimeInfo,
)
from src.train.backends.parameters import build_trainable_parameter_manifest
from src.train.execution.sinks import JsonlMetricSink
from src.train.finalize import evaluate_run
from src.train.pipeline import run_training
from src.train.preparation import open_run_store
from tests.test_train_data import _toy_config, _write_toy_source


class _Parameter:
    requires_grad = True

    def numel(self) -> int:
        return 16


class _ParameterModel:
    def named_parameters(self) -> list[tuple[str, _Parameter]]:
        return [
            ("model.layers.0.self_attn.q_proj.lora_A", _Parameter()),
            ("model.layers.0.mlp.up_proj.lora_A", _Parameter()),
        ]


class _FakeBackend:
    """Produce three valid epoch checkpoints without importing CUDA."""

    name = "fake"

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
        if resume_from_checkpoint is not None:
            raise AssertionError("Fresh integration run unexpectedly resumed")
        mask_path = (
            request.trainer.output_dir.parent
            / "manifests"
            / "assistant_mask_audit.json"
        )
        mask_path.write_text('{"passed": true}\n', encoding="utf-8")
        events: list[CheckpointEvent] = []
        for epoch in range(1, 4):
            step = epoch * 10
            checkpoint = request.trainer.output_dir / f"checkpoint-{step}"
            checkpoint.mkdir(parents=True)
            _write_checkpoint(checkpoint, epoch=epoch, step=step)
            event = CheckpointEvent(checkpoint, step, float(epoch))
            checkpoint_observer.on_checkpoint(event)
            events.append(event)
            metric_sink.write(
                MetricEvent("loss", 1.0 / epoch, step=step, epoch=float(epoch))
            )
            metric_sink.write(
                MetricEvent(
                    "eval_loss",
                    0.4 / epoch,
                    step=step,
                    epoch=float(epoch),
                )
            )
            metric_sink.write(
                MetricEvent(
                    "learning_rate",
                    2e-4 / epoch,
                    step=step,
                    epoch=float(epoch),
                )
            )
        final_adapter = request.trainer.output_dir / "final_adapter"
        final_adapter.mkdir()
        return BackendFitResult(
            global_step=30,
            training_loss=1.0 / 3.0,
            metrics={
                "train_loss": 1.0 / 3.0,
                "train_runtime": 12.0,
                "train_samples_per_second": 5.25,
            },
            checkpoints=tuple(events),
            final_adapter_dir=final_adapter,
            trainable_parameters=build_trainable_parameter_manifest(_ParameterModel()),
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
        samples: Sequence[PredictionSample],
        *,
        generation: GenerationSpec | None = None,
    ) -> tuple[BackendPrediction, ...]:
        del loaded, generation
        return tuple(
            BackendPrediction(sample.sample_id, _label_for(sample.sample_id))
            for sample in samples
        )

    def release(self, loaded: LoadedCheckpoint) -> None:
        del loaded


class _Monitor:
    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


class _Plotter:
    """Create deterministic placeholder figures for orchestration testing."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def _write(self, name: str) -> None:
        (self._directory / f"{name}.png").write_bytes(b"png")
        (self._directory / f"{name}.svg").write_text("<svg/>", encoding="utf-8")
        (self._directory / f"{name}.csv").write_text(
            "metric,value\ntest,1\n", encoding="utf-8"
        )

    def training_history(self, points: object) -> None:
        del points
        self._write("training_history")

    def checkpoint_metrics(self, points: object) -> None:
        del points
        self._write("checkpoint_metrics")

    def per_class_metrics(self, metrics: object) -> None:
        del metrics
        self._write("per_class_metrics")

    def confusion_matrix(self, metrics: object) -> None:
        del metrics
        self._write("confusion_matrix")

    def class_distribution(self, points: object) -> None:
        del points
        self._write("class_distribution")

    def source_distribution(self, points: object) -> None:
        del points
        self._write("source_distribution")

    def trainable_parameters(self, points: object) -> None:
        del points
        self._write("trainable_parameters")

    def resource_usage(self, points: object) -> None:
        del points
        self._write("resource_usage")


def _runtime() -> RuntimeInfo:
    return RuntimeInfo("test", "12.8", "NVIDIA fake", 1, True, 80_000_000_000)


def _label_for(sample_id: str) -> str:
    index = int(sample_id.rsplit("_", maxsplit=1)[1])
    return "class_a" if index % 2 == 0 else "class_b"


def _write_checkpoint(path: Path, *, epoch: int, step: int) -> None:
    state = {
        "epoch": float(epoch),
        "global_step": step,
        "log_history": [{"step": step, "eval_loss": 0.4 / epoch}],
    }
    (path / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    for filename in (
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "adapter_config.json",
        "adapter_model.safetensors",
    ):
        (path / filename).write_bytes(filename.encode())


def _metric_sink(run_directory: Path, *, require_tensorboard: bool) -> MetricSink:
    if not require_tensorboard:
        raise AssertionError("Scientific runs must require TensorBoard")
    (run_directory / "tensorboard" / "events.fake").write_bytes(b"event")
    return JsonlMetricSink(run_directory / "logs" / "metrics.jsonl")


class TrainingPipelineIntegrationTests(unittest.TestCase):
    def test_fake_backend_builds_complete_thesis_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            release_root = root / "release"
            _write_toy_source(source)
            config = _toy_config(root, source, release_root)
            from src.train.data import prepare_data_release

            prepare_data_release(config)
            with (
                patch(
                    "src.train.pipeline.create_default_metric_sink",
                    side_effect=_metric_sink,
                ),
                patch(
                    "src.train.pipeline.LocalResourceMonitor",
                    return_value=_Monitor(),
                ),
                patch("src.train.reporting.ThesisPlotter", _Plotter),
                patch.dict(
                    "os.environ",
                    {"MPLCONFIGDIR": str(root / "matplotlib-cache")},
                ),
            ):
                result = run_training(
                    config,
                    backend=_FakeBackend(),
                    run_id="cpu-integration",
                )

            run = result.run_directory
            self.assertEqual(result.best_checkpoint.name, "checkpoint-30")
            self.assertEqual(result.best_metrics.macro_f1, 1.0)
            for relative in (
                "manifests/config.resolved.json",
                "manifests/dataset_release.json",
                "manifests/environment.json",
                "manifests/best_checkpoint.json",
                "manifests/backend_result.json",
                "metrics/classification.json",
                "predictions/sft_dev__checkpoint-30.parquet",
                "figures/checkpoint_metrics.png",
                "figures/checkpoint_metrics.csv",
                "tables/checkpoint_metrics.tex",
                "tables/resource_summary.csv",
                "report/thesis_summary.md",
                "report/report.html",
                "tensorboard/events.fake",
            ):
                self.assertTrue((run / relative).is_file(), relative)
            status = json.loads(
                (run / "manifests" / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "completed")
            tampered = result.best_checkpoint / "adapter_model.safetensors"
            tampered.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "corrupted"):
                evaluate_run(run, backend=_FakeBackend())

    def test_smoke_profile_is_validated_and_cannot_resume_as_full(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            release_root = root / "release"
            _write_toy_source(source)
            config = _toy_config(root, source, release_root)
            from src.train.data import prepare_data_release

            prepare_data_release(config)
            with (
                patch(
                    "src.train.pipeline.create_default_metric_sink",
                    side_effect=_metric_sink,
                ),
                patch(
                    "src.train.pipeline.LocalResourceMonitor",
                    return_value=_Monitor(),
                ),
                patch("src.train.reporting.ThesisPlotter", _Plotter),
            ):
                result = run_training(
                    config,
                    backend=_FakeBackend(),
                    smoke=True,
                    run_id="cpu-smoke",
                )

            validation = result.run_directory / "manifests" / "smoke_validation.json"
            self.assertTrue(validation.is_file())
            with self.assertRaisesRegex(ValueError, "smoke run as full"):
                open_run_store(
                    config,
                    result.best_checkpoint,
                    False,
                    requested_run_id=None,
                )

    def test_standalone_smoke_evaluation_revalidates_before_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            release_root = root / "release"
            _write_toy_source(source)
            config = _toy_config(root, source, release_root)
            from src.train.data import prepare_data_release

            release = prepare_data_release(config)
            with (
                patch(
                    "src.train.pipeline.create_default_metric_sink",
                    side_effect=_metric_sink,
                ),
                patch(
                    "src.train.pipeline.LocalResourceMonitor",
                    return_value=_Monitor(),
                ),
                patch("src.train.reporting.ThesisPlotter", _Plotter),
            ):
                result = run_training(
                    config,
                    backend=_FakeBackend(),
                    smoke=True,
                    run_id="cpu-smoke-reevaluate",
                )

            run = result.run_directory
            store = ArtifactStore.at(run)
            store.write_status("failed", detail="prior smoke failure")
            (run / "manifests" / "smoke_validation.json").unlink()
            with (
                patch("src.train.reporting.ThesisPlotter", _Plotter),
                patch(
                    "src.train.finalize.validate_smoke_run",
                    side_effect=RuntimeError("smoke postcondition failed"),
                ) as validate_smoke,
                self.assertRaisesRegex(RuntimeError, "smoke postcondition failed"),
            ):
                evaluate_run(
                    run,
                    backend=_FakeBackend(),
                    config=config,
                    release=release,
                )

            validate_smoke.assert_called_once()
            status = json.loads(
                (run / "manifests" / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "failed")
            self.assertFalse((run / "manifests" / "smoke_validation.json").exists())


if __name__ == "__main__":
    unittest.main()
