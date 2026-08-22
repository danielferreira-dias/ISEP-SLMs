"""CPU-only contracts for the E3 student post-training namespace."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import ClassVar

import pytest
import yaml
from PIL import Image

from project.post_training.common import (
    DatasetReference,
    load_sft_stage_config,
    prepare_dataset_rows,
)
from project.post_training.sft import audit_sft_configuration, run_sft
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
from src.train.backends.contracts import TrainableParameterManifest

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "configs" / "training" / "e3_qwen3_5_4b_sft.yaml"


def _png_bytes(*, width: int = 20, height: int = 10) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "brown").save(buffer, format="PNG")
    return buffer.getvalue()


def _messages(target: str) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "text": ""},
                {"type": "text", "text": "Classify.\n\n/no_think"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target}],
        },
    ]


def _row(index: int, *, task: str = "diagnosis") -> dict[str, object]:
    target = "melanoma" if task == "diagnosis" else f"target-{task}"
    prompt = "Classify.\n\n/no_think"
    encoded = _png_bytes()
    return {
        "image": {"bytes": encoded, "path": None},
        "row_id": f"row-{index}-{task}",
        "sample_id": f"sample-{index}",
        "source_sample_id": f"source-{index}",
        "leakage_group_id": f"group-{index}",
        "disease_id": "D001",
        "gold_diagnosis": "melanoma",
        "source_dataset": "test",
        "image_sha256": hashlib.sha256(encoded).hexdigest(),
        "split": "sft_train",
        "task": task,
        "task_id": f"e3-{task}",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "target_text": target,
        "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
        "messages": _messages(target),
        "schema_version": "e3_multitask_sft_v1",
        "quality_status": "accepted",
    }


class _OneRowDataset:
    column_names: ClassVar[list[str]] = [
        "image",
        "row_id",
        "sample_id",
        "task",
        "target_text",
    ]

    def __init__(self, row: dict[str, object]) -> None:
        self._row = row

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> object:
        if index != 0:
            raise IndexError(index)
        return self._row


class _TaskColumn(Sequence[str]):
    _PARTS = (
        (6312, "diagnosis"),
        (6312, "morphology"),
        (6312, "caption"),
        (6127, "grounded_differential"),
        (21, "request_new_image"),
    )

    def __len__(self) -> int:
        return 25084

    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]  # type: ignore[misc]
        normalized = index if index >= 0 else len(self) + index
        if normalized < 0 or normalized >= len(self):
            raise IndexError(index)
        offset = normalized
        for count, task in self._PARTS:
            if offset < count:
                return task
            offset -= count
        raise AssertionError("unreachable")


class _VirtualDataset:
    def __init__(self, *, train: bool) -> None:
        self._train = train
        self.column_names = (
            ["image", "sample_id", "row_id", "task", "target_text"]
            if train
            else ["image", "sample_id", "target_text"]
        )

    def __len__(self) -> int:
        return 25084 if self._train else 1229

    def __getitem__(self, index: int | str) -> object:
        if isinstance(index, str):
            if index == "task" and self._train:
                return _TaskColumn()
            raise KeyError(index)
        task = _TaskColumn()[index] if self._train else "diagnosis"
        assert isinstance(task, str)
        row = _row(index, task=task)
        if not self._train:
            row.pop("row_id")
            row.pop("task")
            row.pop("target_sha256")
            row["split"] = "sft_dev"
        return row


class _Sink(MetricSink):
    def write(self, event: MetricEvent) -> None:
        del event

    def close(self) -> None:
        return None


class _Monitor:
    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


def _runtime() -> RuntimeInfo:
    return RuntimeInfo("test", "12.8", "NVIDIA test", 1, True, 1_000_000)


class _Backend:
    name = "unsloth"

    def __init__(self) -> None:
        self.request: FineTuneRequest | None = None

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
        assert resume_from_checkpoint is None
        assert len(request.train_dataset) == 16  # type: ignore[arg-type]
        assert len(request.eval_dataset) == 8  # type: ignore[arg-type]
        assert request.trainer.eval_steps == 2
        assert request.trainer.max_steps == -1
        assert request.lora.finetune_vision_layers is True
        self.request = request
        checkpoint = request.trainer.output_dir / "checkpoint-2"
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
        event = CheckpointEvent(checkpoint, global_step=2, epoch=1.0)
        checkpoint_observer.on_checkpoint(event)
        metric_sink.write(MetricEvent("loss", 0.5, step=2, epoch=1.0))
        final_adapter = request.trainer.output_dir / "final_adapter"
        final_adapter.mkdir()
        return BackendFitResult(
            global_step=2,
            training_loss=0.5,
            metrics={"train_loss": 0.5},
            checkpoints=(event,),
            final_adapter_dir=final_adapter,
            trainable_parameters=TrainableParameterManifest((), 0, {}),
            runtime=_runtime(),
        )

    def load_base(self, model: ModelLoadSpec) -> LoadedCheckpoint:
        del model
        return LoadedCheckpoint(object(), object(), _runtime(), None)

    def load_checkpoint(
        self, *, model: ModelLoadSpec, checkpoint_path: Path
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
        return tuple(BackendPrediction(item.sample_id, "melanoma") for item in samples)

    def release(self, loaded: LoadedCheckpoint) -> None:
        del loaded


class _IncompleteFullBackend(_Backend):
    """Return only the final epoch checkpoint to exercise fail-closed validation."""

    def fit(
        self,
        request: FineTuneRequest,
        *,
        metric_sink: MetricSink,
        checkpoint_observer: CheckpointObserver,
        resume_from_checkpoint: Path | None = None,
    ) -> BackendFitResult:
        assert resume_from_checkpoint is None
        assert len(request.train_dataset) == 25084  # type: ignore[arg-type]
        assert request.trainer.eval_steps == 3136
        checkpoint = request.trainer.output_dir / "checkpoint-12544"
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
        event = CheckpointEvent(checkpoint, global_step=12544, epoch=4.0)
        checkpoint_observer.on_checkpoint(event)
        metric_sink.write(MetricEvent("loss", 0.5, step=12544, epoch=4.0))
        final_adapter = request.trainer.output_dir / "final_adapter"
        final_adapter.mkdir()
        return BackendFitResult(
            global_step=12544,
            training_loss=0.5,
            metrics={"train_loss": 0.5},
            checkpoints=(event,),
            final_adapter_dir=final_adapter,
            trainable_parameters=TrainableParameterManifest((), 0, {}),
            runtime=_runtime(),
        )


def test_real_frozen_stage_and_student_configs_validate() -> None:
    audit = audit_sft_configuration(_CONFIG)

    assert audit.config.stage.initialization == "official_base"
    assert audit.config.datasets.train.expected_rows == 25084
    assert audit.config.datasets.dev.expected_rows == 1229
    assert audit.config.training.num_train_epochs == 4
    assert audit.config.checkpointing.save_total_limit == 4
    assert audit.student.student.precision.bf16 is True
    assert audit.student.student.precision.allow_precision_fallback is False
    assert len(audit.config_sha256) == 64
    assert len(audit.dataset_contract_sha256) == 64


def test_adapter_injects_image_target_and_cost_audit_metadata() -> None:
    config = load_sft_stage_config(_CONFIG)
    reference = DatasetReference(
        repo_id=config.datasets.train.repo_id,
        revision=config.datasets.train.revision,
        config=config.datasets.train.config,
        split=config.datasets.train.split,
        expected_rows=1,
        expected_task_counts={"diagnosis": 1},
    )
    dataset = prepare_dataset_rows(_OneRowDataset(_row(0)), reference)

    record = dataset[0]
    image_item = record["messages"][0]["content"][0]  # type: ignore[index]
    assert isinstance(image_item["image"], Image.Image)  # type: ignore[index]
    assert record["sample_id"] == "row-0-diagnosis"
    assert record["source_sample_id"] == "source-0"
    assert record["label"] == "melanoma"
    assert record["prompt"] == "Classify.\n\n/no_think"
    assert record["phase"] == "e3_multitask_sft"
    assert record["annotation_availability"] == ["diagnosis"]
    assert record["image_integrity_verified"] is True
    assert (record["image_width"], record["image_height"]) == (20, 10)
    assert (record["resized_width"], record["resized_height"]) == (20, 10)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("target_hash", "target_sha256 differs"),
        ("user_prompt", "User message differs from prompt"),
    ],
)
def test_adapter_rejects_frozen_text_integrity_drift(
    mutation: str,
    error: str,
) -> None:
    config = load_sft_stage_config(_CONFIG)
    reference = DatasetReference(
        repo_id=config.datasets.train.repo_id,
        revision=config.datasets.train.revision,
        config=config.datasets.train.config,
        split=config.datasets.train.split,
        expected_rows=1,
        expected_task_counts={"diagnosis": 1},
    )
    row = _row(0)
    if mutation == "target_hash":
        row["target_sha256"] = "0" * 64
    else:
        messages = row["messages"]
        assert isinstance(messages, list)
        messages[0]["content"][1]["text"] = "Different prompt"  # type: ignore[index]
    dataset = prepare_dataset_rows(_OneRowDataset(row), reference)

    with pytest.raises(ValueError, match=error):
        dataset[0]


def test_historical_dev_uses_gold_equality_when_no_target_hash_was_published() -> None:
    config = load_sft_stage_config(_CONFIG)
    row = _row(0)
    row.pop("target_sha256")
    row["split"] = "sft_dev"
    dataset = prepare_dataset_rows(
        _OneRowDataset(row),
        DatasetReference(
            repo_id=config.datasets.dev.repo_id,
            revision=config.datasets.dev.revision,
            config=config.datasets.dev.config,
            split=config.datasets.dev.split,
            expected_rows=1,
        ),
    )

    record = dataset[0]
    assert record["target_integrity_method"] == "gold_diagnosis_equality"
    assert record["target_sha256"] == hashlib.sha256(b"melanoma").hexdigest()

    row["gold_diagnosis"] = "nevus"
    corrupted = prepare_dataset_rows(
        _OneRowDataset(row),
        DatasetReference(
            repo_id=config.datasets.dev.repo_id,
            revision=config.datasets.dev.revision,
            config=config.datasets.dev.config,
            split=config.datasets.dev.split,
            expected_rows=1,
        ),
    )
    with pytest.raises(ValueError, match="differs from gold_diagnosis"):
        corrupted[0]


def test_smoke_runner_reuses_backend_and_leaves_selection_pending() -> None:
    config_document = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(config_document, Mapping)
    outputs = _ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=outputs) as temporary:
        temporary_path = Path(temporary)
        relative = temporary_path.relative_to(_ROOT)
        config_document["checkpointing"]["output_dir"] = str(relative / "runs")  # type: ignore[index]
        config_path = temporary_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(config_document, sort_keys=False),
            encoding="utf-8",
        )

        def loader(repo_id: str, **kwargs: object) -> _VirtualDataset:
            assert repo_id == "danielfdias98/ISEPDistillDataset"
            return _VirtualDataset(train=kwargs["name"] == "e3_multitask_sft_v1")

        backend = _Backend()
        result = run_sft(
            config_path,
            smoke=True,
            run_id="cpu-contract",
            dataset_loader=loader,
            backend=backend,
            metric_sink=_Sink(),
            resource_monitor=_Monitor(),
        )

        assert backend.request is not None
        assert result.checkpoint_selection_status == (
            "pending_sft_dev_generative_evaluation"
        )
        selection = json.loads(
            (
                result.run_directory / "manifests" / "checkpoint_selection.json"
            ).read_text(encoding="utf-8")
        )
        assert selection["best_checkpoint"] is None
        assert selection["external_benchmark_selection"] is False
        assert selection["selection_dataset"]["split"] == "sft_dev"
        status = json.loads(
            (result.run_directory / "manifests" / "run_status.json").read_text(
                encoding="utf-8"
            )
        )
        assert status["status"] == "completed"
        dataset_manifest = json.loads(
            (result.run_directory / "manifests" / "dataset_contract.json").read_text(
                encoding="utf-8"
            )
        )
        assert dataset_manifest["observed_audits"]["train"]["observed_task_counts"] == {
            "caption": 6312,
            "diagnosis": 6312,
            "grounded_differential": 6127,
            "morphology": 6312,
            "request_new_image": 21,
        }
        with pytest.raises(FileExistsError, match="already exists"):
            run_sft(
                config_path,
                smoke=True,
                run_id="cpu-contract",
                dataset_loader=loader,
                backend=_Backend(),
                metric_sink=_Sink(),
                resource_monitor=_Monitor(),
            )


def test_resume_identity_mismatch_does_not_rewrite_run_manifests() -> None:
    config_document = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(config_document, Mapping)
    outputs = _ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=outputs) as temporary:
        temporary_path = Path(temporary)
        relative = temporary_path.relative_to(_ROOT)
        config_document["checkpointing"]["output_dir"] = str(relative / "runs")  # type: ignore[index]
        config_path = temporary_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(config_document, sort_keys=False),
            encoding="utf-8",
        )

        def loader(repo_id: str, **kwargs: object) -> _VirtualDataset:
            del repo_id
            return _VirtualDataset(train=kwargs["name"] == "e3_multitask_sft_v1")

        result = run_sft(
            config_path,
            smoke=True,
            run_id="immutable-resume",
            dataset_loader=loader,
            backend=_Backend(),
            metric_sink=_Sink(),
            resource_monitor=_Monitor(),
        )
        status_path = result.run_directory / "manifests" / "run_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["status"] = "failed"
        status_path.write_text(json.dumps(status), encoding="utf-8")
        manifests = result.run_directory / "manifests"
        before = {
            path.name: path.read_bytes()
            for path in manifests.iterdir()
            if path.is_file()
        }

        config_document["logging"]["logging_steps"] = 11  # type: ignore[index]
        altered_path = temporary_path / "altered.yaml"
        altered_path.write_text(
            yaml.safe_dump(config_document, sort_keys=False),
            encoding="utf-8",
        )
        checkpoint = result.run_directory / "checkpoints" / "checkpoint-2"
        with pytest.raises(RuntimeError, match="immutable run identity"):
            run_sft(
                altered_path,
                smoke=True,
                run_id="immutable-resume",
                resume_from_checkpoint=checkpoint,
                dataset_loader=loader,
                backend=_Backend(),
                metric_sink=_Sink(),
                resource_monitor=_Monitor(),
            )
        after = {
            path.name: path.read_bytes()
            for path in manifests.iterdir()
            if path.is_file()
        }
        assert after == before

        later_checkpoint = result.run_directory / "checkpoints" / "checkpoint-3"
        later_checkpoint.mkdir()
        with pytest.raises(ValueError, match="latest checkpoint"):
            run_sft(
                config_path,
                smoke=True,
                run_id="immutable-resume",
                resume_from_checkpoint=checkpoint,
                dataset_loader=loader,
                backend=_Backend(),
                metric_sink=_Sink(),
                resource_monitor=_Monitor(),
            )
        assert {
            path.name: path.read_bytes()
            for path in manifests.iterdir()
            if path.is_file()
        } == before


def test_full_run_fails_before_completion_when_four_checkpoints_are_missing() -> None:
    config_document = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(config_document, Mapping)
    outputs = _ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=outputs) as temporary:
        temporary_path = Path(temporary)
        relative = temporary_path.relative_to(_ROOT)
        config_document["checkpointing"]["output_dir"] = str(relative / "runs")  # type: ignore[index]
        config_path = temporary_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(config_document, sort_keys=False),
            encoding="utf-8",
        )

        def loader(repo_id: str, **kwargs: object) -> _VirtualDataset:
            del repo_id
            return _VirtualDataset(train=kwargs["name"] == "e3_multitask_sft_v1")

        run_directory = temporary_path / "runs" / "missing-epochs"
        with pytest.raises(RuntimeError, match="expected 4 epoch checkpoints"):
            run_sft(
                config_path,
                run_id="missing-epochs",
                dataset_loader=loader,
                backend=_IncompleteFullBackend(),
                metric_sink=_Sink(),
                resource_monitor=_Monitor(),
            )
        status = json.loads(
            (run_directory / "manifests" / "run_status.json").read_text(
                encoding="utf-8"
            )
        )
        assert status["status"] == "failed"
        selection = json.loads(
            (run_directory / "manifests" / "checkpoint_selection.json").read_text(
                encoding="utf-8"
            )
        )
        assert selection["status"] == "pending_training"
        assert selection["eligible_checkpoints"] == []
