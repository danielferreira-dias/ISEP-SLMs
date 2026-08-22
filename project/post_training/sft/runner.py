"""Fail-fast orchestration for the frozen E3 multitask LoRA experiment."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast, overload

from project.metrics.resources import ResourceMonitor
from project.post_training.common.config import (
    E3SFTStageConfig,
    StudentRecipe,
    load_sft_stage_config,
    load_student_recipe,
)
from project.post_training.common.data import (
    DatasetLoader,
    PreparedDatasets,
    load_training_datasets,
)
from src.train.backends import (
    BackendFitResult,
    CheckpointObserver,
    FineTuneRequest,
    FineTuningBackend,
    LoraSpec,
    MetricSink,
    ModelLoadSpec,
    TrainerSpec,
)
from src.train.execution import (
    RunIdentity,
    TrainingExecutor,
    stable_json_hash,
    validate_resume_checkpoint,
)
from src.train.execution.io import (
    JsonValue,
    atomic_write_json,
    atomic_write_text,
    read_json_object,
)

_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_ADAPTER_CONTRACT = "e3_multimodal_messages_with_decoded_image_v1"


class _SizedDataset(Protocol):
    """Minimal dataset surface needed by the backend and smoke view."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> object: ...


@dataclass(frozen=True, slots=True)
class SFTConfigurationAudit:
    """Resolved, hashable configuration before any dataset or CUDA work."""

    config: E3SFTStageConfig
    student: StudentRecipe
    config_path: Path
    student_config_path: Path
    project_root: Path
    config_sha256: str
    dataset_contract_sha256: str


@dataclass(frozen=True, slots=True)
class SFTExecutionResult:
    """Durable result of one full or smoke E3 fit invocation."""

    run_directory: Path
    backend_result: BackendFitResult
    config_sha256: str
    dataset_contract_sha256: str
    checkpoint_selection_status: str


class _PrefixDataset(Sequence[dict[str, object]]):
    """Non-copying prefix used only by the explicit smoke profile."""

    def __init__(self, backing: _SizedDataset, limit: int) -> None:
        if limit <= 0:
            raise ValueError("Smoke dataset limit must be positive")
        self._backing = backing
        self._length = min(len(backing), limit)

    def __len__(self) -> int:
        return self._length

    def mask_audit_records(self) -> tuple[dict[str, object], ...]:
        """Return one row per task present in the explicit smoke prefix."""

        records: dict[str, dict[str, object]] = {}
        for index in range(self._length):
            record = self[index]
            task = record.get("task")
            if not isinstance(task, str) or not task:
                raise ValueError("Smoke row has no task for mask audit")
            records.setdefault(task, record)
        return tuple(records[task] for task in sorted(records))

    @overload
    def __getitem__(self, index: int) -> dict[str, object]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, object]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> dict[str, object] | list[dict[str, object]]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(self._length))]
        normalized = index if index >= 0 else self._length + index
        if normalized < 0 or normalized >= self._length:
            raise IndexError("Smoke dataset index out of range")
        row = self._backing[normalized]
        if not isinstance(row, Mapping):
            raise TypeError("Prepared training dataset returned a non-mapping row")
        return {str(key): value for key, value in row.items()}


def audit_sft_configuration(config_path: Path) -> SFTConfigurationAudit:
    """Load both frozen YAML files and compute their scientific identities."""

    resolved_config = config_path.resolve()
    config = load_sft_stage_config(resolved_config)
    project_root = _find_project_root(resolved_config)
    student_path = _resolve_project_path(project_root, config.student.config_path)
    student = load_student_recipe(student_path)
    config_document = {
        "stage": _json_value(config.model_dump(mode="json")),
        "student": _json_value(student.model_dump(mode="json")),
    }
    dataset_document = {
        "adapter_contract": _ADAPTER_CONTRACT,
        "datasets": _json_value(config.datasets.model_dump(mode="json")),
    }
    return SFTConfigurationAudit(
        config=config,
        student=student,
        config_path=resolved_config,
        student_config_path=student_path,
        project_root=project_root,
        config_sha256=stable_json_hash(cast(JsonValue, config_document)),
        dataset_contract_sha256=stable_json_hash(cast(JsonValue, dataset_document)),
    )


def run_sft(
    config_path: Path,
    *,
    smoke: bool = False,
    run_id: str | None = None,
    resume_from_checkpoint: Path | None = None,
    dataset_loader: DatasetLoader | None = None,
    backend: FineTuningBackend | None = None,
    metric_sink: MetricSink | None = None,
    resource_monitor: ResourceMonitor | None = None,
    checkpoint_observer: CheckpointObserver | None = None,
) -> SFTExecutionResult:
    """Train E3 from the pinned official base without selecting on benchmarks.

    Model loading happens only inside :class:`TrainingExecutor`, after the
    resolved config, data lineage, and checkpoint-selection policy have been
    persisted.  The returned run deliberately has no selected best checkpoint:
    that decision requires a separate deterministic generative evaluation on
    the frozen ``sft_dev`` split.
    """

    audit = audit_sft_configuration(config_path)
    profile = "smoke" if smoke else "full"
    if resume_from_checkpoint is not None and run_id is None:
        raise ValueError("Resuming requires the original explicit run_id")
    selected_run_id = run_id or _default_run_id(audit.config, profile)
    _validate_run_id(selected_run_id)
    run_dir = _run_directory(audit, selected_run_id)
    identity = _run_identity(
        audit,
        run_id=selected_run_id,
        execution_profile=profile,
    )
    _validate_run_directory(
        run_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        expected_identity=identity,
    )
    if resume_from_checkpoint is None:
        _write_preflight_manifests(run_dir, audit, profile=profile)

    prepared = load_training_datasets(
        audit.config,
        dataset_loader=dataset_loader,
    )
    train_dataset: _SizedDataset = prepared.train
    eval_dataset: _SizedDataset = prepared.dev
    max_steps = -1
    epochs = float(audit.config.training.num_train_epochs)
    if smoke:
        # Complete one small epoch and one complete dev pass.  A partial
        # max_steps run would violate the production collator's exact
        # train+dev sample-cost coverage contract.
        train_dataset = _PrefixDataset(train_dataset, 16)
        eval_dataset = _PrefixDataset(eval_dataset, 8)
        epochs = 1.0
    if resume_from_checkpoint is None:
        _write_dataset_manifest(
            run_dir,
            audit,
            prepared,
            train_rows=len(train_dataset),
            dev_rows=len(eval_dataset),
            profile=profile,
        )

    request = _fine_tune_request(
        audit,
        run_dir=run_dir,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        epochs=epochs,
        max_steps=max_steps,
    )
    selected_backend = backend or _unsloth_backend()
    if selected_backend.name != audit.config.stage.backend:
        raise ValueError(
            "Backend differs from the frozen stage config: "
            f"{selected_backend.name!r} != {audit.config.stage.backend!r}"
        )
    result = TrainingExecutor(
        backend=selected_backend,
        run_dir=run_dir,
        identity=identity,
        metric_sink=metric_sink,
        resource_monitor=resource_monitor,
        checkpoint_observer=checkpoint_observer,
    ).execute(
        request,
        resume_from_checkpoint=(
            resume_from_checkpoint.resolve()
            if resume_from_checkpoint is not None
            else None
        ),
        result_validator=lambda backend_result: _validate_checkpoint_contract(
            backend_result,
            run_dir=run_dir,
            identity=identity,
            steps_per_epoch=request.trainer.eval_steps,
            epochs=int(epochs),
        ),
    )
    selection_status = "pending_sft_dev_generative_evaluation"
    _write_checkpoint_selection(
        run_dir,
        audit,
        status=selection_status,
        checkpoints=tuple(str(item.path) for item in result.checkpoints),
    )
    return SFTExecutionResult(
        run_directory=run_dir,
        backend_result=result,
        config_sha256=audit.config_sha256,
        dataset_contract_sha256=audit.dataset_contract_sha256,
        checkpoint_selection_status=selection_status,
    )


def _fine_tune_request(
    audit: SFTConfigurationAudit,
    *,
    run_dir: Path,
    train_dataset: object,
    eval_dataset: object,
    epochs: float,
    max_steps: int,
) -> FineTuneRequest:
    config = audit.config
    student = audit.student.student
    effective_batch = (
        config.training.per_device_train_batch_size
        * config.training.gradient_accumulation_steps
    )
    train_rows = len(cast(_SizedDataset, train_dataset))
    steps_per_epoch = max(1, math.ceil(train_rows / effective_batch))
    return FineTuneRequest(
        model=ModelLoadSpec(
            model_id=student.model.id,
            revision=student.model.revision,
            processor_id=student.model.id,
            processor_revision=student.model.revision,
            dtype=student.precision.dtype,
            load_in_4bit=student.precision.load_in_4bit,
        ),
        lora=LoraSpec(
            finetune_vision_layers=student.vision.finetune_vision_layers,
            finetune_language_layers=student.vision.finetune_language_layers,
            finetune_attention_modules=student.vision.finetune_attention_modules,
            finetune_mlp_modules=student.vision.finetune_mlp_modules,
            rank=student.lora.r,
            alpha=student.lora.lora_alpha,
            dropout=student.lora.lora_dropout,
            bias=student.lora.bias,
            target_modules=student.lora.target_modules,
            use_rslora=student.lora.use_rslora,
            loftq_config=student.lora.loftq_config,
        ),
        trainer=TrainerSpec(
            output_dir=run_dir / "checkpoints",
            per_device_train_batch_size=config.training.per_device_train_batch_size,
            per_device_eval_batch_size=config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            num_train_epochs=epochs,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            warmup_ratio=config.training.warmup_ratio,
            max_grad_norm=config.training.max_grad_norm,
            logging_steps=config.logging.logging_steps,
            eval_steps=steps_per_epoch,
            seed=config.training.seed,
            max_length=config.training.max_length,
            max_steps=max_steps,
            dataset_num_proc=config.training.dataset_num_proc,
            save_total_limit=config.checkpointing.save_total_limit,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


def _validate_checkpoint_contract(
    result: BackendFitResult,
    *,
    run_dir: Path,
    identity: RunIdentity,
    steps_per_epoch: int,
    epochs: int,
) -> None:
    """Require one valid, resumable checkpoint at every completed epoch.

    This validator runs inside :class:`TrainingExecutor` before the lifecycle
    can transition to ``completed``.  A missing, duplicated, out-of-order, or
    corrupted checkpoint therefore leaves the run explicitly ``failed`` and
    never exposes it to checkpoint selection.
    """

    if steps_per_epoch <= 0 or epochs <= 0:
        raise ValueError("Checkpoint contract requires positive epoch geometry")
    expected = tuple(
        (steps_per_epoch * epoch, float(epoch)) for epoch in range(1, epochs + 1)
    )
    events = result.checkpoints
    if len(events) != len(expected):
        raise RuntimeError(
            "Checkpoint contract violation: expected "
            f"{len(expected)} epoch checkpoints, found {len(events)}"
        )
    if result.global_step != expected[-1][0]:
        raise RuntimeError(
            "Checkpoint contract violation: final global_step "
            f"{result.global_step} != {expected[-1][0]}"
        )

    checkpoint_root = (run_dir / "checkpoints").resolve()
    observed_paths: set[Path] = set()
    observed_steps: set[int] = set()
    for event, (expected_step, expected_epoch) in zip(events, expected, strict=True):
        resolved_path = event.path.resolve()
        expected_path = (checkpoint_root / f"checkpoint-{expected_step}").resolve()
        if resolved_path in observed_paths or event.global_step in observed_steps:
            raise RuntimeError("Checkpoint contract violation: duplicate checkpoint")
        observed_paths.add(resolved_path)
        observed_steps.add(event.global_step)
        if event.global_step != expected_step or resolved_path != expected_path:
            raise RuntimeError(
                "Checkpoint contract violation: expected "
                f"checkpoint-{expected_step}, found {event.path.name} "
                f"at step {event.global_step}"
            )
        if not resolved_path.is_dir() or not resolved_path.is_relative_to(
            checkpoint_root
        ):
            raise RuntimeError(
                f"Checkpoint contract violation: invalid directory {resolved_path}"
            )
        if event.epoch is None or not math.isclose(
            event.epoch,
            expected_epoch,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise RuntimeError(
                "Checkpoint contract violation: expected epoch "
                f"{expected_epoch}, found {event.epoch}"
            )
        checkpoint_manifest = read_json_object(resolved_path / "isep_checkpoint.json")
        manifest_step = checkpoint_manifest.get("global_step")
        manifest_epoch = checkpoint_manifest.get("epoch")
        if manifest_step != expected_step or (
            isinstance(manifest_epoch, bool)
            or not isinstance(manifest_epoch, int | float)
            or not math.isclose(
                float(manifest_epoch),
                expected_epoch,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
        ):
            raise RuntimeError(
                "Checkpoint contract violation: manifest coordinates do not "
                f"match epoch {expected_epoch} / step {expected_step}"
            )
        validate_resume_checkpoint(resolved_path, identity)


def _write_preflight_manifests(
    run_dir: Path,
    audit: SFTConfigurationAudit,
    *,
    profile: str,
) -> None:
    manifests = run_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        manifests / "config.original.yaml",
        audit.config_path.read_text(encoding="utf-8"),
    )
    atomic_write_text(
        manifests / "student.original.yaml",
        audit.student_config_path.read_text(encoding="utf-8"),
    )
    atomic_write_json(
        manifests / "config.resolved.json",
        _json_value(audit.config.model_dump(mode="json")),
    )
    atomic_write_json(
        manifests / "student.resolved.json",
        _json_value(audit.student.model_dump(mode="json")),
    )
    atomic_write_json(
        manifests / "execution_context.json",
        {
            "config_path": str(audit.config_path),
            "student_config_path": str(audit.student_config_path),
            "project_root": str(audit.project_root),
            "config_sha256": audit.config_sha256,
            "dataset_contract_sha256": audit.dataset_contract_sha256,
            "execution_profile": profile,
            "initialization": "official_base",
        },
    )
    atomic_write_json(
        manifests / "model_initialization.json",
        {
            "model_id": audit.student.student.model.id,
            "model_revision": audit.student.student.model.revision,
            "adapter_parent": None,
            "starts_from_official_base": True,
        },
    )
    _write_checkpoint_selection(
        run_dir,
        audit,
        status="pending_training",
        checkpoints=(),
    )


def _write_dataset_manifest(
    run_dir: Path,
    audit: SFTConfigurationAudit,
    prepared: PreparedDatasets,
    *,
    train_rows: int,
    dev_rows: int,
    profile: str,
) -> None:
    atomic_write_json(
        run_dir / "manifests" / "dataset_contract.json",
        {
            "adapter_contract": _ADAPTER_CONTRACT,
            "contract_sha256": audit.dataset_contract_sha256,
            "profile": profile,
            "train": _json_value(audit.config.datasets.train.model_dump(mode="json")),
            "dev": _json_value(audit.config.datasets.dev.model_dump(mode="json")),
            "loaded_train_rows": train_rows,
            "loaded_dev_rows": dev_rows,
            "full_train_rows": len(prepared.train),
            "full_dev_rows": len(prepared.dev),
            "observed_audits": _json_value(prepared.as_manifest()),
            "selection_scope": "checkpoint_selection_only",
            "teacher_generated_dev_targets": False,
        },
    )


def _write_checkpoint_selection(
    run_dir: Path,
    audit: SFTConfigurationAudit,
    *,
    status: str,
    checkpoints: tuple[str, ...],
) -> None:
    evaluation = audit.config.evaluation
    atomic_write_json(
        run_dir / "manifests" / "checkpoint_selection.json",
        {
            "status": status,
            "best_checkpoint": None,
            "eligible_checkpoints": list(checkpoints),
            "selection_dataset": _json_value(
                audit.config.datasets.dev.model_dump(mode="json")
            ),
            "selection_metric": evaluation.selection_metric,
            "greater_is_better": evaluation.greater_is_better,
            "tie_break_metrics": list(evaluation.tie_break_metrics),
            "external_benchmark_selection": False,
            "external_benchmark_policy": evaluation.external_benchmark_selection,
        },
    )


def _run_directory(audit: SFTConfigurationAudit, run_id: str) -> Path:
    root = _resolve_project_path(
        audit.project_root,
        audit.config.checkpointing.output_dir,
    )
    return (root / run_id).resolve()


def _validate_run_directory(
    run_dir: Path,
    *,
    resume_from_checkpoint: Path | None,
    expected_identity: RunIdentity,
) -> None:
    """Prevent accidental reuse or cross-run checkpoint continuation."""

    if resume_from_checkpoint is None:
        if run_dir.exists():
            raise FileExistsError(
                f"Run directory already exists; choose a new run_id: {run_dir}"
            )
        return
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot resume because the original run directory is missing: {run_dir}"
        )
    checkpoint = resume_from_checkpoint.resolve()
    checkpoint_root = (run_dir / "checkpoints").resolve()
    if not checkpoint.is_relative_to(checkpoint_root):
        raise ValueError("Resume checkpoint must belong to the original run_id")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Resume checkpoint is missing: {checkpoint}")
    available = tuple(
        (int(match.group(1)), candidate.resolve())
        for candidate in checkpoint_root.glob("checkpoint-*")
        if candidate.is_dir()
        and (match := re.fullmatch(r"checkpoint-(\d+)", candidate.name)) is not None
    )
    if not available:
        raise FileNotFoundError(
            f"No Trainer checkpoints exist in the original run: {checkpoint_root}"
        )
    latest_step, latest_checkpoint = max(available, key=lambda item: item[0])
    if checkpoint != latest_checkpoint:
        raise ValueError(
            "Resume must use the latest checkpoint to prevent an accidental "
            f"rewind: checkpoint-{latest_step}"
        )
    identity_path = run_dir / "manifests" / "run_identity.json"
    if not identity_path.is_file():
        raise FileNotFoundError(
            "Cannot resume because the immutable run identity is missing: "
            f"{identity_path}"
        )
    observed_identity = read_json_object(identity_path)
    if observed_identity != _identity_document(expected_identity):
        raise RuntimeError(
            "Resume config, dataset, model, run_id, or profile differs from "
            "the immutable run identity"
        )
    status_path = run_dir / "manifests" / "run_status.json"
    if status_path.is_file():
        status = read_json_object(status_path).get("status")
        if status == "completed":
            raise RuntimeError("A completed E3 run cannot be resumed")


def _run_identity(
    audit: SFTConfigurationAudit,
    *,
    run_id: str,
    execution_profile: str,
) -> RunIdentity:
    return RunIdentity(
        experiment_id=audit.config.stage.id,
        run_id=run_id,
        config_hash=audit.config_sha256,
        dataset_hash=audit.dataset_contract_sha256,
        model_id=audit.student.student.model.id,
        model_revision=audit.student.student.model.revision,
        execution_profile=execution_profile,
    )


def _identity_document(identity: RunIdentity) -> dict[str, JsonValue]:
    return {
        "experiment_id": identity.experiment_id,
        "run_id": identity.run_id,
        "config_hash": identity.config_hash,
        "dataset_hash": identity.dataset_hash,
        "model_id": identity.model_id,
        "model_revision": identity.model_revision,
        "execution_profile": identity.execution_profile,
    }


def _default_run_id(config: E3SFTStageConfig, profile: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{profile}-seed{config.training.seed}-{timestamp}"


def _validate_run_id(value: str) -> None:
    if _SAFE_RUN_ID.fullmatch(value) is None:
        raise ValueError("run_id must contain only letters, digits, '.', '_' and '-'")


def _find_project_root(config_path: Path) -> Path:
    for parent in (config_path.parent, *config_path.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent.resolve()
    raise ValueError(f"Cannot find the ISEP project root from {config_path}")


def _resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    if not resolved.is_relative_to(project_root):
        raise ValueError(f"Configured path escapes the project root: {value}")
    return resolved


def _json_value(value: object) -> JsonValue:
    """Normalize Pydantic output through JSON and reject exotic objects."""

    return cast(
        JsonValue,
        json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False)),
    )


def _unsloth_backend() -> FineTuningBackend:
    from src.train.backends import UnslothBackend

    return UnslothBackend()


__all__ = [
    "SFTConfigurationAudit",
    "SFTExecutionResult",
    "audit_sft_configuration",
    "run_sft",
]
