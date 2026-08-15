"""End-to-end E1 orchestration over typed, independently tested modules."""

from __future__ import annotations

from pathlib import Path

from src.train.artifacts import ArtifactStore
from src.train.backends import (
    FineTuneRequest,
    FineTuningBackend,
)
from src.train.checkpoint_hub import create_checkpoint_mirror
from src.train.config import TrainingConfig
from src.train.continuation import stage_continuation_checkpoint
from src.train.data import load_taxonomy
from src.train.domain import PreparedRelease
from src.train.environment import collect_environment
from src.train.evaluate import model_spec
from src.train.execution import RunIdentity, TrainingExecutor
from src.train.execution.resources import LocalResourceMonitor
from src.train.execution.sinks import create_default_metric_sink
from src.train.finalize import ensure_preupdate_panel, evaluate_run
from src.train.phases.label_only import LabelOnlyPhase
from src.train.preparation import (
    lora_spec,
    open_run_store,
    trainer_spec,
    training_datasets,
)
from src.train.run_domain import TrainingRunResult
from src.train.run_io import open_frozen_release, write_run_manifests
from src.train.scientific import config_hash


def run_training(
    config: TrainingConfig,
    *,
    backend: FineTuningBackend | None = None,
    resume_from: Path | None = None,
    smoke: bool = False,
    run_id: str | None = None,
) -> TrainingRunResult:
    """Train, evaluate every model state, and build thesis artefacts.

    The function never creates a data split. The frozen release and all hashes
    are validated before the backend can import Unsloth or reserve CUDA.
    """

    selected_backend = backend or _unsloth_backend()
    release = open_frozen_release(config)
    phase = LabelOnlyPhase(load_taxonomy(config))
    store = open_run_store(config, resume_from, smoke, run_id)
    run_directory = store.layout.run_directory
    identity = RunIdentity(
        experiment_id=config.experiment.id,
        run_id=store.layout.run_id,
        config_hash=config_hash(config),
        dataset_hash=release.audit.assignment_sha256,
        model_id=config.model.repo_id,
        model_revision=config.model.revision,
        execution_profile="smoke" if smoke else "full",
    )
    store.write_status("created")
    try:
        _ensure_run_manifests(
            store,
            config,
            release,
            phase.prompt,
            resume_from,
            smoke=smoke,
        )
        if config.continuation is not None and smoke:
            raise ValueError(
                "Continuation runs do not use the fresh-training smoke profile"
            )
        if config.continuation is not None and resume_from is None:
            training_checkpoint = stage_continuation_checkpoint(
                config,
                run_directory=run_directory,
                identity=identity,
            )
        else:
            training_checkpoint = resume_from
        checkpoint_mirror = create_checkpoint_mirror(
            config=config.artifacts.checkpoint_hub,
            identity=identity,
            seed=config.trainer.seed,
            manifests_directory=store.layout.manifests,
            smoke=smoke,
        )
        if checkpoint_mirror is not None:
            checkpoint_mirror.validate_destination()
        store.write_status("running")
        ensure_preupdate_panel(
            backend=selected_backend,
            config=config,
            release=release,
            store=store,
            smoke=smoke,
            checkpoint_path=(
                run_directory / "checkpoints" / config.continuation.parent_checkpoint_id
                if config.continuation is not None
                else None
            ),
        )
        train_dataset, eval_dataset = training_datasets(
            config, release, run_directory, smoke=smoke
        )
        request = FineTuneRequest(
            model=model_spec(config),
            lora=lora_spec(config),
            trainer=trainer_spec(
                config,
                run_directory,
                train_size=len(train_dataset),
                smoke=smoke,
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        sink = create_default_metric_sink(
            run_directory,
            require_tensorboard=config.artifacts.tensorboard,
        )
        monitor = LocalResourceMonitor(
            output_dir=run_directory / "logs",
            metric_sink=sink,
            interval_seconds=(config.artifacts.resource_sample_interval_seconds),
        )
        initializing_from_parent = (
            config.continuation is not None
            and training_checkpoint is not None
            and training_checkpoint.name == config.continuation.parent_checkpoint_id
            and not (run_directory / "manifests" / "run_status.json").exists()
        )
        TrainingExecutor(
            backend=selected_backend,
            run_dir=run_directory,
            identity=identity,
            metric_sink=sink,
            resource_monitor=monitor,
            checkpoint_observer=checkpoint_mirror,
        ).execute(
            request,
            resume_from_checkpoint=training_checkpoint,
            initialize_from_checkpoint=initializing_from_parent,
        )
        result = evaluate_run(
            run_directory,
            backend=selected_backend,
            config=config,
            release=release,
            smoke=smoke,
        )
        store.write_status("completed")
        return result
    except KeyboardInterrupt:
        store.write_status("interrupted", detail="KeyboardInterrupt")
        raise
    except Exception as exc:
        store.write_status("failed", detail=f"{type(exc).__name__}: {exc}")
        raise


def _ensure_run_manifests(
    store: ArtifactStore,
    config: TrainingConfig,
    release: PreparedRelease,
    prompt: str,
    resume_from: Path | None,
    *,
    smoke: bool,
) -> None:
    if resume_from is not None:
        run_status = store.layout.manifests / "run_status.json"
        if config.continuation is not None and not run_status.exists():
            store.write_json(
                "manifests",
                "environment.json",
                collect_environment(config.project_root),
            )
        return
    write_run_manifests(
        store=store,
        config=config,
        release=release,
        prompt=prompt,
        execution_profile="smoke" if smoke else "full",
    )


def _unsloth_backend() -> FineTuningBackend:
    from src.train.backends import UnslothBackend

    return UnslothBackend()
