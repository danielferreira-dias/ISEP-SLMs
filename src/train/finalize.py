"""Checkpoint evaluation, selection, comparison snapshot, and reporting."""

from __future__ import annotations

import json
from pathlib import Path

from src.train.artifacts import (
    ArtifactStore,
    export_thesis_artifacts,
    write_comparable_run_snapshot,
)
from src.train.backends import FineTuningBackend
from src.train.config import TrainingConfig
from src.train.data import load_taxonomy
from src.train.domain import PreparedRelease, ReleaseSubset
from src.train.evaluate import (
    EvaluationResult,
    checkpoint_training_state,
    evaluate_model_state,
)
from src.train.evaluation import (
    CheckpointScore,
    ComparableRun,
    RunContract,
    select_best_checkpoint,
)
from src.train.execution import RunIdentity, validate_resume_checkpoint
from src.train.phases.label_only import LabelOnlyPhase
from src.train.reporting import build_run_report
from src.train.run_domain import TrainingRunResult
from src.train.run_io import (
    checkpoint_directories,
    load_execution_profile,
    load_run_config,
    open_frozen_release,
    persist_evaluation,
    resource_summary,
)
from src.train.scientific import (
    config_hash,
    controlled_training_hash,
    label_contract_hash,
    prompt_hash,
)
from src.train.smoke import validate_smoke_run


def evaluate_run(
    run_directory: Path,
    *,
    backend: FineTuningBackend | None = None,
    config: TrainingConfig | None = None,
    release: PreparedRelease | None = None,
    smoke: bool | None = None,
) -> TrainingRunResult:
    """Evaluate base and all checkpoints, select the best, and report."""

    selected_config = config or load_run_config(run_directory)
    selected_release = release or open_frozen_release(selected_config)
    selected_backend = backend or _unsloth_backend()
    store = ArtifactStore.at(run_directory)
    execution_profile = load_execution_profile(run_directory)
    selected_smoke = execution_profile == "smoke" if smoke is None else smoke
    if selected_smoke != (execution_profile == "smoke"):
        raise ValueError("Requested evaluation profile differs from the run profile")
    checkpoints = checkpoint_directories(run_directory)
    if not checkpoints:
        raise RuntimeError("Run has no checkpoints to evaluate")
    identity = RunIdentity(
        experiment_id=selected_config.experiment.id,
        run_id=store.layout.run_id,
        config_hash=config_hash(selected_config),
        dataset_hash=selected_release.audit.assignment_sha256,
        model_id=selected_config.model.repo_id,
        model_revision=selected_config.model.revision,
        execution_profile=execution_profile,
    )
    for checkpoint in checkpoints:
        validate_resume_checkpoint(checkpoint, identity)
    if not selected_smoke and len(checkpoints) != selected_config.trainer.epochs:
        raise RuntimeError(
            f"Expected {selected_config.trainer.epochs} epoch checkpoints, "
            f"found {len(checkpoints)}"
        )
    if not selected_smoke:
        epochs = tuple(checkpoint_training_state(path)[0] for path in checkpoints)
        expected_epochs = tuple(
            float(epoch) for epoch in range(1, selected_config.trainer.epochs + 1)
        )
        if epochs != expected_epochs:
            raise RuntimeError(
                f"Expected epoch checkpoints {expected_epochs}, found {epochs}"
            )
    limit = 32 if selected_smoke else None
    for checkpoint in checkpoints:
        panel = evaluate_model_state(
            backend=selected_backend,
            config=selected_config,
            release=selected_release,
            subset=ReleaseSubset.DEV_PANEL,
            checkpoint_id=checkpoint.name,
            checkpoint_path=checkpoint,
            max_samples=limit,
        )
        persist_evaluation(store, panel)
    dev_results = _evaluate_full_development(
        backend=selected_backend,
        config=selected_config,
        release=selected_release,
        checkpoints=checkpoints,
        max_samples=limit,
        store=store,
    )
    best, best_result = _select_best(dev_results)
    best_path = next(path for path in checkpoints if path.name == best.checkpoint_id)
    store.write_json(
        "manifests",
        "best_checkpoint.json",
        {
            "checkpoint_id": best.checkpoint_id,
            "path": str(best_path),
            "epoch": best.epoch,
            "eval_loss": best.eval_loss,
            "selection_metric": "macro_f1",
            "macro_f1": best.metrics.macro_f1,
            "balanced_accuracy": best.metrics.balanced_accuracy,
        },
    )
    _write_comparable_snapshot(
        store=store,
        config=selected_config,
        release=selected_release,
        result=best_result,
    )
    build_run_report(run_directory)
    export_directory = selected_config.artifacts.thesis_export_directory
    if export_directory is not None:
        export_thesis_artifacts(
            run_directory,
            selected_config.resolve_path(export_directory),
        )
    if selected_smoke:
        validate_smoke_run(
            run_directory=run_directory,
            backend=selected_backend,
            config=selected_config,
            release=selected_release,
            checkpoint=best_path,
        )
    store.write_status("completed")
    return TrainingRunResult(run_directory, best_path, best.metrics)


def ensure_preupdate_panel(
    *,
    backend: FineTuningBackend,
    config: TrainingConfig,
    release: PreparedRelease,
    store: ArtifactStore,
    smoke: bool,
) -> None:
    """Evaluate the base on the 210-case panel before the first update."""

    path = store.path("metrics", "dev_panel__base.json")
    if path.is_file():
        return
    baseline = evaluate_model_state(
        backend=backend,
        config=config,
        release=release,
        subset=ReleaseSubset.DEV_PANEL,
        checkpoint_id="base",
        checkpoint_path=None,
        max_samples=32 if smoke else None,
    )
    persist_evaluation(store, baseline)


def _evaluate_full_development(
    *,
    backend: FineTuningBackend,
    config: TrainingConfig,
    release: PreparedRelease,
    checkpoints: tuple[Path, ...],
    max_samples: int | None,
    store: ArtifactStore,
) -> tuple[EvaluationResult, ...]:
    paths: tuple[Path | None, ...] = (None, *checkpoints)
    results: list[EvaluationResult] = []
    for path in paths:
        checkpoint_id = "base" if path is None else path.name
        result = evaluate_model_state(
            backend=backend,
            config=config,
            release=release,
            subset=ReleaseSubset.SFT_DEV,
            checkpoint_id=checkpoint_id,
            checkpoint_path=path,
            max_samples=max_samples,
        )
        persist_evaluation(store, result)
        results.append(result)
    return tuple(results)


def _select_best(
    results: tuple[EvaluationResult, ...],
) -> tuple[CheckpointScore, EvaluationResult]:
    checkpoints = tuple(result for result in results if result.checkpoint_id != "base")
    scores: list[CheckpointScore] = []
    for result in checkpoints:
        if result.epoch is None or result.eval_loss is None:
            raise RuntimeError(
                f"Checkpoint {result.checkpoint_id} lacks epoch/eval_loss"
            )
        scores.append(
            CheckpointScore(
                checkpoint_id=result.checkpoint_id,
                epoch=result.epoch,
                eval_loss=result.eval_loss,
                metrics=result.metrics,
            )
        )
    best = select_best_checkpoint(tuple(scores))
    result = next(
        item for item in checkpoints if item.checkpoint_id == best.checkpoint_id
    )
    return best, result


def _write_comparable_snapshot(
    *,
    store: ArtifactStore,
    config: TrainingConfig,
    release: PreparedRelease,
    result: EvaluationResult,
) -> None:
    taxonomy = load_taxonomy(config)
    phase = LabelOnlyPhase(taxonomy)
    resources = resource_summary(store.layout.run_directory)
    run = ComparableRun(
        experiment_id=config.experiment.id,
        run_id=store.layout.run_id,
        seed=config.trainer.seed,
        contract=RunContract(
            dataset_revision=config.dataset.hub_revision,
            split_hash=release.audit.assignment_sha256,
            prompt_hash=prompt_hash(phase.prompt),
            model_revision=config.model.revision,
            label_contract_hash=label_contract_hash(taxonomy),
            training_contract_hash=controlled_training_hash(config),
        ),
        predictions=result.predictions,
        metrics=result.metrics,
        duration_seconds=resources.duration_seconds,
        gpu_hours=resources.gpu_hours,
        peak_vram_gib=resources.peak_vram_gib,
        trainable_parameters=_trainable_count(store.layout.run_directory),
    )
    write_comparable_run_snapshot(store, run)


def _trainable_count(run_directory: Path) -> int | None:
    path = run_directory / "manifests" / "backend_result.json"
    if not path.is_file():
        return None
    document: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        return None
    trainable = document.get("trainable_parameters")
    if not isinstance(trainable, dict):
        return None
    value = trainable.get("total")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _unsloth_backend() -> FineTuningBackend:
    from src.train.backends import UnslothBackend

    return UnslothBackend()
