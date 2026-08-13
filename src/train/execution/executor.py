"""Fail-fast orchestration for one immutable training run."""

from __future__ import annotations

from pathlib import Path

from src.train.backends.contracts import (
    BackendFitResult,
    CheckpointObserver,
    FineTuneRequest,
    FineTuningBackend,
    MetricSink,
)
from src.train.execution.identity import (
    CheckpointRecorder,
    RunIdentity,
    RunIdentityStore,
    RunLayout,
    RunStatus,
    RunStatusStore,
    read_checkpoint_event,
    validate_resume_checkpoint,
)
from src.train.execution.io import (
    JsonValue,
    atomic_write_json,
    read_json_array,
)
from src.train.execution.resources import (
    LocalResourceMonitor,
    ResourceMonitor,
)
from src.train.execution.sinks import create_default_metric_sink


class ExecutionCleanupError(RuntimeError):
    """Report cleanup failure without concealing the primary training error."""

    def __init__(
        self,
        primary: BaseException | None,
        cleanup: BaseException,
    ) -> None:
        """Describe both primary and cleanup failures."""
        if primary is None:
            message = f"Execution cleanup failed: {cleanup}"
        else:
            message = (
                f"Training failed with {type(primary).__name__}: {primary}; "
                f"cleanup also failed with {type(cleanup).__name__}: {cleanup}"
            )
        super().__init__(message)
        self.primary = primary
        self.cleanup = cleanup


class TrainingExecutor:
    """Execute one backend fit with durable identity and lifecycle state."""

    def __init__(
        self,
        *,
        backend: FineTuningBackend,
        run_dir: Path,
        identity: RunIdentity,
        metric_sink: MetricSink | None = None,
        resource_monitor: ResourceMonitor | None = None,
        checkpoint_observer: CheckpointObserver | None = None,
    ) -> None:
        """Configure execution without loading a model or reserving CUDA."""
        self._backend = backend
        self._layout = RunLayout(run_dir)
        self._identity = identity
        self._metric_sink = metric_sink
        self._resource_monitor = resource_monitor
        self._checkpoint_observer = checkpoint_observer

    def execute(
        self,
        request: FineTuneRequest,
        *,
        resume_from_checkpoint: Path | None = None,
    ) -> BackendFitResult:
        """Run a fit and atomically record completion, failure, or interruption.

        Args:
            request: Immutable backend request with already prepared datasets.
            resume_from_checkpoint: Explicit checkpoint validated against the
                run's config, dataset, and model identity.

        Returns:
            Backend result after durable local manifests are written.

        Raises:
            RuntimeError: If identity, output directory, status, or resume
                checkpoint validation fails.
        """
        self._layout.create()
        expected_output = self._layout.checkpoints.resolve()
        actual_output = request.trainer.output_dir.resolve()
        if actual_output != expected_output:
            raise RuntimeError(
                "Trainer output_dir must equal the run checkpoint directory: "
                f"{expected_output}"
            )

        identity_store = RunIdentityStore(self._layout.manifests / "run_identity.json")
        status_store = RunStatusStore(self._layout.manifests / "run_status.json")
        identity_store.ensure(self._identity)
        status_store.initialize()
        self._prepare_status(
            status_store,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        if resume_from_checkpoint is not None:
            validate_resume_checkpoint(
                resume_from_checkpoint,
                self._identity,
            )
            if self._checkpoint_observer is not None:
                self._checkpoint_observer.on_checkpoint(
                    read_checkpoint_event(resume_from_checkpoint)
                )

        metric_sink = self._metric_sink or create_default_metric_sink(self._layout.root)
        monitor = self._resource_monitor or LocalResourceMonitor(
            output_dir=self._layout.logs,
            metric_sink=metric_sink,
            interval_seconds=5.0,
        )
        recorder = CheckpointRecorder(
            self._identity,
            downstream=self._checkpoint_observer,
        )
        status_store.transition(RunStatus.RUNNING)

        try:
            monitor.start()
            result = self._backend.fit(
                request,
                metric_sink=metric_sink,
                checkpoint_observer=recorder,
                resume_from_checkpoint=resume_from_checkpoint,
            )
        except KeyboardInterrupt as exc:
            failure = _cleanup(monitor, metric_sink, primary=exc)
            status_store.transition(RunStatus.INTERRUPTED, error=failure)
            if failure is not None and failure is not exc:
                raise failure from exc
            raise
        except Exception as exc:
            failure = _cleanup(monitor, metric_sink, primary=exc)
            status_store.transition(RunStatus.FAILED, error=failure)
            if failure is not None and failure is not exc:
                raise failure from exc
            raise

        cleanup_error = _cleanup(monitor, metric_sink, primary=None)
        if cleanup_error is not None:
            status_store.transition(RunStatus.FAILED, error=cleanup_error)
            raise cleanup_error
        _record_backend_result(self._layout.manifests, result)
        status_store.transition(RunStatus.COMPLETED)
        return result

    def _prepare_status(
        self,
        status_store: RunStatusStore,
        *,
        resume_from_checkpoint: Path | None,
    ) -> None:
        current = status_store.current()
        if resume_from_checkpoint is None:
            if current != RunStatus.CREATED:
                raise RuntimeError(
                    f"Run is already {current.value}; explicit resume is required"
                )
            return
        if current == RunStatus.RUNNING:
            status_store.transition(
                RunStatus.INTERRUPTED,
                error=RuntimeError("Recovered stale running state for resume"),
            )
            current = RunStatus.INTERRUPTED
        if current not in {RunStatus.FAILED, RunStatus.INTERRUPTED}:
            raise RuntimeError(f"Cannot resume a run in state {current.value}")


def _cleanup(
    monitor: ResourceMonitor,
    sink: MetricSink,
    *,
    primary: BaseException | None,
) -> BaseException | None:
    first_cleanup_error: BaseException | None = None
    try:
        monitor.stop()
    except BaseException as exc:  # cleanup must preserve KeyboardInterrupt too
        first_cleanup_error = exc
    try:
        sink.close()
    except BaseException as exc:
        if first_cleanup_error is None:
            first_cleanup_error = exc
    if first_cleanup_error is None:
        return primary
    return ExecutionCleanupError(primary, first_cleanup_error)


def _result_json(result: BackendFitResult) -> dict[str, JsonValue]:
    return {
        "global_step": result.global_step,
        "training_loss": result.training_loss,
        "metrics": dict(result.metrics),
        "checkpoints": [
            {
                "path": str(event.path),
                "global_step": event.global_step,
                "epoch": event.epoch,
            }
            for event in result.checkpoints
        ],
        "final_adapter_dir": str(result.final_adapter_dir),
        "runtime": {
            "torch_version": result.runtime.torch_version,
            "cuda_version": result.runtime.cuda_version,
            "device_name": result.runtime.device_name,
            "device_count": result.runtime.device_count,
            "bf16_supported": result.runtime.bf16_supported,
            "total_memory_bytes": result.runtime.total_memory_bytes,
        },
        "trainable_parameters": {
            "total": result.trainable_parameters.total_trainable,
            "by_component": {
                str(component): count
                for component, count in result.trainable_parameters.by_component.items()
            },
            "parameters": [
                {
                    "name": parameter.name,
                    "component": parameter.component,
                    "count": parameter.count,
                }
                for parameter in result.trainable_parameters.parameters
            ],
        },
    }


def _record_backend_result(
    manifests_directory: Path,
    result: BackendFitResult,
) -> None:
    """Persist the latest fit and append it to the resumed-session history.

    ``backend_result.json`` remains the backwards-compatible view of the most
    recent successful backend invocation. ``backend_sessions.json`` is the
    canonical cumulative accounting input for runs that required one or more
    resumes. The complete list is atomically replaced so readers never observe
    a partially appended JSON document.

    Args:
        manifests_directory: Canonical run manifest directory.
        result: Successful result returned by one backend fit invocation.

    Raises:
        ValueError: If an existing session history is not a JSON array of
            backend result objects.
    """

    payload = _result_json(result)
    sessions_path = manifests_directory / "backend_sessions.json"
    sessions: list[JsonValue] = []
    if sessions_path.is_file():
        sessions = read_json_array(sessions_path)
        if not all(isinstance(item, dict) for item in sessions):
            raise ValueError(
                f"Backend session history contains a non-object: {sessions_path}"
            )
    sessions.append(payload)
    atomic_write_json(sessions_path, sessions)
    atomic_write_json(manifests_directory / "backend_result.json", payload)
