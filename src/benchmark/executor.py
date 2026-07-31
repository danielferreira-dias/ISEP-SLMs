"""Generic, resumable execution for every dermatology benchmark task."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
import time
from typing import Any, Iterable, Protocol

from src.benchmark.results import (
    RunWriter,
    count_statuses,
    read_jsonl,
)
from src.benchmark.runner import (
    BenchmarkPrediction,
    BenchmarkSample,
    ModelResponse,
)
from src.benchmark.selection import task_seed
from src.inference.base import (
    InferenceBackend,
    InferenceRequest,
    InferenceResult,
    InferenceSafetyRefusal,
    ReasoningTrace,
    TokenUsage,
)


class PreparedTaskProtocol(Protocol):
    system_prompt: str
    user_prompt: str
    schema: dict[str, Any]


class TaskAdapterProtocol(Protocol):
    def prepare(self, sample: BenchmarkSample) -> PreparedTaskProtocol:
        ...

    def parse_response(
        self,
        model_id: str,
        raw_text: str,
        prepared_task: PreparedTaskProtocol,
        reasoning_text: str | None = None,
    ) -> ModelResponse:
        ...

    def compute_metrics(
        self,
        predictions: Iterable[BenchmarkPrediction],
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Resolved execution settings shared by all requests in one run."""

    batch_size: int
    max_output_tokens: int
    run_seed: int = 42
    save_rendered_prompts: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")


@dataclass(frozen=True, slots=True)
class ExecutionSummary:
    """Terminal state returned by the benchmark executor."""

    metrics: dict[str, Any]
    counts: dict[str, int]
    predictions: tuple[BenchmarkPrediction, ...]


@dataclass(frozen=True, slots=True)
class _PreparedRequest:
    sample: BenchmarkSample
    task: PreparedTaskProtocol
    request: InferenceRequest
    rendered_prompt_record: dict[str, Any]


class BenchmarkExecutor:
    """Execute selected tasks with per-case isolation and durable artifacts."""

    def __init__(
        self,
        *,
        backend: InferenceBackend,
        adapter: TaskAdapterProtocol,
        image_loader: Any,
        writer: RunWriter,
        execution: ExecutionConfig,
        generation: Any | None = None,
    ) -> None:
        self.backend = backend
        self.adapter = adapter
        self.image_loader = image_loader
        self.writer = writer
        self.execution = execution
        self.generation = generation

    def run(
        self,
        samples: Iterable[BenchmarkSample],
    ) -> ExecutionSummary:
        """Synchronous CLI wrapper around :meth:`arun`."""

        return asyncio.run(self.arun(samples))

    async def arun(
        self,
        samples: Iterable[BenchmarkSample],
    ) -> ExecutionSummary:
        """Run pending samples asynchronously and compute final metrics."""

        selected = list(samples)
        completed_ids = self.writer.completed_task_ids()
        pending = [
            sample
            for sample in selected
            if _task_id(sample) not in completed_ids
        ]
        try:
            await self._run_pending(pending)
        except KeyboardInterrupt:
            records = read_jsonl(self.writer.paths.predictions)
            self.writer.finalize(
                status="interrupted",
                counts=count_statuses(records),
            )
            raise
        except Exception as exc:
            records = read_jsonl(self.writer.paths.predictions)
            self.writer.finalize(
                status="failed",
                counts=count_statuses(records),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        records = read_jsonl(self.writer.paths.predictions)
        predictions = tuple(_record_to_prediction(record) for record in records)
        metrics = self.adapter.compute_metrics(predictions)
        counts = count_statuses(records)
        self.writer.write_metrics(metrics)
        self.writer.finalize(status="completed", counts=counts)
        return ExecutionSummary(
            metrics=metrics,
            counts=counts,
            predictions=predictions,
        )

    async def _run_pending(
        self,
        pending: list[BenchmarkSample],
    ) -> None:
        for batch in _batches(pending, self.execution.batch_size):
            await self._run_batch(batch)

    async def _run_batch(
        self,
        samples: list[BenchmarkSample],
    ) -> None:
        prepared: list[_PreparedRequest] = []
        for sample in samples:
            try:
                image_bytes = self.image_loader(sample.image_uri)
            except Exception as exc:
                self._write_failure(
                    sample=sample,
                    status="image_error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            task = self.adapter.prepare(sample)
            effective_generation = _generation_override(
                self.generation,
                max_output_tokens=self.execution.max_output_tokens,
                seed=task_seed(
                    self.execution.run_seed,
                    _task_id(sample),
                ),
            )
            request = InferenceRequest(
                system_prompt=task.system_prompt,
                user_prompt=task.user_prompt,
                image_bytes=image_bytes,
                schema=task.schema,
                generation=effective_generation,
                request_id=_task_id(sample),
            )
            prompt_record = {
                "task_id": _task_id(sample),
                "sample_id": sample.sample_id,
                "system_prompt": task.system_prompt,
                "user_prompt": task.user_prompt,
                "schema": task.schema,
                "image_uri": sample.image_uri,
                "image_embedded": False,
            }
            prepared.append(
                _PreparedRequest(
                    sample=sample,
                    task=task,
                    request=request,
                    rendered_prompt_record=prompt_record,
                )
            )

        if not prepared:
            return
        for item in prepared:
            if self.execution.save_rendered_prompts:
                self.writer.append_rendered_prompt(
                    item.rendered_prompt_record
                )

        semaphore = asyncio.Semaphore(
            min(self.execution.batch_size, len(prepared))
        )
        tasks = [
            asyncio.create_task(self._complete_one(item, semaphore))
            for item in prepared
        ]
        for completed in asyncio.as_completed(tasks):
            item, result, error = await completed
            if isinstance(error, InferenceSafetyRefusal):
                self._write_failure(
                    sample=item.sample,
                    status="safety_refusal",
                    error=f"{type(error).__name__}: {error}",
                    response_metadata={
                        "safety_refusal": dict(error.details),
                    },
                )
            elif error is not None:
                self._write_failure(
                    sample=item.sample,
                    status="backend_error",
                    error=f"{type(error).__name__}: {error}",
                )
            elif result is not None:
                self._write_result(item.sample, item.task, result)

    async def _complete_one(
        self,
        item: _PreparedRequest,
        semaphore: asyncio.Semaphore,
    ) -> tuple[
        _PreparedRequest,
        InferenceResult | None,
        Exception | None,
    ]:
        async with semaphore:
            try:
                result = await self.backend.acomplete(item.request)
            except Exception as exc:
                return item, None, exc
            return item, result, None

    def _write_result(
        self,
        sample: BenchmarkSample,
        task: PreparedTaskProtocol,
        result: InferenceResult,
    ) -> None:
        reasoning_text = result.reasoning.text
        response = self.adapter.parse_response(
            self.backend.model_id,
            result.final_text,
            prepared_task=task,
            reasoning_text=reasoning_text,
        )
        truncated = _is_truncated(result)
        if truncated:
            response.schema_valid = False
            if "semantic_valid" in response.metadata:
                response.metadata["semantic_valid"] = False
            if "truncated_output" not in response.validation_errors:
                response.validation_errors.append("truncated_output")
        status = _classify_response_status(
            response,
            truncated=truncated,
        )
        prediction = BenchmarkPrediction(
            task_id=_task_id(sample),
            sample_id=sample.sample_id,
            model_id=self.backend.model_id,
            ground_truth_disease_id=sample.disease_id,
            response=response,
            metadata=dict(sample.metadata),
        )
        self.writer.append_prediction(
            _prediction_to_record(
                prediction,
                sample=sample,
                status=status,
                result=result,
            )
        )

    def _write_failure(
        self,
        *,
        sample: BenchmarkSample,
        status: str,
        error: str,
        response_metadata: dict[str, Any] | None = None,
    ) -> None:
        response = ModelResponse(
            model_id=self.backend.model_id,
            raw_text="",
            parsed_output=None,
            json_valid=False,
            schema_valid=False,
            recoverable_json_valid=False,
            validation_errors=[f"{status}:{error}"],
            metadata=dict(response_metadata or {}),
        )
        prediction = BenchmarkPrediction(
            task_id=_task_id(sample),
            sample_id=sample.sample_id,
            model_id=self.backend.model_id,
            ground_truth_disease_id=sample.disease_id,
            response=response,
            metadata=dict(sample.metadata),
        )
        self.writer.append_prediction(
            _prediction_to_record(
                prediction,
                sample=sample,
                status=status,
                result=None,
            )
        )


def _prediction_to_record(
    prediction: BenchmarkPrediction,
    *,
    sample: BenchmarkSample,
    status: str,
    result: InferenceResult | None,
) -> dict[str, Any]:
    response = prediction.response
    record: dict[str, Any] = {
        "task_id": prediction.task_id,
        "sample_id": prediction.sample_id,
        "model_id": prediction.model_id,
        "status": status,
        "image_uri": sample.image_uri,
        "ground_truth_disease_id": prediction.ground_truth_disease_id,
        "metadata": prediction.metadata,
        "response": {
            "final_text": response.raw_text,
            "parsed_output": response.parsed_output,
            "canonical_output": response.canonical_output,
            "json_valid": response.json_valid,
            "recoverable_json_valid": response.recoverable_json_valid,
            "schema_valid": response.schema_valid,
            "canonical_schema_valid": response.canonical_schema_valid,
            "canonicalization_rules": response.canonicalization_rules,
            "validation_errors": response.validation_errors,
            "metadata": response.metadata,
        },
    }
    if result is None:
        record["response"].update(
            {
                "reasoning": _reasoning_dict(
                    ReasoningTrace(capture_mode="none")
                ),
                "usage": _usage_dict(TokenUsage()),
                "finish_reason": None,
                "provider_response_id": None,
                "provider_metadata": {},
            }
        )
    else:
        record["response"].update(
            {
                "reasoning": _reasoning_dict(result.reasoning),
                "usage": _usage_dict(result.usage),
                "finish_reason": result.finish_reason,
                "provider_response_id": result.provider_response_id,
                "provider_metadata": dict(result.metadata),
            }
        )
    return record


def _record_to_prediction(record: dict[str, Any]) -> BenchmarkPrediction:
    response_value = record.get("response", {})
    if not isinstance(response_value, dict):
        response_value = {}
    response = ModelResponse(
        model_id=str(record.get("model_id", "")),
        raw_text=str(response_value.get("final_text", "")),
        parsed_output=(
            response_value.get("parsed_output")
            if isinstance(response_value.get("parsed_output"), dict)
            else None
        ),
        canonical_output=(
            response_value.get("canonical_output")
            if isinstance(response_value.get("canonical_output"), dict)
            else None
        ),
        json_valid=bool(response_value.get("json_valid")),
        recoverable_json_valid=bool(
            response_value.get(
                "recoverable_json_valid",
                response_value.get("json_valid"),
            )
        ),
        schema_valid=bool(response_value.get("schema_valid")),
        canonical_schema_valid=bool(
            response_value.get(
                "canonical_schema_valid",
                response_value.get("schema_valid"),
            )
        ),
        canonicalization_rules=[
            str(value)
            for value in response_value.get(
                "canonicalization_rules",
                [],
            )
        ],
        validation_errors=[
            str(value)
            for value in response_value.get("validation_errors", [])
        ],
        metadata=(
            dict(response_value.get("metadata", {}))
            if isinstance(response_value.get("metadata"), dict)
            else {}
        ),
    )
    metadata = record.get("metadata", {})
    return BenchmarkPrediction(
        task_id=str(record.get("task_id", "")),
        sample_id=str(record.get("sample_id", "")),
        model_id=str(record.get("model_id", "")),
        ground_truth_disease_id=str(
            record.get("ground_truth_disease_id", "")
        ),
        response=response,
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


def _generation_override(
    generation: Any | None,
    *,
    max_output_tokens: int,
    seed: int,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if isinstance(generation, dict):
        values.update(generation)
    elif is_dataclass(generation) and not isinstance(generation, type):
        values.update(asdict(generation))
    elif generation is not None:
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "do_sample",
        ):
            value = getattr(generation, key, None)
            if value is not None:
                values[key] = value
    values["max_output_tokens"] = max_output_tokens
    values["seed"] = seed
    return values


def _is_truncated(result: InferenceResult) -> bool:
    reason = (result.finish_reason or "").casefold()
    if reason in {"length", "max_tokens", "max_output_tokens"}:
        return True
    incomplete_reason = result.metadata.get("incomplete_reason")
    return (
        isinstance(incomplete_reason, str)
        and incomplete_reason.casefold() == "max_output_tokens"
    )


def _classify_response_status(
    response: ModelResponse,
    *,
    truncated: bool,
) -> str:
    """Return the most fundamental failed output-contract layer."""

    if truncated:
        return "truncated_output"
    if not response.json_valid:
        return "format_invalid"
    if not response.schema_valid:
        return "schema_invalid"
    if response.metadata.get("semantic_valid", True) is False:
        return "semantic_noncompliant"
    return "ok"


def _reasoning_dict(value: ReasoningTrace) -> dict[str, Any]:
    availability = "none"
    if value.text:
        availability = (
            "summary"
            if value.capture_mode == "summary"
            else "full"
        )
    elif value.token_count is not None:
        availability = "tokens_only"
    return {
        "capture_mode": value.capture_mode,
        "availability": availability,
        "text": value.text,
        "token_count": value.token_count,
        "source": value.source_field,
    }


def _usage_dict(value: TokenUsage) -> dict[str, int | None]:
    return {
        "input_tokens": value.input_tokens,
        "output_tokens": value.output_tokens,
        "total_tokens": value.total_tokens,
        "reasoning_tokens": value.reasoning_tokens,
    }


def _task_id(sample: BenchmarkSample) -> str:
    return str(sample.task_id or sample.sample_id)


def _batches(
    values: list[BenchmarkSample],
    size: int,
) -> Iterable[list[BenchmarkSample]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]
