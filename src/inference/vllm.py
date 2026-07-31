"""vLLM OpenAI-server transport, readiness checks, and process lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from src.inference.base import (
    InferenceConfigurationError,
    InferencePreflightError,
    InferenceTransportError,
    PreflightResult,
    read_field,
)
from src.inference.openai_compatible import OpenAICompatibleChatBackend


@dataclass(frozen=True, slots=True)
class VllmServerConfig:
    """Safe command-line configuration for one managed vLLM server."""

    model: str
    host: str = "127.0.0.1"
    port: int = 8000
    executable: str = "vllm"
    dtype: str | None = None
    tensor_parallel_size: int | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    reasoning_parser: str | None = None
    served_model_name: str | None = None
    additional_args: tuple[str, ...] = ()
    startup_timeout_seconds: float = 900.0
    shutdown_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not self.model:
            raise InferenceConfigurationError(
                "A Hugging Face model repository is required for vLLM"
            )
        if not 1 <= self.port <= 65535:
            raise InferenceConfigurationError(
                "vLLM port must be between 1 and 65535"
            )
        if self.tensor_parallel_size is not None and (
            self.tensor_parallel_size < 1
        ):
            raise InferenceConfigurationError(
                "tensor_parallel_size must be positive"
            )
        _reject_secret_cli_args(self.additional_args)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def command(self) -> tuple[str, ...]:
        command: list[str] = [
            self.executable,
            "serve",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.tensor_parallel_size is not None:
            command.extend(
                [
                    "--tensor-parallel-size",
                    str(self.tensor_parallel_size),
                ]
            )
        if self.dtype:
            command.extend(["--dtype", self.dtype])
        if self.max_model_len is not None:
            command.extend(
                ["--max-model-len", str(self.max_model_len)]
            )
        if self.gpu_memory_utilization is not None:
            command.extend(
                [
                    "--gpu-memory-utilization",
                    str(self.gpu_memory_utilization),
                ]
            )
        if self.reasoning_parser:
            command.extend(
                ["--reasoning-parser", self.reasoning_parser]
            )
        if self.served_model_name:
            command.extend(
                ["--served-model-name", self.served_model_name]
            )
        command.extend(self.additional_args)
        return tuple(command)


class VllmBackend(OpenAICompatibleChatBackend):
    """OpenAI-compatible client with vLLM-specific endpoint preflight."""

    def __init__(
        self,
        *,
        model_id: str,
        request_model: str | None = None,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str | None = None,
        api_key_env: str | None = "VLLM_API_KEY",
        client: Any | None = None,
        health_probe: Callable[[str, float], bool] | None = None,
        health_timeout_seconds: float = 2.0,
        **kwargs: Any,
    ) -> None:
        # A timed-out generation is still expensive server-side. Do not let
        # the OpenAI SDK silently duplicate local vLLM requests.
        kwargs.setdefault("max_retries", 0)
        # Long thinking generations can leave reverse proxies idle for
        # minutes. Stream vLLM deltas so the connection remains active.
        kwargs.setdefault("stream_responses", True)
        resolved_api_key = api_key
        resolved_api_key_env = api_key_env
        if (
            resolved_api_key is None
            and (
                resolved_api_key_env is None
                or not os.environ.get(resolved_api_key_env)
            )
        ):
            # vLLM does not require authentication unless launched with an
            # API key. The OpenAI SDK still requires a non-empty placeholder.
            resolved_api_key = "EMPTY"
            resolved_api_key_env = None
        super().__init__(
            model_id=model_id,
            request_model=request_model,
            base_url=base_url,
            api_key=resolved_api_key,
            api_key_env=resolved_api_key_env,
            client=client,
            **kwargs,
        )
        self._health_url = _health_url(base_url)
        self._health_probe = health_probe or _default_health_probe
        self.health_timeout_seconds = health_timeout_seconds

    def preflight(self) -> PreflightResult:
        """Check `/health` and verify that the configured model is served."""

        checks: list[str] = []
        errors: list[str] = []
        try:
            healthy = bool(
                self._health_probe(
                    self._health_url,
                    self.health_timeout_seconds,
                )
            )
        except Exception:
            healthy = False
        if healthy:
            checks.append("health_endpoint")
        else:
            errors.append("health_endpoint_unavailable")

        model_available = False
        if healthy:
            try:
                response = self.client.models.list()
                served_ids = _served_model_ids(response)
                model_available = self.request_model in served_ids
            except Exception:
                errors.append("model_registry_unavailable")
            else:
                if model_available:
                    checks.append("model_available")
                else:
                    errors.append("configured_model_not_served")
        return PreflightResult(
            healthy=healthy,
            model_available=model_available,
            checks=tuple(checks),
            errors=tuple(errors),
        )

    def require_ready(self) -> None:
        """Raise a sanitized failure if endpoint preflight does not pass."""

        result = self.preflight()
        if not result.ok:
            detail = ", ".join(result.errors) or "unknown_preflight_error"
            raise InferencePreflightError(
                f"vLLM preflight failed for model {self.model_id!r}: "
                f"{detail}"
            )


class ManagedVllmServer:
    """Manage one local vLLM server without invoking a shell.

    Secrets must be supplied through the process environment. Secret-bearing
    CLI flags are rejected because command lines can be visible to other
    processes and may be included in diagnostics.
    """

    def __init__(
        self,
        config: VllmServerConfig,
        *,
        environment: Mapping[str, str] | None = None,
        process_factory: Callable[..., Any] | None = None,
        ready_check: Callable[[], bool] | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.config = config
        self._environment = dict(environment or {})
        self._process_factory = process_factory or subprocess.Popen
        self._ready_check = ready_check or self._probe_default_endpoint
        self.log_path = log_path
        self._log_handle: Any | None = None
        self._process: Any | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def pid(self) -> int | None:
        if self._process is None:
            return None
        pid = getattr(self._process, "pid", None)
        return pid if isinstance(pid, int) else None

    def start(self, *, wait_until_ready: bool = True) -> None:
        if self.is_running:
            raise InferenceConfigurationError(
                "The managed vLLM server is already running"
            )
        environment = os.environ.copy()
        environment.update(self._environment)
        output: Any = subprocess.DEVNULL
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("ab", buffering=0)
            output = self._log_handle
        try:
            self._process = self._process_factory(
                list(self.config.command()),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=output,
                start_new_session=True,
            )
        except Exception:
            self._process = None
            self._close_log()
            raise InferenceTransportError(
                f"Failed to start the vLLM server for "
                f"{self.config.model!r}"
            ) from None
        if wait_until_ready:
            try:
                self.wait_until_ready()
            except Exception:
                self.stop()
                raise

    def wait_until_ready(
        self,
        timeout_seconds: float | None = None,
    ) -> None:
        if self._process is None:
            raise InferenceConfigurationError(
                "The managed vLLM server has not been started"
            )
        timeout = (
            self.config.startup_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            exit_code = self._process.poll()
            if exit_code is not None:
                raise InferenceTransportError(
                    f"vLLM exited before becoming ready for model "
                    f"{self.config.model!r} (exit code {exit_code})"
                )
            try:
                if self._ready_check():
                    return
            except Exception:
                pass
            time.sleep(0.1)
        raise InferencePreflightError(
            f"vLLM did not become ready for model "
            f"{self.config.model!r} within {timeout:g} seconds"
        )

    def stop(self) -> None:
        process = self._process
        self._process = None
        try:
            if process is None or process.poll() is not None:
                return
            process.terminate()
            try:
                process.wait(
                    timeout=self.config.shutdown_timeout_seconds
                )
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(
                    timeout=self.config.shutdown_timeout_seconds
                )
        finally:
            self._close_log()

    def __enter__(self) -> ManagedVllmServer:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        self.stop()

    def _probe_default_endpoint(self) -> bool:
        return _default_health_probe(
            _health_url(self.config.base_url),
            2.0,
        )

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


def server_config_from_model(
    config: Any,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    executable: str = "vllm",
    max_model_len: int | None = None,
    additional_args: Sequence[str] = (),
) -> VllmServerConfig:
    """Build a managed-server command from normalized model configuration."""

    backend = _field(config, "backend")
    profile = _field(backend, "active_profile")
    if _field(profile, "engine") != "vllm":
        raise InferenceConfigurationError(
            "Managed vLLM requires a local profile with engine='vllm'"
        )
    if _field(profile, "managed", True) is False:
        raise InferenceConfigurationError(
            "The selected vLLM profile does not allow managed startup"
        )
    if _field(profile, "managed_allowed", True) is False:
        raise InferenceConfigurationError(
            "The selected vLLM profile forbids managed startup"
        )
    source = _field(config, "source")
    model = _field(source, "repo_id")
    if not isinstance(model, str) or not model:
        raise InferenceConfigurationError(
            "Managed vLLM requires a Hugging Face source.repo_id"
        )
    reasoning = _field(config, "reasoning")
    resolved_max_model_len = (
        max_model_len
        if max_model_len is not None
        else _field(profile, "max_model_len")
    )
    image_limit = _field(profile, "limit_images_per_prompt")
    if image_limit is None:
        image_limit = _field(profile, "limit_mm_per_prompt_image")
    resolved_additional_args = list(additional_args)
    if isinstance(image_limit, int) and image_limit > 0:
        resolved_additional_args.extend(
            [
                "--limit-mm-per-prompt",
                json.dumps({"image": image_limit}),
            ]
        )
    processor = _field(config, "processor")
    image_processor = _field(processor, "image")
    if image_processor is not None:
        processor_kwargs = {
            "downsample_mode": _field(
                image_processor,
                "downsample_mode",
            ),
            "max_slice_nums": _field(
                image_processor,
                "max_slice_nums",
            ),
        }
        processor_kwargs = {
            key: value
            for key, value in processor_kwargs.items()
            if value is not None
        }
        if processor_kwargs:
            resolved_additional_args.extend(
                [
                    "--mm-processor-kwargs",
                    json.dumps(processor_kwargs),
                ]
            )
    return VllmServerConfig(
        model=model,
        host=host,
        port=port,
        executable=executable,
        dtype=_field(profile, "dtype"),
        tensor_parallel_size=_field(
            profile,
            "tensor_parallel_size",
        ),
        max_model_len=resolved_max_model_len,
        gpu_memory_utilization=_field(
            profile,
            "gpu_memory_utilization",
        ),
        reasoning_parser=_field(reasoning, "parser"),
        additional_args=tuple(resolved_additional_args),
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a dataclass-like object or mapping."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _health_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            f"{path}/health" if path else "/health",
            "",
            "",
        )
    )


def _default_health_probe(url: str, timeout: float) -> bool:
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            return 200 <= status < 300
    except (HTTPError, URLError, TimeoutError):
        return False


def _served_model_ids(response: Any) -> set[str]:
    data = read_field(response, "data", ())
    if not isinstance(data, Sequence) or isinstance(
        data,
        (str, bytes, bytearray),
    ):
        return set()
    return {
        model_id
        for item in data
        if isinstance(
            (model_id := read_field(item, "id")),
            str,
        )
    }


def _reject_secret_cli_args(arguments: Sequence[str]) -> None:
    secret_flags = {
        "--api-key",
        "--hf-token",
        "--token",
        "--access-token",
    }
    for argument in arguments:
        normalized = argument.split("=", maxsplit=1)[0].lower()
        if normalized in secret_flags:
            raise InferenceConfigurationError(
                "Secret-bearing vLLM CLI arguments are forbidden; "
                "provide credentials through the process environment"
            )
