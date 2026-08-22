"""Strict CUDA and BF16 validation without importing Unsloth."""

from __future__ import annotations

import importlib
from types import ModuleType

from src.train.backends.contracts import RuntimeInfo


class RuntimeValidationError(RuntimeError):
    """Raised when the host cannot execute the declared training recipe."""


def validate_nvidia_bf16_runtime() -> RuntimeInfo:
    """Validate CUDA, NVIDIA device presence, and native BF16 support.

    Returns:
        Auditable runtime details for device zero.

    Raises:
        RuntimeValidationError: If PyTorch, CUDA, NVIDIA, or BF16 is missing.
    """
    try:
        torch = importlib.import_module("torch")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeValidationError(
            "PyTorch is unavailable; install the training extra on Linux"
        ) from exc

    cuda = getattr(torch, "cuda", None)
    if cuda is None or not _bool_method(cuda, "is_available"):
        raise RuntimeValidationError(
            "CUDA is unavailable; CPU fallback is intentionally disabled"
        )
    count = _int_method(cuda, "device_count")
    if count != 1:
        raise RuntimeValidationError(
            "Controlled ISEP training requires exactly one visible NVIDIA GPU; "
            f"CUDA reported {count} devices"
        )
    device_name = _string_method(cuda, "get_device_name", 0)
    if "NVIDIA" not in device_name.upper():
        raise RuntimeValidationError(
            f"Device zero is not reported as NVIDIA: {device_name}"
        )
    bf16_supported = _bool_method(cuda, "is_bf16_supported")
    if not bf16_supported:
        raise RuntimeValidationError(
            f"Device does not support the required BF16 recipe: {device_name}"
        )
    cuda_version = _cuda_version(torch)
    if not cuda_version:
        raise RuntimeValidationError("PyTorch has no CUDA runtime version")

    properties = _call(cuda, "get_device_properties", 0)
    memory = getattr(properties, "total_memory", None)
    total_memory = (
        int(memory)
        if isinstance(memory, (int, float)) and not isinstance(memory, bool)
        else None
    )
    torch_version = getattr(torch, "__version__", "unknown")
    return RuntimeInfo(
        torch_version=str(torch_version),
        cuda_version=cuda_version,
        device_name=device_name,
        device_count=count,
        bf16_supported=True,
        total_memory_bytes=total_memory,
    )


def _cuda_version(torch: ModuleType) -> str:
    version = getattr(torch, "version", None)
    cuda_version = getattr(version, "cuda", None)
    return str(cuda_version) if cuda_version is not None else ""


def _call(instance: object, name: str, *args: object) -> object:
    method = getattr(instance, name, None)
    if not callable(method):
        raise RuntimeValidationError(f"PyTorch runtime is missing {name}()")
    return method(*args)


def _bool_method(instance: object, name: str) -> bool:
    return bool(_call(instance, name))


def _int_method(instance: object, name: str) -> int:
    value = _call(instance, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeValidationError(f"PyTorch {name}() returned a non-number")
    return int(value)


def _string_method(instance: object, name: str, *args: object) -> str:
    return str(_call(instance, name, *args))
