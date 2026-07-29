"""Local multimodal model backend scaffold."""

from __future__ import annotations

from typing import Any

from src.config.models import LocalModelConfig


class LocalBackend:
    """Lazy-loading local backend pending model-specific chat templates."""

    def __init__(self, config: LocalModelConfig) -> None:
        self.config = config
        self._model: Any = None
        self._processor: Any = None

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        schema: dict[str, Any],
    ) -> str:
        raise NotImplementedError(
            "Local generation requires a model-family-specific processor and "
            "chat template. Validate one model in the real-model pilot before "
            "enabling this backend."
        )
