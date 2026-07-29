"""Azure-hosted multimodal model backend scaffold."""

from __future__ import annotations

import os
from typing import Any

from src.config.models import AzureModelConfig


class AzureBackend:
    """Azure adapter with explicit credential checks and deferred API binding."""

    def __init__(
        self,
        config: AzureModelConfig,
        client: Any | None = None,
    ) -> None:
        self.config = config
        endpoint = os.environ.get(config.endpoint_env)
        api_key = os.environ.get(config.api_key_env)
        if not endpoint:
            raise ValueError(
                f"Environment variable {config.endpoint_env!r} is missing"
            )
        if not api_key:
            raise ValueError(
                f"Environment variable {config.api_key_env!r} is missing"
            )
        self._client = client

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
            "Azure generation requires the selected deployment API contract. "
            "Bind it after the teacher and API provider are finalized."
        )
