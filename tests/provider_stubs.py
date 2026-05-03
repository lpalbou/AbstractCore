from __future__ import annotations

from typing import List

from abstractcore.providers.base import BaseProvider


class StaticProvider(BaseProvider):
    """Minimal provider stub for tests.

    This avoids pulling optional provider SDK dependencies (e.g. OpenAI) into the
    default CI environment while still exercising BaseProvider initialization
    logic (model capabilities, token limits, etc).
    """

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        self.provider = "static"

    def get_capabilities(self) -> List[str]:
        return ["chat"]

    def list_available_models(self) -> List[str]:
        return []

    def unload_model(self, model_name: str) -> None:
        _ = model_name
        return None

