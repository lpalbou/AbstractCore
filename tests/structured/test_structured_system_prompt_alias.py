from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from abstractcore.structured.handler import StructuredOutputHandler


class _User(BaseModel):
    name: str


class _PromptedProvider:
    model = "prompted-test-model"
    model_capabilities = {"structured_output": "prompted"}

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def _generate_internal(self, *, prompt: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        return SimpleNamespace(content='{"name": "Ada"}')


class _NativeProvider:
    model = "native-test-model"
    model_capabilities = {"structured_output": "native"}

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def _generate_internal(self, *, prompt: str, response_model: Any = None, **kwargs: Any) -> SimpleNamespace:
        self.calls.append({"prompt": prompt, "response_model": response_model, "kwargs": dict(kwargs)})
        return SimpleNamespace(content='{"name": "Ada"}')


def test_generate_structured_accepts_system_alias_for_prompted_provider() -> None:
    provider = _PromptedProvider()
    handler = StructuredOutputHandler()

    with pytest.warns(UserWarning, match="alias for `system_prompt=`"):
        result = handler.generate_structured(
            provider=provider,
            prompt="Extract user info",
            response_model=_User,
            system="You extract concise user records.",
        )

    assert result.name == "Ada"
    assert provider.calls
    assert provider.calls[0]["kwargs"]["system_prompt"] == "You extract concise user records."
    assert "system" not in provider.calls[0]["kwargs"]


def test_generate_structured_prefers_system_prompt_when_both_system_forms_are_provided() -> None:
    provider = _NativeProvider()
    handler = StructuredOutputHandler()

    with pytest.warns(UserWarning, match="Both `system_prompt=` and `system=`"):
        result = handler.generate_structured(
            provider=provider,
            prompt="Extract user info",
            response_model=_User,
            system_prompt="Use this system prompt.",
            system="Ignore this alias.",
        )

    assert result.name == "Ada"
    assert provider.calls
    assert provider.calls[0]["response_model"] is _User
    assert provider.calls[0]["kwargs"]["system_prompt"] == "Use this system prompt."
    assert "system" not in provider.calls[0]["kwargs"]
