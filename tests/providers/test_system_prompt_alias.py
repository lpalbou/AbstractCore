from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider


class _CaptureProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(model="stub", enable_tracing=False)
        self.last_messages: Any = None
        self.last_system_prompt: Any = None
        self.last_kwargs: Dict[str, Any] = {}

    def get_capabilities(self) -> List[str]:
        return ["chat"]

    def list_available_models(self, **kwargs: Any) -> List[str]:
        _ = kwargs
        return [self.model]

    def unload_model(self, model_name: str) -> None:
        _ = model_name

    def _generate_internal(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> GenerateResponse:
        _ = (prompt, tools, media, stream)
        self.last_messages = messages
        self.last_system_prompt = system_prompt
        self.last_kwargs = dict(kwargs)
        return GenerateResponse(content="ok", model=self.model, finish_reason="stop")


class _NativeAsyncCaptureProvider(_CaptureProvider):
    async def _agenerate_internal(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        system_prompt: Optional[str],
        tools: Optional[List[Any]],
        media: Optional[List[Any]],
        stream: bool,
        **kwargs: Any,
    ) -> GenerateResponse:
        _ = (prompt, tools, media, stream)
        self.last_messages = messages
        self.last_system_prompt = system_prompt
        self.last_kwargs = dict(kwargs)
        return GenerateResponse(content="ok", model=self.model, finish_reason="stop")


def test_generate_accepts_system_alias_with_warning() -> None:
    provider = _CaptureProvider()

    with pytest.warns(UserWarning, match="alias for `system_prompt=`"):
        response = provider.generate("hi", system="You are a pirate.")

    assert response.content == "ok"
    assert provider.last_system_prompt == "You are a pirate."
    assert "system" not in provider.last_kwargs


def test_generate_prefers_system_prompt_when_both_system_forms_are_provided() -> None:
    provider = _CaptureProvider()

    with pytest.warns(UserWarning, match="Both `system_prompt=` and `system=`"):
        _ = provider.generate(
            "hi",
            system_prompt="Use this system prompt.",
            system="Ignore this alias.",
        )

    assert provider.last_system_prompt == "Use this system prompt."
    assert "system" not in provider.last_kwargs


def test_generate_normalizes_developer_messages_to_system_prompt() -> None:
    provider = _CaptureProvider()

    response = provider.generate(
        "",
        messages=[
            {"role": "developer", "content": "Answer in JSON."},
            {"role": "user", "content": "ping"},
        ],
        system_prompt="You are concise.",
    )

    assert response.content == "ok"
    assert provider.last_system_prompt == "You are concise.\n\nAnswer in JSON."
    assert provider.last_messages == [{"role": "user", "content": "ping"}]


@pytest.mark.asyncio
async def test_agenerate_accepts_system_alias_for_native_async_provider() -> None:
    provider = _NativeAsyncCaptureProvider()

    with pytest.warns(UserWarning, match="alias for `system_prompt=`"):
        response = await provider.agenerate("hi", system="You are concise.")

    assert response.content == "ok"
    assert provider.last_system_prompt == "You are concise."
    assert "system" not in provider.last_kwargs


@pytest.mark.asyncio
async def test_agenerate_normalizes_developer_messages_for_native_async_provider() -> None:
    provider = _NativeAsyncCaptureProvider()

    response = await provider.agenerate(
        "",
        messages=[
            {"role": "developer", "content": "Use terse answers."},
            {"role": "user", "content": "ping"},
        ],
    )

    assert response.content == "ok"
    assert provider.last_system_prompt == "Use terse answers."
    assert provider.last_messages == [{"role": "user", "content": "ping"}]
