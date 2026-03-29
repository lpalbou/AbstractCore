from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider, PromptCacheUnsupportedError


class DummyPromptCacheProvider(BaseProvider):
    def __init__(self, model: str = "dummy", **kwargs):
        super().__init__(model, **kwargs)
        self.last_kwargs: Dict[str, Any] = {}

    def supports_prompt_cache(self) -> bool:
        return True

    def _generate_internal(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        stream: bool = False,
        **kwargs,
    ) -> GenerateResponse | Iterator[GenerateResponse]:
        _ = (prompt, messages, system_prompt, tools, media, stream)
        self.last_kwargs = dict(kwargs)
        return GenerateResponse(content="ok", model=self.model, finish_reason="stop")

    def get_capabilities(self) -> List[str]:
        return ["chat"]

    def unload_model(self, model_name: str) -> None:
        _ = model_name

    def list_available_models(self, **kwargs) -> List[str]:
        _ = kwargs
        return ["dummy"]


def test_prompt_cache_default_key_is_applied():
    llm = DummyPromptCacheProvider()
    assert llm.prompt_cache_set("cache-a") is True

    resp = llm.generate("hello")
    assert isinstance(resp, GenerateResponse)
    assert llm.last_kwargs.get("prompt_cache_key") == "cache-a"


def test_prompt_cache_explicit_key_overrides_default():
    llm = DummyPromptCacheProvider()
    assert llm.prompt_cache_set("cache-a") is True

    llm.generate("hello", prompt_cache_key="cache-b")
    assert llm.last_kwargs.get("prompt_cache_key") == "cache-b"


def test_prompt_cache_explicit_none_disables_default():
    llm = DummyPromptCacheProvider()
    assert llm.prompt_cache_set("cache-a") is True

    llm.generate("hello", prompt_cache_key=None)
    assert llm.last_kwargs.get("prompt_cache_key") is None


def test_prompt_cache_clear_resets_default():
    llm = DummyPromptCacheProvider()
    assert llm.prompt_cache_set("cache-a") is True
    assert llm.prompt_cache_clear() is True

    llm.generate("hello")
    assert "prompt_cache_key" not in llm.last_kwargs


def test_prompt_cache_capabilities_for_keyed_provider() -> None:
    llm = DummyPromptCacheProvider()

    caps = llm.get_prompt_cache_capabilities()

    assert caps.supported is True
    assert caps.mode == "keyed"
    assert caps.supports_set is True
    assert caps.supports_clear is True
    assert caps.supports_stats is True
    assert caps.supports_update is False
    assert caps.supports_fork is False
    assert caps.supports_prepare_modules is False


def test_prompt_cache_prepare_modules_raises_clear_error_for_keyed_provider() -> None:
    llm = DummyPromptCacheProvider()

    with pytest.raises(PromptCacheUnsupportedError) as exc:
        llm.prompt_cache_prepare_modules(
            namespace="tenant:model",
            modules=[{"module_id": "system", "system_prompt": "hello"}],
        )

    assert exc.value.operation == "prepare_modules"
    assert exc.value.code == "prompt_cache_unsupported"
