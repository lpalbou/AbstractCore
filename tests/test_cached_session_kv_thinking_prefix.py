"""Regression pin (2026-09-17): KV-mode CachedSession keeps thinking controls OUT of its prefix.

KV mode serializes the system bloc without any thinking control and is honest about it:
`generate()` is told the system region is prefilled, and provider hooks DECLINE an effort
level for a prefilled system block (`handled_level=False` plus a warning).

`prompt_cache_prepare_modules(thinking=None)` resolves None to the reasoning effort
configured on the text route — which is right for full-context callers and was wrong
here: on a machine configured `reasoning: minimal`, a Qwen3.8 session got "Reasoning
effort is set to low. …" baked into its system bloc's KV while every turn reported that
no level could be applied. Found by the adversarial pass on the fix that introduced the
default resolution; CachedSession now says `thinking="auto"` (no control, no bytes).
"""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any, Dict, List

import pytest

from abstractcore.core.cached_session import CachedSession
from abstractcore.providers.base import (
    PromptCacheModule,
    PromptCacheOperationError,
    ThinkingControlHandling,
)
from tests.test_cached_session_kv_mode import _FakeCache, _StubKVProvider

_EFFORT_LINE = "Reasoning effort is set to low. Keep your thinking brief."


class _EffortLineProvider(_StubKVProvider):
    """A provider whose model renders its effort level as a sentence in the system block,
    on a host configured with a reasoning default — the shape that exposed the bug."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._abstractcore_config_file = os.path.join(
            tempfile.gettempdir(), f"acore-absent-config-{uuid.uuid4().hex}.json"
        )
        self._abstractcore_capability_defaults = {
            "input.text": {
                "key": "input.text",
                "provider": "stubkv",
                "model": "stub",
                "reasoning": "low",
                "source": "abstractcore.capability_defaults",
            }
        }
        self.architecture_config = dict(self.architecture_config or {})
        self.architecture_config["thinking_control"] = {"effort_system_lines": {"low": _EFFORT_LINE}}
        self.architecture_config["reasoning_levels"] = ["low"]
        self.prepare_thinking: List[Any] = []

    def _apply_provider_thinking_kwargs(self, *, enabled, level=None, kwargs: Dict[str, Any]):
        new_kwargs = dict(kwargs or {})
        prefilled = new_kwargs.get("prompt_cache_prefilled_modules") or ()
        if isinstance(level, str) and "system" not in prefilled:
            new_kwargs["_acore_stub_reasoning_effort"] = level
            return new_kwargs, ThinkingControlHandling(handled_enable_disable=True, handled_level=True)
        return new_kwargs, ThinkingControlHandling()

    def prompt_cache_prepare_modules(self, **kwargs: Any) -> Dict[str, Any]:
        self.prepare_thinking.append(kwargs.get("thinking", "<absent>"))
        return super().prompt_cache_prepare_modules(**kwargs)


def _system_texts(llm: _EffortLineProvider, key: str) -> List[str]:
    cache = llm._prompt_cache_store.get(key)
    assert isinstance(cache, _FakeCache)
    return [str(c.get("system_prompt") or "") for c in cache.chunks if c.get("system_prompt")]


def test_the_configured_default_WOULD_reach_a_full_context_prefix() -> None:
    """Not vacuous: on this provider `thinking=None` really does bake the sentence in."""
    llm = _EffortLineProvider()
    mods = [PromptCacheModule(module_id="system", system_prompt="You are helpful.").normalized()]
    out, applied = llm._prompt_cache_modules_with_thinking(mods, None)
    assert applied is True and out[0].system_prompt.startswith(_EFFORT_LINE)


def test_kv_mode_prefix_carries_no_thinking_control() -> None:
    llm = _EffortLineProvider()
    session = CachedSession(provider=llm, system_prompt="You are helpful.", prompt_cache_strategy="kv")

    assert llm.prepare_thinking == ["auto"]
    texts = _system_texts(llm, session.prompt_cache_key)
    assert texts == ["You are helpful."]
    assert not any("Reasoning effort" in t for t in texts)


def test_an_invalid_thinking_value_fails_instead_of_planning_without_it() -> None:
    llm = _EffortLineProvider()
    with pytest.raises(PromptCacheOperationError) as exc:
        llm.prompt_cache_prepare_modules(
            namespace="ns",
            modules=[{"module_id": "system", "system_prompt": "You are helpful."}],
            thinking="banana",
        )
    assert exc.value.code == "prompt_cache_invalid_thinking"
