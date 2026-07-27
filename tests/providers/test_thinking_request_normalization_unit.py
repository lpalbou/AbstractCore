"""Unit tests for `thinking=` request normalization and Harmony system-prompt control.

Regression coverage for two escaped-regex defects: the separator-normalization
character class matched literal backslashes instead of whitespace (so the documented
"extra high" alias raised ValueError), and the Harmony `Reasoning:` line replacement
never matched an existing line (so conflicting Reasoning lines stacked up).
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Union

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("extra high", (True, "xhigh")),
        ("extra-high", (True, "xhigh")),
        ("extra_high", (True, "xhigh")),
        ("x high", (True, "xhigh")),
        ("x-high", (True, "xhigh")),
        ("EXTRA  HIGH", (True, "xhigh")),
        ("xhigh", (True, "xhigh")),
        ("minimal", (True, "minimal")),
        ("low", (True, "low")),
        ("medium", (True, "medium")),
        ("high", (True, "high")),
        ("on", (True, None)),
        ("off", (False, None)),
        ("none", (False, None)),
        ("auto", (None, None)),
        ("", (None, None)),
        (None, (None, None)),
        (True, (True, None)),
        (False, (False, None)),
    ],
)
def test_normalize_thinking_request(raw, expected) -> None:
    assert BaseProvider._normalize_thinking_request(raw) == expected


@pytest.mark.parametrize("raw", ["ultra", "max", "highest", "xxhigh"])
def test_normalize_thinking_request_rejects_unknown_levels(raw) -> None:
    with pytest.raises(ValueError):
        BaseProvider._normalize_thinking_request(raw)


class _HarmonyStubProvider(BaseProvider):
    def get_capabilities(self) -> list[str]:
        return []

    def list_available_models(self, **kwargs) -> list[str]:
        return [self.model]

    def unload_model(self, model_name: str) -> None:
        return None

    def _generate_internal(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        stream: bool = False,
        response_model: Optional[Any] = None,
        execute_tools: Optional[bool] = None,
        media_metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        return GenerateResponse(content="ok", model=self.model, finish_reason="stop")


def _harmony_provider() -> _HarmonyStubProvider:
    provider = _HarmonyStubProvider(model="gpt-oss-unit")
    provider.model_capabilities = {
        "thinking_support": True,
        "response_format": "harmony",
        "reasoning_levels": ["low", "medium", "high"],
    }
    return provider


def test_harmony_reasoning_line_replaces_existing_line() -> None:
    provider = _harmony_provider()
    _, _, system_prompt, _, meta = provider._apply_thinking_request(
        thinking="high",
        prompt="hi",
        messages=None,
        system_prompt="Reasoning: low\nBe brief.",
        kwargs={},
    )

    assert isinstance(system_prompt, str)
    assert system_prompt.count("Reasoning:") == 1
    assert "Reasoning: high" in system_prompt
    assert "Reasoning: low" not in system_prompt
    assert "Be brief." in system_prompt
    assert isinstance(meta, dict)
    assert meta.get("thinking_effective") == "high"


def test_harmony_reasoning_line_prepends_when_absent() -> None:
    provider = _harmony_provider()
    _, _, system_prompt, _, _ = provider._apply_thinking_request(
        thinking="medium",
        prompt="hi",
        messages=None,
        system_prompt="Be brief.",
        kwargs={},
    )

    assert isinstance(system_prompt, str)
    assert system_prompt.startswith("Reasoning: medium")
    assert system_prompt.count("Reasoning:") == 1
