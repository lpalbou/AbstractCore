from __future__ import annotations

import uuid

import pytest


@pytest.mark.basic
def test_chat_completion_request_unload_after_defaults_false() -> None:
    from abstractcore.server.app import ChatCompletionRequest, ChatMessage

    req = ChatCompletionRequest(
        model="openai/gpt-4o-mini",
        messages=[ChatMessage(role="user", content="Hello")],
    )
    assert req.unload_after is False


@pytest.mark.basic
def test_huggingface_provider_has_no_destructor_unload() -> None:
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    assert hasattr(HuggingFaceProvider, "__del__") is False


@pytest.mark.basic
def test_ollama_inflight_defers_unload_until_last_request_finishes() -> None:
    from abstractcore.server.app import _ollama_inflight_enter, _ollama_inflight_exit, _ollama_inflight_key

    key = _ollama_inflight_key("ollama", "http://localhost:11434", f"model-{uuid.uuid4().hex}")

    _ollama_inflight_enter(key)
    _ollama_inflight_enter(key)

    assert _ollama_inflight_exit(key, unload_after_requested=True) is False
    assert _ollama_inflight_exit(key, unload_after_requested=False) is True

