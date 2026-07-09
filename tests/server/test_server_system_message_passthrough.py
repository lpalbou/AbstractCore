"""Server regression: client system messages must reach native providers.

OpenAI-format clients send their system prompt inside `messages` (usually messages[0]).
The server passes `messages` verbatim to the provider with no `system_prompt` extraction,
so the provider layer MUST deliver `role:"system"` entries instead of dropping them —
previously the native OpenAI/Anthropic providers silently deleted every system message in
`messages`, which meant server-mediated clients lost personas/guardrails entirely on
those backends (while ollama/lmstudio backends worked), with no error and no warning.

This test pins the server half of the contract: role fidelity from the HTTP request all
the way into the provider's `messages` argument. The provider half (system delivery to
the wire payload) is pinned in tests/providers/test_mid_stream_system_messages_unit.py.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

from fastapi.testclient import TestClient

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider


class _CaptureProvider(BaseProvider):
    """Minimal provider stub capturing what the server passes to generate()."""

    captured: Dict[str, Any] = {}

    def __init__(self, model: str = "stub-model", **kwargs: Any):
        super().__init__(model, **kwargs)
        self.provider = "openai"

    def _generate_internal(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        stream: bool = False,
        **kwargs: Any,
    ):
        _CaptureProvider.captured = {
            "prompt": prompt,
            "messages": messages,
            "system_prompt": system_prompt,
        }
        return GenerateResponse(
            content="ok",
            model=self.model,
            finish_reason="stop",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    def get_capabilities(self) -> List[str]:
        return ["chat"]

    def list_available_models(self, **kwargs: Any) -> List[str]:
        return [self.model]

    def unload_model(self, model_name: str) -> None:
        return None


def test_chat_completions_system_message_reaches_native_provider(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    _CaptureProvider.captured = {}

    monkeypatch.setattr(
        server_app,
        "create_llm",
        lambda provider, model=None, **kwargs: _CaptureProvider(model=model or "stub-model"),
    )

    client = TestClient(server_app.app)
    resp = client.post(
        "/v1/chat/completions",
        # Explicit per-request provider key: keeps the test independent of any
        # server-held OPENAI_API_KEY / inbound-auth configuration on the host.
        headers={"X-AbstractCore-Provider-API-Key": "sk-test"},
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a pirate."},
                {"role": "user", "content": "Hello."},
                {"role": "system", "content": "[Attachment index] notes.txt already attached."},
            ],
            "stream": False,
        },
    )
    assert resp.status_code == 200, resp.text

    sent = _CaptureProvider.captured.get("messages") or []
    pairs = [(m.get("role"), m.get("content")) for m in sent if isinstance(m, dict)]

    # Both the leading system prompt and the tail system hint must reach the provider,
    # in their original positions.
    assert pairs[0] == ("system", "You are a pirate.")
    assert ("system", "[Attachment index] notes.txt already attached.") in pairs
    assert pairs.index(("system", "[Attachment index] notes.txt already attached.")) == len(pairs) - 1
