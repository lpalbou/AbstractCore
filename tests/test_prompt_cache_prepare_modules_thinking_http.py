"""Regression pin (2026-09-17): `thinking` reaches `prompt_cache_prepare_modules` over HTTP.

`thinking` is part of a prepared prefix's identity — for models that render the effort
level into the head of the system block, a chain planned under a different request is a
prefix of nothing `generate()` sends. The request model had no such field, and pydantic
drops unknown keys silently, so a remote host that named the level had it discarded
without an error: the adversarial pass found the parameter lost on every HTTP path.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi.testclient import TestClient

from abstractcore.endpoint.app import create_app
from tests.test_prompt_cache_control_plane import _StubModularCacheProvider


class _RecordingProvider(_StubModularCacheProvider):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prepare_thinking: List[Any] = []

    def prompt_cache_prepare_modules(self, **kwargs: Any) -> Dict[str, Any]:
        self.prepare_thinking.append(kwargs.get("thinking", "<absent>"))
        return super().prompt_cache_prepare_modules(**kwargs)


_BODY = {"namespace": "ns", "modules": [{"module_id": "system", "system_prompt": "You are helpful."}]}


def test_endpoint_forwards_the_thinking_request_to_the_planner() -> None:
    llm = _RecordingProvider(model="stub-model")
    client = TestClient(create_app(provider_instance=llm))

    named = client.post("/acore/prompt_cache/prepare_modules", json={**_BODY, "thinking": "low"})
    unnamed = client.post("/acore/prompt_cache/prepare_modules", json=dict(_BODY))

    assert named.status_code == 200 and named.json()["supported"] is True
    assert unnamed.status_code == 200 and unnamed.json()["supported"] is True
    assert llm.prepare_thinking == ["low", None]


def test_endpoint_accepts_the_boolean_spelling() -> None:
    llm = _RecordingProvider(model="stub-model")
    client = TestClient(create_app(provider_instance=llm))
    r = client.post("/acore/prompt_cache/prepare_modules", json={**_BODY, "thinking": False})
    assert r.status_code == 200 and r.json()["supported"] is True
    assert llm.prepare_thinking == [False]
