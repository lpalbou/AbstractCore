from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from fastapi.testclient import TestClient

from abstractcore.core.types import GenerateResponse
from abstractcore.endpoint.app import create_app
from abstractcore.providers.base import BaseProvider


@dataclass
class _FakeCache:
    chunks: List[Dict[str, Any]] = field(default_factory=list)


class _StubModularCacheProvider(BaseProvider):
    def __init__(self, model: str = "stub", **kwargs: Any):
        super().__init__(model, **kwargs)
        self.append_calls = 0
        self.last_kwargs: Dict[str, Any] = {}

    def supports_prompt_cache(self) -> bool:
        return True

    def _prompt_cache_backend_create(self) -> Optional[Any]:
        return _FakeCache()

    def _prompt_cache_backend_clone(self, cache_value: Any) -> Optional[Any]:
        if not isinstance(cache_value, _FakeCache):
            return None
        return _FakeCache(chunks=list(cache_value.chunks))

    def _prompt_cache_backend_append(
        self,
        cache_value: Any,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> bool:
        _ = kwargs
        if not isinstance(cache_value, _FakeCache):
            return False
        self.append_calls += 1
        cache_value.chunks.append(
            {
                "prompt": prompt,
                "messages": messages,
                "system_prompt": system_prompt,
                "tools": tools,
                "add_generation_prompt": bool(add_generation_prompt),
            }
        )
        return True

    def _prompt_cache_backend_token_count(self, cache_value: Any) -> Optional[int]:
        if not isinstance(cache_value, _FakeCache):
            return None
        return len(cache_value.chunks)

    def _generate_internal(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> GenerateResponse | Iterator[GenerateResponse]:
        _ = (prompt, messages, system_prompt, tools, media, stream)
        self.last_kwargs = dict(kwargs)
        return GenerateResponse(content="ok", model=self.model, finish_reason="stop")

    def get_capabilities(self) -> List[str]:
        return ["chat"]

    def unload_model(self, model_name: str) -> None:
        _ = model_name

    @classmethod
    def list_available_models(cls, **kwargs: Any) -> List[str]:
        _ = kwargs
        return ["stub"]


def test_prompt_cache_update_fork_and_prepare_modules():
    llm = _StubModularCacheProvider()

    assert llm.prompt_cache_update("k1", prompt="hello") is True
    cache1 = llm._prompt_cache_store.get("k1")
    assert isinstance(cache1, _FakeCache)
    assert cache1.chunks and cache1.chunks[-1]["prompt"] == "hello"

    assert llm.prompt_cache_fork("k1", "k2") is True
    cache2 = llm._prompt_cache_store.get("k2")
    assert isinstance(cache2, _FakeCache)
    assert cache2.chunks == cache1.chunks
    cache2.chunks.append({"prompt": "mutate"})
    assert cache2.chunks != cache1.chunks

    llm.append_calls = 0
    result = llm.prompt_cache_prepare_modules(
        namespace="tenantA:modelX",
        modules=[
            {"module_id": "system", "system_prompt": "You are helpful"},
            {"module_id": "tools", "tools": [{"type": "function", "function": {"name": "t", "parameters": {}}}]},
            {"module_id": "turn", "prompt": "hi", "add_generation_prompt": True},
        ],
        make_default=True,
        version=1,
    )
    assert result["supported"] is True
    assert result["final_cache_key"]
    assert llm.get_prompt_cache_stats()["default_key"] == result["final_cache_key"]
    assert llm.append_calls == 3

    # Second call should reuse existing prefix caches (no additional appends).
    llm.append_calls = 0
    result2 = llm.prompt_cache_prepare_modules(
        namespace="tenantA:modelX",
        modules=[
            {"module_id": "system", "system_prompt": "You are helpful"},
            {"module_id": "tools", "tools": [{"type": "function", "function": {"name": "t", "parameters": {}}}]},
            {"module_id": "turn", "prompt": "hi", "add_generation_prompt": True},
        ],
        make_default=False,
        version=1,
    )
    assert result2["supported"] is True
    assert result2["final_cache_key"] == result["final_cache_key"]
    assert llm.append_calls == 0


def test_abstractendpoint_prompt_cache_control_plane_endpoints():
    llm = _StubModularCacheProvider(model="stub-model")
    app = create_app(provider_instance=llm)
    client = TestClient(app)

    r = client.post("/acore/prompt_cache/set", json={"key": "k1", "make_default": True})
    assert r.status_code == 200
    assert r.json()["supported"] is True
    assert r.json()["ok"] is True

    r = client.post("/acore/prompt_cache/update", json={"key": "k1", "prompt": "hello"})
    assert r.status_code == 200
    assert r.json()["supported"] is True
    assert r.json()["ok"] is True

    r = client.post("/acore/prompt_cache/fork", json={"from_key": "k1", "to_key": "k2"})
    assert r.status_code == 200
    assert r.json()["supported"] is True
    assert r.json()["ok"] is True

    r = client.post(
        "/acore/prompt_cache/prepare_modules",
        json={
            "namespace": "tenantA:stub-model",
            "modules": [{"module_id": "system", "system_prompt": "You are helpful"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["supported"] is True
    assert body["final_cache_key"]

    r = client.get("/acore/prompt_cache/stats")
    assert r.status_code == 200
    stats = r.json()
    assert stats["supported"] is True
    assert stats["stats"]["entries"] >= 1

    r = client.post("/acore/prompt_cache/clear", json={"key": "k1"})
    assert r.status_code == 200
    assert r.json()["supported"] is True
    assert r.json()["ok"] is True

