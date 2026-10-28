"""`/v1/chat/completions` reuses a model already resident in this process.

Fresh-Linux release test (2026-10-28): the gateway loaded a 2.5 GB GGUF through its runtime pool;
`/v1/chat/completions` built a SECOND llama.cpp instance per request (the server registry did not
know the pool's) and failed with "Failed to create llama_context" on a 16 GB GPU; on Metal each
request added ~11 GB. The chat route now reuses the alive huggingface holder of the same model.
"""
from __future__ import annotations

import importlib
import sys
import types
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from abstractcore.core.types import GenerateResponse
from abstractcore.providers import hf_residency
from abstractcore.providers.base import BaseProvider

MODEL = "unsloth/Qwen3.5-4B-GGUF"


class _Holder(BaseProvider):
    """Stands in for a resident HuggingFaceProvider: alive (`llm` set), named by `model`."""

    def __init__(self, model: str, *, alive: bool = True) -> None:
        super().__init__(model=model, enable_tracing=False)
        self.llm: Optional[Any] = types.SimpleNamespace(model_path=f"/cache/{model.split('/')[-1]}/model-Q4_K_M.gguf") if alive else None
        self.calls = 0
        self.unloaded = False

    def get_capabilities(self) -> List[str]:
        return ["chat"]

    def list_available_models(self, **kwargs: Any) -> List[str]:
        return [self.model]

    def unload_model(self, model_name: str) -> None:
        self.unloaded = True

    def _generate_internal(self, prompt: str, messages: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> GenerateResponse:
        self.calls += 1
        return GenerateResponse(content="pong", model=self.model, finish_reason="stop",
                                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})


@pytest.fixture()
def server(monkeypatch):
    for name in ("ABSTRACTCORE_AUTH_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    # The huggingface backend counts as loaded only once its provider module is imported.
    if "abstractcore.providers.huggingface_provider" not in sys.modules:
        monkeypatch.setitem(sys.modules, "abstractcore.providers.huggingface_provider", types.ModuleType("hf_stub"))
    app = importlib.import_module("abstractcore.server.app")
    created: List[_Holder] = []

    def fake_create_llm(provider: str, model: str, **kwargs: Any) -> _Holder:
        llm = _Holder(model)
        created.append(llm)
        return llm

    monkeypatch.setattr(app, "create_llm", fake_create_llm)
    return app, created


def _chat(app, model: str = f"huggingface/{MODEL}", **extra: Any) -> Dict[str, Any]:
    resp = TestClient(app.app).post("/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": "ping"}], **extra})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_chat_reuses_the_resident_holder_instead_of_a_second_copy(server) -> None:
    app, created = server
    holder = _Holder(MODEL)
    hf_residency.register_provider(holder)
    try:
        for _ in range(3):
            assert _chat(app)["choices"][0]["message"]["content"] == "pong"
        assert created == [] and holder.calls == 3
        # Matched by path too (a request naming the cached file's repo spelled differently).
        assert app._resident_provider_for_request("huggingface", MODEL.lower(), {}) is holder
    finally:
        hf_residency._HF_PROVIDERS.discard(holder)


def test_no_reuse_when_it_would_be_wrong(server) -> None:
    app, created = server
    holder = _Holder(MODEL)
    hf_residency.register_provider(holder)
    try:
        # unload_after would unload the holder's model: the request builds (and unloads) its own.
        # (Its process-wide eject is the existing unload_after rule: a claimed model, e.g. a
        # runtime pool's, is spared by `eject_unclaimed`; this bare test holder is not claimed.)
        _chat(app, unload_after=True)
        assert len(created) == 1 and holder.calls == 0
        # That request's own instance is marked request-scoped and never lent to another request.
        assert created[0]._abstractcore_request_scoped is True
        hf_residency.register_provider(created[0])
        assert app._resident_provider_for_request("huggingface", MODEL, {}) is holder
        holder.llm = None
        assert app._resident_provider_for_request("huggingface", MODEL, {}) is None
        hf_residency._HF_PROVIDERS.discard(created[0])
        holder.llm = types.SimpleNamespace(model_path=None)  # alive again for the next checks
        # Another model, and request-specific provider settings, never borrow.
        _chat(app, model="huggingface/other/Model-GGUF")
        assert len(created) == 2
        assert app._resident_provider_for_request("huggingface", MODEL, {"base_url": "http://x"}) is None
        assert app._resident_provider_for_request("huggingface", MODEL, {"api_key": "k"}) is None
        # Only the huggingface backend lends (MLX instances already share weights).
        assert app._resident_provider_for_request("mlx", MODEL, {}) is None
    finally:
        hf_residency._HF_PROVIDERS.discard(holder)
    # A holder whose weights were released is not alive: never lent.
    dead = _Holder(MODEL, alive=False)
    hf_residency.register_provider(dead)
    try:
        _chat(app)
        assert len(created) == 3 and dead.calls == 0
    finally:
        hf_residency._HF_PROVIDERS.discard(dead)
