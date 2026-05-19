from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from fastapi.testclient import TestClient

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider


@dataclass
class _FakePersistentCache:
    chunks: List[Dict[str, Any]] = field(default_factory=list)


class _StubGatewayMLXProvider(BaseProvider):
    def __init__(self, model: str = "mlx-community/Qwen3.6-27B-4bit", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.provider = "mlx"
        self._resolved_model_id = f"/resolved/{model}"
        self.last_kwargs: Dict[str, Any] = {}
        self.unload_model_calls: List[str] = []

    def supports_prompt_cache(self) -> bool:
        return True

    def prompt_cache_supports_kv_source_of_truth(self) -> bool:
        return True

    def _prompt_cache_backend_create(self) -> Optional[Any]:
        return _FakePersistentCache()

    def _prompt_cache_backend_clone(self, cache_value: Any) -> Optional[Any]:
        if not isinstance(cache_value, _FakePersistentCache):
            return None
        return _FakePersistentCache(chunks=[dict(chunk) for chunk in cache_value.chunks])

    def _build_prompt_fragment(
        self,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        prefilled_modules: Optional[List[str]] = None,
    ) -> str:
        _ = (tools, prefilled_modules)
        parts: List[str] = []
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
        for msg in messages or []:
            parts.append(f"<|im_start|>{msg.get('role','user')}\n{msg.get('content','')}<|im_end|>\n")
        if prompt:
            parts.append(f"<|im_start|>user\n{prompt}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

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
        if not isinstance(cache_value, _FakePersistentCache):
            return False
        cache_value.chunks.append(
            {
                "serialized": self._build_prompt_fragment(
                    prompt=prompt,
                    messages=messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    add_generation_prompt=add_generation_prompt,
                )
            }
        )
        return True

    def _prompt_cache_backend_token_count(self, cache_value: Any) -> Optional[int]:
        if not isinstance(cache_value, _FakePersistentCache):
            return None
        return len(cache_value.chunks)

    def prompt_cache_save(self, key: str, filename: str, *, q8: bool = False, meta: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        _ = (q8, kwargs)
        cache_value = self._prompt_cache_store.get(key)
        if not isinstance(cache_value, _FakePersistentCache):
            raise ValueError("missing cache")
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"chunks": cache_value.chunks, "meta": dict(meta or {})}), encoding="utf-8")
        return {"key": key, "filename": str(path), "meta": dict(meta or {})}

    def prompt_cache_load(self, filename: str, *, key: Optional[str] = None, make_default: bool = True, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        target_key = str(key or "loaded")
        self._prompt_cache_store.set(
            target_key,
            _FakePersistentCache(chunks=list(payload.get("chunks") or [])),
            meta=dict(payload.get("meta") or {}) | {"loaded_from": str(filename)},
        )
        if make_default:
            self._default_prompt_cache_key = target_key
        return {"key": target_key, "filename": str(filename), "meta": dict(payload.get("meta") or {})}

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
        return GenerateResponse(
            content="ok",
            model=self.model,
            finish_reason="stop",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    def get_capabilities(self) -> List[str]:
        return ["chat"]

    def unload_model(self, model_name: str) -> None:
        self.unload_model_calls.append(str(model_name))

    @classmethod
    def list_available_models(cls, **kwargs: Any) -> List[str]:
        _ = kwargs
        return ["mlx-community/Qwen3.6-27B-4bit"]


def test_server_loaded_runtime_supports_local_bloc_kv_and_chat_reuse(monkeypatch, tmp_path: Path) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    server_app._GATEWAY_LOADED_RUNTIMES.clear()
    server_app._GATEWAY_RUNTIME_IDS.clear()
    server_app._SERVER_BLOC_STORE = server_app.FileBlocStore(root_dir=tmp_path / "gateway-blocs")

    created: List[_StubGatewayMLXProvider] = []

    def fake_create_llm(provider: str, model: str, **kwargs: Any) -> _StubGatewayMLXProvider:
        _ = (provider, kwargs)
        llm = _StubGatewayMLXProvider(model=model)
        created.append(llm)
        return llm

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)

    load = client.post(
        "/acore/models/load",
        json={"provider": "mlx", "model": "mlx-community/Qwen3.6-27B-4bit"},
    )
    assert load.status_code == 200
    load_body = load.json()
    assert load_body["ok"] is True
    assert load_body["loaded_new"] is True
    runtime_id = load_body["runtime"]["runtime_id"]
    assert len(created) == 1

    loaded = client.get("/acore/models/loaded?provider=mlx")
    assert loaded.status_code == 200
    assert len(loaded.json()["data"]) == 1

    upsert = client.post(
        "/acore/blocs/upsert_text",
        json={
            "path": str(tmp_path / "orbit.txt"),
            "content": "Document title: Orbit Notes\n\nThe launch window is Tuesday at 14:30 UTC.\n",
            "media_type": "text",
            "format": "text/plain",
        },
    )
    assert upsert.status_code == 200
    sha256 = upsert.json()["record"]["sha256"]

    ensured = client.post(
        "/acore/blocs/kv/ensure",
        json={
            "provider": "mlx",
            "model": "mlx-community/Qwen3.6-27B-4bit",
            "sha256": sha256,
        },
    )
    assert ensured.status_code == 200
    assert Path(ensured.json()["artifact"]["artifact_path"]).exists()

    loaded_bloc = client.post(
        "/acore/blocs/kv/load",
        json={
            "provider": "mlx",
            "model": "mlx-community/Qwen3.6-27B-4bit",
            "sha256": sha256,
            "stable_cache_key": "stable:orbit",
            "key": "work:orbit",
        },
    )
    assert loaded_bloc.status_code == 200
    loaded_bloc_body = loaded_bloc.json()
    assert loaded_bloc_body["artifact"]["key"] == "work:orbit"
    assert loaded_bloc_body["artifact"]["stable_cache_key"] == "stable:orbit"

    stats = client.get(
        "/acore/prompt_cache/stats",
        params={"provider": "mlx", "model": "mlx-community/Qwen3.6-27B-4bit"},
    )
    assert stats.status_code == 200
    assert stats.json()["supported"] is True

    chat = client.post(
        "/v1/chat/completions",
        json={
            "model": "mlx/mlx-community/Qwen3.6-27B-4bit",
            "messages": [{"role": "user", "content": "Summarize the loaded notes."}],
            "thinking": "off",
            "prompt_cache_key": "work:orbit",
        },
    )
    assert chat.status_code == 200
    assert len(created) == 1
    assert created[0].last_kwargs.get("prompt_cache_key") == "work:orbit"

    unload = client.post("/acore/models/unload", json={"runtime_id": runtime_id})
    assert unload.status_code == 200
    assert created[0].unload_model_calls == ["mlx-community/Qwen3.6-27B-4bit"]

    loaded_after = client.get("/acore/models/loaded?provider=mlx")
    assert loaded_after.status_code == 200
    assert loaded_after.json()["data"] == []


def test_server_local_control_plane_can_target_runtime_loaded_with_base_url(monkeypatch, tmp_path: Path) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    server_app._GATEWAY_LOADED_RUNTIMES.clear()
    server_app._GATEWAY_RUNTIME_IDS.clear()
    server_app._SERVER_BLOC_STORE = server_app.FileBlocStore(root_dir=tmp_path / "gateway-blocs")

    created: List[_StubGatewayMLXProvider] = []

    def fake_create_llm(provider: str, model: str, **kwargs: Any) -> _StubGatewayMLXProvider:
        _ = (provider, kwargs)
        llm = _StubGatewayMLXProvider(model=model)
        created.append(llm)
        return llm

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    base_url = "http://127.0.0.1:1234/v1"

    load = client.post(
        "/acore/models/load",
        json={"provider": "mlx", "model": "mlx-community/Qwen3.6-27B-4bit", "base_url": base_url},
    )
    assert load.status_code == 200
    runtime_id = load.json()["runtime"]["runtime_id"]
    assert len(created) == 1

    set_resp = client.post(
        "/acore/prompt_cache/set",
        json={
            "runtime_id": runtime_id,
            "provider": "mlx",
            "model": "mlx-community/Qwen3.6-27B-4bit",
            "base_url": base_url,
            "key": "shared",
            "make_default": True,
        },
    )
    assert set_resp.status_code == 200
    assert set_resp.json()["ok"] is True

    stats = client.get(
        "/acore/prompt_cache/stats",
        params={
            "provider": "mlx",
            "model": "mlx-community/Qwen3.6-27B-4bit",
            "base_url": base_url,
        },
    )
    assert stats.status_code == 200
    assert stats.json()["supported"] is True
    assert len(created) == 1


def test_server_local_prompt_cache_update_applies_thinking_before_cache_append(monkeypatch, tmp_path: Path) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    server_app._GATEWAY_LOADED_RUNTIMES.clear()
    server_app._GATEWAY_RUNTIME_IDS.clear()
    server_app._SERVER_BLOC_STORE = server_app.FileBlocStore(root_dir=tmp_path / "gateway-blocs")

    created: List[_StubGatewayMLXProvider] = []

    def fake_create_llm(provider: str, model: str, **kwargs: Any) -> _StubGatewayMLXProvider:
        _ = (provider, kwargs)
        llm = _StubGatewayMLXProvider(model=model)
        llm.model_capabilities = {"thinking_support": True}
        llm.architecture_config = {"thinking_control": "/nothink"}
        created.append(llm)
        return llm

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    load = client.post(
        "/acore/models/load",
        json={"provider": "mlx", "model": "mlx-community/Qwen3.6-27B-4bit"},
    )
    assert load.status_code == 200

    update = client.post(
        "/acore/prompt_cache/update",
        json={
            "provider": "mlx",
            "model": "mlx-community/Qwen3.6-27B-4bit",
            "key": "session-1",
            "prompt": "Remember the launch window.",
            "thinking": "off",
        },
    )
    assert update.status_code == 200
    assert update.json()["supported"] is True
    assert update.json()["ok"] is True

    cache = created[0]._prompt_cache_store.get("session-1")
    assert isinstance(cache, _FakePersistentCache)
    assert cache.chunks[-1]["serialized"].endswith("Remember the launch window.\n/nothink<|im_end|>\n")
