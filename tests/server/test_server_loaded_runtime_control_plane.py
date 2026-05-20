from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
import threading
from typing import Any, Dict, Iterator, List, Optional

from fastapi.testclient import TestClient

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider, PromptCacheRenderedFragment


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
        self.prompt_cache_save_thread_ids: List[int] = []
        self.prompt_cache_load_thread_ids: List[int] = []
        self.prompt_cache_append_thread_ids: List[int] = []
        self.generate_thread_ids: List[int] = []
        self.stream_generate_thread_ids: List[int] = []

    def supports_prompt_cache(self) -> bool:
        return True

    def prompt_cache_supports_kv_source_of_truth(self) -> bool:
        return True

    def prompt_cache_cache_backend(self) -> str:
        return "mlx"

    def prompt_cache_artifact_format(self) -> str:
        return "abstractcore-mlx-prompt-cache/v1"

    def prompt_cache_render_fragment(
        self,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        prefilled_modules: Optional[List[str]] = None,
    ) -> Optional[PromptCacheRenderedFragment]:
        serialized = self._build_prompt_fragment(
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            prefilled_modules=prefilled_modules,
        )
        if not serialized:
            return None
        return PromptCacheRenderedFragment(
            serialized_prompt=serialized,
            serializer_version="mlx-prompt-fragment/v1:qwen-chatml",
            cache_backend="mlx",
            artifact_format=self.prompt_cache_artifact_format(),
            meta={"prompt_format": "qwen-chatml"},
        )

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
        self.prompt_cache_append_thread_ids.append(threading.get_ident())
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
        self.prompt_cache_save_thread_ids.append(threading.get_ident())
        cache_value = self._prompt_cache_store.get(key)
        if not isinstance(cache_value, _FakePersistentCache):
            raise ValueError("missing cache")
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"chunks": cache_value.chunks, "meta": dict(meta or {})}), encoding="utf-8")
        return {"key": key, "filename": str(path), "meta": dict(meta or {})}

    def prompt_cache_load(self, filename: str, *, key: Optional[str] = None, make_default: bool = True, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        self.prompt_cache_load_thread_ids.append(threading.get_ident())
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
        if stream:
            def _iter() -> Iterator[GenerateResponse]:
                self.stream_generate_thread_ids.append(threading.get_ident())
                yield GenerateResponse(
                    content="ok",
                    model=self.model,
                    finish_reason=None,
                    usage={"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
                )
                yield GenerateResponse(
                    content=" done",
                    model=self.model,
                    finish_reason="stop",
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                )

            return _iter()
        self.generate_thread_ids.append(threading.get_ident())
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


class _FloodStreamingGatewayMLXProvider(_StubGatewayMLXProvider):
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
        _ = (prompt, messages, system_prompt, tools, media)
        self.last_kwargs = dict(kwargs)
        if stream:
            def _iter() -> Iterator[GenerateResponse]:
                self.stream_generate_thread_ids.append(threading.get_ident())
                for index in range(128):
                    yield GenerateResponse(
                        content=f"chunk-{index}",
                        model=self.model,
                        finish_reason=None,
                        usage={"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
                    )
                yield GenerateResponse(
                    content="done",
                    model=self.model,
                    finish_reason="stop",
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                )

            return _iter()
        return super()._generate_internal(
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            media=media,
            stream=stream,
            **kwargs,
        )


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

    loaded = client.get("/acore/models/loaded?task=text_generation&provider=mlx")
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
            "debug": True,
        },
    )
    assert ensured.status_code == 200
    ensured_artifact = ensured.json()["artifact"]
    assert Path(ensured_artifact["artifact_path"]).exists()
    assert ensured_artifact["binding_id"]
    assert ensured_artifact["debug"]["operation"] == "ensure"

    loaded_bloc = client.post(
        "/acore/blocs/kv/load",
        json={
            "provider": "mlx",
            "model": "mlx-community/Qwen3.6-27B-4bit",
            "sha256": sha256,
            "stable_cache_key": "stable:orbit",
            "key": "work:orbit",
            "debug": True,
        },
    )
    assert loaded_bloc.status_code == 200
    loaded_bloc_body = loaded_bloc.json()
    assert loaded_bloc_body["artifact"]["key"] == "work:orbit"
    assert loaded_bloc_body["artifact"]["stable_cache_key"] == "stable:orbit"
    assert loaded_bloc_body["artifact"]["binding_id"] == ensured_artifact["binding_id"]
    assert loaded_bloc_body["artifact"]["prompt_cache_binding"]["key"] == "work:orbit"
    assert loaded_bloc_body["artifact"]["debug"]["operation"] == "load"

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
            "prompt_cache_binding": loaded_bloc_body["artifact"]["prompt_cache_binding"],
        },
    )
    assert chat.status_code == 200
    assert len(created) == 1
    assert created[0].last_kwargs.get("prompt_cache_key") == "work:orbit"

    clear = client.post(
        "/acore/prompt_cache/clear",
        json={"provider": "mlx", "model": "mlx-community/Qwen3.6-27B-4bit", "key": "work:orbit"},
    )
    assert clear.status_code == 200
    stale_chat = client.post(
        "/v1/chat/completions",
        json={
            "model": "mlx/mlx-community/Qwen3.6-27B-4bit",
            "messages": [{"role": "user", "content": "Summarize the loaded notes."}],
            "prompt_cache_key": "work:orbit",
            "prompt_cache_binding": loaded_bloc_body["artifact"]["prompt_cache_binding"],
        },
    )
    assert stale_chat.status_code == 409

    unload = client.post("/acore/models/unload", json={"runtime_id": runtime_id})
    assert unload.status_code == 200
    assert created[0].unload_model_calls == ["mlx-community/Qwen3.6-27B-4bit"]

    loaded_after = client.get("/acore/models/loaded?task=text_generation&provider=mlx")
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


def test_server_loaded_runtime_runs_cache_and_chat_on_stable_provider_thread(
    monkeypatch,
    tmp_path: Path,
) -> None:
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

    update = client.post(
        "/acore/prompt_cache/update",
        json={
            "provider": "mlx",
            "model": "mlx-community/Qwen3.6-27B-4bit",
            "key": "work:orbit",
            "prompt": "Remember the orbit notes.",
            "thinking": "off",
        },
    )
    assert update.status_code == 200

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

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "mlx/mlx-community/Qwen3.6-27B-4bit",
            "messages": [{"role": "user", "content": "Stream the summary."}],
            "thinking": "off",
            "prompt_cache_key": "work:orbit",
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert any("data:" in line for line in response.iter_lines())

    provider = created[0]
    thread_ids = {
        *provider.prompt_cache_save_thread_ids,
        *provider.prompt_cache_load_thread_ids,
        *provider.prompt_cache_append_thread_ids,
        *provider.generate_thread_ids,
        *provider.stream_generate_thread_ids,
    }
    assert len(thread_ids) == 1
    assert next(iter(thread_ids)) != threading.get_ident()


def test_loaded_runtime_stream_bridge_does_not_wedge_provider_executor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    server_app._GATEWAY_LOADED_RUNTIMES.clear()
    server_app._GATEWAY_RUNTIME_IDS.clear()
    server_app._SERVER_BLOC_STORE = server_app.FileBlocStore(root_dir=tmp_path / "gateway-blocs")

    created: List[_FloodStreamingGatewayMLXProvider] = []

    def fake_create_llm(
        provider: str,
        model: str,
        **kwargs: Any,
    ) -> _FloodStreamingGatewayMLXProvider:
        _ = (provider, kwargs)
        llm = _FloodStreamingGatewayMLXProvider(model=model)
        created.append(llm)
        return llm

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    load = client.post(
        "/acore/models/load",
        json={"provider": "mlx", "model": "mlx-community/Qwen3.6-27B-4bit"},
    )
    assert load.status_code == 200

    runtime = server_app._get_loaded_gateway_runtime(
        provider="mlx",
        model="mlx-community/Qwen3.6-27B-4bit",
    )
    assert runtime is not None

    stream = server_app._stream_loaded_gateway_runtime(
        runtime,
        lambda: created[0]._generate_internal(
            prompt="hello",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        ),
    )
    first = next(stream)
    assert getattr(first, "content", "") == "chunk-0"

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(
            lambda: server_app._run_loaded_gateway_runtime(runtime, lambda: {"ok": True})
        ).result(timeout=2)
    assert result == {"ok": True}


def test_loaded_runtime_prompt_cache_stats_do_not_wait_for_busy_provider_executor(
    monkeypatch,
    tmp_path: Path,
) -> None:
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

    runtime = server_app._get_loaded_gateway_runtime(
        provider="mlx",
        model="mlx-community/Qwen3.6-27B-4bit",
    )
    assert runtime is not None

    started = threading.Event()
    release = threading.Event()

    def block_worker() -> bool:
        started.set()
        release.wait(timeout=2)
        return True

    runtime.provider_executor.submit(block_worker)
    assert started.wait(timeout=1)

    stats = client.get(
        "/acore/prompt_cache/stats",
        params={"provider": "mlx", "model": "mlx-community/Qwen3.6-27B-4bit"},
    )
    release.set()

    assert stats.status_code == 200
    assert stats.json()["supported"] is True
