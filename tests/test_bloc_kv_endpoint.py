from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from fastapi.testclient import TestClient

from abstractcore.core.file_blocs import FileBlocStore
from abstractcore.core.types import GenerateResponse
from abstractcore.endpoint.app import create_app
from abstractcore.providers.base import BaseProvider


@dataclass
class _FakePersistentCache:
    chunks: List[Dict[str, Any]] = field(default_factory=list)


class _StubEndpointMLXProvider(BaseProvider):
    def __init__(self, model: str = "qwen3-test", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.provider = "mlx"
        self._resolved_model_id = f"/resolved/{model}"

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
        self._prompt_cache_store.set(target_key, _FakePersistentCache(chunks=list(payload.get("chunks") or [])), meta=dict(payload.get("meta") or {}) | {"loaded_from": str(filename)})
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
        _ = (prompt, messages, system_prompt, tools, media, stream, kwargs)
        return GenerateResponse(content="ok", model=self.model, finish_reason="stop")

    def get_capabilities(self) -> List[str]:
        return ["chat"]

    def unload_model(self, model_name: str) -> None:
        _ = model_name

    @classmethod
    def list_available_models(cls, **kwargs: Any) -> List[str]:
        _ = kwargs
        return ["stub"]


def test_endpoint_bloc_kv_routes_work_end_to_end(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path / "blocs")
    app = create_app(provider_instance=_StubEndpointMLXProvider(), bloc_store=store)
    client = TestClient(app)

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
    upsert_body = upsert.json()
    assert upsert_body["ok"] is True
    sha256 = upsert_body["record"]["sha256"]
    bloc_id = upsert_body["record"]["bloc_id"]
    assert isinstance(bloc_id, int) and bloc_id > 0

    record = client.get(f"/acore/blocs/record?sha256={sha256}")
    assert record.status_code == 200
    assert record.json()["record"]["sha256"] == sha256

    ensured = client.post("/acore/blocs/kv/ensure", json={"sha256": sha256})
    assert ensured.status_code == 200
    ensured_body = ensured.json()
    assert ensured_body["ok"] is True
    assert Path(ensured_body["artifact"]["artifact_path"]).exists()

    manifest = client.get(f"/acore/blocs/kv/manifest?bloc_id={bloc_id}")
    assert manifest.status_code == 200
    manifest_body = manifest.json()
    assert manifest_body["ok"] is True
    assert manifest_body["manifest"]["bloc_id"] == bloc_id

    loaded = client.post(
        "/acore/blocs/kv/load",
        json={
            "sha256": sha256,
            "stable_cache_key": "stable:orbit",
            "key": "work:orbit",
            "make_default": False,
        },
    )
    assert loaded.status_code == 200
    loaded_body = loaded.json()
    assert loaded_body["ok"] is True
    assert loaded_body["artifact"]["key"] == "work:orbit"
    assert loaded_body["artifact"]["stable_cache_key"] == "stable:orbit"
