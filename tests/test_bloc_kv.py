from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pytest

import abstractcore.core.bloc_kv as bloc_kv_module
from abstractcore.core.bloc_kv import (
    BlocKVArtifactInUseError,
    _tmp_path,
    delete_bloc,
    delete_bloc_kv_artifact,
    ensure_bloc_kv_artifact,
    find_bloc_kv_live_bindings,
    list_bloc_kv_artifacts,
    load_bloc_kv_artifact,
    prune_bloc_kv_artifacts,
    read_bloc_kv_manifest,
)
from abstractcore.core.bloc_metadata import generate_bloc_metadata_jsonld
from abstractcore.core.file_blocs import FileBlocStore
from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider, PromptCacheOperationError, PromptCacheRenderedFragment


@dataclass
class _FakePersistentCache:
    chunks: List[Dict[str, Any]] = field(default_factory=list)


class _StubPersistentMLXProvider(BaseProvider):
    def __init__(
        self,
        model: str = "qwen3-test",
        *,
        resolved_model_id: Optional[str] = None,
        provider_name: str = "mlx",
        cache_backend: Optional[str] = None,
        artifact_extension: str = ".safetensors",
        artifact_format: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self.provider = provider_name
        self._cache_backend = cache_backend or provider_name
        self._artifact_extension = artifact_extension
        self._artifact_format = artifact_format or f"abstractcore-{self._cache_backend}-prompt-cache/v1"
        self._resolved_model_id = resolved_model_id or f"/resolved/{model}"
        self.load_calls = 0
        self.save_calls = 0
        self.fork_calls = 0
        self.generate_calls = 0
        self.generated_payloads: List[str] = []
        self.last_kwargs: Dict[str, Any] = {}
        self.last_call: Dict[str, Any] = {}
        self.save_nonce: Optional[str] = None

    def supports_prompt_cache(self) -> bool:
        return True

    def prompt_cache_supports_kv_source_of_truth(self) -> bool:
        return True

    def prompt_cache_artifact_extension(self) -> str:
        return self._artifact_extension

    def prompt_cache_cache_backend(self) -> str:
        return self._cache_backend

    def prompt_cache_artifact_format(self) -> str:
        return self._artifact_format

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
        fmt = "qwen-chatml" if "qwen" in self.model.lower() else "plain-chat"
        return PromptCacheRenderedFragment(
            serialized_prompt=serialized,
            serializer_version=f"{self._cache_backend}-prompt-fragment/v1:{fmt}",
            cache_backend=self._cache_backend,
            artifact_format=self._artifact_format,
            meta={"prompt_format": fmt},
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
        is_qwen = "qwen" in self.model.lower()
        parts: List[str] = []
        if isinstance(system_prompt, str) and system_prompt:
            if is_qwen:
                parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
            else:
                parts.append(f"{system_prompt}\n\n")
        for msg in messages or []:
            role = str(msg.get("role") or "user")
            content = str(msg.get("content") or "")
            if is_qwen:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            else:
                parts.append(f"{role}: {content}\n")
        if isinstance(prompt, str) and prompt:
            if is_qwen:
                parts.append(f"<|im_start|>user\n{prompt}<|im_end|>\n")
            else:
                parts.append(f"user: {prompt}\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n" if is_qwen else "assistant:")
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
                ),
                "messages": messages,
            }
        )
        return True

    def _prompt_cache_backend_token_count(self, cache_value: Any) -> Optional[int]:
        if not isinstance(cache_value, _FakePersistentCache):
            return None
        return len(cache_value.chunks)

    def prompt_cache_save(
        self,
        key: str,
        filename: str,
        *,
        q8: bool = False,
        meta: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = kwargs
        cache_value = self._prompt_cache_store.get(key)
        if not isinstance(cache_value, _FakePersistentCache):
            raise ValueError(f"prompt cache key '{key}' does not exist")
        payload = {
            "chunks": list(cache_value.chunks),
            "meta": dict(meta or {}),
            "quantization": "q8" if q8 else "fp",
        }
        if isinstance(self.save_nonce, str) and self.save_nonce:
            payload["nonce"] = self.save_nonce
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        self.save_calls += 1
        return {"key": key, "filename": str(path), "meta": dict(meta or {})}

    def prompt_cache_load(
        self,
        filename: str,
        *,
        key: Optional[str] = None,
        make_default: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = kwargs
        path = Path(filename)
        payload = json.loads(path.read_text(encoding="utf-8"))
        cache = _FakePersistentCache(chunks=[dict(chunk) for chunk in payload.get("chunks") or []])
        target_key = str(key or f"cache:{len(self._prompt_cache_store.keys()) + 1}")
        meta = dict(payload.get("meta") or {})
        meta["loaded_from"] = str(path)
        meta["quantization"] = payload.get("quantization")
        self._prompt_cache_store.set(target_key, cache, meta=meta)
        if make_default:
            self._default_prompt_cache_key = target_key
        self.load_calls += 1
        return {"key": target_key, "filename": str(path), "meta": meta}

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
        self.last_call = {
            "prompt": prompt,
            "messages": messages,
            "system_prompt": system_prompt,
            "tools": tools,
            "media": media,
            "stream": stream,
        }
        self.generate_calls += 1
        self.last_kwargs = dict(kwargs)
        if self.generated_payloads:
            content = self.generated_payloads.pop(0)
        else:
            content = (
                '{"t":"Title","d":"Desc","kind":"s:Report","mod":"text","lang":"en",'
                '"q":{"snr":0.91,"clar":0.82,"coh":0.87,"conc":0.76,"struct":0.88,"arg":0.71,"evid":0.69},'
                '"tp":["topic"],"kw":["keyword"]}'
            )
        return GenerateResponse(content=content, model=self.model, finish_reason="stop")

    def get_capabilities(self) -> List[str]:
        return ["chat"]

    def unload_model(self, model_name: str) -> None:
        _ = model_name

    @classmethod
    def list_available_models(cls, **kwargs: Any) -> List[str]:
        _ = kwargs
        return ["stub"]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _upsert_record(store: FileBlocStore, root: Path, *, sha: str, path_name: str, content: str) -> Any:
    file_path = root / path_name
    file_meta = {
        "path": str(file_path),
        "media_type": "text",
        "size_bytes": len(content.encode("utf-8")),
        "mtime_ns": 1,
        "sha256": sha,
        "content_sha256": _sha256_text(content),
        "format": "text/plain",
        "content_length": len(content),
        "estimated_tokens": 3,
    }
    return store.upsert(file_meta=file_meta, content=content, relpath_base=root)


def test_bloc_kv_compile_creates_manifest_and_keeps_default_key(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="a" * 64, path_name="doc.txt", content="hello world\n")
    provider = _StubPersistentMLXProvider(model="qwen3-test")

    result = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)

    assert result.compiled is True
    assert result.artifact_path.exists()
    assert result.manifest_path.exists()
    assert result.manifest.provider == "mlx"
    assert result.manifest.model == "qwen3-test"
    assert result.manifest.path_in_prompt.endswith("doc.txt")
    assert result.manifest.artifact_sha256 == _sha256_file(result.artifact_path)
    assert store.has_kv_cache_manifest(record.sha256, provider="mlx", model="qwen3-test") is True
    assert provider._default_prompt_cache_key is None


def test_bloc_kv_rebuilds_when_prompt_path_changes(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    provider = _StubPersistentMLXProvider(model="qwen3-test")

    record_a = _upsert_record(store, tmp_path, sha="b" * 64, path_name="a.txt", content="same content\n")
    first = ensure_bloc_kv_artifact(provider=provider, store=store, record=record_a)

    record_b = _upsert_record(store, tmp_path, sha="b" * 64, path_name="b.txt", content="same content\n")
    second = ensure_bloc_kv_artifact(provider=provider, store=store, record=record_b)

    assert second.compiled is True
    assert second.manifest.path_in_prompt.endswith("b.txt")
    assert first.manifest.rendered_recipe_sha256 != second.manifest.rendered_recipe_sha256


def test_bloc_kv_serializer_hash_differs_between_qwen_and_non_qwen(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="c" * 64, path_name="doc.txt", content="hello world\n")

    qwen = _StubPersistentMLXProvider(model="qwen3-test")
    mistral = _StubPersistentMLXProvider(model="mistral-test")

    qwen_result = ensure_bloc_kv_artifact(provider=qwen, store=store, record=record)
    mistral_result = ensure_bloc_kv_artifact(provider=mistral, store=store, record=record)

    assert qwen_result.manifest.serializer_version != mistral_result.manifest.serializer_version
    assert qwen_result.manifest.rendered_recipe_sha256 != mistral_result.manifest.rendered_recipe_sha256


@pytest.mark.parametrize(
    ("provider_name", "cache_backend", "artifact_extension", "artifact_format"),
    [
        ("mlx", "mlx", ".safetensors", "abstractcore-mlx-prompt-cache/v1"),
        ("huggingface", "hf-transformers", ".safetensors", "abstractcore-transformers-prompt-cache/v1"),
        ("huggingface", "hf-gguf", ".npz", "abstractcore-gguf-prompt-cache/v1"),
    ],
)
def test_bloc_kv_contract_spans_supported_local_provider_backends(
    tmp_path: Path,
    provider_name: str,
    cache_backend: str,
    artifact_extension: str,
    artifact_format: str,
) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha=cache_backend[0] * 64, path_name="doc.txt", content="hello world\n")
    provider = _StubPersistentMLXProvider(
        model=f"{cache_backend}-test",
        provider_name=provider_name,
        cache_backend=cache_backend,
        artifact_extension=artifact_extension,
        artifact_format=artifact_format,
    )

    ensured = ensure_bloc_kv_artifact(provider=provider, store=store, record=record, debug=True)
    loaded = load_bloc_kv_artifact(provider=provider, store=store, record=record, key="work", debug=True)

    assert ensured.artifact_path.suffix == artifact_extension
    assert ensured.manifest.provider == provider_name
    assert ensured.manifest.cache_backend == cache_backend
    assert ensured.manifest.artifact_format == artifact_format
    assert ensured.binding_id
    assert ensured.prompt_cache_binding["binding_id"] == ensured.binding_id
    assert ensured.debug and ensured.debug["cache_backend"] == cache_backend
    assert loaded.key == "work"
    assert loaded.prompt_cache_binding["key"] == "work"
    assert loaded.prompt_cache_binding["binding_id"] == ensured.binding_id
    assert loaded.debug and loaded.debug["binding_id"] == ensured.binding_id
    assert provider.prompt_cache_validate_binding("work", loaded.prompt_cache_binding)["key"] == "work"


def test_bloc_kv_prompt_cache_binding_guards_generation(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="5" * 64, path_name="doc.txt", content="hello world\n")
    provider = _StubPersistentMLXProvider(model="qwen3-test")

    loaded = load_bloc_kv_artifact(provider=provider, store=store, record=record, key="work")
    provider.generate(prompt="question", prompt_cache_binding=loaded.prompt_cache_binding)
    assert provider.last_kwargs["prompt_cache_key"] == "work"

    provider.prompt_cache_clear("work")
    with pytest.raises(PromptCacheOperationError, match="not loaded|no binding metadata"):
        provider.generate(prompt="question", prompt_cache_key="work", prompt_cache_binding=loaded.prompt_cache_binding)


def test_bloc_kv_delete_blocks_live_binding_then_clears_and_deletes(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="a1" * 32, path_name="doc.txt", content="hello world\n")
    provider = _StubPersistentMLXProvider(model="qwen3-test")

    loaded = load_bloc_kv_artifact(provider=provider, store=store, record=record, key="work")
    artifacts = list_bloc_kv_artifacts(store=store, sha256=record.sha256, provider="mlx", model="qwen3-test")
    assert len(artifacts) == 1
    assert find_bloc_kv_live_bindings(provider=provider, manifest=loaded.manifest)

    with pytest.raises(BlocKVArtifactInUseError):
        delete_bloc_kv_artifact(provider=provider, store=store, sha256=record.sha256, model="qwen3-test")

    result = delete_bloc_kv_artifact(
        provider=provider,
        store=store,
        sha256=record.sha256,
        model="qwen3-test",
        clear_loaded=True,
    )

    assert result.deleted is True
    assert result.cleared_keys == ["work"]
    assert result.artifact_path and not result.artifact_path.exists()
    assert result.manifest_path and not result.manifest_path.exists()
    assert provider._prompt_cache_store.get("work") is None


def test_bloc_kv_prune_and_delete_bloc_support_dry_run(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="a2" * 32, path_name="doc.txt", content="hello world\n")
    provider = _StubPersistentMLXProvider(model="qwen3-test")
    ensured = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)

    dry = prune_bloc_kv_artifacts(store=store, sha256=record.sha256, dry_run=True)
    assert len(dry) == 1
    assert dry[0].dry_run is True
    assert ensured.artifact_path.exists()

    deleted = delete_bloc(store=store, sha256=record.sha256, dry_run=True)
    assert deleted.deleted is False
    assert store.get(record.sha256) is not None


def test_bloc_kv_incomplete_commit_is_rebuilt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    provider = _StubPersistentMLXProvider(model="qwen3-test")
    record = _upsert_record(store, tmp_path, sha="d" * 64, path_name="doc.txt", content="hello world\n")

    original_commit = bloc_kv_module._commit_manifest_last

    def _broken_commit(*, artifact_tmp: Path, artifact_path: Path, manifest_tmp: Path, manifest_path: Path) -> None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_tmp.replace(artifact_path)
        raise RuntimeError("boom")

    monkeypatch.setattr(bloc_kv_module, "_commit_manifest_last", _broken_commit)
    with pytest.raises(RuntimeError, match="boom"):
        ensure_bloc_kv_artifact(provider=provider, store=store, record=record)

    broken_artifact = store.kv_cache_path(record.sha256, provider="mlx", model="qwen3-test")
    broken_manifest = store.kv_cache_manifest_path(record.sha256, provider="mlx", model="qwen3-test")
    assert broken_artifact.exists() is True
    assert broken_manifest.exists() is False

    monkeypatch.setattr(bloc_kv_module, "_commit_manifest_last", original_commit)
    rebuilt = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)
    assert rebuilt.compiled is True
    assert rebuilt.manifest_path.exists() is True


def test_bloc_kv_failed_rebuild_preserves_last_committed_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    provider = _StubPersistentMLXProvider(model="qwen3-test")
    record = _upsert_record(store, tmp_path, sha="e" * 64, path_name="doc.txt", content="hello world\n")

    first = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)
    manifest_before = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    artifact_hash_before = _sha256_file(first.artifact_path)

    def _boom(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("save failed")

    monkeypatch.setattr(provider, "prompt_cache_save", _boom)
    with pytest.raises(RuntimeError, match="save failed"):
        ensure_bloc_kv_artifact(provider=provider, store=store, record=record, force_rebuild=True)

    assert json.loads(first.manifest_path.read_text(encoding="utf-8")) == manifest_before
    assert _sha256_file(first.artifact_path) == artifact_hash_before


def test_bloc_kv_commit_failure_restores_previous_committed_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    provider = _StubPersistentMLXProvider(model="qwen3-test")
    record = _upsert_record(store, tmp_path, sha="6" * 64, path_name="doc.txt", content="hello world\n")

    first = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)
    manifest_before = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    artifact_hash_before = _sha256_file(first.artifact_path)
    original_replace = Path.replace
    failed = {"done": False}

    def _flaky_replace(self: Path, target: Path) -> Path:
        target_path = Path(target)
        if (
            not failed["done"]
            and self.name.startswith(f"{first.manifest_path.stem}.tmp.")
            and self.suffix == first.manifest_path.suffix
            and target_path == first.manifest_path
        ):
            failed["done"] = True
            raise RuntimeError("manifest replace failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", _flaky_replace)
    with pytest.raises(RuntimeError, match="manifest replace failed"):
        ensure_bloc_kv_artifact(provider=provider, store=store, record=record, force_rebuild=True)

    assert json.loads(first.manifest_path.read_text(encoding="utf-8")) == manifest_before
    assert _sha256_file(first.artifact_path) == artifact_hash_before


def test_bloc_kv_load_reloads_on_miss_and_preserves_default_key(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    provider = _StubPersistentMLXProvider(model="qwen3-test")
    record = _upsert_record(store, tmp_path, sha="f" * 64, path_name="doc.txt", content="hello world\n")
    provider.prompt_cache_set("keep", make_default=True)

    first = load_bloc_kv_artifact(
        provider=provider,
        store=store,
        record=record,
        stable_cache_key="stable",
        key="work",
        make_default=False,
    )
    assert first.stable_cache_key == "stable"
    assert first.forked_from == "stable"
    assert provider.load_calls == 1
    assert provider._default_prompt_cache_key == "keep"

    provider.prompt_cache_clear("stable")
    provider.prompt_cache_clear("work")

    second = load_bloc_kv_artifact(
        provider=provider,
        store=store,
        record=record,
        stable_cache_key="stable",
        key="work2",
        make_default=False,
    )
    assert second.reloaded_stable_key is True
    assert second.forked_from == "stable"
    assert provider.load_calls == 2
    assert provider._default_prompt_cache_key == "keep"


def test_bloc_kv_reloads_stable_key_when_artifact_changes(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    provider = _StubPersistentMLXProvider(model="qwen3-test")
    provider.prompt_cache_set("keep", make_default=True)

    record_a = _upsert_record(store, tmp_path, sha="7" * 64, path_name="a.txt", content="hello world\n")
    first = load_bloc_kv_artifact(
        provider=provider,
        store=store,
        record=record_a,
        stable_cache_key="stable",
        key="work-a",
        make_default=False,
    )
    assert first.reloaded_stable_key is True
    load_calls_before = provider.load_calls

    record_b = _upsert_record(store, tmp_path, sha="7" * 64, path_name="b.txt", content="hello world\n")
    second = load_bloc_kv_artifact(
        provider=provider,
        store=store,
        record=record_b,
        stable_cache_key="stable",
        key="work-b",
        make_default=False,
    )

    assert second.reloaded_stable_key is True
    assert provider.load_calls == load_calls_before + 1
    assert provider._default_prompt_cache_key == "keep"


def test_bloc_kv_reloads_stable_key_when_artifact_bytes_change_only(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    provider = _StubPersistentMLXProvider(model="qwen3-test")
    record = _upsert_record(store, tmp_path, sha="9" * 64, path_name="doc.txt", content="hello world\n")

    first = load_bloc_kv_artifact(
        provider=provider,
        store=store,
        record=record,
        stable_cache_key="stable",
        key="work-a",
        make_default=False,
    )
    assert first.reloaded_stable_key is True
    load_calls_before = provider.load_calls

    provider.save_nonce = "rebuilt"
    second = load_bloc_kv_artifact(
        provider=provider,
        store=store,
        record=record,
        stable_cache_key="stable",
        key="work-b",
        make_default=False,
        force_rebuild=True,
    )

    assert second.reloaded_stable_key is True
    assert provider.load_calls == load_calls_before + 1


def test_bloc_kv_load_failure_restores_default_key_and_cleans_working_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    provider = _StubPersistentMLXProvider(model="qwen3-test")
    record = _upsert_record(store, tmp_path, sha="1" * 64, path_name="doc.txt", content="hello world\n")
    provider.prompt_cache_set("keep", make_default=True)

    ensure_bloc_kv_artifact(provider=provider, store=store, record=record)

    def _boom(from_key: str, to_key: str, *, make_default: bool = False, ttl_s: Optional[float] = None, **kwargs: Any) -> bool:
        _ = (from_key, to_key, make_default, ttl_s, kwargs)
        raise RuntimeError("fork failed")

    monkeypatch.setattr(provider, "prompt_cache_fork", _boom)
    with pytest.raises(RuntimeError, match="fork failed"):
        load_bloc_kv_artifact(
            provider=provider,
            store=store,
            record=record,
            stable_cache_key="stable",
            key="tmp-work",
            make_default=False,
        )

    assert provider._default_prompt_cache_key == "keep"
    assert provider._prompt_cache_store.get("tmp-work") is None


def test_bloc_kv_uses_actual_bloc_content_hash_when_meta_drifted(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    provider = _StubPersistentMLXProvider(model="qwen3-test")
    record = _upsert_record(store, tmp_path, sha="8" * 64, path_name="doc.txt", content="hello world\n")

    meta_path = store.meta_path(record.sha256)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["content_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    drifted_record = store.get(record.sha256)
    assert drifted_record is not None

    result = ensure_bloc_kv_artifact(provider=provider, store=store, record=drifted_record)

    assert result.manifest.content_sha256 == _sha256_text("hello world\n")
    assert result.manifest.content_sha256 != "0" * 64


def test_generate_bloc_metadata_uses_bloc_kv_loader_and_preserves_default_key(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="2" * 64, path_name="doc.txt", content="hello world\n")
    provider = _StubPersistentMLXProvider(model="qwen3-test")
    provider.prompt_cache_set("keep", make_default=True)

    result = generate_bloc_metadata_jsonld(
        provider=provider,
        model_id=provider.model,
        stable_cache_key="stable-meta",
        kv_path=None,
        record=record,
        store=store,
        enabled=True,
    )

    assert result.ok is True
    assert isinstance(result.jsonld, dict)
    assert provider._default_prompt_cache_key == "keep"
    assert provider.last_call["system_prompt"] is None
    assert "You are generating metadata for ONE bloc of text already in context." in str(provider.last_call["prompt"])
    assert all(not key.startswith("tmp:meta:") for key in provider._prompt_cache_store.keys())
    assert store.get_jsonld(record.sha256) is not None


def test_bloc_kv_rebuilds_when_resolved_model_id_changes(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    provider = _StubPersistentMLXProvider(model="qwen3-test", resolved_model_id="/resolved/v1")
    record = _upsert_record(store, tmp_path, sha="3" * 64, path_name="doc.txt", content="hello world\n")

    first = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)
    provider._resolved_model_id = "/resolved/v2"
    second = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)

    assert first.manifest.model_resolved_id == "/resolved/v1"
    assert second.compiled is True
    assert second.manifest.model_resolved_id == "/resolved/v2"


def test_read_bloc_kv_manifest_returns_none_for_missing_manifest(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    provider = _StubPersistentMLXProvider(model="qwen3-test")
    record = _upsert_record(store, tmp_path, sha="4" * 64, path_name="doc.txt", content="hello world\n")

    assert read_bloc_kv_manifest(provider=provider, store=store, model=provider.model, record=record) is None


def test_tmp_path_preserves_original_suffix() -> None:
    artifact = Path("/tmp/example.safetensors")
    tmp = _tmp_path(artifact)
    assert tmp.suffix == ".safetensors"
    assert ".tmp." in tmp.name


def test_bloc_kv_helpers_are_public_package_exports() -> None:
    import abstractcore
    import abstractcore.core as core

    assert abstractcore.ensure_bloc_kv_artifact is ensure_bloc_kv_artifact
    assert abstractcore.load_bloc_kv_artifact is load_bloc_kv_artifact
    assert abstractcore.read_bloc_kv_manifest is read_bloc_kv_manifest
    assert abstractcore.delete_bloc_kv_artifact is delete_bloc_kv_artifact
    assert abstractcore.list_bloc_kv_artifacts is list_bloc_kv_artifacts
    assert abstractcore.prune_bloc_kv_artifacts is prune_bloc_kv_artifacts
    assert abstractcore.delete_bloc is delete_bloc
    assert core.ensure_bloc_kv_artifact is ensure_bloc_kv_artifact
    assert core.load_bloc_kv_artifact is load_bloc_kv_artifact
    assert core.read_bloc_kv_manifest is read_bloc_kv_manifest
    assert core.delete_bloc_kv_artifact is delete_bloc_kv_artifact
    assert core.list_bloc_kv_artifacts is list_bloc_kv_artifacts
    assert core.prune_bloc_kv_artifacts is prune_bloc_kv_artifacts
    assert core.delete_bloc is delete_bloc
