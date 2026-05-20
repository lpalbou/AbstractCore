"""Durable bloc-scoped KV artifacts for local prompt-cache backends."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

from .file_blocs import FileBlocRecord, FileBlocStore
from .file_boxes import FileBox, render_file_box_message


_MANIFEST_VERSION = 1
_RECIPE_ID = "attached_file_box"
_RECIPE_VERSION = 1
_RENDERER_VERSION = 1


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _provider_name(provider: Any) -> str:
    return str(getattr(provider, "provider", "") or "").strip().lower()


def _artifact_path_for(
    *,
    store: FileBlocStore,
    record: FileBlocRecord,
    provider: Any,
    model: str,
    artifact_path: Optional[Union[str, Path]] = None,
) -> Path:
    if artifact_path is not None:
        return Path(artifact_path).expanduser()
    ext = ".safetensors"
    getter = getattr(provider, "prompt_cache_artifact_extension", None)
    if callable(getter):
        try:
            got = str(getter() or "").strip()
            if got:
                ext = got if got.startswith(".") else f".{got}"
        except Exception:
            ext = ".safetensors"
    return store.kv_cache_path(record.sha256, provider=_provider_name(provider), model=model, ext=ext)


def _manifest_path_for(
    *,
    store: FileBlocStore,
    record: FileBlocRecord,
    provider: Any,
    model: str,
    artifact_path: Optional[Union[str, Path]] = None,
) -> Path:
    if artifact_path is not None:
        p = Path(artifact_path).expanduser()
        stem = p.name[: -len(p.suffix)] if p.suffix else p.name
        return p.with_name(f"{stem}.manifest.json")
    return store.kv_cache_manifest_path(record.sha256, provider=_provider_name(provider), model=model)


def _tmp_path(path: Path) -> Path:
    token = uuid.uuid4().hex[:12]
    if path.suffix:
        return path.with_name(f"{path.stem}.tmp.{token}{path.suffix}")
    return path.with_name(f"{path.name}.tmp.{token}")


def _provider_supports(provider: Any, operation: str) -> bool:
    supports = getattr(provider, "prompt_cache_supports_operation", None)
    return callable(supports) and bool(supports(operation))


def _resolved_model_id(provider: Any) -> str:
    resolved = getattr(provider, "_resolved_model_id", None)
    if isinstance(resolved, str) and resolved.strip():
        return resolved.strip()
    return str(getattr(provider, "model", "") or "").strip()


def _restore_default_key(provider: Any, previous_key: Optional[str]) -> None:
    if hasattr(provider, "_default_prompt_cache_key"):
        provider._default_prompt_cache_key = previous_key  # type: ignore[attr-defined]


@contextmanager
def _preserve_default_key(provider: Any, *, enabled: bool = True):
    previous_key = getattr(provider, "_default_prompt_cache_key", None)
    try:
        yield previous_key
    finally:
        if enabled:
            _restore_default_key(provider, previous_key)


def _prompt_cache_key_exists(provider: Any, key: Optional[str]) -> bool:
    if not isinstance(key, str) or not key.strip():
        return False
    store = getattr(provider, "_prompt_cache_store", None)
    if store is None or not hasattr(store, "get"):
        return False
    try:
        return store.get(key.strip()) is not None
    except Exception:
        return False


def _clear_cache_key(provider: Any, key: Optional[str]) -> None:
    if not isinstance(key, str) or not key.strip():
        return
    if not _provider_supports(provider, "clear"):
        return
    try:
        provider.prompt_cache_clear(key.strip())
    except Exception:
        pass


def _prompt_cache_key_meta(provider: Any, key: Optional[str]) -> Dict[str, Any]:
    getter = getattr(provider, "prompt_cache_key_meta", None)
    if callable(getter):
        try:
            meta = getter(key)
            return dict(meta or {}) if isinstance(meta, dict) else {}
        except Exception:
            return {}
    if not isinstance(key, str) or not key.strip():
        return {}
    store = getattr(provider, "_prompt_cache_store", None)
    if store is None or not hasattr(store, "meta"):
        return {}
    try:
        meta = store.meta(key.strip())
        return dict(meta or {}) if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _augment_prompt_cache_key_meta(provider_obj: Any, key: Optional[str], **updates: Any) -> None:
    updater = getattr(provider_obj, "prompt_cache_update_key_meta", None)
    if callable(updater):
        try:
            if updater(key, **updates):
                return
        except Exception:
            pass
    if not isinstance(key, str) or not key.strip():
        return
    store = getattr(provider_obj, "_prompt_cache_store", None)
    if store is None:
        return
    lock = getattr(store, "_lock", None)
    entries = getattr(store, "_entries", None)
    if lock is None or entries is None:
        return
    try:
        with lock:
            entry = entries.get(key.strip())
            if entry is None:
                return
            meta = dict(getattr(entry, "meta", {}) or {})
            for field_name, field_value in updates.items():
                if field_value is not None:
                    meta[str(field_name)] = field_value
            entry.meta = meta
    except Exception:
        return


def _read_bloc_content(store: FileBlocStore, record: FileBlocRecord) -> str:
    path = store.content_path(record.sha256)
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Bloc content is unavailable for sha256='{record.sha256}': {e}") from e


def _content_sha256(content: str) -> str:
    return _sha256_bytes(str(content or "").encode("utf-8"))


def _resolve_record(
    *,
    store: FileBlocStore,
    record: Optional[FileBlocRecord] = None,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
) -> FileBlocRecord:
    if isinstance(record, FileBlocRecord) and record.sha256:
        return record
    if isinstance(sha256, str) and sha256.strip():
        resolved = store.get(sha256.strip().lower())
        if resolved is not None:
            return resolved
    if bloc_id is not None:
        try:
            resolved = store.get_by_bloc_id(int(bloc_id))
        except Exception:
            resolved = None
        if resolved is not None:
            return resolved
    raise ValueError("Could not resolve bloc record; provide record, sha256, or bloc_id.")


def _reconstruct_file_box(record: FileBlocRecord, *, content: str) -> FileBox:
    return FileBox(
        path=str(record.path or ""),
        media_type=str(record.media_type or ""),
        size_bytes=int(record.size_bytes or 0),
        mtime_ns=int(record.mtime_ns or 0),
        sha256=str(record.sha256 or ""),
        content=str(content or ""),
        content_sha256=_content_sha256(content),
        format=str(record.format) if isinstance(record.format, str) and record.format else None,
        content_length=int(record.content_length or len(content or "")),
        estimated_tokens=int(record.estimated_tokens) if isinstance(record.estimated_tokens, int) else None,
    )


@dataclass(frozen=True)
class _RenderedRecipe:
    recipe_id: str
    recipe_version: int
    renderer_version: int
    serializer_version: str
    path_in_prompt: str
    file_box_prompt: str
    serialized_prompt: str
    rendered_recipe_sha256: str
    cache_backend: str
    artifact_format: str
    provider_meta: Dict[str, Any]


def _render_attached_file_box_recipe(*, provider: Any, record: FileBlocRecord, content: str) -> _RenderedRecipe:
    file_box = _reconstruct_file_box(record, content=content)
    file_box_prompt = render_file_box_message(file_box)
    renderer = getattr(provider, "prompt_cache_render_fragment", None)
    if not callable(renderer):
        raise ValueError("Bloc KV compilation requires provider.prompt_cache_render_fragment(...).")
    rendered = renderer(
        messages=[{"role": "user", "content": file_box_prompt}],
        prefilled_modules=None,
    )
    serialized_prompt = str(getattr(rendered, "serialized_prompt", "") or "")
    if not serialized_prompt:
        raise ValueError("Provider cannot render an exact prompt-cache fragment for bloc KV compilation.")
    serializer_version = str(getattr(rendered, "serializer_version", "") or "").strip()
    if not serializer_version:
        raise ValueError("Provider did not report a serializer version for bloc KV compilation.")
    cache_backend = str(getattr(rendered, "cache_backend", "") or "").strip()
    if not cache_backend:
        getter = getattr(provider, "prompt_cache_cache_backend", None)
        cache_backend = str(getter() if callable(getter) else _provider_name(provider)).strip()
    artifact_format = str(getattr(rendered, "artifact_format", "") or "").strip()
    if not artifact_format:
        getter = getattr(provider, "prompt_cache_artifact_format", None)
        artifact_format = str(getter() if callable(getter) else "abstractcore-prompt-cache/v1").strip()
    provider_meta = getattr(rendered, "meta", None)
    return _RenderedRecipe(
        recipe_id=_RECIPE_ID,
        recipe_version=_RECIPE_VERSION,
        renderer_version=_RENDERER_VERSION,
        serializer_version=serializer_version,
        path_in_prompt=str(file_box.path or ""),
        file_box_prompt=file_box_prompt,
        serialized_prompt=serialized_prompt,
        rendered_recipe_sha256=_sha256_bytes(serialized_prompt.encode("utf-8")),
        cache_backend=cache_backend,
        artifact_format=artifact_format,
        provider_meta=dict(provider_meta or {}) if isinstance(provider_meta, dict) else {},
    )


@dataclass(frozen=True)
class BlocKVArtifactManifest:
    version: int
    provider: str
    model: str
    model_resolved_id: str
    cache_backend: str
    artifact_format: str
    bloc_sha256: str
    bloc_id: Optional[int]
    content_sha256: str
    path_in_prompt: str
    recipe_id: str
    recipe_version: int
    rendered_recipe_sha256: str
    renderer_version: int
    serializer_version: str
    artifact_filename: str
    artifact_sha256: str
    quantization: Optional[str]
    created_at: str
    token_count: Optional[int]
    binding_id: str = ""
    provider_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "provider": self.provider,
            "model": self.model,
            "model_resolved_id": self.model_resolved_id,
            "cache_backend": self.cache_backend,
            "artifact_format": self.artifact_format,
            "bloc_sha256": self.bloc_sha256,
            "bloc_id": int(self.bloc_id) if isinstance(self.bloc_id, int) and self.bloc_id > 0 else None,
            "content_sha256": self.content_sha256,
            "path_in_prompt": self.path_in_prompt,
            "recipe_id": self.recipe_id,
            "recipe_version": int(self.recipe_version),
            "rendered_recipe_sha256": self.rendered_recipe_sha256,
            "renderer_version": int(self.renderer_version),
            "serializer_version": self.serializer_version,
            "artifact_filename": self.artifact_filename,
            "artifact_sha256": self.artifact_sha256,
            "quantization": self.quantization,
            "created_at": self.created_at,
            "token_count": int(self.token_count) if isinstance(self.token_count, int) and self.token_count >= 0 else None,
            "binding_id": self.binding_id,
            "provider_meta": dict(self.provider_meta or {}),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlocKVArtifactManifest":
        bloc_id = data.get("bloc_id")
        try:
            bloc_id_i = int(bloc_id) if bloc_id is not None else None
        except Exception:
            bloc_id_i = None
        if not (isinstance(data, dict) and isinstance(data.get("provider"), str) and isinstance(data.get("model"), str)):
            raise ValueError("Invalid bloc KV manifest payload.")
        provider_meta = data.get("provider_meta")
        manifest = cls(
            version=int(data.get("version") or 0),
            provider=str(data.get("provider") or ""),
            model=str(data.get("model") or ""),
            model_resolved_id=str(data.get("model_resolved_id") or ""),
            cache_backend=str(data.get("cache_backend") or data.get("provider") or ""),
            artifact_format=str(data.get("artifact_format") or "abstractcore-prompt-cache/v1"),
            bloc_sha256=str(data.get("bloc_sha256") or ""),
            bloc_id=bloc_id_i if isinstance(bloc_id_i, int) and bloc_id_i > 0 else None,
            content_sha256=str(data.get("content_sha256") or ""),
            path_in_prompt=str(data.get("path_in_prompt") or ""),
            recipe_id=str(data.get("recipe_id") or ""),
            recipe_version=int(data.get("recipe_version") or 0),
            rendered_recipe_sha256=str(data.get("rendered_recipe_sha256") or ""),
            renderer_version=int(data.get("renderer_version") or 0),
            serializer_version=str(data.get("serializer_version") or ""),
            artifact_filename=str(data.get("artifact_filename") or ""),
            artifact_sha256=str(data.get("artifact_sha256") or ""),
            quantization=str(data.get("quantization")) if isinstance(data.get("quantization"), str) and data.get("quantization") else None,
            created_at=str(data.get("created_at") or ""),
            token_count=int(data.get("token_count")) if isinstance(data.get("token_count"), int) else None,
            binding_id=str(data.get("binding_id") or ""),
            provider_meta=dict(provider_meta or {}) if isinstance(provider_meta, dict) else {},
        )
        if not manifest.binding_id:
            return cls(
                **{
                    **manifest.__dict__,
                    "binding_id": _compute_binding_id(manifest.to_dict(), include_binding=False),
                }
            )
        return manifest


def _compute_binding_id(payload: Dict[str, Any], *, include_binding: bool = False) -> str:
    fields = {
        "version": payload.get("version"),
        "provider": payload.get("provider"),
        "model": payload.get("model"),
        "model_resolved_id": payload.get("model_resolved_id"),
        "cache_backend": payload.get("cache_backend"),
        "artifact_format": payload.get("artifact_format"),
        "bloc_sha256": payload.get("bloc_sha256"),
        "bloc_id": payload.get("bloc_id"),
        "content_sha256": payload.get("content_sha256"),
        "path_in_prompt": payload.get("path_in_prompt"),
        "recipe_id": payload.get("recipe_id"),
        "recipe_version": payload.get("recipe_version"),
        "rendered_recipe_sha256": payload.get("rendered_recipe_sha256"),
        "renderer_version": payload.get("renderer_version"),
        "serializer_version": payload.get("serializer_version"),
        "artifact_filename": payload.get("artifact_filename"),
        "artifact_sha256": payload.get("artifact_sha256"),
        "quantization": payload.get("quantization"),
    }
    if include_binding:
        fields["binding_id"] = payload.get("binding_id")
    raw = json.dumps(fields, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _debug_enabled(enabled: bool = False) -> bool:
    if enabled:
        return True
    raw = str(os.getenv("ABSTRACTCORE_BLOC_KV_DEBUG", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on", "debug"}


def _debug_payload(
    *,
    operation: str,
    manifest: BlocKVArtifactManifest,
    artifact_path: Path,
    manifest_path: Path,
    rendered: Optional[_RenderedRecipe] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "operation": operation,
        "provider": manifest.provider,
        "model": manifest.model,
        "cache_backend": manifest.cache_backend,
        "artifact_format": manifest.artifact_format,
        "artifact_path": str(artifact_path),
        "manifest_path": str(manifest_path),
        "artifact_sha256": manifest.artifact_sha256,
        "binding_id": manifest.binding_id,
        "bloc_sha256": manifest.bloc_sha256,
        "content_sha256": manifest.content_sha256,
        "rendered_recipe_sha256": manifest.rendered_recipe_sha256,
        "token_count": manifest.token_count,
    }
    if rendered is not None:
        payload["rendered"] = {
            "serializer_version": rendered.serializer_version,
            "serialized_chars": len(rendered.serialized_prompt),
            "file_box_chars": len(rendered.file_box_prompt),
            "provider_meta": dict(rendered.provider_meta or {}),
        }
    if extra:
        payload.update(dict(extra))
    return payload


def bloc_kv_binding_payload(*, manifest: BlocKVArtifactManifest, key: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "binding_id": manifest.binding_id,
        "provider": manifest.provider,
        "model": manifest.model,
        "manifest_version": str(manifest.version),
        "artifact_sha256": manifest.artifact_sha256,
        "bloc_sha256": manifest.bloc_sha256,
        "content_sha256": manifest.content_sha256,
        "rendered_recipe_sha256": manifest.rendered_recipe_sha256,
    }
    if isinstance(key, str) and key.strip():
        payload["key"] = key.strip()
    return payload


def _manifest_key_meta(*, manifest: BlocKVArtifactManifest, artifact_path: Path) -> Dict[str, Any]:
    return {
        "loaded_from": str(artifact_path),
        "provider": manifest.provider,
        "model": manifest.model,
        "model_resolved_id": manifest.model_resolved_id,
        "cache_backend": manifest.cache_backend,
        "artifact_format": manifest.artifact_format,
        "bloc_sha256": manifest.bloc_sha256,
        "content_sha256": manifest.content_sha256,
        "path_in_prompt": manifest.path_in_prompt,
        "recipe_id": manifest.recipe_id,
        "recipe_version": str(manifest.recipe_version),
        "rendered_recipe_sha256": manifest.rendered_recipe_sha256,
        "renderer_version": str(manifest.renderer_version),
        "serializer_version": manifest.serializer_version,
        "manifest_version": str(manifest.version),
        "artifact_sha256": manifest.artifact_sha256,
        "binding_id": manifest.binding_id,
        "quantization": str(manifest.quantization or "fp"),
    }


@dataclass(frozen=True)
class BlocKVCompileResult:
    artifact_path: Path
    manifest_path: Path
    manifest: BlocKVArtifactManifest
    compiled: bool
    rebuilt: bool
    source_cache_key: Optional[str] = None
    binding_id: str = ""
    prompt_cache_binding: Dict[str, Any] = field(default_factory=dict)
    debug: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class BlocKVLoadResult:
    artifact_path: Path
    manifest_path: Path
    manifest: BlocKVArtifactManifest
    key: str
    stable_cache_key: Optional[str]
    compiled: bool
    loaded: bool
    reloaded_stable_key: bool
    forked_from: Optional[str] = None
    binding_id: str = ""
    prompt_cache_binding: Dict[str, Any] = field(default_factory=dict)
    debug: Optional[Dict[str, Any]] = None


class BlocKVArtifactInUseError(ValueError):
    def __init__(self, message: str, *, live_bindings: Optional[List[Dict[str, Any]]] = None) -> None:
        super().__init__(message)
        self.live_bindings = list(live_bindings or [])


@dataclass(frozen=True)
class BlocKVDeleteResult:
    operation: str
    deleted: bool
    artifact_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    manifest: Optional[BlocKVArtifactManifest] = None
    live_bindings: List[Dict[str, Any]] = field(default_factory=list)
    cleared_keys: List[str] = field(default_factory=list)
    deleted_paths: List[str] = field(default_factory=list)
    missing_paths: List[str] = field(default_factory=list)
    dry_run: bool = False
    debug: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "operation": self.operation,
            "deleted": bool(self.deleted),
            "dry_run": bool(self.dry_run),
            "live_bindings": [dict(item) for item in self.live_bindings],
            "cleared_keys": list(self.cleared_keys),
            "deleted_paths": list(self.deleted_paths),
            "missing_paths": list(self.missing_paths),
        }
        if self.artifact_path is not None:
            out["artifact_path"] = str(self.artifact_path)
        if self.manifest_path is not None:
            out["manifest_path"] = str(self.manifest_path)
        if self.manifest is not None:
            out["manifest"] = self.manifest.to_dict()
        if self.debug is not None:
            out["debug"] = dict(self.debug)
        return out


@dataclass(frozen=True)
class BlocDeleteResult:
    operation: str
    deleted: bool
    record: Optional[FileBlocRecord] = None
    deleted_path: Optional[Path] = None
    kv_results: List[BlocKVDeleteResult] = field(default_factory=list)
    live_bindings: List[Dict[str, Any]] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "operation": self.operation,
            "deleted": bool(self.deleted),
            "dry_run": bool(self.dry_run),
            "kv_results": [item.to_dict() for item in self.kv_results],
            "live_bindings": [dict(item) for item in self.live_bindings],
        }
        if self.record is not None:
            out["record"] = self.record.to_dict()
        if self.deleted_path is not None:
            out["deleted_path"] = str(self.deleted_path)
        return out


def read_bloc_kv_manifest(
    *,
    store: FileBlocStore,
    provider: Any,
    model: str,
    record: Optional[FileBlocRecord] = None,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
    artifact_path: Optional[Union[str, Path]] = None,
) -> Optional[BlocKVArtifactManifest]:
    record = _resolve_record(store=store, record=record, sha256=sha256, bloc_id=bloc_id)
    manifest_path = _manifest_path_for(
        store=store,
        record=record,
        provider=provider,
        model=model,
        artifact_path=artifact_path,
    )
    try:
        if not manifest_path.exists():
            return None
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return BlocKVArtifactManifest.from_dict(data) if isinstance(data, dict) else None
    except Exception:
        return None


def list_bloc_kv_artifacts(
    *,
    store: FileBlocStore,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return store.list_kv_artifacts(sha256=sha256, bloc_id=bloc_id, provider=provider, model=model)


def _manifest_from_artifact_entry(entry: Dict[str, Any]) -> Optional[BlocKVArtifactManifest]:
    data = entry.get("manifest") if isinstance(entry, dict) else None
    try:
        return BlocKVArtifactManifest.from_dict(data) if isinstance(data, dict) else None
    except Exception:
        return None


def _path_from_entry(entry: Dict[str, Any], name: str) -> Optional[Path]:
    value = entry.get(name) if isinstance(entry, dict) else None
    if isinstance(value, str) and value.strip():
        return Path(value).expanduser()
    return None


def _provider_prompt_cache_keys(provider: Any) -> List[str]:
    keys: List[str] = []
    store = getattr(provider, "_prompt_cache_store", None)
    if store is not None and hasattr(store, "keys"):
        try:
            keys.extend(str(k) for k in (store.keys() or []) if isinstance(k, str) and k.strip())
        except Exception:
            pass
    stats_getter = getattr(provider, "get_prompt_cache_stats", None)
    if callable(stats_getter):
        try:
            stats = stats_getter()
            stats_keys = stats.get("keys") if isinstance(stats, dict) else None
            if isinstance(stats_keys, list):
                keys.extend(str(k) for k in stats_keys if isinstance(k, str) and k.strip())
        except Exception:
            pass
    return list(dict.fromkeys(keys))


def find_bloc_kv_live_bindings(
    *,
    provider: Any,
    manifest: Optional[BlocKVArtifactManifest] = None,
    artifact_path: Optional[Union[str, Path]] = None,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
    binding_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return prompt-cache keys whose metadata points at a bloc KV artifact."""
    artifact_s = str(Path(artifact_path).expanduser()) if artifact_path is not None else None
    out: List[Dict[str, Any]] = []
    for key in _provider_prompt_cache_keys(provider):
        meta = _prompt_cache_key_meta(provider, key)
        if not meta:
            continue
        if manifest is not None:
            if str(meta.get("binding_id") or "") != manifest.binding_id:
                continue
            if str(meta.get("artifact_sha256") or "") != manifest.artifact_sha256:
                continue
            if str(meta.get("bloc_sha256") or "") != manifest.bloc_sha256:
                continue
        if artifact_s is not None and str(meta.get("loaded_from") or "") != artifact_s:
            continue
        if isinstance(sha256, str) and sha256.strip() and str(meta.get("bloc_sha256") or "") != sha256.strip().lower():
            continue
        if bloc_id is not None:
            try:
                meta_bloc_id = int(meta.get("bloc_id"))
            except Exception:
                meta_bloc_id = None
            if meta_bloc_id != int(bloc_id):
                continue
        if isinstance(binding_id, str) and binding_id.strip() and str(meta.get("binding_id") or "") != binding_id.strip():
            continue
        out.append(
            {
                "key": key,
                "binding_id": meta.get("binding_id"),
                "loaded_from": meta.get("loaded_from"),
                "artifact_sha256": meta.get("artifact_sha256"),
                "bloc_sha256": meta.get("bloc_sha256"),
                "provider": meta.get("provider"),
                "model": meta.get("model"),
            }
        )
    return out


def _select_artifact_entries(
    *,
    store: FileBlocStore,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    artifact_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    entries = store.list_kv_artifacts(sha256=sha256, bloc_id=bloc_id, provider=provider, model=model)
    if artifact_path is None:
        return entries
    artifact_s = str(Path(artifact_path).expanduser())
    return [e for e in entries if str(e.get("artifact_path") or "") == artifact_s or str(e.get("manifest_path") or "") == artifact_s]


def delete_bloc_kv_artifact(
    *,
    store: FileBlocStore,
    provider: Optional[Any] = None,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    artifact_path: Optional[Union[str, Path]] = None,
    clear_loaded: bool = False,
    force: bool = False,
    dry_run: bool = False,
    debug: bool = False,
) -> BlocKVDeleteResult:
    provider_filter = str(provider_name or "").strip().lower()
    if provider is not None and not provider_filter:
        provider_filter = _provider_name(provider)
    model_s = str(model or getattr(provider, "model", "") or "").strip()

    entries = _select_artifact_entries(
        store=store,
        sha256=sha256,
        bloc_id=bloc_id,
        provider=provider_filter or None,
        model=model_s or None,
        artifact_path=artifact_path,
    )
    if not entries:
        raise ValueError("bloc KV artifact not found")
    if len(entries) > 1:
        raise ValueError("delete_bloc_kv_artifact requires a selector that resolves to exactly one artifact")

    entry = entries[0]
    manifest = _manifest_from_artifact_entry(entry)
    artifact = _path_from_entry(entry, "artifact_path")
    manifest_path = _path_from_entry(entry, "manifest_path")
    live: List[Dict[str, Any]] = []
    if provider is not None:
        live = find_bloc_kv_live_bindings(provider=provider, manifest=manifest, artifact_path=artifact)
    if live and not (clear_loaded or force):
        raise BlocKVArtifactInUseError("bloc KV artifact is loaded in a live prompt-cache key", live_bindings=live)

    cleared: List[str] = []
    if live and clear_loaded and provider is not None and not dry_run:
        for binding in live:
            key = binding.get("key")
            if isinstance(key, str) and key.strip():
                provider.prompt_cache_clear(key.strip())
                cleared.append(key.strip())

    deleted_paths: List[str] = []
    missing_paths: List[str] = []
    if not dry_run:
        deleted = store.delete_kv_artifact_paths(artifact_path=artifact, manifest_path=manifest_path)
        deleted_paths = list(deleted.get("deleted_paths") or [])
        missing_paths = list(deleted.get("missing_paths") or [])

    return BlocKVDeleteResult(
        operation="kv_delete",
        deleted=bool(deleted_paths) if not dry_run else False,
        artifact_path=artifact,
        manifest_path=manifest_path,
        manifest=manifest,
        live_bindings=live,
        cleared_keys=cleared,
        deleted_paths=deleted_paths,
        missing_paths=missing_paths,
        dry_run=bool(dry_run),
        debug={
            "provider_checked": bool(provider is not None),
            "force": bool(force),
            "clear_loaded": bool(clear_loaded),
        }
        if _debug_enabled(debug)
        else None,
    )


def prune_bloc_kv_artifacts(
    *,
    store: FileBlocStore,
    provider: Optional[Any] = None,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    clear_loaded: bool = False,
    force: bool = False,
    dry_run: bool = False,
    debug: bool = False,
) -> List[BlocKVDeleteResult]:
    entries = store.list_kv_artifacts(sha256=sha256, bloc_id=bloc_id, provider=provider_name, model=model)
    results: List[BlocKVDeleteResult] = []
    for entry in entries:
        artifact = entry.get("artifact_path")
        manifest = _manifest_from_artifact_entry(entry)
        result = delete_bloc_kv_artifact(
            store=store,
            provider=provider,
            sha256=manifest.bloc_sha256 if manifest is not None else sha256,
            provider_name=manifest.provider if manifest is not None else provider_name,
            model=manifest.model if manifest is not None else model,
            artifact_path=artifact if isinstance(artifact, str) else None,
            clear_loaded=clear_loaded,
            force=force,
            dry_run=dry_run,
            debug=debug,
        )
        results.append(result)
    return results


def delete_bloc(
    *,
    store: FileBlocStore,
    provider: Optional[Any] = None,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
    delete_kv: bool = True,
    clear_loaded: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> BlocDeleteResult:
    record = _resolve_record(store=store, sha256=sha256, bloc_id=bloc_id)
    kv_results: List[BlocKVDeleteResult] = []
    live: List[Dict[str, Any]] = []
    if delete_kv:
        for entry in store.list_kv_artifacts(sha256=record.sha256):
            manifest = _manifest_from_artifact_entry(entry)
            artifact = _path_from_entry(entry, "artifact_path")
            if provider is not None:
                live.extend(find_bloc_kv_live_bindings(provider=provider, manifest=manifest, artifact_path=artifact))
        if live and not (clear_loaded or force):
            raise BlocKVArtifactInUseError("bloc has loaded KV artifacts in live prompt-cache keys", live_bindings=live)
        if provider is not None:
            kv_results = prune_bloc_kv_artifacts(
                store=store,
                provider=provider,
                sha256=record.sha256,
                clear_loaded=clear_loaded,
                force=force,
                dry_run=dry_run,
            )
    if not dry_run:
        store.delete(record.sha256, delete_kv=delete_kv)
    return BlocDeleteResult(
        operation="bloc_delete",
        deleted=not bool(dry_run),
        record=record,
        deleted_path=store._bloc_dir(record.sha256),
        kv_results=kv_results,
        live_bindings=live,
        dry_run=bool(dry_run),
    )


def _validate_existing_manifest(
    *,
    provider: Any,
    store: FileBlocStore,
    record: FileBlocRecord,
    model: str,
    artifact_path: Path,
    manifest_path: Path,
) -> Optional[BlocKVArtifactManifest]:
    try:
        if not manifest_path.exists() or not artifact_path.exists():
            return None
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = BlocKVArtifactManifest.from_dict(data) if isinstance(data, dict) else None
        if manifest is None:
            return None
    except Exception:
        return None

    content = _read_bloc_content(store, record)
    content_sha256 = _content_sha256(content)
    rendered = _render_attached_file_box_recipe(provider=provider, record=record, content=content)

    if manifest.version != _MANIFEST_VERSION:
        return None
    if manifest.provider != _provider_name(provider):
        return None
    if manifest.model != model:
        return None
    if manifest.model_resolved_id != _resolved_model_id(provider):
        return None
    if manifest.cache_backend != rendered.cache_backend:
        return None
    if manifest.artifact_format != rendered.artifact_format:
        return None
    if manifest.bloc_sha256 != record.sha256:
        return None
    if (manifest.bloc_id or None) != (record.bloc_id or None):
        return None
    if manifest.content_sha256 != content_sha256:
        return None
    if manifest.path_in_prompt != rendered.path_in_prompt:
        return None
    if manifest.recipe_id != rendered.recipe_id or manifest.recipe_version != rendered.recipe_version:
        return None
    if manifest.renderer_version != rendered.renderer_version:
        return None
    if manifest.serializer_version != rendered.serializer_version:
        return None
    if manifest.rendered_recipe_sha256 != rendered.rendered_recipe_sha256:
        return None
    if manifest.artifact_filename != artifact_path.name:
        return None
    try:
        artifact_sha256 = _sha256_file(artifact_path)
    except Exception:
        return None
    if manifest.artifact_sha256 != artifact_sha256:
        return None
    if manifest.binding_id != _compute_binding_id(manifest.to_dict(), include_binding=False):
        return None
    if manifest.quantization not in {None, "", "fp"}:
        return None
    return manifest


def _save_temp_manifest(path: Path, manifest: BlocKVArtifactManifest) -> None:
    payload = json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2)
    path.write_text(payload, encoding="utf-8")


def _commit_manifest_last(*, artifact_tmp: Path, artifact_path: Path, manifest_tmp: Path, manifest_path: Path) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_backup = _tmp_path(artifact_path)
    manifest_backup = _tmp_path(manifest_path)
    artifact_backed_up = False
    manifest_backed_up = False
    artifact_committed = False
    try:
        if artifact_path.exists():
            artifact_path.replace(artifact_backup)
            artifact_backed_up = True
        if manifest_path.exists():
            manifest_path.replace(manifest_backup)
            manifest_backed_up = True
        artifact_tmp.replace(artifact_path)
        artifact_committed = True
        manifest_tmp.replace(manifest_path)
    except Exception:
        if artifact_committed:
            try:
                if artifact_path.exists():
                    artifact_path.unlink()
            except Exception:
                pass
        if artifact_backed_up:
            try:
                artifact_backup.replace(artifact_path)
            except Exception:
                pass
        if manifest_backed_up:
            try:
                manifest_backup.replace(manifest_path)
            except Exception:
                pass
        raise
    finally:
        for backup in (artifact_backup, manifest_backup):
            try:
                if backup.exists():
                    backup.unlink()
            except Exception:
                pass


def _loaded_cache_matches_manifest(
    *,
    provider: Any,
    key: str,
    manifest: BlocKVArtifactManifest,
    artifact_path: Path,
) -> bool:
    meta = _prompt_cache_key_meta(provider, key)
    if not meta:
        return False

    def _meta_str(name: str) -> str:
        value = meta.get(name)
        return str(value).strip() if value is not None else ""

    expected = {
        "loaded_from": str(artifact_path),
        "provider": manifest.provider,
        "model": manifest.model,
        "model_resolved_id": manifest.model_resolved_id,
        "cache_backend": manifest.cache_backend,
        "artifact_format": manifest.artifact_format,
        "bloc_sha256": manifest.bloc_sha256,
        "content_sha256": manifest.content_sha256,
        "path_in_prompt": manifest.path_in_prompt,
        "recipe_id": manifest.recipe_id,
        "recipe_version": str(manifest.recipe_version),
        "rendered_recipe_sha256": manifest.rendered_recipe_sha256,
        "renderer_version": str(manifest.renderer_version),
        "serializer_version": manifest.serializer_version,
        "manifest_version": str(manifest.version),
        "artifact_sha256": manifest.artifact_sha256,
        "binding_id": manifest.binding_id,
        "quantization": str(manifest.quantization or "fp"),
    }
    for field_name, expected_value in expected.items():
        if _meta_str(field_name) != str(expected_value).strip():
            return False
    return True


def _require_operations(provider: Any, operations: List[str], *, context: str) -> None:
    missing = [op for op in operations if not _provider_supports(provider, op)]
    if not missing:
        return
    raise ValueError(
        f"Bloc KV {context} requires prompt-cache operation(s): {', '.join(missing)} "
        f"(provider={_provider_name(provider) or type(provider).__name__})."
    )


def ensure_bloc_kv_artifact(
    *,
    provider: Any,
    store: FileBlocStore,
    model: Optional[str] = None,
    record: Optional[FileBlocRecord] = None,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
    artifact_path: Optional[Union[str, Path]] = None,
    force_rebuild: bool = False,
    debug: bool = False,
) -> BlocKVCompileResult:
    _require_operations(provider, ["set", "update", "save"], context="compilation")
    model_id = str(model or getattr(provider, "model", "") or "").strip()
    if not model_id:
        raise ValueError("Bloc KV compilation requires a non-empty model id.")
    record = _resolve_record(store=store, record=record, sha256=sha256, bloc_id=bloc_id)
    final_artifact_path = _artifact_path_for(
        store=store,
        record=record,
        provider=provider,
        model=model_id,
        artifact_path=artifact_path,
    )
    final_manifest_path = _manifest_path_for(
        store=store,
        record=record,
        provider=provider,
        model=model_id,
        artifact_path=artifact_path,
    )
    had_existing_artifact = final_artifact_path.exists() or final_manifest_path.exists()

    if not force_rebuild:
        existing = _validate_existing_manifest(
            provider=provider,
            store=store,
            record=record,
            model=model_id,
            artifact_path=final_artifact_path,
            manifest_path=final_manifest_path,
        )
        if existing is not None:
            return BlocKVCompileResult(
                artifact_path=final_artifact_path,
                manifest_path=final_manifest_path,
                manifest=existing,
                compiled=False,
                rebuilt=False,
                source_cache_key=None,
                binding_id=existing.binding_id,
                prompt_cache_binding=bloc_kv_binding_payload(manifest=existing),
                debug=_debug_payload(
                    operation="ensure",
                    manifest=existing,
                    artifact_path=final_artifact_path,
                    manifest_path=final_manifest_path,
                    extra={"compiled": False, "reused_existing": True},
                )
                if _debug_enabled(debug)
                else None,
            )

    content = _read_bloc_content(store, record)
    content_sha256 = _content_sha256(content)
    rendered = _render_attached_file_box_recipe(provider=provider, record=record, content=content)
    tmp_cache_key = f"tmp:bloc-kv:build:{uuid.uuid4().hex[:12]}"
    artifact_tmp = _tmp_path(final_artifact_path)
    manifest_tmp = _tmp_path(final_manifest_path)

    with _preserve_default_key(provider):
        try:
            provider.prompt_cache_set(tmp_cache_key, make_default=False)
            provider.prompt_cache_update(
                tmp_cache_key,
                messages=[{"role": "user", "content": rendered.file_box_prompt}],
            )
            artifact_tmp.parent.mkdir(parents=True, exist_ok=True)
            out_meta = {
                "format": rendered.artifact_format,
                "bloc_kv_format": "abstractcore-bloc-kv/v1",
                "provider": _provider_name(provider),
                "model": model_id,
                "model_resolved_id": _resolved_model_id(provider),
                "cache_backend": rendered.cache_backend,
                "artifact_format": rendered.artifact_format,
                "bloc_sha256": record.sha256,
                "bloc_id": record.bloc_id,
                "content_sha256": content_sha256,
                "path_in_prompt": rendered.path_in_prompt,
                "recipe_id": rendered.recipe_id,
                "recipe_version": rendered.recipe_version,
                "rendered_recipe_sha256": rendered.rendered_recipe_sha256,
                "renderer_version": rendered.renderer_version,
                "serializer_version": rendered.serializer_version,
                "quantization": "fp",
                "provider_meta": dict(rendered.provider_meta or {}),
            }
            provider.prompt_cache_save(tmp_cache_key, str(artifact_tmp), meta=out_meta)
            artifact_sha256 = _sha256_file(artifact_tmp)
            token_count = None
            token_counter = getattr(provider, "prompt_cache_token_count", None)
            if callable(token_counter):
                try:
                    tok = token_counter(tmp_cache_key)
                    token_count = int(tok) if isinstance(tok, int) and tok >= 0 else None
                except Exception:
                    token_count = None
            manifest = BlocKVArtifactManifest(
                version=_MANIFEST_VERSION,
                provider=_provider_name(provider),
                model=model_id,
                model_resolved_id=_resolved_model_id(provider),
                cache_backend=rendered.cache_backend,
                artifact_format=rendered.artifact_format,
                bloc_sha256=record.sha256,
                bloc_id=record.bloc_id,
                content_sha256=content_sha256,
                path_in_prompt=rendered.path_in_prompt,
                recipe_id=rendered.recipe_id,
                recipe_version=rendered.recipe_version,
                rendered_recipe_sha256=rendered.rendered_recipe_sha256,
                renderer_version=rendered.renderer_version,
                serializer_version=rendered.serializer_version,
                artifact_filename=final_artifact_path.name,
                artifact_sha256=artifact_sha256,
                quantization="fp",
                created_at=datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat(),
                token_count=token_count,
                provider_meta=dict(rendered.provider_meta or {}),
            )
            manifest = BlocKVArtifactManifest(
                **{
                    **manifest.__dict__,
                    "binding_id": _compute_binding_id(manifest.to_dict(), include_binding=False),
                }
            )
            _save_temp_manifest(manifest_tmp, manifest)
            _commit_manifest_last(
                artifact_tmp=artifact_tmp,
                artifact_path=final_artifact_path,
                manifest_tmp=manifest_tmp,
                manifest_path=final_manifest_path,
            )
            return BlocKVCompileResult(
                artifact_path=final_artifact_path,
                manifest_path=final_manifest_path,
                manifest=manifest,
                compiled=True,
                rebuilt=bool(had_existing_artifact),
                source_cache_key=tmp_cache_key,
                binding_id=manifest.binding_id,
                prompt_cache_binding=bloc_kv_binding_payload(manifest=manifest),
                debug=_debug_payload(
                    operation="ensure",
                    manifest=manifest,
                    artifact_path=final_artifact_path,
                    manifest_path=final_manifest_path,
                    rendered=rendered,
                    extra={"compiled": True, "rebuilt": bool(had_existing_artifact)},
                )
                if _debug_enabled(debug)
                else None,
            )
        finally:
            _clear_cache_key(provider, tmp_cache_key)
            for tmp in (artifact_tmp, manifest_tmp):
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass


def compile_bloc_kv_artifact(**kwargs: Any) -> BlocKVCompileResult:
    kwargs["force_rebuild"] = True
    return ensure_bloc_kv_artifact(**kwargs)


def load_bloc_kv_artifact(
    *,
    provider: Any,
    store: FileBlocStore,
    model: Optional[str] = None,
    record: Optional[FileBlocRecord] = None,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
    artifact_path: Optional[Union[str, Path]] = None,
    stable_cache_key: Optional[str] = None,
    key: Optional[str] = None,
    make_default: bool = False,
    force_rebuild: bool = False,
    debug: bool = False,
) -> BlocKVLoadResult:
    _require_operations(provider, ["load"], context="loading")
    record = _resolve_record(store=store, record=record, sha256=sha256, bloc_id=bloc_id)
    model_id = str(model or getattr(provider, "model", "") or "").strip()
    ensured = ensure_bloc_kv_artifact(
        provider=provider,
        store=store,
        model=model_id,
        record=record,
        artifact_path=artifact_path,
        force_rebuild=force_rebuild,
        debug=debug,
    )

    target_key = str(key or stable_cache_key or f"cache:bloc:{uuid.uuid4().hex[:12]}").strip()
    stable_key = str(stable_cache_key).strip() if isinstance(stable_cache_key, str) and stable_cache_key.strip() else None
    loaded = False
    reloaded_stable_key = False
    forked_from: Optional[str] = None

    with _preserve_default_key(provider, enabled=not make_default):
        try:
            if stable_key:
                if not _prompt_cache_key_exists(provider, stable_key) or not _loaded_cache_matches_manifest(
                    provider=provider,
                    key=stable_key,
                    manifest=ensured.manifest,
                    artifact_path=ensured.artifact_path,
                ):
                    provider.prompt_cache_load(
                        str(ensured.artifact_path),
                        key=stable_key,
                        make_default=bool(make_default and target_key == stable_key),
                    )
                    _augment_prompt_cache_key_meta(
                        provider,
                        stable_key,
                        **_manifest_key_meta(manifest=ensured.manifest, artifact_path=ensured.artifact_path),
                    )
                    loaded = True
                    reloaded_stable_key = True
                if target_key == stable_key:
                    return BlocKVLoadResult(
                        artifact_path=ensured.artifact_path,
                        manifest_path=ensured.manifest_path,
                        manifest=ensured.manifest,
                        key=stable_key,
                        stable_cache_key=stable_key,
                        compiled=ensured.compiled,
                        loaded=loaded,
                        reloaded_stable_key=reloaded_stable_key,
                        forked_from=None,
                        binding_id=ensured.manifest.binding_id,
                        prompt_cache_binding=bloc_kv_binding_payload(manifest=ensured.manifest, key=stable_key),
                        debug=_debug_payload(
                            operation="load",
                            manifest=ensured.manifest,
                            artifact_path=ensured.artifact_path,
                            manifest_path=ensured.manifest_path,
                            extra={
                                "key": stable_key,
                                "stable_cache_key": stable_key,
                                "loaded": loaded,
                                "reloaded_stable_key": reloaded_stable_key,
                                "forked_from": None,
                            },
                        )
                        if _debug_enabled(debug)
                        else None,
                    )
                if not _provider_supports(provider, "fork"):
                    raise ValueError("Bloc KV forking requires prompt-cache fork support.")
                provider.prompt_cache_fork(
                    stable_key,
                    target_key,
                    make_default=bool(make_default),
                )
                forked_from = stable_key
                return BlocKVLoadResult(
                    artifact_path=ensured.artifact_path,
                    manifest_path=ensured.manifest_path,
                    manifest=ensured.manifest,
                    key=target_key,
                    stable_cache_key=stable_key,
                    compiled=ensured.compiled,
                    loaded=loaded,
                    reloaded_stable_key=reloaded_stable_key,
                    forked_from=forked_from,
                    binding_id=ensured.manifest.binding_id,
                    prompt_cache_binding=bloc_kv_binding_payload(manifest=ensured.manifest, key=target_key),
                    debug=_debug_payload(
                        operation="load",
                        manifest=ensured.manifest,
                        artifact_path=ensured.artifact_path,
                        manifest_path=ensured.manifest_path,
                        extra={
                            "key": target_key,
                            "stable_cache_key": stable_key,
                            "loaded": loaded,
                            "reloaded_stable_key": reloaded_stable_key,
                            "forked_from": forked_from,
                        },
                    )
                    if _debug_enabled(debug)
                    else None,
                )

            provider.prompt_cache_load(
                str(ensured.artifact_path),
                key=target_key,
                make_default=bool(make_default),
            )
            _augment_prompt_cache_key_meta(
                provider,
                target_key,
                **_manifest_key_meta(manifest=ensured.manifest, artifact_path=ensured.artifact_path),
            )
            loaded = True
            return BlocKVLoadResult(
                artifact_path=ensured.artifact_path,
                manifest_path=ensured.manifest_path,
                manifest=ensured.manifest,
                key=target_key,
                stable_cache_key=None,
                compiled=ensured.compiled,
                loaded=loaded,
                reloaded_stable_key=False,
                forked_from=None,
                binding_id=ensured.manifest.binding_id,
                prompt_cache_binding=bloc_kv_binding_payload(manifest=ensured.manifest, key=target_key),
                debug=_debug_payload(
                    operation="load",
                    manifest=ensured.manifest,
                    artifact_path=ensured.artifact_path,
                    manifest_path=ensured.manifest_path,
                    extra={
                        "key": target_key,
                        "stable_cache_key": None,
                        "loaded": loaded,
                        "reloaded_stable_key": False,
                        "forked_from": None,
                    },
                )
                if _debug_enabled(debug)
                else None,
            )
        except Exception:
            if target_key != stable_key:
                _clear_cache_key(provider, target_key)
            raise
