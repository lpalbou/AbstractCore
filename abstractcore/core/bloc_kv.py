"""Durable bloc-scoped KV artifacts for local prompt-cache backends."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
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
    return store.kv_cache_path(record.sha256, provider=_provider_name(provider), model=model)


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


def _require_mlx_provider(provider: Any) -> None:
    if _provider_name(provider) != "mlx":
        raise ValueError("Bloc KV artifacts are currently implemented only for provider='mlx'.")
    if not hasattr(provider, "_build_prompt_fragment"):
        raise ValueError("MLX bloc KV compilation requires provider._build_prompt_fragment(...).")


def _resolved_model_id(provider: Any) -> str:
    resolved = getattr(provider, "_resolved_model_id", None)
    if isinstance(resolved, str) and resolved.strip():
        return resolved.strip()
    return str(getattr(provider, "model", "") or "").strip()


def _serializer_version(provider: Any) -> str:
    model = str(getattr(provider, "model", "") or "").strip().lower()
    fmt = "qwen-chatml" if "qwen" in model else "plain-chat"
    return f"mlx-prompt-fragment/v1:{fmt}"


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


def _augment_prompt_cache_key_meta(provider: Any, key: Optional[str], **updates: Any) -> None:
    if not isinstance(key, str) or not key.strip():
        return
    store = getattr(provider, "_prompt_cache_store", None)
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


def _render_attached_file_box_recipe(*, provider: Any, record: FileBlocRecord, content: str) -> _RenderedRecipe:
    file_box = _reconstruct_file_box(record, content=content)
    file_box_prompt = render_file_box_message(file_box)
    builder = getattr(provider, "_build_prompt_fragment", None)
    if not callable(builder):
        raise ValueError("MLX bloc KV compilation requires provider._build_prompt_fragment(...).")
    serialized_prompt = str(
        builder(
            messages=[{"role": "user", "content": file_box_prompt}],
            prefilled_modules=None,
        )
        or ""
    )
    if not serialized_prompt:
        raise ValueError("Failed to serialize the bloc prompt for MLX prompt-cache compilation.")
    return _RenderedRecipe(
        recipe_id=_RECIPE_ID,
        recipe_version=_RECIPE_VERSION,
        renderer_version=_RENDERER_VERSION,
        serializer_version=_serializer_version(provider),
        path_in_prompt=str(file_box.path or ""),
        file_box_prompt=file_box_prompt,
        serialized_prompt=serialized_prompt,
        rendered_recipe_sha256=_sha256_bytes(serialized_prompt.encode("utf-8")),
    )


@dataclass(frozen=True)
class BlocKVArtifactManifest:
    version: int
    provider: str
    model: str
    model_resolved_id: str
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "provider": self.provider,
            "model": self.model,
            "model_resolved_id": self.model_resolved_id,
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
        return cls(
            version=int(data.get("version") or 0),
            provider=str(data.get("provider") or ""),
            model=str(data.get("model") or ""),
            model_resolved_id=str(data.get("model_resolved_id") or ""),
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
        )


@dataclass(frozen=True)
class BlocKVCompileResult:
    artifact_path: Path
    manifest_path: Path
    manifest: BlocKVArtifactManifest
    compiled: bool
    rebuilt: bool
    source_cache_key: Optional[str] = None


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
        "quantization": str(manifest.quantization or "fp"),
    }
    for field_name, expected_value in expected.items():
        if _meta_str(field_name) != str(expected_value).strip():
            return False
    return True


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
) -> BlocKVCompileResult:
    _require_mlx_provider(provider)
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
            )

    if not _provider_supports(provider, "save") or not _provider_supports(provider, "update"):
        raise ValueError("MLX bloc KV compilation requires prompt-cache save and update operations.")

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
                "format": "abstractcore-bloc-kv/v1",
                "provider": _provider_name(provider),
                "model": model_id,
                "model_resolved_id": _resolved_model_id(provider),
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
) -> BlocKVLoadResult:
    _require_mlx_provider(provider)
    record = _resolve_record(store=store, record=record, sha256=sha256, bloc_id=bloc_id)
    model_id = str(model or getattr(provider, "model", "") or "").strip()
    ensured = ensure_bloc_kv_artifact(
        provider=provider,
        store=store,
        model=model_id,
        record=record,
        artifact_path=artifact_path,
        force_rebuild=force_rebuild,
    )

    target_key = str(key or stable_cache_key or f"cache:bloc:{uuid.uuid4().hex[:12]}").strip()
    stable_key = str(stable_cache_key).strip() if isinstance(stable_cache_key, str) and stable_cache_key.strip() else None
    loaded = False
    reloaded_stable_key = False
    forked_from: Optional[str] = None

    if not _provider_supports(provider, "load"):
        raise ValueError("MLX bloc KV loading requires prompt-cache load support.")

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
                        manifest_version=str(ensured.manifest.version),
                        artifact_sha256=ensured.manifest.artifact_sha256,
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
                    )
                if not _provider_supports(provider, "fork"):
                    raise ValueError("MLX bloc KV forking requires prompt-cache fork support.")
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
                )

            provider.prompt_cache_load(
                str(ensured.artifact_path),
                key=target_key,
                make_default=bool(make_default),
            )
            _augment_prompt_cache_key_meta(
                provider,
                target_key,
                manifest_version=str(ensured.manifest.version),
                artifact_sha256=ensured.manifest.artifact_sha256,
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
            )
        except Exception:
            if target_key != stable_key:
                _clear_cache_key(provider, target_key)
            raise
