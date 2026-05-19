"""
AbstractEndpoint (AbstractCore) - single-model OpenAI-compatible server.

Unlike `abstractcore.server.app` (multi-provider gateway), this server loads one provider+model
once per worker and reuses it across requests. It is intended for hosting local inference
backends (HF GGUF / MLX) as a `/v1` endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..core.bloc_kv import ensure_bloc_kv_artifact, load_bloc_kv_artifact, read_bloc_kv_manifest
from ..core.file_blocs import FileBlocStore
from ..core.factory import create_llm
from ..core.types import GenerateResponse
from ..providers.base import PromptCacheCapabilities, PromptCacheError


@dataclass(frozen=True)
class EndpointConfig:
    provider: str
    model: str
    host: str = "0.0.0.0"
    port: int = 8001


class ChatMessage(BaseModel):
    role: str
    content: Optional[Any] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(description="Model identifier (ignored/validated in single-model mode)")
    messages: List[ChatMessage]

    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    stream: bool = False
    thinking: Optional[Union[bool, str]] = Field(
        default=None,
        description="Unified thinking/reasoning control (best-effort across providers/models).",
    )

    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None

    stop: Optional[Any] = None
    seed: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    # OpenAI prompt caching (2025+): supported by OpenAI and forwarded by AbstractCore providers.
    prompt_cache_key: Optional[str] = None
    prompt_cache_retention: Optional[str] = None


class PromptCacheSetRequest(BaseModel):
    key: str = Field(description="Prompt cache key to create/select")
    make_default: bool = Field(default=True, description="Set this key as the default for subsequent calls")
    ttl_s: Optional[float] = Field(default=None, description="Optional in-process TTL (seconds) for this key")


class PromptCacheUpdateRequest(BaseModel):
    key: str = Field(description="Prompt cache key to update/append into")
    prompt: Optional[str] = Field(default=None, description="Raw prompt text (treated as a user message for chat templates)")
    messages: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional message list to append (provider-dependent)")
    system_prompt: Optional[str] = Field(default=None, description="Optional system prompt to append")
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional tool definitions to append")
    thinking: Optional[Union[bool, str]] = Field(
        default=None,
        description="Optional unified thinking/reasoning control to apply while preparing the cache state.",
    )
    add_generation_prompt: bool = Field(default=False, description="If true, append an assistant preamble (backend-dependent)")
    ttl_s: Optional[float] = Field(default=None, description="Optional TTL update (seconds)")


class PromptCacheForkRequest(BaseModel):
    from_key: str = Field(description="Source prompt cache key")
    to_key: str = Field(description="Destination prompt cache key")
    make_default: bool = Field(default=False, description="Set the new key as default")
    ttl_s: Optional[float] = Field(default=None, description="Optional TTL for the new key (seconds)")


class PromptCacheClearRequest(BaseModel):
    key: Optional[str] = Field(default=None, description="If omitted, clears all in-process caches for this worker")


class PromptCachePrepareModulesRequest(BaseModel):
    namespace: str = Field(description="Namespace used as a stable prefix for derived keys (e.g. tenant_id:model_id)")
    modules: List[Dict[str, Any]] = Field(description="Ordered list of cache modules (see abstractcore.providers.base.PromptCacheModule)")
    make_default: bool = Field(default=False, description="Set the final derived key as default")
    ttl_s: Optional[float] = Field(default=None, description="Optional TTL for derived keys (seconds)")
    version: int = Field(default=1, description="Hash version for key derivation (bump on formatting changes)")


class BlocUpsertTextRequest(BaseModel):
    path: str = Field(description="Logical source path for the extracted text snapshot.")
    content: str = Field(description="Extracted text content to persist in the bloc store.")
    sha256: Optional[str] = Field(default=None, description="Optional caller-supplied bloc sha256. Defaults to a hash of `content` when omitted.")
    content_sha256: Optional[str] = Field(default=None, description="Optional caller-supplied content hash. Defaults to a hash of `content` when omitted.")
    media_type: str = Field(default="text", description="Logical media type for the stored bloc.", example="text")
    size_bytes: Optional[int] = Field(default=None, description="Optional original source size in bytes. Defaults to the UTF-8 size of `content`.")
    mtime_ns: Optional[int] = Field(default=None, description="Optional source modification time in nanoseconds.")
    format: Optional[str] = Field(default=None, description="Optional source format identifier, for example `text/plain`.")
    estimated_tokens: Optional[int] = Field(default=None, description="Optional estimated token count for the stored content.")
    relpath_base: Optional[str] = Field(default=None, description="Optional base path used to derive a stable relative path in the record.")
    summary: Optional[str] = Field(default=None, description="Optional explicit summary override.")
    keywords: Optional[List[str]] = Field(default=None, description="Optional explicit keywords override.")


class BlocLookupRequest(BaseModel):
    sha256: Optional[str] = Field(default=None, description="Bloc sha256 selector.")
    bloc_id: Optional[int] = Field(default=None, description="Stable integer bloc selector.")


class BlocKVEnsureRequest(BlocLookupRequest):
    artifact_path: Optional[str] = Field(default=None, description="Optional explicit artifact path override.")
    force_rebuild: bool = Field(default=False, description="Force a rebuild even if a valid artifact already exists.")


class BlocKVLoadRequest(BlocLookupRequest):
    artifact_path: Optional[str] = Field(default=None, description="Optional explicit artifact path override.")
    stable_cache_key: Optional[str] = Field(default=None, description="Optional long-lived cache key to keep loaded for reuse.")
    key: Optional[str] = Field(default=None, description="Target cache key to load or fork into.")
    make_default: bool = Field(default=False, description="Set the loaded/forked key as the provider default.")
    force_rebuild: bool = Field(default=False, description="Force artifact rebuild before loading.")


def _config_examples() -> Dict[str, Any]:
    return {
        "examples": [
            {
                "path": "/tmp/orbit.txt",
                "content": "Document title: Orbit Notes\n\nThe launch window is Tuesday at 14:30 UTC.\n",
                "media_type": "text",
                "format": "text/plain",
            }
        ]
    }


BlocUpsertTextRequest.Config = type("Config", (), {"json_schema_extra": _config_examples()})  # type: ignore[attr-defined]
BlocKVEnsureRequest.Config = type(  # type: ignore[attr-defined]
    "Config",
    (),
    {"json_schema_extra": {"examples": [{"sha256": "abababababababababababababababababababababababababababababababab"}]}},
)
BlocKVLoadRequest.Config = type(  # type: ignore[attr-defined]
    "Config",
    (),
    {
        "json_schema_extra": {
            "examples": [
                {
                    "sha256": "abababababababababababababababababababababababababababababababab",
                    "stable_cache_key": "stable:orbit",
                    "key": "work:orbit",
                    "make_default": False,
                }
            ]
        }
    },
)


def _extract_system_prompt(messages: List[ChatMessage]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    system_parts: List[str] = []
    out: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            if isinstance(msg.content, str) and msg.content.strip():
                system_parts.append(msg.content.strip())
            continue
        out.append(msg.model_dump(exclude_none=True))

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, out


def _format_tool_calls(tool_calls: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    formatted = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        formatted.append(
            {
                "id": tc.get("id"),
                "type": tc.get("type") or "function",
                "function": {
                    "name": tc.get("name"),
                    "arguments": tc.get("arguments", ""),
                },
            }
        )
    return formatted or None


def _usage_to_openai(usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    if not isinstance(usage, dict) or not usage:
        return None
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        try:
            total_tokens = int(prompt_tokens) + int(completion_tokens)
        except Exception:
            total_tokens = 0
    return {
        "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else 0,
        "completion_tokens": int(completion_tokens) if completion_tokens is not None else 0,
        "total_tokens": int(total_tokens) if total_tokens is not None else 0,
    }


def _maybe_strip_provider_prefix(model: str) -> str:
    if not isinstance(model, str):
        return ""
    s = model.strip()
    if not s:
        return ""
    # If the prefix looks like an AbstractCore provider (first segment), strip it.
    if "/" in s:
        head, tail = s.split("/", 1)
        if head.lower() in {
            "openai",
            "anthropic",
            "openrouter",
            "portkey",
            "ollama",
            "lmstudio",
            "vllm",
            "openai-compatible",
            "huggingface",
            "mlx",
        }:
            return tail
    return s


def create_app(
    *,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    provider_factory: Optional[callable] = None,
    provider_instance: Optional[Any] = None,
    bloc_store: Optional[FileBlocStore] = None,
) -> FastAPI:
    if provider_instance is not None:
        provider = provider_instance
    else:
        if provider_factory is not None:
            provider = provider_factory()
        else:
            if not provider_name or not model:
                raise ValueError("provider_name and model are required when no provider_instance is provided")
            provider = create_llm(provider_name, model=model)

    app = FastAPI(title="AbstractEndpoint", version="0.1.0")
    lock = threading.Lock()
    provider_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="abstractendpoint-provider")
    bloc_store = bloc_store or FileBlocStore()
    created_at = int(time.time())
    model_id = getattr(provider, "model", model or "unknown")

    def _provider_call(operation):
        """Run provider work on a stable thread.

        MLX prompt caches contain stream-local state. Running cache creation, cache updates, and
        generation on the same dedicated thread prevents cross-request FastAPI worker threads from
        reusing a cache created on a different MLX stream.
        """
        return provider_executor.submit(operation).result()

    def _provider_stream(operation):
        out: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=32)

        def _run():
            try:
                chunks = operation()
                if not hasattr(chunks, "__iter__"):
                    out.put(("error", RuntimeError("provider did not return an iterator for stream=True")))
                    return
                for chunk in chunks:
                    out.put(("chunk", chunk))
            except Exception as e:
                out.put(("error", e))
            finally:
                out.put(("done", None))

        provider_executor.submit(_run)
        while True:
            kind, value = out.get()
            if kind == "done":
                break
            if kind == "error":
                yield GenerateResponse(content=f"Error: {value}", model=model_id, finish_reason="error")
                continue
            yield value

    @app.on_event("shutdown")
    def _shutdown_provider_executor():
        provider_executor.shutdown(wait=False, cancel_futures=True)

    def _bloc_error(operation: str, error: Union[Exception, str], *, status_code: int) -> JSONResponse:
        return JSONResponse(
            status_code=int(status_code),
            content={
                "ok": False,
                "operation": operation,
                "error": str(error),
            },
        )

    def _resolve_bloc_selector(*, sha256: Optional[str], bloc_id: Optional[int]):
        if isinstance(sha256, str) and sha256.strip():
            rec = bloc_store.get(sha256.strip().lower())
            if rec is not None:
                return rec
        if bloc_id is not None:
            try:
                rec = bloc_store.get_by_bloc_id(int(bloc_id))
            except Exception:
                rec = None
            if rec is not None:
                return rec
        return None

    def _prompt_cache_capabilities() -> PromptCacheCapabilities:
        getter = getattr(provider, "get_prompt_cache_capabilities", None)
        if callable(getter):
            try:
                caps = getter()
                if isinstance(caps, PromptCacheCapabilities):
                    return caps
            except Exception:
                pass

        try:
            supported = bool(getattr(provider, "supports_prompt_cache", lambda: False)())
        except Exception:
            supported = False
        if supported:
            return PromptCacheCapabilities(
                supported=True,
                mode="keyed",
                supports_set=True,
                supports_clear=True,
                supports_stats=True,
                notes=("Provider only exposed legacy prompt-cache detection.",),
            )
        return PromptCacheCapabilities()

    def _prompt_cache_error_payload(error: Exception, *, operation: str) -> Dict[str, Any]:
        caps = _prompt_cache_capabilities()
        if isinstance(error, PromptCacheError):
            payload = error.to_dict()
            payload.setdefault("capabilities", caps.to_dict())
            return {
                "supported": False,
                "operation": payload.get("operation") or operation,
                "code": payload.get("code") or "prompt_cache_error",
                "error": payload.get("message") or str(error),
                "capabilities": payload.get("capabilities") or caps.to_dict(),
            }
        return {
            "supported": False,
            "operation": operation,
            "code": "prompt_cache_error",
            "error": str(error),
            "capabilities": caps.to_dict(),
        }

    @app.get("/health")
    def health():
        return {"status": "healthy", "model": model_id}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": created_at,
                    "owned_by": "abstractendpoint",
                }
            ],
        }

    @app.get("/acore/prompt_cache/stats")
    def prompt_cache_stats():
        caps = _prompt_cache_capabilities()
        if not caps.supports_operation("stats") or not hasattr(provider, "get_prompt_cache_stats"):
            return {
                "supported": False,
                "operation": "stats",
                "code": "prompt_cache_unsupported",
                "error": "provider does not expose prompt cache stats",
                "capabilities": caps.to_dict(),
            }
        try:
            return _provider_call(
                lambda: {
                    "supported": True,
                    "operation": "stats",
                    "capabilities": caps.to_dict(),
                    "stats": provider.get_prompt_cache_stats(),
                }  # type: ignore[no-any-return]
            )
        except Exception as e:
            return _prompt_cache_error_payload(e, operation="stats")

    @app.get("/acore/prompt_cache/capabilities")
    def prompt_cache_capabilities():
        caps = _prompt_cache_capabilities()
        return {
            "supported": bool(caps.supported),
            "operation": "capabilities",
            "capabilities": caps.to_dict(),
        }

    @app.post("/acore/prompt_cache/set")
    def prompt_cache_set(req: PromptCacheSetRequest):
        caps = _prompt_cache_capabilities()
        if not caps.supports_operation("set") or not hasattr(provider, "prompt_cache_set"):
            return {
                "supported": False,
                "operation": "set",
                "code": "prompt_cache_unsupported",
                "error": "provider does not support prompt cache control plane",
                "capabilities": caps.to_dict(),
            }
        try:
            def _op():
                ok = provider.prompt_cache_set(req.key, make_default=req.make_default, ttl_s=req.ttl_s)  # type: ignore[arg-type]
                return {
                    "supported": True,
                    "operation": "set",
                    "ok": bool(ok),
                    "capabilities": caps.to_dict(),
                }

            return _provider_call(_op)
        except Exception as e:
            return _prompt_cache_error_payload(e, operation="set")

    @app.post("/acore/prompt_cache/update")
    def prompt_cache_update(req: PromptCacheUpdateRequest):
        caps = _prompt_cache_capabilities()
        if not caps.supports_operation("update") or not hasattr(provider, "prompt_cache_update"):
            return {
                "supported": False,
                "operation": "update",
                "code": "prompt_cache_unsupported",
                "error": "provider does not support prompt cache control plane",
                "capabilities": caps.to_dict(),
            }
        try:
            def _op():
                ok = provider.prompt_cache_update(  # type: ignore[arg-type]
                    req.key,
                    prompt=req.prompt or "",
                    messages=req.messages,
                    system_prompt=req.system_prompt,
                    tools=req.tools,
                    thinking=req.thinking,
                    add_generation_prompt=bool(req.add_generation_prompt),
                    ttl_s=req.ttl_s,
                )
                return {
                    "supported": True,
                    "operation": "update",
                    "ok": bool(ok),
                    "capabilities": caps.to_dict(),
                }

            return _provider_call(_op)
        except Exception as e:
            return _prompt_cache_error_payload(e, operation="update")

    @app.post("/acore/prompt_cache/fork")
    def prompt_cache_fork(req: PromptCacheForkRequest):
        caps = _prompt_cache_capabilities()
        if not caps.supports_operation("fork") or not hasattr(provider, "prompt_cache_fork"):
            return {
                "supported": False,
                "operation": "fork",
                "code": "prompt_cache_unsupported",
                "error": "provider does not support prompt cache control plane",
                "capabilities": caps.to_dict(),
            }
        try:
            def _op():
                ok = provider.prompt_cache_fork(  # type: ignore[arg-type]
                    req.from_key,
                    req.to_key,
                    make_default=bool(req.make_default),
                    ttl_s=req.ttl_s,
                )
                return {
                    "supported": True,
                    "operation": "fork",
                    "ok": bool(ok),
                    "capabilities": caps.to_dict(),
                }

            return _provider_call(_op)
        except Exception as e:
            return _prompt_cache_error_payload(e, operation="fork")

    @app.post("/acore/prompt_cache/clear")
    def prompt_cache_clear(req: PromptCacheClearRequest):
        caps = _prompt_cache_capabilities()
        if not caps.supports_operation("clear") or not hasattr(provider, "prompt_cache_clear"):
            return {
                "supported": False,
                "operation": "clear",
                "code": "prompt_cache_unsupported",
                "error": "provider does not support prompt cache control plane",
                "capabilities": caps.to_dict(),
            }
        try:
            def _op():
                ok = provider.prompt_cache_clear(req.key)  # type: ignore[arg-type]
                return {
                    "supported": True,
                    "operation": "clear",
                    "ok": bool(ok),
                    "capabilities": caps.to_dict(),
                }

            return _provider_call(_op)
        except Exception as e:
            return _prompt_cache_error_payload(e, operation="clear")

    @app.post("/acore/prompt_cache/prepare_modules")
    def prompt_cache_prepare_modules(req: PromptCachePrepareModulesRequest):
        caps = _prompt_cache_capabilities()
        if not caps.supports_operation("prepare_modules") or not hasattr(provider, "prompt_cache_prepare_modules"):
            return {
                "supported": False,
                "operation": "prepare_modules",
                "code": "prompt_cache_unsupported",
                "error": "provider does not support prompt cache module preparation",
                "capabilities": caps.to_dict(),
            }
        try:
            def _op():
                return provider.prompt_cache_prepare_modules(  # type: ignore[arg-type]
                    namespace=req.namespace,
                    modules=req.modules,
                    make_default=bool(req.make_default),
                    ttl_s=req.ttl_s,
                    version=int(req.version),
                )

            return _provider_call(_op)
        except Exception as e:
            return _prompt_cache_error_payload(e, operation="prepare_modules")

    @app.post("/acore/blocs/upsert_text")
    def bloc_upsert_text(req: BlocUpsertTextRequest):
        with lock:
            try:
                content = str(req.content or "")
                if not content:
                    return _bloc_error("upsert_text", "content is required", status_code=400)
                path = str(req.path or "").strip()
                if not path:
                    return _bloc_error("upsert_text", "path is required", status_code=400)
                content_sha256 = (
                    str(req.content_sha256).strip().lower()
                    if isinstance(req.content_sha256, str) and req.content_sha256.strip()
                    else hashlib.sha256(content.encode("utf-8")).hexdigest()
                )
                sha256 = (
                    str(req.sha256).strip().lower()
                    if isinstance(req.sha256, str) and req.sha256.strip()
                    else hashlib.sha256(content.encode("utf-8")).hexdigest()
                )
                rec = bloc_store.upsert(
                    file_meta={
                        "path": path,
                        "media_type": str(req.media_type or "text"),
                        "size_bytes": int(req.size_bytes) if req.size_bytes is not None else len(content.encode("utf-8")),
                        "mtime_ns": int(req.mtime_ns) if req.mtime_ns is not None else time.time_ns(),
                        "sha256": sha256,
                        "content_sha256": content_sha256,
                        "format": req.format,
                        "content_length": len(content),
                        "estimated_tokens": int(req.estimated_tokens) if req.estimated_tokens is not None else None,
                    },
                    content=content,
                    relpath_base=Path(os.path.expanduser(req.relpath_base)) if isinstance(req.relpath_base, str) and req.relpath_base.strip() else None,
                    summary=req.summary,
                    keywords=req.keywords,
                )
                bloc_store.ensure_bloc_ids()
                rec = bloc_store.get(rec.sha256) or rec
                return {"ok": True, "operation": "upsert_text", "record": rec.to_dict()}
            except Exception as e:
                return _bloc_error("upsert_text", e, status_code=500)

    @app.get("/acore/blocs/record")
    def bloc_record(sha256: Optional[str] = None, bloc_id: Optional[int] = None):
        with lock:
            rec = _resolve_bloc_selector(sha256=sha256, bloc_id=bloc_id)
            if rec is None:
                return _bloc_error("record", "bloc not found", status_code=404)
            return {"ok": True, "operation": "record", "record": rec.to_dict()}

    @app.get("/acore/blocs/kv/manifest")
    def bloc_kv_manifest(sha256: Optional[str] = None, bloc_id: Optional[int] = None, artifact_path: Optional[str] = None):
        with lock:
            rec = _resolve_bloc_selector(sha256=sha256, bloc_id=bloc_id)
            if rec is None:
                return _bloc_error("kv_manifest", "bloc not found", status_code=404)
            try:
                manifest = _provider_call(
                    lambda: read_bloc_kv_manifest(
                        provider=provider,
                        store=bloc_store,
                        model=model_id,
                        record=rec,
                        artifact_path=artifact_path,
                    )
                )
                if manifest is None:
                    return _bloc_error("kv_manifest", "manifest not found", status_code=404)
                return {"ok": True, "operation": "kv_manifest", "manifest": manifest.to_dict()}
            except Exception as e:
                return _bloc_error("kv_manifest", e, status_code=500)

    @app.post("/acore/blocs/kv/ensure")
    def bloc_kv_ensure(req: BlocKVEnsureRequest):
        with lock:
            rec = _resolve_bloc_selector(sha256=req.sha256, bloc_id=req.bloc_id)
            if rec is None:
                return _bloc_error("kv_ensure", "bloc not found", status_code=404)
            try:
                result = _provider_call(
                    lambda: ensure_bloc_kv_artifact(
                        provider=provider,
                        store=bloc_store,
                        model=model_id,
                        record=rec,
                        artifact_path=req.artifact_path,
                        force_rebuild=bool(req.force_rebuild),
                    )
                )
                return {
                    "ok": True,
                    "operation": "kv_ensure",
                    "artifact": {
                        "artifact_path": str(result.artifact_path),
                        "manifest_path": str(result.manifest_path),
                        "compiled": bool(result.compiled),
                        "rebuilt": bool(result.rebuilt),
                        "source_cache_key": result.source_cache_key,
                        "manifest": result.manifest.to_dict(),
                    },
                }
            except Exception as e:
                return _bloc_error("kv_ensure", e, status_code=500)

    @app.post("/acore/blocs/kv/load")
    def bloc_kv_load(req: BlocKVLoadRequest):
        with lock:
            rec = _resolve_bloc_selector(sha256=req.sha256, bloc_id=req.bloc_id)
            if rec is None:
                return _bloc_error("kv_load", "bloc not found", status_code=404)
            try:
                result = _provider_call(
                    lambda: load_bloc_kv_artifact(
                        provider=provider,
                        store=bloc_store,
                        model=model_id,
                        record=rec,
                        artifact_path=req.artifact_path,
                        stable_cache_key=req.stable_cache_key,
                        key=req.key,
                        make_default=bool(req.make_default),
                        force_rebuild=bool(req.force_rebuild),
                    )
                )
                return {
                    "ok": True,
                    "operation": "kv_load",
                    "artifact": {
                        "artifact_path": str(result.artifact_path),
                        "manifest_path": str(result.manifest_path),
                        "compiled": bool(result.compiled),
                        "loaded": bool(result.loaded),
                        "reloaded_stable_key": bool(result.reloaded_stable_key),
                        "key": result.key,
                        "stable_cache_key": result.stable_cache_key,
                        "forked_from": result.forked_from,
                        "manifest": result.manifest.to_dict(),
                    },
                }
            except Exception as e:
                return _bloc_error("kv_load", e, status_code=500)

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        requested_model = _maybe_strip_provider_prefix(request.model)
        if requested_model and requested_model != model_id:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"This endpoint serves model '{model_id}', but request asked for '{requested_model}'.",
                        "type": "invalid_request_error",
                    }
                },
            )

        system_prompt, messages = _extract_system_prompt(request.messages)

        gen_kwargs: Dict[str, Any] = {}
        if request.temperature is not None:
            gen_kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            gen_kwargs["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            gen_kwargs["top_p"] = request.top_p
        if request.thinking is not None:
            gen_kwargs["thinking"] = request.thinking
        if request.seed is not None:
            gen_kwargs["seed"] = request.seed
        if request.frequency_penalty is not None:
            gen_kwargs["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            gen_kwargs["presence_penalty"] = request.presence_penalty
        if request.stop is not None:
            gen_kwargs["stop"] = request.stop
        if isinstance(request.prompt_cache_key, str) and request.prompt_cache_key.strip():
            gen_kwargs["prompt_cache_key"] = request.prompt_cache_key.strip()
        if isinstance(request.prompt_cache_retention, str) and request.prompt_cache_retention.strip():
            gen_kwargs["prompt_cache_retention"] = request.prompt_cache_retention.strip()

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        response_created = int(time.time())

        def _non_streaming_response(resp: GenerateResponse) -> JSONResponse:
            tool_calls = _format_tool_calls(resp.tool_calls)
            message: Dict[str, Any] = {
                "role": "assistant",
                "content": resp.content,
            }
            if tool_calls:
                message["tool_calls"] = tool_calls

            body: Dict[str, Any] = {
                "id": completion_id,
                "object": "chat.completion",
                "created": response_created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": resp.finish_reason or "stop",
                    }
                ],
            }
            usage = _usage_to_openai(resp.usage)
            if usage:
                body["usage"] = usage
            return JSONResponse(content=body)

        def _event_stream(chunks: Iterable[GenerateResponse]):
            # Initial delta with role, matches OpenAI stream behavior.
            yield "data: " + json.dumps(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": response_created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
            ) + "\n\n"

            for chunk in chunks:
                delta: Dict[str, Any] = {}
                if chunk.content:
                    delta["content"] = chunk.content

                tool_calls = _format_tool_calls(chunk.tool_calls)
                if tool_calls:
                    delta["tool_calls"] = tool_calls

                if not delta:
                    continue

                yield "data: " + json.dumps(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": response_created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    }
                ) + "\n\n"

            yield "data: [DONE]\n\n"

        if request.stream:
            def _locked_event_stream():
                resp = _provider_stream(
                    lambda: provider.generate(
                        prompt="",
                        messages=messages,
                        system_prompt=system_prompt,
                        tools=request.tools,
                        stream=True,
                        **gen_kwargs,
                    )
                )
                yield from _event_stream(resp)
            return StreamingResponse(
                _locked_event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        resp = _provider_call(
            lambda: provider.generate(
                    prompt="",
                    messages=messages,
                    system_prompt=system_prompt,
                    tools=request.tools,
                    stream=False,
                    **gen_kwargs,
                )
            )

        if not isinstance(resp, GenerateResponse):
            # Defensive: structured outputs or other provider behaviors.
            resp = GenerateResponse(content=str(resp), model=model_id, finish_reason="stop")

        return _non_streaming_response(resp)

    return app


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip()
    return v if v else default


def _parse_args(argv: Optional[List[str]] = None) -> EndpointConfig:
    parser = argparse.ArgumentParser(description="AbstractEndpoint: single-model /v1 server")
    parser.add_argument("--provider", default=_env("ABSTRACTENDPOINT_PROVIDER", "mlx"))
    parser.add_argument("--model", default=_env("ABSTRACTENDPOINT_MODEL", "mlx-community/Qwen3-4B"))
    parser.add_argument("--host", default=_env("ABSTRACTENDPOINT_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(_env("ABSTRACTENDPOINT_PORT", "8001") or 8001))
    args = parser.parse_args(argv)
    return EndpointConfig(provider=args.provider, model=args.model, host=args.host, port=args.port)


def main(argv: Optional[List[str]] = None) -> None:
    cfg = _parse_args(argv)
    app = create_app(provider_name=cfg.provider, model=cfg.model)
    import uvicorn

    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="error")


if __name__ == "__main__":  # pragma: no cover
    main()
