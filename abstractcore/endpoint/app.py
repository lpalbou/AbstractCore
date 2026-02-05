"""
AbstractEndpoint (AbstractCore) - single-model OpenAI-compatible server.

Unlike `abstractcore.server.app` (multi-provider gateway), this server loads one provider+model
once per worker and reuses it across requests. It is intended for hosting local inference
backends (HF GGUF / MLX) as a `/v1` endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..core.factory import create_llm
from ..core.types import GenerateResponse


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

    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None

    stop: Optional[Any] = None
    seed: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    # OpenAI prompt caching (2025+): supported by OpenAI and forwarded by AbstractCore providers.
    prompt_cache_key: Optional[str] = None


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
    created_at = int(time.time())
    model_id = getattr(provider, "model", model or "unknown")

    def _has_cache_api() -> bool:
        return bool(getattr(provider, "supports_prompt_cache", lambda: False)())

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
        if not _has_cache_api() or not hasattr(provider, "get_prompt_cache_stats"):
            return {"supported": False, "error": "provider does not expose prompt cache stats"}
        with lock:
            try:
                return {"supported": True, "stats": provider.get_prompt_cache_stats()}  # type: ignore[no-any-return]
            except Exception as e:
                return {"supported": False, "error": str(e)}

    @app.post("/acore/prompt_cache/set")
    def prompt_cache_set(req: PromptCacheSetRequest):
        if not _has_cache_api() or not hasattr(provider, "prompt_cache_set"):
            return {"supported": False, "error": "provider does not support prompt cache control plane"}
        with lock:
            try:
                ok = provider.prompt_cache_set(req.key, make_default=req.make_default, ttl_s=req.ttl_s)  # type: ignore[arg-type]
                return {"supported": True, "ok": bool(ok)}
            except Exception as e:
                return {"supported": False, "error": str(e)}

    @app.post("/acore/prompt_cache/update")
    def prompt_cache_update(req: PromptCacheUpdateRequest):
        if not _has_cache_api() or not hasattr(provider, "prompt_cache_update"):
            return {"supported": False, "error": "provider does not support prompt cache control plane"}
        with lock:
            try:
                ok = provider.prompt_cache_update(  # type: ignore[arg-type]
                    req.key,
                    prompt=req.prompt or "",
                    messages=req.messages,
                    system_prompt=req.system_prompt,
                    tools=req.tools,
                    add_generation_prompt=bool(req.add_generation_prompt),
                    ttl_s=req.ttl_s,
                )
                return {"supported": True, "ok": bool(ok)}
            except Exception as e:
                return {"supported": False, "error": str(e)}

    @app.post("/acore/prompt_cache/fork")
    def prompt_cache_fork(req: PromptCacheForkRequest):
        if not _has_cache_api() or not hasattr(provider, "prompt_cache_fork"):
            return {"supported": False, "error": "provider does not support prompt cache control plane"}
        with lock:
            try:
                ok = provider.prompt_cache_fork(  # type: ignore[arg-type]
                    req.from_key,
                    req.to_key,
                    make_default=bool(req.make_default),
                    ttl_s=req.ttl_s,
                )
                return {"supported": True, "ok": bool(ok)}
            except Exception as e:
                return {"supported": False, "error": str(e)}

    @app.post("/acore/prompt_cache/clear")
    def prompt_cache_clear(req: PromptCacheClearRequest):
        if not _has_cache_api() or not hasattr(provider, "prompt_cache_clear"):
            return {"supported": False, "error": "provider does not support prompt cache control plane"}
        with lock:
            try:
                ok = provider.prompt_cache_clear(req.key)  # type: ignore[arg-type]
                return {"supported": True, "ok": bool(ok)}
            except Exception as e:
                return {"supported": False, "error": str(e)}

    @app.post("/acore/prompt_cache/prepare_modules")
    def prompt_cache_prepare_modules(req: PromptCachePrepareModulesRequest):
        if not _has_cache_api() or not hasattr(provider, "prompt_cache_prepare_modules"):
            return {"supported": False, "error": "provider does not support prompt cache module preparation"}
        with lock:
            try:
                result = provider.prompt_cache_prepare_modules(  # type: ignore[arg-type]
                    namespace=req.namespace,
                    modules=req.modules,
                    make_default=bool(req.make_default),
                    ttl_s=req.ttl_s,
                    version=int(req.version),
                )
                return result
            except Exception as e:
                return {"supported": False, "error": str(e)}

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

        with lock:
            resp = provider.generate(
                prompt="",
                messages=messages,
                system_prompt=system_prompt,
                tools=request.tools,
                stream=request.stream,
                **gen_kwargs,
            )

        if request.stream:
            if not hasattr(resp, "__iter__"):
                raise HTTPException(status_code=500, detail="provider did not return an iterator for stream=True")
            return StreamingResponse(
                _event_stream(resp),  # type: ignore[arg-type]
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
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
