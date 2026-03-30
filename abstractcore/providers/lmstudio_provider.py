"""
LM Studio provider implementation (OpenAI-compatible API).

LM Studio exposes an OpenAI-compatible server (by default at `http://localhost:1234/v1`).
This provider is a thin wrapper around `OpenAICompatibleProvider` with LM Studio defaults.
"""

import time
import warnings
from typing import Any, Dict, Iterator, List, Optional, Union, Type, TYPE_CHECKING

import httpx

if TYPE_CHECKING:  # pragma: no cover
    from pydantic import BaseModel
    from ..media.types import MediaContent

from .openai_compatible_provider import OpenAICompatibleProvider
from ..core.types import GenerateResponse
from ..exceptions import ProviderAPIError


class LMStudioProvider(OpenAICompatibleProvider):
    """LM Studio provider using OpenAI-compatible API."""

    PROVIDER_ID = "lmstudio"
    PROVIDER_DISPLAY_NAME = "LMStudio"
    BASE_URL_ENV_VAR = "LMSTUDIO_BASE_URL"
    API_KEY_ENV_VAR = None
    DEFAULT_BASE_URL = "http://localhost:1234/v1"

    _TIMEOUT_UNSET = object()

    def __init__(
        self,
        model: str = "local-model",
        base_url: Optional[str] = None,
        timeout: Any = _TIMEOUT_UNSET,
        **kwargs: Any,
    ):
        # ADR-0027: avoid silent low timeouts; timeouts must be explicit and attributable.
        #
        # Semantics:
        # - If the caller explicitly provides `timeout` (including `None`), we forward it.
        # - If the caller omits `timeout`, BaseProvider will use AbstractCore config
        #   `timeouts.default_timeout` (see `~/.abstractcore/config/abstractcore.json`).
        super_kwargs = dict(kwargs)
        if timeout is not self._TIMEOUT_UNSET:
            super_kwargs["timeout"] = timeout

        super().__init__(model=model, base_url=base_url, **super_kwargs)

    def _native_rest_base_url(self) -> str:
        """Derive LM Studio native REST base URL from the OpenAI-compatible base_url."""
        base = str(getattr(self, "base_url", "") or "").strip().rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return base.rstrip("/")

    def _native_rest_chat_generate(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str],
        stream: bool,
        **kwargs: Any,
    ) -> GenerateResponse:
        """Call LM Studio native REST endpoint `POST /api/v1/chat` (best-effort).

        This endpoint is the only LM Studio surface that *documents* per-request `reasoning`
        control (`off|low|medium|high|on`). The OpenAI-compatible endpoint does not document
        `chat_template_kwargs` or `reasoning`, and may ignore them depending on backend/template.
        """
        if stream:
            raise UnsupportedOperation("LM Studio native REST streaming not implemented yet")

        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(generation_kwargs)
        reasoning = kwargs.get("reasoning")

        payload: Dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "stream": False,
            "temperature": generation_kwargs.get("temperature", self.temperature),
            "max_output_tokens": int(max_output_tokens),
        }
        if isinstance(system_prompt, str) and system_prompt.strip():
            payload["system_prompt"] = system_prompt.strip()
        if isinstance(reasoning, str) and reasoning.strip():
            payload["reasoning"] = reasoning.strip()

        request_url = f"{self._native_rest_base_url()}/api/v1/chat"
        start = time.time()
        try:
            resp = httpx.post(request_url, json=payload, timeout=self._timeout)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:  # pragma: no cover
                body = ""
                try:
                    body = resp.text or ""
                except Exception:
                    body = ""
                body = body.strip()
                if len(body) > 800:
                    body = body[:799] + "…"
                raise ProviderAPIError(
                    f"LM Studio native REST API error ({resp.status_code}) for {request_url}: {body or '(empty response body)'}"
                ) from e

            gen_time = round((time.time() - start) * 1000, 1)
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            raise ProviderAPIError(f"LM Studio native REST API error: {e}") from e

        output_items = data.get("output") if isinstance(data, dict) else None
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        if isinstance(output_items, list):
            for item in output_items:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type") or "").strip().lower()
                if item_type == "message":
                    c = item.get("content")
                    if isinstance(c, str) and c:
                        content_parts.append(c)
                elif item_type == "reasoning":
                    c = item.get("content")
                    if isinstance(c, str) and c:
                        reasoning_parts.append(c)

        content = "\n".join([p for p in content_parts if isinstance(p, str) and p.strip()]).strip()
        reasoning_text = "\n\n".join([p for p in reasoning_parts if isinstance(p, str) and p.strip()]).strip()

        stats = data.get("stats") if isinstance(data, dict) else {}
        usage = {
            "input_tokens": stats.get("input_tokens", 0) if isinstance(stats, dict) else 0,
            "output_tokens": stats.get("total_output_tokens", 0) if isinstance(stats, dict) else 0,
            "total_tokens": (
                (stats.get("input_tokens", 0) if isinstance(stats, dict) else 0)
                + (stats.get("total_output_tokens", 0) if isinstance(stats, dict) else 0)
            ),
            # Back-compat keys
            "prompt_tokens": stats.get("input_tokens", 0) if isinstance(stats, dict) else 0,
            "completion_tokens": stats.get("total_output_tokens", 0) if isinstance(stats, dict) else 0,
        }

        metadata: Dict[str, Any] = {
            "_provider_request": {"url": request_url, "payload": payload},
        }
        if reasoning_text:
            metadata["reasoning"] = reasoning_text

        return GenerateResponse(
            content=content,
            model=self.model,
            finish_reason="stop",
            raw_response=data,
            metadata=metadata,
            usage=usage,
            gen_time=gen_time,
        )

    def _generate_internal(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List["MediaContent"]] = None,
        stream: bool = False,
        response_model: Optional[Type["BaseModel"]] = None,
        execute_tools: Optional[bool] = None,
        tool_call_tags: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        # When `thinking=` is used, BaseProvider sets a best-effort `reasoning` kwarg for LM Studio.
        # Prefer the native REST endpoint which documents this control surface.
        _ = (execute_tools, tool_call_tags)
        if (
            kwargs.get("reasoning") is not None
            and tools is None
            and response_model is None
            and media is None
        ):
            try:
                # LM Studio native REST `/api/v1/chat` does not support injecting assistant history.
                # When callers provide a chat history, fall back to OpenAI-compatible mode.
                if messages:
                    roles = [m.get("role") for m in messages if isinstance(m, dict)]
                    if any(r for r in roles if r not in {None, "user"}):
                        return super()._generate_internal(
                            prompt=prompt,
                            messages=messages,
                            system_prompt=system_prompt,
                            tools=tools,
                            media=media,
                            stream=stream,
                            response_model=response_model,
                            execute_tools=execute_tools,
                            tool_call_tags=tool_call_tags,
                            **kwargs,
                        )

                return self._native_rest_chat_generate(
                    prompt=str(prompt or ""),
                    system_prompt=system_prompt,
                    stream=stream,
                    **kwargs,
                )
            except Exception as e:  # noqa: BLE001
                # Fall back to OpenAI-compatible path if the native REST endpoint is unavailable.
                if not getattr(self, "_native_rest_fallback_warned", False):
                    setattr(self, "_native_rest_fallback_warned", True)
                    warnings.warn(
                        f"LM Studio native REST fallback failed; using OpenAI-compatible endpoint instead. "
                        f"Error: {type(e).__name__}: {e}",
                        RuntimeWarning,
                        stacklevel=3,
                    )

        return super()._generate_internal(
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            media=media,
            stream=stream,
            response_model=response_model,
            execute_tools=execute_tools,
            tool_call_tags=tool_call_tags,
            **kwargs,
        )


class UnsupportedOperation(RuntimeError):
    pass
