"""
LM Studio provider implementation (OpenAI-compatible API).

LM Studio exposes an OpenAI-compatible server (by default at `http://localhost:1234/v1`).
This provider is a thin wrapper around `OpenAICompatibleProvider` with LM Studio defaults.
"""

import json
import time
import warnings
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union, Type, TYPE_CHECKING

import httpx

if TYPE_CHECKING:  # pragma: no cover
    from pydantic import BaseModel
    from ..media.types import MediaContent

from .openai_compatible_provider import OpenAICompatibleProvider
from .base import ThinkingControlHandling
from ..core.types import GenerateResponse
from ..exceptions import ProviderAPIError


class LMStudioProvider(OpenAICompatibleProvider):
    """LM Studio provider using OpenAI-compatible API."""

    PROVIDER_ID = "lmstudio"
    PROVIDER_DISPLAY_NAME = "LMStudio"
    TEXT_MODEL_RESIDENCY_CONTROL_PLANE = "server"
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

        # Register-at-first-use: makes LM Studio's model dir (report-only row)
        # and the HF hub cache visible in the machine-level data registry.
        from ..utils.data_registry import ensure_core_data_homes
        ensure_core_data_homes()

    def _native_rest_base_url(self) -> str:
        """Derive LM Studio native REST base URL from the OpenAI-compatible base_url."""
        base = str(getattr(self, "base_url", "") or "").strip().rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return base.rstrip("/")

    def _apply_provider_thinking_kwargs(
        self,
        *,
        enabled: Optional[bool],
        level: Optional[str],
        kwargs: Dict[str, Any],
    ) -> tuple[Dict[str, Any], ThinkingControlHandling]:
        """Map the unified `thinking=` control to LM Studio's native REST `reasoning` field.

        LM Studio's native REST endpoint (`POST /api/v1/chat`) is the only LM Studio surface
        that *documents* per-request reasoning control (`off|low|medium|high|on`; the accepted
        subset is model-specific). The OpenAI-compatible endpoint does not document
        `chat_template_kwargs` and ignores it for some models (verified for Gemma 4), so for
        models whose only declared surface is a chat-template kwarg — or that declare no
        surface at all — the native `reasoning` field is the authoritative control.

        Template-level semantics (assistant prefill, prompt disable tokens, budget kwargs)
        keep priority when declared: those transports are pinned by existing model behavior
        (Qwen3 family, GLM, Seed-OSS) and remain unchanged.
        """
        new_kwargs, handling = super()._apply_provider_thinking_kwargs(enabled=enabled, level=level, kwargs=kwargs)
        if enabled is None and level is None:
            return new_kwargs, handling

        # Harmony models (GPT-OSS) control reasoning via a system-prompt line handled in
        # BaseProvider; never reroute them to the native REST endpoint.
        arch = self.architecture_config if isinstance(self.architecture_config, dict) else {}
        caps = self.model_capabilities if isinstance(self.model_capabilities, dict) else {}
        msg_fmt = str(arch.get("message_format") or "").strip().lower()
        resp_fmt = str(caps.get("response_format") or "").strip().lower()
        if msg_fmt == "harmony" or resp_fmt == "harmony":
            return new_kwargs, handling

        # Only reasoning-capable models expose LM Studio's reasoning toggle; for anything else
        # the request would be rejected and the base warning ladder already reports the no-op.
        if not self._model_supports_reasoning_output():
            return new_kwargs, handling

        surfaces = self._thinking_control_surfaces()
        if surfaces.prompt_disable_token or surfaces.assistant_prefill_disable or surfaces.budget_template_kwarg:
            return new_kwargs, handling

        levels = self._model_reasoning_levels()
        handled_level = False
        value: Optional[str] = None
        if enabled is False:
            value = "off"
        elif isinstance(level, str) and level.strip():
            lvl = level.strip().lower()
            if levels and lvl in levels:
                value = lvl
                handled_level = True
            else:
                # Clamp undeclared effort levels to plain "on" (LM Studio rejects unsupported
                # settings with HTTP 400); BaseProvider warns that effort scaling is degraded.
                value = "on"
        elif enabled is True:
            value = "on"

        if value is None:
            return new_kwargs, handling

        out_kwargs = dict(new_kwargs)
        # An explicit caller-provided `reasoning=` kwarg always wins.
        out_kwargs.setdefault("reasoning", value)
        return out_kwargs, ThinkingControlHandling(
            handled_enable_disable=True,
            handled_level=handled_level or handling.handled_level,
        )

    @staticmethod
    def _native_rest_image_parts(media: Optional[List[Any]]) -> Optional[List[Dict[str, str]]]:
        """Build `/api/v1/chat` image input parts from processed media.

        Returns a (possibly empty) list of `{"type": "image", "data_url": ...}` parts, or
        ``None`` when any media item cannot be represented on the native endpoint (which
        accepts only text/image input parts — verified empirically: other part types are
        rejected with `invalid_union`).
        """
        if not media:
            return []
        parts: List[Dict[str, str]] = []
        for item in media:
            media_type = getattr(item, "media_type", None)
            media_type_value = str(getattr(media_type, "value", media_type) or "").strip().lower()
            mime_type = str(getattr(item, "mime_type", "") or "").strip()
            content = getattr(item, "content", None)
            if (
                media_type_value != "image"
                or not mime_type.lower().startswith("image/")
                or not isinstance(content, str)
                or not content.strip()
            ):
                return None
            parts.append({"type": "image", "data_url": f"data:{mime_type};base64,{content.strip()}"})
        return parts

    def _native_rest_build_chat_payload(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str],
        stream: bool,
        media: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build the `POST /api/v1/chat` payload shared by streaming and non-streaming calls."""
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(generation_kwargs)
        reasoning = kwargs.get("reasoning")

        image_parts = self._native_rest_image_parts(media)
        input_value: Any
        if image_parts:
            input_value = []
            if isinstance(prompt, str) and prompt.strip():
                input_value.append({"type": "text", "content": prompt})
            input_value.extend(image_parts)
        else:
            input_value = prompt

        payload: Dict[str, Any] = {
            "model": self.model,
            "input": input_value,
            "stream": bool(stream),
            "temperature": generation_kwargs.get("temperature", self.temperature),
            "max_output_tokens": int(max_output_tokens),
        }
        for key in ("top_p", "top_k"):
            value = generation_kwargs.get(key)
            if value is not None:
                payload[key] = value
        if isinstance(system_prompt, str) and system_prompt.strip():
            payload["system_prompt"] = system_prompt.strip()
        if isinstance(reasoning, str) and reasoning.strip():
            payload["reasoning"] = reasoning.strip()
        return payload

    @staticmethod
    def _native_rest_usage_from_stats(stats: Any) -> Dict[str, Any]:
        """Normalize native REST `stats` into the AbstractCore usage dict.

        `reasoning_output_tokens` is preserved as `completion_tokens_details.reasoning_tokens`
        so reasoning billing evidence stays observable (matching the OpenAI-compatible path).
        """
        stats = stats if isinstance(stats, dict) else {}
        input_tokens = int(stats.get("input_tokens", 0) or 0)
        output_tokens = int(stats.get("total_output_tokens", 0) or 0)
        usage: Dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            # Back-compat keys
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
        }
        reasoning_tokens = stats.get("reasoning_output_tokens")
        if isinstance(reasoning_tokens, (int, float)) and not isinstance(reasoning_tokens, bool):
            usage["completion_tokens_details"] = {"reasoning_tokens": int(reasoning_tokens)}
        return usage

    def _native_rest_chat_generate(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str],
        stream: bool,
        media: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> GenerateResponse:
        """Call LM Studio native REST endpoint `POST /api/v1/chat` (non-streaming).

        This endpoint is the only LM Studio surface that *documents* per-request `reasoning`
        control (`off|low|medium|high|on`; accepted subset is model-specific). The
        OpenAI-compatible endpoint does not document `chat_template_kwargs` or `reasoning`,
        and may ignore them depending on backend/template.
        """
        _ = stream
        payload = self._native_rest_build_chat_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            stream=False,
            media=media,
            **kwargs,
        )

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

        usage = self._native_rest_usage_from_stats(data.get("stats") if isinstance(data, dict) else None)

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

    @staticmethod
    def _iter_native_sse_events(lines: Iterable[Any]) -> Iterator[Dict[str, Any]]:
        """Parse LM Studio native REST SSE lines into event dicts (with a `type` key)."""
        event_name: Optional[str] = None
        for raw in lines:
            line = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            line = line.strip()
            if not line:
                continue
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
                continue
            if not line.startswith("data:"):
                continue
            try:
                data = json.loads(line[len("data:"):].strip())
            except Exception:
                continue
            if isinstance(data, dict):
                if not data.get("type") and event_name:
                    data["type"] = event_name
                yield data

    def _native_rest_chat_stream(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str],
        media: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Iterator[GenerateResponse]:
        """Stream from `POST /api/v1/chat` (SSE).

        LM Studio emits typed events (`reasoning.delta`, `message.delta`, `chat.end`, ...)
        so reasoning arrives pre-separated: message deltas map to chunk `content`, reasoning
        deltas to chunk `metadata["reasoning"]`, and `chat.end` carries the aggregated stats
        (incl. `reasoning_output_tokens`).

        The HTTP stream is opened (and its status checked) eagerly so connection/validation
        failures surface to the caller before any chunk is yielded — the routing gate can
        then fall back to the OpenAI-compatible endpoint with an explicit warning.
        """
        payload = self._native_rest_build_chat_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            stream=True,
            media=media,
            **kwargs,
        )
        request_url = f"{self._native_rest_base_url()}/api/v1/chat"

        stream_cm = httpx.stream("POST", request_url, json=payload, timeout=self._timeout)
        try:
            resp = stream_cm.__enter__()
        except Exception as e:  # noqa: BLE001
            raise ProviderAPIError(f"LM Studio native REST API error: {e}") from e

        try:
            if resp.status_code >= 400:
                body = ""
                try:
                    resp.read()
                    body = (resp.text or "").strip()
                except Exception:
                    body = ""
                if len(body) > 800:
                    body = body[:799] + "…"
                raise ProviderAPIError(
                    f"LM Studio native REST API error ({resp.status_code}) for {request_url}: {body or '(empty response body)'}"
                )
        except BaseException:
            stream_cm.__exit__(None, None, None)
            raise

        def _chunks() -> Iterator[GenerateResponse]:
            try:
                for event in self._iter_native_sse_events(resp.iter_lines()):
                    event_type = str(event.get("type") or "").strip().lower()
                    if event_type == "message.delta":
                        delta = event.get("content")
                        if isinstance(delta, str) and delta:
                            yield GenerateResponse(content=delta, model=self.model, raw_response=event)
                    elif event_type == "reasoning.delta":
                        delta = event.get("content")
                        if isinstance(delta, str) and delta:
                            yield GenerateResponse(
                                content="",
                                model=self.model,
                                metadata={"reasoning": delta},
                                raw_response=event,
                            )
                    elif event_type == "error":
                        # Per LM Studio docs the final aggregated payload is still delivered
                        # in `chat.end`; surface the error loudly without killing the stream.
                        warnings.warn(
                            f"#FALLBACK: LM Studio native REST stream reported an error event: "
                            f"{json.dumps(event, ensure_ascii=False)[:300]}; continuing to chat.end.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    elif event_type == "chat.end":
                        result = event.get("result") if isinstance(event.get("result"), dict) else {}
                        yield GenerateResponse(
                            content="",
                            model=self.model,
                            finish_reason="stop",
                            usage=self._native_rest_usage_from_stats(result.get("stats")),
                            raw_response=result,
                            metadata={"_provider_request": {"url": request_url, "payload": payload}},
                        )
            finally:
                stream_cm.__exit__(None, None, None)

        return _chunks()

    def unload_model(self, model_name: str) -> None:
        """Best-effort unload via LM Studio native REST (`POST /api/v1/models/unload`)."""
        target = str(model_name or getattr(self, "model", "") or "").strip()
        if target:
            try:
                self._native_rest_unload_model(target)
            except Exception as e:
                # Unload must remain best-effort; fall back to closing clients.
                if hasattr(self, "logger"):
                    self.logger.debug(f"LM Studio native REST unload failed for {target!r}: {e}")

        super().unload_model(model_name)

    def load_model(self, model_name: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Load/warm a model through LM Studio native REST (`POST /api/v1/models/load`)."""
        _ = kwargs
        target = str(model_name or getattr(self, "model", "") or "").strip()
        if not target:
            raise ValueError("model_name is required")

        load_url = f"{self._native_rest_base_url()}/api/v1/models/load"
        resp = httpx.post(
            load_url,
            json={"model": target},
            headers=self._get_headers(),
            timeout=self._timeout,
        )
        try:
            data = resp.json()
        except Exception:
            data = None

        if isinstance(data, dict) and data.get("error"):
            raise ProviderAPIError(str(data.get("error")))

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = ""
            try:
                body = resp.text or ""
            except Exception:
                body = ""
            raise ProviderAPIError(
                f"LM Studio native REST load failed ({resp.status_code}) for {target!r}: {body[:800]}"
            ) from e

        return {
            "supported": True,
            "operation": "load",
            "provider": "lmstudio",
            "model": target,
            "source": "abstractcore.provider.lmstudio.native_rest",
            "raw": data if isinstance(data, dict) else {},
        }

    def _native_rest_unload_model(self, target: str) -> None:
        """Unload a model by instance id, resolving model keys to loaded instances when needed."""
        unload_url = f"{self._native_rest_base_url()}/api/v1/models/unload"
        headers = self._get_headers()

        if self._post_native_unload(unload_url, target, headers=headers):
            return

        instance_ids = self._native_rest_loaded_instance_ids_for_model(target)
        for instance_id in instance_ids:
            self._post_native_unload(unload_url, instance_id, headers=headers, raise_on_failure=True)

    def _post_native_unload(
        self,
        unload_url: str,
        instance_id: str,
        *,
        headers: Dict[str, str],
        raise_on_failure: bool = False,
    ) -> bool:
        resp = httpx.post(
            unload_url,
            json={"instance_id": instance_id},
            headers=headers,
            timeout=self._timeout,
        )
        try:
            data = resp.json()
        except Exception:
            data = None

        if isinstance(data, dict) and data.get("error"):
            if raise_on_failure:
                raise ProviderAPIError(str(data.get("error")))
            return False

        try:
            resp.raise_for_status()
        except Exception:
            if raise_on_failure:
                raise
            return False

        return True

    def _native_rest_loaded_instance_ids_for_model(self, target: str) -> List[str]:
        """Resolve an LM Studio model key/variant/display id to currently loaded instance ids."""
        needle = str(target or "").strip().lower()
        if not needle:
            return []

        url = f"{self._native_rest_base_url()}/api/v1/models"
        resp = httpx.get(url, headers=self._get_headers(), timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()

        items: Any = None
        if isinstance(data, dict):
            items = data.get("models") or data.get("data") or data.get("items")
        if not isinstance(items, list):
            return []

        def _coerce_str(value: Any) -> str:
            return value.strip() if isinstance(value, str) else ""

        def _candidate_names(item: Dict[str, Any]) -> List[str]:
            names: List[str] = []
            for key in ("key", "id", "model", "name", "model_id", "modelId", "display_name", "selected_variant"):
                value = _coerce_str(item.get(key))
                if value:
                    names.append(value)

            variants = item.get("variants")
            if isinstance(variants, list):
                names.extend(v.strip() for v in variants if isinstance(v, str) and v.strip())

            nested = item.get("model") if isinstance(item.get("model"), dict) else None
            if isinstance(nested, dict):
                for key in ("key", "id", "name", "identifier"):
                    value = _coerce_str(nested.get(key))
                    if value:
                        names.append(value)
            return names

        out: List[str] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            names = [name.lower() for name in _candidate_names(item)]
            if not any(needle == name or needle in name for name in names):
                continue

            direct_instance_id = _coerce_str(item.get("instance_id") or item.get("instanceId") or item.get("instance"))
            if direct_instance_id and direct_instance_id not in seen:
                out.append(direct_instance_id)
                seen.add(direct_instance_id)

            loaded_instances = item.get("loaded_instances") or item.get("loadedInstances")
            if not isinstance(loaded_instances, list):
                continue
            for inst in loaded_instances:
                if not isinstance(inst, dict):
                    continue
                instance_id = _coerce_str(inst.get("id") or inst.get("instance_id") or inst.get("instanceId"))
                if instance_id and instance_id not in seen:
                    out.append(instance_id)
                    seen.add(instance_id)
        return out

    def get_model_residency(self, *, task: str = "text_generation", model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Return LM Studio loaded-instance truth through the Core provider boundary."""
        _ = kwargs
        task_s = str(task or "text_generation").strip() or "text_generation"
        model_s = str(model or self.model or "").strip()
        try:
            instance_ids = self._native_rest_loaded_instance_ids_for_model(model_s)
        except Exception as e:  # noqa: BLE001
            return {
                "task": task_s,
                "provider": "lmstudio",
                "model": model_s,
                "provider_residency_verified": False,
                "provider_resident": None,
                "loaded": False,
                "state": "provider_residency_unknown",
                "source": "abstractcore.provider.lmstudio.native_rest",
                "warnings": [f"LM Studio loaded-instance query failed: {e}"],
            }

        loaded = bool(instance_ids)
        return {
            "task": task_s,
            "provider": "lmstudio",
            "model": model_s,
            "provider_residency_verified": True,
            "provider_resident": loaded,
            "provider_instance_ids": instance_ids,
            "loaded": loaded,
            "state": "loaded" if loaded else "not_loaded",
            "source": "abstractcore.provider.lmstudio.native_rest",
        }

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
        # `LMStudioProvider._apply_provider_thinking_kwargs` maps the unified `thinking=` control
        # to the native REST `reasoning` kwarg (callers may also pass `reasoning=` explicitly).
        # The native REST endpoint is the only LM Studio surface that documents this control.
        # It supports streaming (typed SSE events) and text+image input parts, but rejects
        # custom `tools`, assistant-history messages, and `response_format` with HTTP 400
        # (verified live 2026-07-07; matches the official /api/v1/chat feature table).
        _ = (execute_tools, tool_call_tags)
        reasoning_value = kwargs.get("reasoning")
        if reasoning_value is not None:
            image_parts = self._native_rest_image_parts(media)
            route_blockers: List[str] = []
            if tools is not None:
                route_blockers.append("custom tools")
            if response_model is not None:
                route_blockers.append("structured output (response_format)")
            if messages:
                # /api/v1/chat statefulness is via response_id; request-side message replay
                # (incl. all-user multi-turn) is not accepted.
                route_blockers.append("multi-message conversation history")
            if media and image_parts is None:
                route_blockers.append("non-image media")

            if route_blockers:
                # ADR-0001 (no silent degradation): the native reasoning control cannot ride this
                # request. State what actually happens instead of silently dropping the control.
                kwargs = dict(kwargs)
                kwargs.pop("reasoning", None)
                ctk = kwargs.get("chat_template_kwargs")
                has_template_artifact = isinstance(ctk, dict) and bool(ctk)
                if not getattr(self, "_native_reasoning_route_warned", False):
                    setattr(self, "_native_reasoning_route_warned", True)
                    blockers = ", ".join(route_blockers)
                    if has_template_artifact:
                        warnings.warn(
                            f"LM Studio native REST reasoning={reasoning_value!r} cannot be applied: "
                            f"/api/v1/chat does not accept {blockers}. Falling back to the "
                            "OpenAI-compatible endpoint with best-effort chat_template_kwargs only — "
                            "the server may ignore them.",
                            RuntimeWarning,
                            stacklevel=3,
                        )
                    else:
                        warnings.warn(
                            f"LM Studio native REST reasoning={reasoning_value!r} cannot be applied: "
                            f"/api/v1/chat does not accept {blockers}. No thinking control artifact "
                            "remains in the request and the model/server default thinking behavior "
                            "stays in effect.",
                            RuntimeWarning,
                            stacklevel=3,
                        )
            else:
                try:
                    if stream:
                        return self._native_rest_chat_stream(
                            prompt=str(prompt or ""),
                            system_prompt=system_prompt,
                            media=media,
                            **kwargs,
                        )
                    return self._native_rest_chat_generate(
                        prompt=str(prompt or ""),
                        system_prompt=system_prompt,
                        stream=False,
                        media=media,
                        **kwargs,
                    )
                except Exception as e:  # noqa: BLE001
                    # Fall back to OpenAI-compatible path if the native REST endpoint is unavailable.
                    kwargs = dict(kwargs)
                    kwargs.pop("reasoning", None)
                    if not getattr(self, "_native_rest_fallback_warned", False):
                        setattr(self, "_native_rest_fallback_warned", True)
                        warnings.warn(
                            f"LM Studio native REST request failed; using the OpenAI-compatible endpoint instead "
                            f"WITHOUT the native reasoning={reasoning_value!r} control (best-effort "
                            f"chat_template_kwargs may still apply). Error: {type(e).__name__}: {e}",
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
