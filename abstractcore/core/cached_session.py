"""
CachedSession: BasicSession + provider prompt caching.

Goal: reuse stable prefixes (system/tools) and enable delta-only generation where supported.
"""

from __future__ import annotations

from collections.abc import Generator as GeneratorABC
from collections.abc import Iterator as IteratorABC
import hashlib
from pathlib import Path
import uuid
import warnings
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union

from .session import BasicSession
from .types import GenerateResponse
from .enums import MessageRole
from .file_boxes import attached_file_dedupe_key, extract_file_box, render_file_box_message


class CachedSession(BasicSession):
    """Session with best-effort prompt caching.

    Modes:
    - off: behaves like BasicSession
    - key: keeps a stable `prompt_cache_key` for the session (server-managed or local prefix reuse)
    - kv: cache-as-source-of-truth (send only deltas) when supported by the provider (MLX)
    """

    def __init__(
        self,
        provider=None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        *,
        prompt_cache_strategy: str = "auto",
        prompt_cache_key: Optional[str] = None,
        prompt_cache_namespace: Optional[str] = None,
        prompt_cache_ttl_s: Optional[float] = None,
        prompt_cache_use_modules: Optional[bool] = None,
        **kwargs: Any,
    ):
        super().__init__(provider=provider, system_prompt=system_prompt, tools=tools, **kwargs)

        self.prompt_cache_strategy = str(prompt_cache_strategy or "auto").strip().lower() or "auto"
        self.prompt_cache_mode: str = "off"  # off | key | kv
        self.prompt_cache_key: Optional[str] = str(prompt_cache_key).strip() if isinstance(prompt_cache_key, str) and prompt_cache_key.strip() else None
        self.prompt_cache_namespace: Optional[str] = (
            str(prompt_cache_namespace).strip() if isinstance(prompt_cache_namespace, str) and prompt_cache_namespace.strip() else None
        )
        self.prompt_cache_ttl_s = prompt_cache_ttl_s
        self.prompt_cache_use_modules = prompt_cache_use_modules
        self._prompt_cache_prefix_key: Optional[str] = None
        self._prompt_cache_prefix_namespace: Optional[str] = None
        self._prompt_cache_prefix_modules: Dict[str, Dict[str, Any]] = {}
        # Key-mode bookkeeping: how many transcript messages (excluding system) have been
        # appended into the provider cache via `prompt_cache_update`.
        self._prompt_cache_key_mode_synced_messages: int = 0

        self._init_prompt_caching()

    def _supports_prompt_cache(self) -> bool:
        try:
            fn = getattr(self.provider, "supports_prompt_cache", None)
            return bool(fn and fn())
        except Exception:
            return False

    def _supports_delta_only_generation(self) -> bool:
        """Return True when the provider can treat the prompt cache as context source-of-truth."""
        try:
            fn = getattr(self.provider, "prompt_cache_supports_kv_source_of_truth", None)
            if callable(fn):
                return bool(fn())
        except Exception:
            return False
        return False

    def _default_prompt_cache_namespace(self) -> str:
        provider_name = str(getattr(self.provider, "provider", "") or self.provider.__class__.__name__).strip().lower()
        model_id = str(getattr(self.provider, "model", "") or "").strip()
        model_hash = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:16] if model_id else "unknown"
        return f"acore:{provider_name}:{model_hash}"

    def _tool_schemas_for_fingerprinting(self) -> Optional[List[Dict[str, Any]]]:
        if not self.tools:
            return None
        out: List[Dict[str, Any]] = []
        for tool in self.tools:
            if isinstance(tool, dict):
                out.append(dict(tool))
                continue
            if hasattr(tool, "to_dict") and callable(getattr(tool, "to_dict")):
                try:
                    td = tool.to_dict()
                    if isinstance(td, dict):
                        out.append(dict(td))
                        continue
                except Exception:
                    pass
        return out or None

    def _kv_prefilled_modules(self) -> Optional[tuple[str, ...]]:
        prefilled: List[str] = []
        if isinstance(getattr(self, "system_prompt", None), str) and str(self.system_prompt).strip():
            prefilled.append("system")
        if getattr(self, "tools", None):
            prefilled.append("tools")
        return tuple(prefilled) if prefilled else None

    def _prepare_prefix_cache(self, *, session_key: str) -> bool:
        """Best-effort: prefill stable modules (system/tools) into the provider cache."""
        provider = self.provider
        if provider is None:
            return False

        # Clear any prior prefix bookkeeping; this method may be called during rebuilds.
        self._prompt_cache_prefix_key = None
        self._prompt_cache_prefix_namespace = None
        self._prompt_cache_prefix_modules = {}

        try:
            caps = getattr(provider, "get_prompt_cache_capabilities", None)
            caps = caps() if callable(caps) else None
        except Exception:
            caps = None

        use_modules = self.prompt_cache_use_modules
        if use_modules is None:
            use_modules = bool(getattr(caps, "supports_prepare_modules", False))

        system_prompt = self.system_prompt
        tools_schema = self._tool_schemas_for_fingerprinting()

        if use_modules and bool(getattr(caps, "supports_prepare_modules", False)):
            try:
                from ..providers.base import PromptCacheModule

                modules: List[PromptCacheModule] = []
                if isinstance(system_prompt, str) and system_prompt.strip():
                    modules.append(PromptCacheModule(module_id="system", system_prompt=system_prompt))
                if tools_schema:
                    modules.append(PromptCacheModule(module_id="tools", tools=tools_schema))
                if modules:
                    namespace = self.prompt_cache_namespace or self._default_prompt_cache_namespace()
                    prepared = provider.prompt_cache_prepare_modules(  # type: ignore[attr-defined]
                        namespace=namespace,
                        modules=modules,
                        make_default=False,
                        ttl_s=self.prompt_cache_ttl_s,
                    )
                    self._prompt_cache_prefix_namespace = namespace
                    self._prompt_cache_prefix_modules = {}
                    if isinstance(prepared, dict):
                        for item in prepared.get("modules") or []:
                            if not isinstance(item, dict):
                                continue
                            module_id = str(item.get("module_id") or "").strip()
                            if module_id:
                                self._prompt_cache_prefix_modules[module_id] = dict(item)
                    prefix_key = prepared.get("final_cache_key") if isinstance(prepared, dict) else None
                    if isinstance(prefix_key, str) and prefix_key.strip():
                        provider.prompt_cache_fork(  # type: ignore[attr-defined]
                            prefix_key.strip(),
                            session_key,
                            make_default=True,
                            ttl_s=self.prompt_cache_ttl_s,
                        )
                        self._prompt_cache_prefix_key = prefix_key.strip()
                        return True
            except Exception:
                # Fall back to direct set+update below.
                pass
            finally:
                if not self._prompt_cache_prefix_key:
                    self._prompt_cache_prefix_namespace = None
                    self._prompt_cache_prefix_modules = {}

        # Fallback: direct per-session cache key initialization.
        self._prompt_cache_prefix_namespace = None
        self._prompt_cache_prefix_modules = {}
        try:
            try:
                provider.prompt_cache_set(session_key, make_default=True, ttl_s=self.prompt_cache_ttl_s)  # type: ignore[arg-type]
            except TypeError:
                provider.prompt_cache_set(session_key, make_default=True)  # type: ignore[arg-type]
        except Exception:
            return False

        # Best-effort prefill (only for providers that expose local update semantics).
        try:
            supports_update = bool(getattr(provider, "prompt_cache_supports_operation", lambda _op: False)("update"))
        except Exception:
            supports_update = False
        if supports_update:
            try:
                provider.prompt_cache_update(  # type: ignore[attr-defined]
                    session_key,
                    system_prompt=system_prompt,
                    tools=tools_schema,
                    add_generation_prompt=False,
                    ttl_s=self.prompt_cache_ttl_s,
                )
            except Exception:
                # Keep key-only mode if the backend can't prefill (still useful for server-managed caches).
                pass

        return True

    def _init_prompt_caching(self) -> None:
        provider = self.provider
        if provider is None:
            return

        if self.prompt_cache_strategy in {"off", "none", "false", "0"}:
            self.prompt_cache_mode = "off"
            return

        if not self._supports_prompt_cache():
            self.prompt_cache_mode = "off"
            return

        delta_ok = self._supports_delta_only_generation()
        if self.prompt_cache_strategy == "kv":
            if delta_ok:
                self.prompt_cache_mode = "kv"
            else:
                warnings.warn(
                    "prompt_cache_strategy='kv' requested but provider does not support delta-only KV mode; "
                    "falling back to key-only prompt caching.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self.prompt_cache_mode = "key"
        elif self.prompt_cache_strategy == "key":
            if delta_ok:
                warnings.warn(
                    "prompt_cache_strategy='key' requested for a provider that supports mutable KV prompt caches; "
                    "using prompt_cache_mode='kv' to avoid duplicate context appends. "
                    "Use prompt_cache_strategy='off' to disable prompt caching.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self.prompt_cache_mode = "kv"
            else:
                self.prompt_cache_mode = "key"
        else:  # auto/default
            self.prompt_cache_mode = "kv" if delta_ok else "key"

        if self.prompt_cache_key is None:
            self.prompt_cache_key = f"sess:{uuid.uuid4().hex[:12]}"

        if not self._prepare_prefix_cache(session_key=self.prompt_cache_key):
            self.prompt_cache_mode = "off"
            self.prompt_cache_key = None
            return

        # KV mode uses the provider cache as the context source-of-truth; BasicSession auto-compaction
        # mutates the transcript but does not mutate the KV cache. Disable for correctness.
        if getattr(self, "auto_compact", False) and self.prompt_cache_mode == "kv":
            warnings.warn(
                "auto_compact=True is not compatible with prompt_cache_mode='kv' (KV cache would diverge from the "
                "compacted transcript). Disabling auto_compact for this session.",
                RuntimeWarning,
                stacklevel=3,
            )
            self.auto_compact = False

    def _key_mode_sync_prompt_cache(self) -> bool:
        """Best-effort: append transcript deltas into provider cache in key mode.

        Key mode still sends the full transcript to the provider each call, but keeping an
        in-process prompt cache aligned with the transcript enables fast prefix reuse for
        large contexts (notably GGUF/llama.cpp local control planes).
        """
        if self.prompt_cache_mode != "key":
            return False
        if not self.provider or not self.prompt_cache_key:
            return False

        try:
            supports_update = bool(getattr(self.provider, "prompt_cache_supports_operation", lambda _op: False)("update"))
        except Exception:
            supports_update = False
        if not supports_update:
            return False

        try:
            transcript = self._format_messages_for_provider()
        except Exception:
            transcript = []

        # Transcript shrank (files cleared, manual edits, compaction, etc.). Rebuild to avoid
        # accidentally double-appending mismatched context.
        if self._prompt_cache_key_mode_synced_messages > len(transcript):
            ok = self.rebuild_prompt_cache()
            if ok:
                self._prompt_cache_key_mode_synced_messages = len(transcript)
            return ok

        delta = transcript[self._prompt_cache_key_mode_synced_messages :]
        if not delta:
            return True

        try:
            ok = bool(
                self.provider.prompt_cache_update(  # type: ignore[attr-defined]
                    self.prompt_cache_key,
                    messages=delta,
                    add_generation_prompt=False,
                    ttl_s=self.prompt_cache_ttl_s,
                )
            )
        except Exception:
            ok = False

        if ok:
            self._prompt_cache_key_mode_synced_messages = len(transcript)
        return ok

    def rebuild_prompt_cache(self) -> bool:
        """Clear and rebuild the provider prompt cache from the session transcript (best-effort)."""
        provider = self.provider
        if provider is None or not self.prompt_cache_key:
            return False
        if not self._supports_prompt_cache():
            return False

        try:
            getattr(provider, "prompt_cache_clear")(self.prompt_cache_key)
        except Exception:
            pass

        # Use a fresh key to avoid any stale cache state.
        new_key = f"sess:{uuid.uuid4().hex[:12]}"
        self.prompt_cache_key = new_key
        if not self._prepare_prefix_cache(session_key=new_key):
            return False
        self._prompt_cache_key_mode_synced_messages = 0

        # Replay transcript messages (excluding system) into the rebuilt cache, when possible.
        try:
            supports_update = bool(getattr(provider, "prompt_cache_supports_operation", lambda _op: False)("update"))
        except Exception:
            supports_update = False

        transcript = []
        try:
            transcript = self._format_messages_for_provider()
        except Exception:
            transcript = []

        if transcript:
            if not supports_update:
                warnings.warn(
                    "Provider does not support prompt_cache_update; cannot rebuild prompt cache from transcript.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return False
            try:
                provider.prompt_cache_update(  # type: ignore[attr-defined]
                    new_key,
                    messages=transcript,
                    add_generation_prompt=False,
                    ttl_s=self.prompt_cache_ttl_s,
                )
            except Exception:
                return False
            self._prompt_cache_key_mode_synced_messages = len(transcript)

        return True

    def get_attached_files(self) -> List[Dict[str, Any]]:
        """Return attached file metadata entries from the transcript (best-effort)."""
        out: List[Dict[str, Any]] = []
        for idx, msg in enumerate(getattr(self, "messages", []) or []):
            meta = getattr(msg, "metadata", None)
            attached = meta.get("attached_file") if isinstance(meta, dict) else None
            if isinstance(attached, dict) and isinstance(attached.get("path"), str) and attached.get("path"):
                entry = dict(attached)
                entry.setdefault("message_index", int(idx))
                out.append(entry)
        return out

    def attach_files(
        self,
        file_paths: List[Union[str, Path]],
        *,
        dedupe: bool = True,
        force: bool = False,
        prefill_key_mode_cache: bool = False,
        max_file_size: Optional[int] = None,
        format_output: str = "structured",
    ) -> Dict[str, Any]:
        """Attach local files as cached "boxes" (text extraction) for faster iteration on large contexts.

        This does two things:
        1) Inserts each file's extracted text into the session transcript as a dedicated message
           (1 file = 1 message box), with `metadata["attached_file"]=...`.
        2) In `prompt_cache_mode="kv"`, appends the same message box into the provider KV cache via
           `provider.prompt_cache_update(...)` so the model sees the file without resending history.

        Only TEXT/DOCUMENT files are supported by this helper. Use `media=` for images/audio/video.

        Returns:
            JSON-serializable dict with `attached`, `skipped`, and `errors` lists.

            Also includes `timing` with best-effort durations (milliseconds):
              - `extract_ms`: time spent extracting text boxes
              - `cache_update_ms`: time spent updating provider prompt caches
              - `total_ms`: total time for this call
        """
        import time

        t_total0 = time.perf_counter()
        extract_ms = 0
        cache_update_ms = 0

        attached: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        existing_by_path: Dict[str, Dict[str, Any]] = {}
        if dedupe:
            for item in self.get_attached_files():
                key = attached_file_dedupe_key(item)
                if key is None:
                    continue
                path, size, mtime_ns = key
                existing_by_path[path] = {"size_bytes": size, "mtime_ns": mtime_ns}

        for raw in file_paths or []:
            try:
                p = Path(raw).expanduser() if isinstance(raw, (str, Path)) else None
                if p is None:
                    raise ValueError(f"Unsupported file path type: {type(raw).__name__}")
                if not p.exists():
                    raise FileNotFoundError(str(p))
                resolved = p.resolve()
                stat = resolved.stat()
                size_bytes = int(stat.st_size)
                mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
                resolved_str = str(resolved)

                if dedupe and not force:
                    prior = existing_by_path.get(resolved_str)
                    if prior and int(prior.get("size_bytes", -1)) == size_bytes and int(prior.get("mtime_ns", -2)) == mtime_ns:
                        skipped.append({"path": resolved_str, "reason": "already_attached"})
                        continue
                    if prior and (int(prior.get("size_bytes", -1)) != size_bytes or int(prior.get("mtime_ns", -2)) != mtime_ns):
                        raise RuntimeError(
                            "File appears to have changed since it was attached. "
                            "Refusing to re-attach by default (would duplicate conflicting context). "
                            "Use force=True, or clear/rebuild the session cache."
                        )

                t_extract0 = time.perf_counter()
                box = extract_file_box(
                    resolved,
                    max_file_size=max_file_size,
                    format_output=format_output,
                )
                extract_ms += int((time.perf_counter() - t_extract0) * 1000)
                content = render_file_box_message(box)
                meta = box.to_meta()

                # 1 file = 1 message box in the transcript.
                self.add_message(MessageRole.USER.value, content, attached_file=meta)

                # In KV mode, the provider cache is the context source-of-truth; ensure the file box
                # is appended to the in-process KV cache as well.
                if self.prompt_cache_mode == "kv":
                    if not self.provider or not self.prompt_cache_key:
                        raise RuntimeError("KV mode requires an active provider and prompt_cache_key")

                    try:
                        supports_update = bool(getattr(self.provider, "prompt_cache_supports_operation", lambda _op: False)("update"))
                    except Exception:
                        supports_update = False
                    if not supports_update:
                        raise RuntimeError("Provider does not support prompt_cache_update required for KV file boxes")

                    t_cache0 = time.perf_counter()
                    self.provider.prompt_cache_update(  # type: ignore[attr-defined]
                        self.prompt_cache_key,
                        messages=[{"role": "user", "content": content}],
                        add_generation_prompt=False,
                        ttl_s=self.prompt_cache_ttl_s,
                    )
                    cache_update_ms += int((time.perf_counter() - t_cache0) * 1000)

                attached.append(meta)
                if dedupe:
                    existing_by_path[meta["path"]] = {"size_bytes": meta["size_bytes"], "mtime_ns": meta["mtime_ns"]}

            except Exception as e:
                errors.append({"path": str(raw), "error": str(e)})

        # Key mode: optionally prefill/sync so file boxes participate in prefix reuse
        # before the first generation call.
        if attached and self.prompt_cache_mode == "key" and bool(prefill_key_mode_cache):
            t_cache0 = time.perf_counter()
            self._key_mode_sync_prompt_cache()
            cache_update_ms += int((time.perf_counter() - t_cache0) * 1000)

        total_ms = int((time.perf_counter() - t_total0) * 1000)
        return {
            "attached": attached,
            "skipped": skipped,
            "errors": errors,
            "timing": {
                "extract_ms": int(extract_ms),
                "cache_update_ms": int(cache_update_ms),
                "total_ms": int(total_ms),
            },
        }

    def clear_attached_files(self, *, rebuild_prompt_cache: bool = True) -> int:
        """Remove attached file boxes from the transcript (best-effort).

        In KV mode, removing transcript messages is not enough: the provider KV cache is the
        context source-of-truth. When `rebuild_prompt_cache=True`, this method rebuilds the
        provider cache from the updated transcript so the model no longer sees removed files.
        """
        removed = 0
        kept = []
        for msg in getattr(self, "messages", []) or []:
            meta = getattr(msg, "metadata", None)
            attached = meta.get("attached_file") if isinstance(meta, dict) else None
            if isinstance(attached, dict) and isinstance(attached.get("path"), str) and attached.get("path"):
                removed += 1
                continue
            kept.append(msg)
        self.messages = kept

        if removed and rebuild_prompt_cache and self.prompt_cache_mode in {"kv", "key"}:
            self.rebuild_prompt_cache()

        return removed

    def _append_kv_tool_results_to_cache(self, assistant_text: str) -> None:
        """Best-effort: ensure provider KV cache sees appended tool results text."""
        if not self.provider or not self.prompt_cache_key:
            return
        if not isinstance(assistant_text, str) or not assistant_text:
            return
        marker = "\n\nTool Results:\n"
        idx = assistant_text.find(marker)
        if idx < 0:
            return
        suffix = assistant_text[idx:]
        if not suffix.strip():
            return
        try:
            supports_update = bool(getattr(self.provider, "prompt_cache_supports_operation", lambda _op: False)("update"))
        except Exception:
            supports_update = False
        if not supports_update:
            return
        try:
            self.provider.prompt_cache_update(  # type: ignore[attr-defined]
                self.prompt_cache_key,
                messages=[{"role": "assistant", "content": suffix}],
                add_generation_prompt=False,
                ttl_s=self.prompt_cache_ttl_s,
            )
        except Exception:
            return

    def generate(
        self,
        prompt: str,
        name: Optional[str] = None,
        location: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        if self.prompt_cache_mode == "off":
            return super().generate(prompt, name=name, location=location, **kwargs)

        if not self.provider:
            raise ValueError("No provider configured")

        if self.prompt_cache_mode == "key":
            # Key mode: transcript is still source-of-truth, but we keep the provider-side prompt
            # cache aligned (when supported) for faster prefix reuse.

            if getattr(self, "auto_compact", False) and self.should_compact(getattr(self, "auto_compact_threshold", 8000)):
                compacted = self.compact(reason="auto_threshold")
                self._replace_with_compacted(compacted)

            # Best-effort: sync the *existing* transcript into the provider cache before we append
            # the new user message. In BasicSession-style calls, `prompt` is sent separately and
            # the provider receives `messages=` excluding the current user message. Keeping the
            # provider cache aligned with that prefix avoids doing extra cache work on the hot
            # path before generation (especially important for GGUF/local control planes).
            self._key_mode_sync_prompt_cache()

            self.add_message(MessageRole.USER.value, str(prompt), name=name, location=location)

            media = kwargs.pop("media", None)
            stream = bool(kwargs.pop("stream", False))

            # Avoid per-call overrides that would either error (duplicate kwargs) or desync the cache.
            if "messages" in kwargs:
                warnings.warn(
                    "CachedSession key mode ignores `messages=`; context comes from the session transcript.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            if "system_prompt" in kwargs:
                warnings.warn(
                    "CachedSession key mode ignores `system_prompt=`; use session.system_prompt instead.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            if "tools" in kwargs:
                warnings.warn(
                    "CachedSession key mode ignores per-call `tools=`; tools are session-level cached state.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            kwargs.pop("messages", None)
            kwargs.pop("system_prompt", None)
            kwargs.pop("tools", None)

            if hasattr(self, "tool_call_tags") and self.tool_call_tags is not None and "tool_call_tags" not in kwargs:
                kwargs["tool_call_tags"] = self.tool_call_tags
            if "temperature" not in kwargs and self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if "seed" not in kwargs and isinstance(self.seed, int) and self.seed >= 0:
                kwargs["seed"] = self.seed

            if getattr(self, "enable_tracing", False):
                trace_meta = kwargs.get("trace_metadata")
                if not isinstance(trace_meta, dict):
                    trace_meta = {}
                    kwargs["trace_metadata"] = trace_meta
                trace_meta.update(
                    {
                        "session_id": getattr(self, "id", None),
                        "step_type": kwargs.get("step_type", "chat"),
                        "attempt_number": kwargs.get("attempt_number", 1),
                    }
                )

            messages = self._format_messages_for_provider_excluding_current()

            tools_for_call = self.tools if getattr(self, "tools", None) else None
            response = self.provider.generate(
                prompt=str(prompt),
                messages=messages,
                system_prompt=self.system_prompt,
                tools=tools_for_call,
                media=media,
                stream=stream,
                **kwargs,
            )

            # Streaming: collect for history, then sync the cache with the appended assistant message.
            if hasattr(response, "__iter__") and not hasattr(response, "content"):
                collected = ""

                def _wrap(stream_it: Iterator[GenerateResponse]) -> Iterator[GenerateResponse]:
                    nonlocal collected
                    for chunk in stream_it:
                        if getattr(chunk, "content", None):
                            collected += str(chunk.content)
                        yield chunk
                    if collected:
                        self.add_message(MessageRole.ASSISTANT.value, collected)
                        self._key_mode_sync_prompt_cache()

                return _wrap(response)  # type: ignore[arg-type]

            if hasattr(response, "content") and response.content:
                self.add_message(MessageRole.ASSISTANT.value, response.content)
                self._key_mode_sync_prompt_cache()

            return response  # type: ignore[return-value]

        # KV mode: prompt cache is context source-of-truth; keep transcript for UX only.
        self.add_message(MessageRole.USER.value, str(prompt), name=name, location=location)

        # Extract media parameter explicitly
        media = kwargs.pop("media", None)

        # KV mode does not send system/messages (they are prefixed in the cache).
        if "messages" in kwargs:
            warnings.warn("CachedSession kv mode ignores `messages=`; context comes from prompt cache.", RuntimeWarning, stacklevel=3)
        if "system_prompt" in kwargs:
            warnings.warn("CachedSession kv mode ignores `system_prompt=`; context comes from prompt cache.", RuntimeWarning, stacklevel=3)
        if "tools" in kwargs:
            warnings.warn("CachedSession kv mode ignores per-call `tools=`; tools are session-level cached state.", RuntimeWarning, stacklevel=3)
        kwargs.pop("messages", None)
        kwargs.pop("system_prompt", None)
        kwargs.pop("tools", None)

        if hasattr(self, "tool_call_tags") and self.tool_call_tags is not None and "tool_call_tags" not in kwargs:
            kwargs["tool_call_tags"] = self.tool_call_tags

        if "temperature" not in kwargs and self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if "seed" not in kwargs and isinstance(self.seed, int) and self.seed >= 0:
            kwargs["seed"] = self.seed

        # Tracing parity with BasicSession.
        if getattr(self, "enable_tracing", False):
            trace_meta = kwargs.get("trace_metadata")
            if not isinstance(trace_meta, dict):
                trace_meta = {}
                kwargs["trace_metadata"] = trace_meta
            trace_meta.update(
                {
                    "session_id": getattr(self, "id", None),
                    "step_type": kwargs.get("step_type", "chat"),
                    "attempt_number": kwargs.get("attempt_number", 1),
                }
            )

        # Tools are stable, session-level state in KV mode; pass them for tool parsing/execution
        # while telling the provider not to re-render them into the prompt.
        tools_for_call = self.tools if getattr(self, "tools", None) else None
        prefilled = self._kv_prefilled_modules()
        if prefilled and "prompt_cache_prefilled_modules" not in kwargs:
            kwargs["prompt_cache_prefilled_modules"] = prefilled

        response = self.provider.generate(
            prompt=str(prompt),
            messages=None,
            system_prompt=None,
            tools=tools_for_call,
            media=media,
            **kwargs,
        )

        if isinstance(response, (GeneratorABC, IteratorABC)) or (
            hasattr(response, "__iter__") and not hasattr(response, "content")
        ):
            def _wrap(stream_it: Iterator[GenerateResponse]) -> Iterator[GenerateResponse]:
                collected = ""
                for chunk in stream_it:
                    if getattr(chunk, "content", None):
                        collected += str(chunk.content)
                    yield chunk
                if collected:
                    self.add_message(MessageRole.ASSISTANT.value, collected)
                    self._append_kv_tool_results_to_cache(collected)

            return _wrap(response)  # type: ignore[arg-type]

        if hasattr(response, "content") and response.content:
            self.add_message(MessageRole.ASSISTANT.value, response.content)
            self._append_kv_tool_results_to_cache(str(response.content))

        # Capture provider trace (match BasicSession behavior).
        if getattr(self, "enable_tracing", False) and hasattr(self.provider, "get_traces"):
            try:
                md = getattr(response, "metadata", None)
                trace_id = md.get("trace_id") if isinstance(md, dict) else None
                if trace_id:
                    trace = self.provider.get_traces(trace_id)
                    if trace:
                        getattr(self, "interaction_traces", []).append(trace)
            except Exception:
                pass
        return response

    async def agenerate(
        self,
        prompt: str,
        name: Optional[str] = None,
        location: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[GenerateResponse, AsyncIterator[GenerateResponse]]:
        if self.prompt_cache_mode != "kv":
            return await super().agenerate(prompt, name=name, location=location, **kwargs)  # type: ignore[misc]

        if not self.provider:
            raise ValueError("No provider configured")

        self.add_message(MessageRole.USER.value, str(prompt), name=name, location=location)

        media = kwargs.pop("media", None)

        # KV mode does not send system/messages (they are prefixed in the cache).
        if "messages" in kwargs:
            warnings.warn("CachedSession kv mode ignores `messages=`; context comes from prompt cache.", RuntimeWarning, stacklevel=3)
        if "system_prompt" in kwargs:
            warnings.warn("CachedSession kv mode ignores `system_prompt=`; context comes from prompt cache.", RuntimeWarning, stacklevel=3)
        if "tools" in kwargs:
            warnings.warn("CachedSession kv mode ignores per-call `tools=`; tools are session-level cached state.", RuntimeWarning, stacklevel=3)
        kwargs.pop("tools", None)
        kwargs.pop("messages", None)
        kwargs.pop("system_prompt", None)
        stream = bool(kwargs.pop("stream", False))
        if hasattr(self, "tool_call_tags") and self.tool_call_tags is not None and "tool_call_tags" not in kwargs:
            kwargs["tool_call_tags"] = self.tool_call_tags

        if "temperature" not in kwargs and self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if "seed" not in kwargs and isinstance(self.seed, int) and self.seed >= 0:
            kwargs["seed"] = self.seed

        if getattr(self, "enable_tracing", False):
            trace_meta = kwargs.get("trace_metadata")
            if not isinstance(trace_meta, dict):
                trace_meta = {}
                kwargs["trace_metadata"] = trace_meta
            trace_meta.update(
                {
                    "session_id": getattr(self, "id", None),
                    "step_type": kwargs.get("step_type", "chat"),
                    "attempt_number": kwargs.get("attempt_number", 1),
                }
            )

        tools_for_call = self.tools if getattr(self, "tools", None) else None
        prefilled = self._kv_prefilled_modules()
        if prefilled and "prompt_cache_prefilled_modules" not in kwargs:
            kwargs["prompt_cache_prefilled_modules"] = prefilled

        response = await self.provider.agenerate(
            prompt=str(prompt),
            messages=None,
            system_prompt=None,
            tools=tools_for_call,
            media=media,
            stream=stream,
            **kwargs,
        )

        # Streaming: collect for history (match BasicSession behavior).
        if hasattr(response, "__aiter__"):
            collected = ""

            async def _wrap(stream_it: AsyncIterator[GenerateResponse]) -> AsyncIterator[GenerateResponse]:
                nonlocal collected
                async for chunk in stream_it:
                    if getattr(chunk, "content", None):
                        collected += str(chunk.content)
                    yield chunk
                if collected:
                    self.add_message(MessageRole.ASSISTANT.value, collected)
                    self._append_kv_tool_results_to_cache(collected)

            return _wrap(response)  # type: ignore[return-value]

        if hasattr(response, "content") and response.content:
            self.add_message(MessageRole.ASSISTANT.value, response.content)
            self._append_kv_tool_results_to_cache(str(response.content))
        return response  # type: ignore[return-value]
