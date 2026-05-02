"""
Prompt-cache REPL demo (MLX / HuggingFace transformers / HuggingFace GGUF).

This script is intentionally dev-oriented: it helps you *observe* and *toggle* prompt caching
while iterating on large discussions, with "box caching" boundaries:
  - system prompt (stable)
  - tools (stable)
  - discussion history (grows each turn)
  - file attachments (1 file = 1 message box)

Key ideas:
  - Use `CachedSession(prompt_cache_strategy="auto")` to enable KV mode when supported.
  - Use `@file` attachments to append extracted text boxes via `CachedSession.attach_files(...)`.
  - Use `/cache stats` to inspect in-process prompt-cache entries.

Run:
  python examples/prompt_caching/prompt_cache_repl_demo.py
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import time
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from abstractcore import CachedSession, create_llm
from abstractcore.utils.message_preprocessor import MessagePreprocessor
from abstractcore.utils.model_cache import default_hf_hub_cache_dirs


def _print_json(title: str, payload: Any) -> None:
    print(f"\n== {title} ==")
    try:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception:
        print(payload)


def _maybe_detect_media_type(path: str) -> str:
    try:
        from abstractcore.media.types import detect_media_type

        mt = detect_media_type(Path(path))
        return str(getattr(mt, "value", mt))
    except Exception:
        return "unknown"


def _split_attachments(paths: Sequence[str]) -> tuple[list[str], list[str]]:
    """Return (file_box_paths, media_paths)."""
    file_boxes: list[str] = []
    media: list[str] = []
    for p in paths or []:
        mt = _maybe_detect_media_type(p)
        if mt in {"text", "document"}:
            file_boxes.append(p)
        else:
            media.append(p)
    return file_boxes, media


def _list_cached_hf_models(
    *,
    query: Optional[str] = None,
    limit: int = 60,
) -> Dict[str, List[str]]:
    """Best-effort local HF cache inspection (no network)."""
    q = str(query or "").strip().lower()
    repos: set[str] = set()

    for base in default_hf_hub_cache_dirs():
        try:
            for entry in base.iterdir():
                if not entry.is_dir():
                    continue
                name = entry.name
                if not name.startswith("models--"):
                    continue
                repo_id = name[len("models--") :].replace("--", "/")
                if "/" not in repo_id:
                    continue
                if q and q not in repo_id.lower():
                    continue
                repos.add(repo_id)
        except Exception:
            continue

    # Heuristic categories for convenience.
    mlx: List[str] = []
    gguf: List[str] = []
    hf: List[str] = []

    for repo_id in sorted(repos)[: max(200, limit * 3)]:
        rid = repo_id.lower()
        if "mlx" in rid or rid.startswith("mlx-community/") or rid.endswith("-mlx"):
            mlx.append(repo_id)
            continue
        if "gguf" in rid:
            gguf.append(repo_id)
            continue
        hf.append(repo_id)

    return {
        "mlx": mlx[:limit],
        "hf_transformers": hf[:limit],
        "hf_gguf": gguf[:limit],
    }


class _Ui:
    def __init__(self, *, use_color: Optional[bool] = None) -> None:
        if use_color is None:
            use_color = bool(sys.stdout.isatty()) and os.getenv("NO_COLOR") is None
        self.use_color = bool(use_color)

    def _c(self, s: str, code: str) -> str:
        if not self.use_color:
            return s
        return f"\x1b[{code}m{s}\x1b[0m"

    def bold(self, s: str) -> str:
        return self._c(s, "1")

    def dim(self, s: str) -> str:
        return self._c(s, "90")

    def ok(self, s: str) -> str:
        return self._c(s, "32")

    def warn(self, s: str) -> str:
        return self._c(s, "33")

    def err(self, s: str) -> str:
        return self._c(s, "31")

    def blue(self, s: str) -> str:
        return self._c(s, "34")

    def magenta(self, s: str) -> str:
        return self._c(s, "35")

    def cyan(self, s: str) -> str:
        return self._c(s, "36")

    def hr(self, char: str = "─") -> str:
        width = int(shutil.get_terminal_size((88, 20)).columns)
        width = max(40, min(120, width))
        return str(char) * width


def _short_model_label(model: str) -> str:
    s = str(model or "").strip()
    if not s:
        return "unknown"
    if s.endswith(".gguf"):
        try:
            return Path(s).name
        except Exception:
            return s.split("/")[-1]
    if "/" in s:
        return s.split("/", 1)[1]
    return s


def _sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _module_cache_key(item: Any) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    key = item.get("cache_key") or item.get("derived_key")
    if isinstance(key, str) and key.strip():
        return key.strip()
    return None


def _module_hash(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    h = item.get("module_hash") or item.get("fingerprint")
    return str(h or "")


def _allocate_widths(tokens: Sequence[int], *, width: int) -> List[int]:
    """Allocate a fixed character width across token-weighted segments."""
    total_width = int(width)
    if total_width <= 0:
        return [0 for _ in tokens]
    tok = [max(0, int(t)) for t in (tokens or [])]
    total_tokens = sum(tok)
    if total_tokens <= 0:
        return [0 for _ in tok]

    exact = [t / total_tokens * total_width for t in tok]
    out = [int(x) for x in exact]
    remainder = total_width - sum(out)
    if remainder > 0:
        fracs = sorted(range(len(out)), key=lambda i: (exact[i] - out[i]), reverse=True)
        for i in fracs:
            if remainder <= 0:
                break
            out[i] += 1
            remainder -= 1

    # Ensure tiny but non-zero segments are visible when possible.
    for i, t in enumerate(tok):
        if t > 0 and out[i] == 0 and total_width >= len(tok):
            donor = max(range(len(out)), key=lambda j: out[j])
            if out[donor] > 1:
                out[donor] -= 1
                out[i] = 1

    return out


def _fmt_int(n: Optional[int]) -> str:
    if n is None:
        return "-"
    try:
        return f"{int(n):,}"
    except Exception:
        return "-"

def _fmt_ms(ms: Optional[int]) -> str:
    if ms is None:
        return "-"
    return f"{_fmt_int(ms)}ms"


def _ceil_ms(seconds: float) -> int:
    try:
        return max(0, int(math.ceil(float(seconds) * 1000.0)))
    except Exception:
        return 0


def _ceil_div(n: int, d: int) -> int:
    if d <= 0:
        return 0
    if n <= 0:
        return 0
    return int((n + d - 1) // d)


def _ceil_s(ms: int) -> int:
    if ms <= 0:
        return 0
    return int((ms + 1000 - 1) // 1000)


def _fmt_total_time(total_ms: int) -> str:
    total_s = _ceil_s(int(total_ms))
    if total_s <= 0:
        return "0s (0s)"
    minutes, seconds = divmod(total_s, 60)
    if minutes > 0:
        return f"{minutes}mn{seconds}s ({total_s}s)"
    return f"{total_s}s ({total_s}s)"

def _fmt_duration(ms: Optional[int]) -> str:
    if ms is None:
        return "-"
    try:
        ms_i = max(0, int(ms))
    except Exception:
        return "-"
    if ms_i >= 10_000:
        total_s = _ceil_s(ms_i)
        minutes, seconds = divmod(total_s, 60)
        if minutes > 0:
            return f"{minutes}mn{seconds}s"
        return f"{total_s}s"
    return f"{ms_i:,}ms"


def _fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "-"
    try:
        n_i = int(n)
    except Exception:
        return "-"
    if n_i < 0:
        return "-"
    if n_i < 1024:
        return f"{n_i}B"
    for unit, size in (("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if n_i >= size:
            return f"{_ceil_div(n_i, size)}{unit}"
    return f"{n_i}B"


class _Repl:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        cache_enabled: bool,
        max_tokens: Optional[int],
        max_output_tokens: Optional[int],
    ) -> None:
        self.provider_name = str(provider or "").strip() or "huggingface"
        self.model_name = str(model or "").strip()
        self.cache_enabled = bool(cache_enabled)
        self.max_tokens = max_tokens
        self.max_output_tokens = max_output_tokens

        self.system_prompt = "You are a helpful assistant. Be concise."
        self.temperature: Optional[float] = 0.7
        self.seed: int = -1
        self.stream: bool = False
        # Internal streaming is always used when available to compute TTFT/prefill metrics.
        # `self.stream` toggles *live printing* only.
        self._force_internal_streaming: bool = True

        # Tools are optional in this demo. Default to off to avoid models emitting raw tool-call
        # blocks in responses. Enable via `/tools on` when you want to exercise the tools box.
        self.tools = []

        self.provider = None
        self.session: CachedSession | None = None
        self.ui = _Ui()
        self._reset_provider_and_session(keep_transcript=False)

    def _default_tools(self):
        try:
            from abstractcore.tools.common_tools import list_files, read_file, search_files

            return [list_files, read_file, search_files]
        except Exception:
            return []

    def _provider_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = int(self.max_tokens)
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(self.max_output_tokens)
        return kwargs

    def _provider_label(self) -> str:
        if self.provider_name == "mlx":
            return "mlx"
        if self.provider_name == "huggingface":
            mt = str(getattr(self.provider, "model_type", "") or "").strip().lower() if self.provider is not None else ""
            if mt == "gguf":
                return "hf-gguf"
            if mt == "transformers":
                return "hf-tf"
            return "hf"
        return self.provider_name

    def _prompt(self) -> str:
        if self.session is None:
            return "acore> "
        label = self._provider_label()
        mode = str(getattr(self.session, "prompt_cache_mode", "off") or "off")
        key = str(getattr(self.session, "prompt_cache_key", "") or "")
        key_short = key[-8:] if key else "-"
        return f"acore[{label}|{mode}|{key_short}]> "

    def _cache_meta(self, key: Optional[str]) -> Dict[str, Any]:
        if not key or not self.provider or not hasattr(self.provider, "get_prompt_cache_stats"):
            return {}
        try:
            stats = self.provider.get_prompt_cache_stats()  # type: ignore[union-attr]
        except Exception:
            return {}
        meta = None
        if isinstance(stats, dict):
            meta_by_key = stats.get("meta_by_key")
            if isinstance(meta_by_key, dict):
                meta = meta_by_key.get(key)
        return dict(meta) if isinstance(meta, dict) else {}

    def _cache_token_count(self, key: Optional[str]) -> Optional[int]:
        if not key or not self.provider:
            return None

        fn = getattr(self.provider, "prompt_cache_token_count", None)
        if callable(fn):
            try:
                tok = fn(key)
                if isinstance(tok, int) and tok >= 0:
                    return tok
            except Exception:
                pass

        meta = self._cache_meta(key)
        tok = meta.get("token_count")
        try:
            tok_i = int(tok)
            return tok_i if tok_i >= 0 else None
        except Exception:
            return None

    def _make_session(self, *, keep_transcript: bool) -> None:
        old_messages = []
        if keep_transcript and self.session is not None:
            try:
                old_messages = list(getattr(self.session, "messages", []) or [])
            except Exception:
                old_messages = []
            try:
                self.temperature = getattr(self.session, "temperature", self.temperature)
            except Exception:
                pass
            try:
                self.seed = int(getattr(self.session, "seed", self.seed))
            except Exception:
                pass

        prompt_cache_strategy = "auto" if self.cache_enabled else "off"

        self.session = CachedSession(
            provider=self.provider,
            system_prompt=self.system_prompt,
            tools=self.tools,
            prompt_cache_strategy=prompt_cache_strategy,
            temperature=self.temperature,
            seed=self.seed,
        )

        if keep_transcript and old_messages:
            self.session.messages = old_messages
            # KV mode requires the provider cache to match transcript state.
            if self.session.prompt_cache_mode == "kv":
                self.session.rebuild_prompt_cache()

    def _reset_provider_and_session(self, *, keep_transcript: bool, allow_fallback: bool = True) -> None:
        old_messages = []
        if keep_transcript and self.session is not None:
            try:
                old_messages = list(getattr(self.session, "messages", []) or [])
            except Exception:
                old_messages = []

        kwargs = self._provider_kwargs()
        try:
            self.provider = create_llm(self.provider_name, model=self.model_name, **kwargs)
        except Exception as e:
            if not allow_fallback:
                raise
            # Best-effort fallback to a locally cached model when the default isn't available.
            fallback = None
            try:
                cached = _list_cached_hf_models(limit=40)
                if self.provider_name == "mlx":
                    fallback = (cached.get("mlx") or [None])[0]
                elif self.provider_name == "huggingface":
                    fallback = (cached.get("hf_transformers") or cached.get("hf_gguf") or [None])[0]
            except Exception:
                fallback = None

            if isinstance(fallback, str) and fallback.strip() and fallback != self.model_name:
                print(f"⚠️ Failed to load {self.provider_name}:{self.model_name}: {e}")
                print(f"🔎 Falling back to locally cached model: {fallback}")
                self.model_name = fallback
                self.provider = create_llm(self.provider_name, model=self.model_name, **kwargs)
            else:
                raise
        self._make_session(keep_transcript=False)

        if keep_transcript and old_messages and self.session is not None:
            self.session.messages = old_messages
            if self.session.prompt_cache_mode == "kv":
                self.session.rebuild_prompt_cache()

    def _banner(self) -> None:
        assert self.session is not None
        caps = {}
        try:
            caps = self.provider.get_prompt_cache_capabilities().to_dict()  # type: ignore[union-attr]
        except Exception:
            caps = {}
        print("\n" + self.ui.hr("═"))
        print(self.ui.bold("Prompt-cache REPL demo"))
        print(f"provider={self._provider_label()}  model={self.ui.cyan(_short_model_label(self.model_name))}")
        print(
            f"cache_enabled={self.cache_enabled}  session_mode={self.session.prompt_cache_mode}  "
            f"key={self.session.prompt_cache_key}  prefix={getattr(self.session, '_prompt_cache_prefix_key', None)}"
        )
        print(
            self.ui.dim(
                f"temperature={getattr(self.session, 'temperature', None)} seed={getattr(self.session, 'seed', None)} "
                f"stream={'on' if self.stream else 'off'}"
            )
        )
        try:
            ctx = getattr(self.provider, "max_tokens", None)
        except Exception:
            ctx = None
        try:
            out_cap = getattr(self.provider, "max_output_tokens", None)
        except Exception:
            out_cap = None
        if ctx is not None or out_cap is not None:
            print(self.ui.dim(f"context={ctx} max_output_tokens={out_cap}"))
        try:
            if (
                self.provider_name == "huggingface"
                and str(getattr(self.provider, "model_type", "") or "").strip().lower() == "gguf"
            ):
                ngl = getattr(self.provider, "n_gpu_layers", None)
                print(self.ui.dim(f"gguf.n_gpu_layers={ngl} (0=cpu, -1=all layers)"))
        except Exception:
            pass
        if isinstance(caps, dict) and caps:
            print(f"provider.prompt_cache_capabilities.mode={caps.get('mode')}")
        files_n = len(self.session.get_attached_files())
        msgs_n = len(getattr(self.session, "messages", []) or [])
        print(f"boxes: system=1 tools={len(self.tools)} files={files_n} messages={msgs_n}")
        print(self.ui.dim("Type /help for commands. Attach files with @path."))
        print(self.ui.hr("═") + "\n")

    def _cmd_help(self) -> None:
        print(
            "\nCommands:\n"
            "  /help\n"
            "  /model <provider:model>         (examples: mlx:mlx-community/Qwen3-4B, huggingface:sshleifer/tiny-gpt2)\n"
            "  /cache on|off|clear|stats [json]\n"
            "  /status                         (high-level session + cache summary)\n"
            "  /boxes                          (system/tools/files/history box details)\n"
            "  /stream [on|off]                (toggle streaming output)\n"
            "  /temp [float|none]              (show/set temperature)\n"
            "  /seed [int|random]              (show/set seed)\n"
            "  /system [text]\n"
            "  /tools on|off\n"
            "  /files [clear]\n"
            "  /history [n]\n"
            "  /models [query]                 (local HF cache inspection; no network)\n"
            "  /quit\n"
        )

    def _cmd_temp(self, args: List[str]) -> None:
        assert self.session is not None
        if not args:
            print(f"temperature={getattr(self.session, 'temperature', None)}")
            return
        raw = str(args[0] or "").strip().lower()
        if raw in {"none", "default"}:
            self.temperature = None
            self.session.temperature = None
            print("temperature set to None (provider default)")
            return
        try:
            val = float(raw)
        except Exception:
            print("Usage: /temp [float|none]")
            return
        self.temperature = val
        self.session.temperature = val
        print(f"temperature set to {val}")

    def _cmd_seed(self, args: List[str]) -> None:
        assert self.session is not None
        if not args:
            print(f"seed={getattr(self.session, 'seed', None)}")
            return
        raw = str(args[0] or "").strip().lower()
        if raw in {"random", "none", "-1"}:
            self.seed = -1
            self.session.seed = -1
            print("seed set to random/unset (-1)")
            return
        try:
            val = int(raw)
        except Exception:
            print("Usage: /seed [int|random]")
            return
        self.seed = val if val >= 0 else -1
        self.session.seed = self.seed
        print(f"seed set to {self.seed}")

    def _cmd_stream(self, args: List[str]) -> None:
        if not args:
            self.stream = not self.stream
            print(f"stream {'enabled' if self.stream else 'disabled'}")
            return
        sub = str(args[0] or "").strip().lower()
        if sub in {"on", "1", "true", "yes"}:
            self.stream = True
            print("stream enabled")
            return
        if sub in {"off", "0", "false", "no"}:
            self.stream = False
            print("stream disabled")
            return
        print("Usage: /stream [on|off]")

    def _cmd_status(self) -> None:
        assert self.session is not None
        cache_key = getattr(self.session, "prompt_cache_key", None)
        cache_tokens = self._cache_token_count(cache_key)
        prefix_key = getattr(self.session, "_prompt_cache_prefix_key", None)
        prefix_tokens = self._cache_token_count(prefix_key) if isinstance(prefix_key, str) else None

        transcript_msgs = len(getattr(self.session, "messages", []) or [])
        files_n = len(self.session.get_attached_files())
        try:
            transcript_tokens = int(self.session.get_token_estimate())
        except Exception:
            transcript_tokens = None

        print(self.ui.bold("Session"))
        print(f"- provider: {self._provider_label()}  model: {self.model_name}")
        print(f"- cache: enabled={self.cache_enabled} mode={self.session.prompt_cache_mode} key={cache_key}")
        print(
            f"- generation: temperature={getattr(self.session, 'temperature', None)} seed={getattr(self.session, 'seed', None)} "
            f"stream={'on' if self.stream else 'off'}"
        )
        if cache_tokens is not None:
            print(f"- cache_tokens: {cache_tokens}")
        if prefix_key:
            print(f"- prefix_key: {prefix_key}")
            if prefix_tokens is not None:
                print(f"- prefix_tokens: {prefix_tokens}")
        print(f"- transcript_messages: {transcript_msgs}  file_boxes: {files_n}")
        if transcript_tokens is not None:
            print(f"- transcript_token_estimate: {transcript_tokens}")

    def _cmd_boxes(self) -> None:
        assert self.session is not None
        from abstractcore.utils.token_utils import estimate_tokens

        model_name = str(getattr(self.provider, "model", "") or self.model_name)
        context_window = getattr(self.provider, "max_tokens", None)
        try:
            context_window_i = int(context_window) if context_window is not None else None
            if context_window_i is not None and context_window_i <= 0:
                context_window_i = None
        except Exception:
            context_window_i = None

        files = self.session.get_attached_files()
        msgs = list(getattr(self.session, "messages", []) or [])

        # Split transcript messages into (history vs file boxes) so the box view is stable.
        history_msgs = []
        for m in msgs:
            if getattr(m, "role", None) == "system":
                continue
            meta = getattr(m, "metadata", None)
            attached = meta.get("attached_file") if isinstance(meta, dict) else None
            if isinstance(attached, dict) and attached.get("path"):
                continue
            else:
                history_msgs.append(m)

        prefix_modules = getattr(self.session, "_prompt_cache_prefix_modules", {}) or {}
        sys_mod = prefix_modules.get("system") if isinstance(prefix_modules, dict) else None
        tools_mod = prefix_modules.get("tools") if isinstance(prefix_modules, dict) else None

        sys_mod_key = _module_cache_key(sys_mod)
        tools_mod_key = _module_cache_key(tools_mod)
        sys_mod_tokens = self._cache_token_count(sys_mod_key) if sys_mod_key else None
        tools_prefix_tokens = self._cache_token_count(tools_mod_key) if tools_mod_key else None

        system_tokens = sys_mod_tokens
        if system_tokens is None:
            system_tokens = int(estimate_tokens(str(self.system_prompt or ""), model=model_name))

        tools_tokens = None
        if isinstance(sys_mod_tokens, int) and isinstance(tools_prefix_tokens, int) and tools_prefix_tokens >= sys_mod_tokens:
            tools_tokens = tools_prefix_tokens - sys_mod_tokens
        if tools_tokens is None:
            try:
                tool_defs = list(getattr(self.session, "tools", []) or [])
                tool_schemas = []
                for t in tool_defs:
                    if hasattr(t, "to_dict") and callable(getattr(t, "to_dict")):
                        td = t.to_dict()
                        if isinstance(td, dict):
                            tool_schemas.append(td)
                tools_tokens = int(estimate_tokens(json.dumps(tool_schemas, ensure_ascii=False, sort_keys=True), model=model_name))
            except Exception:
                tools_tokens = 0

        files_tokens = 0
        file_entries: List[Dict[str, Any]] = []
        missing_by_msg_idx: Dict[int, Dict[str, Any]] = {}
        for meta in files:
            try:
                path = meta.get("path")
                if not isinstance(path, str) or not path.strip():
                    continue
                path = path.strip()
                sha = str(meta.get("sha256") or "")[:12]

                size_bytes = meta.get("size_bytes")
                if not isinstance(size_bytes, int) or size_bytes < 0:
                    try:
                        size_bytes = int(Path(path).stat().st_size)
                    except Exception:
                        size_bytes = None

                entry: Dict[str, Any] = {
                    "path": path,
                    "sha": sha,
                    "size_bytes": size_bytes,
                    "message_index": meta.get("message_index"),
                    "tokens": None,
                }

                est = meta.get("estimated_tokens")
                if isinstance(est, int) and est > 0:
                    entry["tokens"] = int(est)
                    files_tokens += int(est)
                else:
                    idx = meta.get("message_index")
                    if isinstance(idx, int):
                        missing_by_msg_idx[idx] = entry

                file_entries.append(entry)
            except Exception:
                continue

        for idx, entry in missing_by_msg_idx.items():
            try:
                if 0 <= idx < len(msgs):
                    content = getattr(msgs[idx], "content", "")
                    tok = int(estimate_tokens(str(content or ""), model=model_name))
                    entry["tokens"] = tok
                    files_tokens += tok
            except Exception:
                continue

        history_tokens = 0
        for m in history_msgs:
            try:
                content = getattr(m, "content", "")
                history_tokens += int(estimate_tokens(str(content or ""), model=model_name))
            except Exception:
                continue

        total_est = int(system_tokens) + int(tools_tokens) + int(files_tokens) + int(history_tokens)
        session_key = getattr(self.session, "prompt_cache_key", None)
        prefix_key = getattr(self.session, "_prompt_cache_prefix_key", None)
        cache_session_tokens = self._cache_token_count(session_key)
        cache_prefix_tokens = self._cache_token_count(prefix_key) if isinstance(prefix_key, str) else None

        # Visual stacked bar for approximate context usage.
        cols = int(shutil.get_terminal_size((88, 20)).columns)
        bar_width = max(28, min(70, cols - 28))
        used_width = bar_width
        if context_window_i is not None and context_window_i > 0:
            ratio = 0.0 if context_window_i <= 0 else min(1.0, float(total_est) / float(context_window_i))
            used_width = int(round(ratio * bar_width))
        used_width = max(0, min(bar_width, used_width))
        if total_est > 0 and used_width == 0:
            used_width = 1
        empty_width = bar_width - used_width

        seg_tokens = [int(system_tokens), int(tools_tokens), int(files_tokens), int(history_tokens)]
        seg_widths = _allocate_widths(seg_tokens, width=used_width) if used_width else [0, 0, 0, 0]

        sys_seg = self.ui.cyan("█" * seg_widths[0])
        tools_seg = self.ui.magenta("█" * seg_widths[1])
        files_seg = self.ui.warn("█" * seg_widths[2])
        hist_seg = self.ui.ok("█" * seg_widths[3])
        empty_seg = self.ui.dim("░" * empty_width) if empty_width else ""
        bar = f"[{sys_seg}{tools_seg}{files_seg}{hist_seg}{empty_seg}]"

        print(self.ui.bold("Boxes"))
        header = f"{bar}  ~{total_est} tokens"
        if context_window_i is not None:
            pct = 0.0 if context_window_i <= 0 else (float(total_est) / float(context_window_i) * 100.0)
            header += f" / {context_window_i} ({pct:.1f}%)"
            if total_est > context_window_i:
                header += f"  {self.ui.err(f'OVER by {total_est - context_window_i}')}"
        if isinstance(cache_session_tokens, int):
            header += f"  {self.ui.dim(f'cache={cache_session_tokens}')}"
        print(header)

        legend = (
            f"{self.ui.cyan('SYS')}={system_tokens}  "
            f"{self.ui.magenta('TOOLS')}={tools_tokens}  "
            f"{self.ui.warn('FILES')}={files_tokens}  "
            f"{self.ui.ok('HIST')}={history_tokens}"
        )
        print(self.ui.dim(legend))

        system_fp = _sha12(self.system_prompt or "")
        print(f"- system: sha={system_fp} chars={len(self.system_prompt or '')} tokens≈{system_tokens}")
        if isinstance(sys_mod, dict):
            extra = f" key={sys_mod_key} hash={_module_hash(sys_mod)[:12]}"
            if sys_mod_key:
                tok = self._cache_token_count(sys_mod_key)
                if isinstance(tok, int):
                    extra += f" cache_tokens={tok}"
            print(self.ui.dim(f"  -{extra}"))

        tool_fp = _sha12(json.dumps([getattr(t, '__name__', str(t)) for t in self.tools], sort_keys=True))
        print(f"- tools: count={len(self.tools)} sha={tool_fp} tokens≈{tools_tokens}")
        if isinstance(tools_mod, dict):
            extra = f" key={tools_mod_key} hash={_module_hash(tools_mod)[:12]}"
            if tools_mod_key:
                tok = self._cache_token_count(tools_mod_key)
                if isinstance(tok, int):
                    extra += f" cache_tokens={tok}"
            print(self.ui.dim(f"  -{extra}"))

        if isinstance(cache_prefix_tokens, int):
            print(self.ui.dim(f"- prefix_cache: key={prefix_key} tokens={cache_prefix_tokens}"))
        if isinstance(cache_session_tokens, int):
            print(self.ui.dim(f"- session_cache: key={session_key} tokens={cache_session_tokens}"))

        print(f"- history: messages={len(history_msgs)} tokens≈{history_tokens}")
        if not file_entries:
            print("- files: 0")
            return

        # Second "FILES" box: visualize file token usage within the FILES slice.
        file_tok_list = [int(e.get("tokens") or 0) for e in file_entries]
        file_bytes_list = [e.get("size_bytes") for e in file_entries]
        files_total_tokens = sum(file_tok_list)
        files_total_bytes = sum(int(b) for b in file_bytes_list if isinstance(b, int) and b >= 0)

        print(self.ui.bold("Files"))
        files_bar_width = bar_width
        file_seg_widths = _allocate_widths(file_tok_list, width=files_bar_width) if files_bar_width else [0 for _ in file_tok_list]

        if self.ui.use_color:
            # 12 distinct-ish ANSI colors (normal + bright) for better per-file separation.
            color_codes = ["34", "35", "36", "32", "33", "31", "94", "95", "96", "92", "93", "91"]

            def paint(i: int, s: str) -> str:
                return self.ui._c(s, color_codes[i % len(color_codes)])

            segs = [paint(i, "█" * w) for i, w in enumerate(file_seg_widths)]
            swatches = [paint(i, "■") for i in range(len(file_entries))]
        else:
            symbols = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            segs = [(symbols[i % len(symbols)] * w) for i, w in enumerate(file_seg_widths)]
            swatches = [symbols[i % len(symbols)] for i in range(len(file_entries))]

        files_bar = f"[{''.join(segs)}]"
        header = f"{files_bar}  ~{_fmt_int(files_total_tokens)} tokens"
        if context_window_i is not None and context_window_i > 0:
            pct = float(files_total_tokens) / float(context_window_i) * 100.0 if files_total_tokens > 0 else 0.0
            header += f"  ({pct:.1f}% ctx)"
        if files_total_bytes > 0:
            header += f"  bytes={_fmt_bytes(files_total_bytes)}"
        print(header)

        print(self.ui.dim(f"- files: {len(file_entries)} tokens≈{files_tokens}"))
        for i, entry in enumerate(file_entries, 1):
            path = entry.get("path")
            sha = str(entry.get("sha") or "")[:12]
            tok = entry.get("tokens")
            tok_i = int(tok) if isinstance(tok, int) else 0
            pct_files = (float(tok_i) / float(files_total_tokens) * 100.0) if files_total_tokens > 0 else 0.0
            size_b = entry.get("size_bytes")
            sw = swatches[i - 1] if 0 <= i - 1 < len(swatches) else ""
            print(
                f"  {i}. {sw} {path} sha={sha} bytes={_fmt_bytes(size_b)} tokens≈{_fmt_int(tok_i)} ({pct_files:.1f}%)"
            )

    def _cmd_cache(self, args: List[str]) -> None:
        assert self.session is not None
        if not args:
            print(f"cache_enabled={self.cache_enabled} mode={self.session.prompt_cache_mode} key={self.session.prompt_cache_key}")
            return

        sub = str(args[0] or "").strip().lower()
        if sub in {"on", "enable"}:
            if self.cache_enabled:
                print("cache already enabled")
                return
            self.cache_enabled = True
            self._make_session(keep_transcript=True)
            print(f"cache enabled (mode={self.session.prompt_cache_mode} key={self.session.prompt_cache_key})")
            return

        if sub in {"off", "disable"}:
            if not self.cache_enabled:
                print("cache already disabled")
                return
            self.cache_enabled = False
            # Best-effort: fully disable caching, including any in-process provider-side reuse.
            # This keeps `/cache off` useful for A/B testing (on vs off) inside a single REPL run.
            try:
                if hasattr(self.provider, "prompt_cache_clear"):
                    self.provider.prompt_cache_clear(None)  # type: ignore[union-attr]
            except Exception:
                pass
            self._make_session(keep_transcript=True)
            print("cache disabled")
            return

        if sub in {"clear", "reset"}:
            try:
                if hasattr(self.provider, "prompt_cache_clear"):
                    self.provider.prompt_cache_clear(None)  # type: ignore[union-attr]
            except Exception as e:
                print(f"cache clear failed: {e}")
            # KV mode needs a rebuild to restore context.
            if self.cache_enabled and self.session.prompt_cache_mode == "kv":
                ok = self.session.rebuild_prompt_cache()
                print(f"cache rebuilt from transcript: {ok}")
            else:
                print("cache cleared (provider-side, best-effort)")
            return

        if sub in {"stats", "info"}:
            if len(args) >= 2 and str(args[1]).strip().lower() == "json":
                caps = {}
                try:
                    caps = self.provider.get_prompt_cache_capabilities().to_dict()  # type: ignore[union-attr]
                except Exception:
                    caps = {}
                stats = {}
                try:
                    stats = self.provider.get_prompt_cache_stats()  # type: ignore[union-attr]
                except Exception:
                    stats = {}
                extra = {
                    "session": {
                        "prompt_cache_mode": getattr(self.session, "prompt_cache_mode", None),
                        "prompt_cache_key": getattr(self.session, "prompt_cache_key", None),
                        "prefix_key": getattr(self.session, "_prompt_cache_prefix_key", None),
                        "prefix_namespace": getattr(self.session, "_prompt_cache_prefix_namespace", None),
                        "prefix_modules": getattr(self.session, "_prompt_cache_prefix_modules", None),
                    },
                    "provider": {
                        "name": getattr(self.provider, "provider", self.provider_name),
                        "model": getattr(self.provider, "model", self.model_name),
                    },
                }
                _print_json("prompt_cache_capabilities", caps)
                _print_json("prompt_cache_stats", stats)
                _print_json("cache_session_state", extra)
                return

            caps = {}
            try:
                caps = self.provider.get_prompt_cache_capabilities().to_dict()  # type: ignore[union-attr]
            except Exception:
                caps = {}
            stats = {}
            try:
                stats = self.provider.get_prompt_cache_stats()  # type: ignore[union-attr]
            except Exception:
                stats = {}
            cache_key = getattr(self.session, "prompt_cache_key", None)
            prefix_key = getattr(self.session, "_prompt_cache_prefix_key", None)
            cache_tok = self._cache_token_count(cache_key)
            prefix_tok = self._cache_token_count(prefix_key) if isinstance(prefix_key, str) else None

            print(self.ui.bold("Prompt Cache"))
            print(f"- provider_mode: {caps.get('mode') if isinstance(caps, dict) else None}")
            print(f"- session_mode: {getattr(self.session, 'prompt_cache_mode', None)} enabled={self.cache_enabled}")
            print(f"- session_key: {cache_key}" + (f" tokens={cache_tok}" if cache_tok is not None else ""))
            if prefix_key:
                print(f"- prefix_key:  {prefix_key}" + (f" tokens={prefix_tok}" if prefix_tok is not None else ""))
            pm = getattr(self.session, "_prompt_cache_prefix_modules", {}) or {}
            if isinstance(pm, dict) and pm:
                print("- modules:")
                for mid in ("system", "tools"):
                    item = pm.get(mid)
                    if not isinstance(item, dict):
                        continue
                    dk = _module_cache_key(item)
                    fp = _module_hash(item)[:12]
                    mtok = self._cache_token_count(dk) if dk else None
                    tok_s = f" tokens={mtok}" if isinstance(mtok, int) else ""
                    print(f"  - {mid}: key={dk}{tok_s} hash={fp}")
            if isinstance(stats, dict):
                entries = stats.get("entries")
                max_entries = stats.get("max_entries")
                default_key = stats.get("default_key")
                if entries is not None and max_entries is not None:
                    print(f"- store: entries={entries}/{max_entries} default_key={default_key}")
                gguf = stats.get("gguf")
                if isinstance(gguf, dict):
                    fmt = gguf.get("control_plane_chat_format")
                    per_key = gguf.get("keys")
                    if isinstance(per_key, dict) and isinstance(cache_key, str):
                        cur = per_key.get(cache_key)
                        if isinstance(cur, dict):
                            cap_b = cur.get("capacity_bytes")
                            n_states = cur.get("cache_state_entries")
                            total_b = cur.get("cache_state_total_bytes")
                            max_b = cur.get("cache_state_max_bytes")
                            print(
                                f"- gguf_cache: chat_format={fmt} capacity={_fmt_bytes(cap_b)} "
                                f"states={_fmt_int(n_states)} total={_fmt_bytes(total_b)} max_state={_fmt_bytes(max_b)}"
                            )
            return

        print("Usage: /cache on|off|clear|stats")

    def _cmd_system(self, raw: str) -> None:
        assert self.session is not None
        text = raw.strip()
        if not text:
            print(f"system_prompt={self.system_prompt!r}")
            return

        self.system_prompt = text
        # Update the visible system message (best-effort) for parity with session.system_prompt.
        try:
            for msg in self.session.messages:
                if getattr(msg, "role", None) == "system":
                    msg.content = text
                    break
        except Exception:
            pass
        self.session.system_prompt = text

        print("system prompt updated")

    def _cmd_tools(self, args: List[str]) -> None:
        assert self.session is not None
        if not args:
            print(f"tools={'on' if self.tools else 'off'} count={len(self.tools)}")
            return
        sub = str(args[0] or "").strip().lower()
        if sub == "on":
            if self.tools:
                print("tools already on")
                return
            self.tools = self._default_tools()
        elif sub == "off":
            if not self.tools:
                print("tools already off")
                return
            self.tools = []
        else:
            print("Usage: /tools on|off")
            return

        # Rebuild session to refresh the tools box (and cache prefix).
        self._make_session(keep_transcript=True)
        print(f"tools set to {'on' if self.tools else 'off'}")

    def _cmd_files(self, args: List[str]) -> None:
        assert self.session is not None
        if args and str(args[0]).strip().lower() == "clear":
            removed = self.session.clear_attached_files(rebuild_prompt_cache=True)
            print(self.ui.ok(f"removed {removed} attached file box(es)"))
            return

        items = self.session.get_attached_files()
        if not items:
            print("no attached file boxes")
            return
        for i, it in enumerate(items, 1):
            sha = str(it.get("sha256") or "")[:12]
            size_b = it.get("size_bytes")
            est = it.get("estimated_tokens")
            est_s = f" tokens≈{est}" if isinstance(est, int) else ""
            print(f"{i}. {it.get('path')}  sha={sha}  bytes={size_b}{est_s}")

    def _cmd_history(self, args: List[str]) -> None:
        assert self.session is not None
        n = None
        if args:
            try:
                n = int(args[0])
            except Exception:
                n = None
        msgs = list(getattr(self.session, "messages", []) or [])
        if n is not None and n > 0:
            msgs = msgs[-n:]
        for m in msgs:
            role = getattr(m, "role", "")
            content = getattr(m, "content", "")
            print(f"- {role}: {content[:200]!r}")

    def _cmd_models(self, args: List[str]) -> None:
        query = " ".join(args).strip() if args else None
        models = _list_cached_hf_models(query=query)
        _print_json("local_hf_cache_models", models)

    def _cmd_model(self, argline: str) -> None:
        spec = argline.strip()
        if not spec:
            print(f"current={self.provider_name}:{self.model_name}")
            return
        if ":" not in spec:
            print("Usage: /model <provider:model>")
            return
        prov, model = spec.split(":", 1)
        prov = prov.strip().lower()
        model = model.strip()
        if prov in {"hf"}:
            prov = "huggingface"
        if prov in {"gguf"}:
            prov = "huggingface"
        if prov not in {"mlx", "huggingface"}:
            print("This demo supports provider=mlx|huggingface (transformers/GGUF).")
            return

        old_provider_name = self.provider_name
        old_model_name = self.model_name
        old_provider = self.provider
        old_session = self.session

        self.provider_name = prov
        self.model_name = model
        try:
            self._reset_provider_and_session(keep_transcript=False, allow_fallback=False)
        except Exception as e:
            self.provider_name = old_provider_name
            self.model_name = old_model_name
            self.provider = old_provider
            self.session = old_session
            print(self.ui.err(f"❌ Failed to load {prov}:{model}: {e}"))
            return

        self._banner()

    def _handle_command(self, line: str) -> bool:
        if not line.startswith("/"):
            return False
        parts = shlex.split(line)
        cmd = parts[0][1:].strip().lower()
        args = parts[1:]

        if cmd in {"quit", "exit", "q"}:
            raise SystemExit(0)
        if cmd == "help":
            self._cmd_help()
            return True
        if cmd == "cache":
            self._cmd_cache(args)
            return True
        if cmd == "status":
            self._cmd_status()
            return True
        if cmd == "boxes":
            self._cmd_boxes()
            return True
        if cmd == "stream":
            self._cmd_stream(args)
            return True
        if cmd == "temp":
            self._cmd_temp(args)
            return True
        if cmd == "seed":
            self._cmd_seed(args)
            return True
        if cmd == "system":
            # Keep the rest of the raw line to preserve spaces.
            raw = line[len("/system") :].lstrip()
            self._cmd_system(raw)
            return True
        if cmd == "tools":
            self._cmd_tools(args)
            return True
        if cmd == "files":
            self._cmd_files(args)
            return True
        if cmd == "history":
            self._cmd_history(args)
            return True
        if cmd == "models":
            self._cmd_models(args)
            return True
        if cmd == "model":
            raw = line[len("/model") :].lstrip()
            self._cmd_model(raw)
            return True

        print("Unknown command. Type /help.")
        return True

    def _estimate_turn_input_tokens(self, *, prompt: str, model_name: str) -> Optional[int]:
        """Best-effort token estimate for the full context *before* this turn's generation."""
        if self.session is None:
            return None
        try:
            from abstractcore.utils.token_utils import estimate_tokens
        except Exception:
            return None

        sys_tokens: Optional[int] = None
        tools_tokens: Optional[int] = None
        files_tokens = 0
        history_tokens = 0

        # Prefer cache-derived system/tools token counts when available.
        prefix_modules = getattr(self.session, "_prompt_cache_prefix_modules", {}) or {}
        sys_mod = prefix_modules.get("system") if isinstance(prefix_modules, dict) else None
        tools_mod = prefix_modules.get("tools") if isinstance(prefix_modules, dict) else None
        sys_mod_key = _module_cache_key(sys_mod)
        tools_mod_key = _module_cache_key(tools_mod)
        sys_mod_tokens = self._cache_token_count(sys_mod_key) if sys_mod_key else None
        tools_prefix_tokens = self._cache_token_count(tools_mod_key) if tools_mod_key else None

        if isinstance(sys_mod_tokens, int):
            sys_tokens = sys_mod_tokens
        else:
            try:
                sys_tokens = int(estimate_tokens(str(self.system_prompt or ""), model=model_name))
            except Exception:
                sys_tokens = None

        if (
            isinstance(sys_mod_tokens, int)
            and isinstance(tools_prefix_tokens, int)
            and tools_prefix_tokens >= sys_mod_tokens
        ):
            tools_tokens = int(tools_prefix_tokens - sys_mod_tokens)
        else:
            try:
                tool_defs = list(getattr(self.session, "tools", []) or [])
                tool_schemas = []
                for t in tool_defs:
                    if hasattr(t, "to_dict") and callable(getattr(t, "to_dict")):
                        td = t.to_dict()
                        if isinstance(td, dict):
                            tool_schemas.append(td)
                tools_tokens = int(
                    estimate_tokens(
                        json.dumps(tool_schemas, ensure_ascii=False, sort_keys=True),
                        model=model_name,
                    )
                )
            except Exception:
                tools_tokens = None

        # Files: prefer the stored per-file estimate (computed at attachment time).
        files = self.session.get_attached_files()
        msgs = list(getattr(self.session, "messages", []) or [])
        missing_file_token_estimates: List[int] = []
        for meta in files:
            try:
                est = meta.get("estimated_tokens")
                if isinstance(est, int) and est > 0:
                    files_tokens += int(est)
                else:
                    idx = meta.get("message_index")
                    if isinstance(idx, int):
                        missing_file_token_estimates.append(idx)
            except Exception:
                continue
        for idx in missing_file_token_estimates:
            try:
                if 0 <= idx < len(msgs):
                    content = getattr(msgs[idx], "content", "")
                    files_tokens += int(estimate_tokens(str(content or ""), model=model_name))
            except Exception:
                continue

        # History: all transcript messages except system and file boxes.
        for m in msgs:
            if getattr(m, "role", None) == "system":
                continue
            meta = getattr(m, "metadata", None)
            attached = meta.get("attached_file") if isinstance(meta, dict) else None
            if isinstance(attached, dict) and attached.get("path"):
                continue
            try:
                history_tokens += int(estimate_tokens(str(getattr(m, "content", "") or ""), model=model_name))
            except Exception:
                continue

        try:
            prompt_tokens = int(estimate_tokens(str(prompt or ""), model=model_name))
        except Exception:
            prompt_tokens = 0

        if not isinstance(sys_tokens, int) or not isinstance(tools_tokens, int):
            return None
        return int(sys_tokens) + int(tools_tokens) + int(files_tokens) + int(history_tokens) + int(prompt_tokens)

    def run(self) -> None:
        self._banner()
        while True:
            try:
                line = input(self._prompt()).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if not line:
                continue
            try:
                if self._handle_command(line):
                    continue
            except SystemExit:
                return

            assert self.session is not None

            # Parse @file attachments.
            t_all0 = time.perf_counter()
            t_pre0 = time.perf_counter()
            clean, attached_paths = MessagePreprocessor.parse_file_attachments(
                line, validate_existence=True, verbose=False
            )
            file_box_paths, media_paths = _split_attachments(attached_paths)

            attached_token_est = 0
            if file_box_paths:
                t_attach0 = time.perf_counter()
                result = self.session.attach_files(file_box_paths, prefill_key_mode_cache=True)
                t_attach_ms = _ceil_ms(time.perf_counter() - t_attach0)
                attached = result.get("attached") if isinstance(result, dict) else None
                skipped = result.get("skipped") if isinstance(result, dict) else None
                errors = result.get("errors") if isinstance(result, dict) else None
                if isinstance(attached, list) and attached:
                    for meta in attached:
                        if not isinstance(meta, dict):
                            continue
                        path = meta.get("path")
                        sha = str(meta.get("sha256") or "")[:12]
                        est = meta.get("estimated_tokens")
                        if isinstance(est, int) and est > 0:
                            attached_token_est += int(est)
                        est_s = f" tokens≈{est}" if isinstance(est, int) else ""
                        print(self.ui.ok(f"📎 attached file box: {path} sha={sha}{est_s}"))
                    timing = result.get("timing") if isinstance(result, dict) else None
                    if isinstance(timing, dict):
                        extract_ms = timing.get("extract_ms")
                        cache_ms = timing.get("cache_update_ms")
                        total_ms = timing.get("total_ms")
                        print(
                            self.ui.dim(
                                f"(attach: extract {_fmt_duration(extract_ms)} | cache {_fmt_duration(cache_ms)} | total {_fmt_duration(total_ms)})"
                            )
                        )
                    else:
                        print(self.ui.dim(f"(attach took {_fmt_duration(t_attach_ms)})"))
                if isinstance(skipped, list) and skipped:
                    for item in skipped:
                        if not isinstance(item, dict):
                            continue
                        print(self.ui.dim(f"↪ skipped: {item.get('path')} ({item.get('reason')})"))
                if isinstance(errors, list) and errors:
                    for item in errors:
                        if not isinstance(item, dict):
                            continue
                        print(self.ui.err(f"✖ attach error: {item.get('path')}: {item.get('error')}"))

            if media_paths:
                shown = ", ".join(str(p) for p in media_paths[:6])
                more = f" (+{len(media_paths) - 6} more)" if len(media_paths) > 6 else ""
                print(self.ui.ok(f"📎 media: {shown}{more}"))

            prompt = clean.strip()
            if not prompt and (file_box_paths or media_paths):
                prompt = "Please analyze the attached file(s)."

            cache_key = getattr(self.session, "prompt_cache_key", None)
            tok_before = self._cache_token_count(cache_key)

            model_name = str(getattr(self.provider, "model", "") or self.model_name)
            in_tok_est = self._estimate_turn_input_tokens(prompt=prompt, model_name=model_name)

            pre_s = time.perf_counter() - t_pre0
            t_gen0 = time.perf_counter()
            resp = self.session.generate(
                prompt,
                media=media_paths if media_paths else None,
                max_output_tokens=self.max_output_tokens,
                # Always stream internally (when possible) so TTFT/prefill metrics exist even when
                # live printing is disabled via `/stream off`.
                stream=bool(self._force_internal_streaming or self.stream),
            )

            is_stream_iter = hasattr(resp, "__next__")
            streamed_text = ""
            last_chunk = None
            ttft_ms: Optional[int] = None  # wall clock to first yielded stream chunk
            tift_ms: Optional[int] = None
            final_reasoning: Optional[str] = None

            if is_stream_iter:
                live_print = bool(self.stream)
                if live_print:
                    print(self.ui.bold("Assistant:"))
                t_first_chunk: Optional[float] = None
                t_first_visible: Optional[float] = None
                placeholder_printed = False

                for chunk in resp:  # type: ignore[union-attr]
                    last_chunk = chunk
                    if t_first_chunk is None:
                        t_first_chunk = time.perf_counter()
                        ttft_ms = max(0, _ceil_ms(t_first_chunk - t_gen0))

                    text = getattr(chunk, "content", None)
                    if isinstance(text, str) and text:
                        if t_first_visible is None:
                            t_first_visible = time.perf_counter()
                            if placeholder_printed and sys.stdout.isatty():
                                # Clear the placeholder line before printing visible output.
                                sys.stdout.write("\r\x1b[2K")
                        if live_print:
                            sys.stdout.write(text)
                            sys.stdout.flush()
                        streamed_text += text
                    elif live_print and t_first_visible is None and not placeholder_printed and t_first_chunk is not None:
                        # We received stream chunks but nothing visible yet (commonly because the model
                        # started with <think>…</think> and AbstractCore strips it).
                        placeholder_printed = True
                        if sys.stdout.isatty():
                            sys.stdout.write(self.ui.dim("…"))
                            sys.stdout.flush()

                    # Reasoning may arrive in a metadata-only terminal chunk (content="") or in-band.
                    chunk_reasoning = getattr(chunk, "reasoning", None)
                    if isinstance(chunk_reasoning, str) and chunk_reasoning.strip():
                        final_reasoning = chunk_reasoning
                if live_print:
                    print("\n")

                if t_first_visible is not None:
                    tift_ms = max(0, _ceil_ms(t_first_visible - t_gen0))
                if ttft_ms is None and t_first_visible is not None:
                    ttft_ms = max(0, _ceil_ms(t_first_visible - t_gen0))

                # Fallback: if we never observed a reasoning-bearing chunk during the stream,
                # read it from the last chunk.
                if not final_reasoning and last_chunk is not None:
                    final_reasoning = getattr(last_chunk, "reasoning", None)
                if not live_print:
                    # Buffered mode: print after stream completes (matches `/stream off` UX).
                    print(self.ui.bold("Assistant:"))
                    print((streamed_text or "").strip() + "\n")
                if isinstance(final_reasoning, str) and final_reasoning.strip():
                    print(self.ui.dim("Reasoning:"))
                    print(self.ui.dim(str(final_reasoning).strip()) + "\n")
            else:
                last_chunk = resp

            gen_s = time.perf_counter() - t_gen0

            tok_after = self._cache_token_count(cache_key)
            delta_tok = None
            if tok_before is not None and tok_after is not None:
                delta_tok = tok_after - tok_before

            content = None
            if is_stream_iter:
                content = streamed_text
            else:
                content = getattr(resp, "content", None) if resp is not None else None
                final_reasoning = getattr(resp, "reasoning", None) if resp is not None else None

                print(self.ui.bold("Assistant:"))
                print((content or "").strip() + "\n")

                if final_reasoning:
                    print(self.ui.dim("Reasoning:"))
                    print(self.ui.dim(str(final_reasoning).strip()) + "\n")

            from abstractcore.utils.token_utils import TokenUtils

            # Token counts:
            # - input: estimated full context tokens for this turn (system+tools+files+history+prompt),
            #          computed BEFORE generation so it does not include the assistant reply.
            # - output: estimated from visible content + extracted reasoning (if present).
            in_tok = int(in_tok_est) if isinstance(in_tok_est, int) else None

            out_tok = getattr(last_chunk, "output_tokens", None) if last_chunk is not None else None
            if not isinstance(out_tok, int):
                try:
                    combined = str(content or "")
                    if isinstance(final_reasoning, str) and final_reasoning.strip():
                        combined = combined + "\n\n" + final_reasoning
                    out_tok = int(TokenUtils.estimate_tokens(combined, model_name))
                except Exception:
                    out_tok = 0

            try:
                tot_tok = int(in_tok or 0) + int(out_tok or 0) if in_tok is not None else None
            except Exception:
                tot_tok = None

            gen_ms = max(1, _ceil_ms(gen_s))
            total_ms = max(0, _ceil_ms(time.perf_counter() - t_all0))

            # Derive prefill vs decode from TTFT when available (streaming mode).
            prefill_ms = int(ttft_ms) if isinstance(ttft_ms, int) else None
            decode_ms: Optional[int] = None
            if prefill_ms is not None:
                decode_ms = max(0, int(gen_ms) - int(prefill_ms))

            # In cache-forwarding modes (KV/keyed prefix), TTFT primarily reflects how many
            # *uncached* tokens were evaluated before generation could start. Using the full
            # logical context token count can make tk/s look wildly inflated when most of the
            # context is already in the cache. Prefer an "uncached" estimate when possible.
            # Estimate how many *uncached* input tokens were processed before generation started.
            # When cache is disabled, use the full logical input token estimate for consistent
            # "prefill tk/s" observability.
            prefill_token_basis: Optional[int] = None
            cache_mode = str(getattr(self.session, "prompt_cache_mode", "off") or "off")
            if isinstance(in_tok, int) and in_tok > 0:
                if cache_mode == "off":
                    prefill_token_basis = int(in_tok)
                elif isinstance(tok_before, int) and tok_before >= 0:
                    diff = int(in_tok) - int(tok_before)
                    if diff > 0:
                        prefill_token_basis = diff
            if prefill_token_basis is None:
                try:
                    prefill_token_basis = int(TokenUtils.estimate_tokens(str(prompt or ""), model_name))
                except Exception:
                    prefill_token_basis = in_tok

            prefill_tps = (
                _ceil_div(int(prefill_token_basis or 0) * 1000, max(1, int(prefill_ms)))
                if (prefill_ms is not None and isinstance(prefill_token_basis, int) and prefill_ms > 0)
                else None
            )
            decode_tps = (
                _ceil_div(int(out_tok or 0) * 1000, max(1, int(decode_ms)))
                if (decode_ms is not None and decode_ms >= 0)
                else _ceil_div(int(out_tok or 0) * 1000, max(1, int(gen_ms)))
            )

            stats = f"{_fmt_total_time(total_ms)}"
            stats += f"  tift {_fmt_ms(tift_ms)}  ttft {_fmt_ms(ttft_ms)}"
            if prefill_ms is not None:
                stats += f"  |  prefill {_fmt_int(prefill_ms)}ms {_fmt_int(prefill_tps)} tk/s"
                stats += f"  |  decode {_fmt_int(decode_ms)}ms {_fmt_int(decode_tps)} tk/s"
            else:
                stats += f"  |  decode {_fmt_int(gen_ms)}ms {_fmt_int(decode_tps)} tk/s"
            stats += f"  |  tok in {_fmt_int(in_tok)} out {_fmt_int(int(out_tok or 0))} tot {_fmt_int(tot_tok)}"
            if isinstance(tok_after, int):
                stats += f"  |  cache {_fmt_int(tok_after)}"
                if isinstance(delta_tok, int):
                    stats += f" (Δ{delta_tok:+d})"
            print(self.ui.dim(stats))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="huggingface", help="mlx|huggingface")
    ap.add_argument("--model", default="sshleifer/tiny-gpt2", help="model id or path (GGUF can be a .gguf file path)")
    ap.add_argument("--cache", default="on", choices=["on", "off"])
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI colors (even in a TTY)")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--max-output-tokens", type=int, default=8192)
    args = ap.parse_args()

    provider = args.provider
    model = args.model

    # Default UX: avoid importing PyTorch/transformers at startup when local models are available.
    # This keeps GGUF Metal offload usable if the user later switches to GGUF in the same REPL.
    import sys

    provider_explicit = "--provider" in sys.argv
    model_explicit = "--model" in sys.argv
    if (not provider_explicit) and (not model_explicit) and provider == "huggingface" and model == "sshleifer/tiny-gpt2":
        try:
            cached = _list_cached_hf_models(limit=40)
            mlx_models = cached.get("mlx") if isinstance(cached, dict) else None
            hf_gguf_models = cached.get("hf_gguf") if isinstance(cached, dict) else None
            hf_tf_models = cached.get("hf_transformers") if isinstance(cached, dict) else None
            if isinstance(mlx_models, list) and mlx_models:
                provider = "mlx"
                model = str(mlx_models[0])
            elif isinstance(hf_gguf_models, list) and hf_gguf_models:
                provider = "huggingface"
                model = str(hf_gguf_models[0])
            elif isinstance(hf_tf_models, list) and hf_tf_models:
                provider = "huggingface"
                model = str(hf_tf_models[0])
        except Exception:
            pass

    repl = _Repl(
        provider=provider,
        model=model,
        cache_enabled=(args.cache == "on"),
        max_tokens=args.max_tokens,
        max_output_tokens=args.max_output_tokens,
    )
    if args.no_color:
        repl.ui = _Ui(use_color=False)
    repl.run()


if __name__ == "__main__":
    main()
