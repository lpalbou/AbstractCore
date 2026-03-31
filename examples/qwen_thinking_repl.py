#!/usr/bin/env python3
"""
Mini REPL for quickly probing Qwen3.5 thinking behavior across local providers.

Goals:
- Switch provider/model quickly (LM Studio, Ollama, HuggingFace GGUF)
- Toggle `thinking=` and inspect both parsed output (content + reasoning) and raw responses
- Keep it simple and dependency-free (ANSI colors only)

This script uses AbstractCore providers and prints:
- Parsed: `response.content` and `response.reasoning`
- Provider request payload (when available) and `response.raw_response` (JSON pretty-printed)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

from abstractcore.core.types import GenerateResponse, Message
from abstractcore.core.session import BasicSession
from abstractcore.config.manager import get_config_manager
from abstractcore.providers.huggingface_provider import HuggingFaceProvider
from abstractcore.providers.lmstudio_provider import LMStudioProvider
from abstractcore.providers.ollama_provider import OllamaProvider


class Ansi:
    RESET = "\x1b[0m"
    DIM = "\x1b[2m"
    BOLD = "\x1b[1m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"
    GRAY = "\x1b[90m"


def _c(text: str, *styles: str) -> str:
    return "".join(styles) + text + Ansi.RESET


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return pformat(obj, width=120)


def _colorize_json(text: str) -> str:
    # Minimal, fast-ish JSON-ish colorizer (regex-free to stay robust on non-JSON pprint).
    out: List[str] = []
    in_str = False
    esc = False
    buf: List[str] = []

    def flush_buf_as_string() -> None:
        if not buf:
            return
        s = "".join(buf)
        out.append(_c(s, Ansi.GREEN))
        buf.clear()

    for ch in text:
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                flush_buf_as_string()
                in_str = False
            continue

        if ch == "\"":
            in_str = True
            buf.append(ch)
            continue

        if ch in "{}[]":
            out.append(_c(ch, Ansi.GRAY))
        elif ch in ":,":
            out.append(_c(ch, Ansi.DIM))
        else:
            out.append(ch)

    # If we ended inside a string, flush whatever we have.
    if buf:
        out.append(_c("".join(buf), Ansi.GREEN))

    # Avoid token replacements that could accidentally modify string content; only add ANSI codes.
    return "".join(out)


def _print_block(title: str, body: str) -> None:
    print(_c(f"\n== {title} ==", Ansi.BOLD, Ansi.CYAN))
    print(body.rstrip("\n"))


def _extract_provider_request(resp: GenerateResponse) -> Optional[Dict[str, Any]]:
    if not isinstance(resp.metadata, dict):
        return None
    pr = resp.metadata.get("_provider_request")
    return pr if isinstance(pr, dict) else None


def _lmstudio_list_models(base_url: str, timeout_s: float = 2.0) -> List[str]:
    url = base_url.rstrip("/") + "/models"
    with httpx.Client(timeout=timeout_s) as client:
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
    items = data.get("data", []) if isinstance(data, dict) else []
    out: List[str] = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("id"), str):
            out.append(it["id"])
    return out


def _ollama_list_models(base_url: str, timeout_s: float = 2.0) -> List[str]:
    url = base_url.rstrip("/") + "/api/tags"
    with httpx.Client(timeout=timeout_s) as client:
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
    models = data.get("models", []) if isinstance(data, dict) else []
    out: List[str] = []
    for it in models:
        name = it.get("name") if isinstance(it, dict) else None
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return out


def _guess_hf_cache_roots() -> List[Path]:
    roots: List[Path] = []
    # AbstractCore config (source of truth for our HF provider).
    try:
        cfg = get_config_manager().config
        hf_base = Path(str(cfg.cache.huggingface_cache_dir)).expanduser()
        roots.append(hf_base / "hub")
    except Exception:
        pass
    # Common env overrides.
    for env in ("HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        v = os.getenv(env)
        if isinstance(v, str) and v.strip():
            p = Path(v).expanduser()
            # HF_HOME usually contains `hub/`.
            roots.append(p / "hub" if env == "HF_HOME" else p)
    # Default.
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    # De-dupe while preserving order.
    seen = set()
    uniq: List[Path] = []
    for r in roots:
        try:
            k = str(r.resolve())
        except Exception:
            k = str(r)
        if k not in seen:
            uniq.append(r)
            seen.add(k)
    return uniq


def _hf_list_gguf_models(max_items: int = 80, name_filter: Optional[str] = None) -> List[str]:
    """List HF cached repos that contain at least one `.gguf` snapshot file."""
    out: List[str] = []
    seen: set[str] = set()
    needle = name_filter.lower() if isinstance(name_filter, str) and name_filter.strip() else None

    for root in _guess_hf_cache_roots():
        if not root.exists():
            continue
        # HF cache layout: hub/models--ORG--REPO/snapshots/<hash>/*.gguf
        # Iterate GGUF files directly to avoid scanning every repo dir.
        for gguf_path in root.glob("models--*/snapshots/*/*.gguf"):
            repo_dir = gguf_path.parent.parent.parent
            if not repo_dir.is_dir():
                continue
            name = repo_dir.name
            if not name.startswith("models--"):
                continue
            rest = name.split("models--", 1)[-1]
            parts = rest.split("--")
            if len(parts) < 2:
                continue
            org = parts[0]
            repo = "--".join(parts[1:])
            model_id = f"{org}/{repo}"
            if needle and needle not in model_id.lower():
                continue
            if model_id in seen:
                continue
            seen.add(model_id)
            out.append(model_id)
            if len(out) >= max_items:
                return out

    return out


def _provider_help() -> str:
    return "\n".join(
        [
            _c("Commands:", Ansi.BOLD),
            "  :help                     Show this help",
            "  :status                   Show current settings",
            "  :history                  Show session messages",
            "  :clear                    Clear session (keeps system)",
            "  :system <TEXT>            Set/replace system prompt",
            "  :save <PATH>              Save session archive (JSON)",
            "  :load <PATH>              Load session archive (JSON)",
            "  :lastreq                  Show last provider request",
            "  :provider <lmstudio|ollama|huggingface>",
            "  :model <MODEL_ID|N>        Set model for current provider (or pick from :list by number)",
            "  :thinking <auto|on|off|none|low|medium|high>",
            "  :list [provider]          List local models (all providers by default)",
            "  :showraw <on|off>          Toggle printing raw response",
            "  :showreq <on|off>          Toggle printing provider request payload (when available)",
            "  :showgen <on|off>          Toggle printing GenerateResponse (parsed) JSON",
            "  :unload                   Unload current model (best-effort)",
            "  :quit                      Exit",
            "",
            _c("Notes:", Ansi.BOLD),
            "- `thinking=` is best-effort and provider/model dependent.",
            "- Raw responses are printed verbatim (no truncation); disable with `:showraw off` if too large.",
        ]
    )


def _parse_on_off(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError("expected on/off")


def _build_provider(
    provider: str,
    model: str,
    *,
    lmstudio_base_url: str,
    ollama_base_url: str,
) -> Any:
    p = provider.strip().lower()
    if p == "lmstudio":
        return LMStudioProvider(model=model, base_url=lmstudio_base_url)
    if p == "ollama":
        return OllamaProvider(model=model, base_url=ollama_base_url)
    if p == "huggingface":
        # Prefer GGUF models in local cache (e.g. unsloth/Qwen3.5-4B-GGUF).
        return HuggingFaceProvider(model=model)
    raise ValueError(f"unknown provider: {provider}")

def _generate_response_to_dict(resp: GenerateResponse) -> Dict[str, Any]:
    # Keep strings verbatim (no truncation).
    return {
        "content": resp.content,
        "reasoning": resp.reasoning,
        "model": resp.model,
        "finish_reason": resp.finish_reason,
        "usage": resp.usage,
        "tool_calls": resp.tool_calls,
        "metadata": resp.metadata,
        "gen_time_ms": resp.gen_time,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="lmstudio", choices=["lmstudio", "ollama", "huggingface"])
    ap.add_argument("--thinking", default="none", help="auto|on|off|none|low|medium|high")
    ap.add_argument("--show-gen", default="on", choices=["on", "off"])
    ap.add_argument("--show-raw", default="off", choices=["on", "off"])
    ap.add_argument("--show-req", default="on", choices=["on", "off"])
    ap.add_argument("--max-output-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--system", default=None, help="Optional system prompt for the session")
    ap.add_argument("--load-session", default=None, help="Load a BasicSession archive JSON file")
    ap.add_argument("--lmstudio-base-url", default=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"))
    ap.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    ap.add_argument(
        "--model",
        default="qwen3.5-4b@q4_k_m",
        help="Initial model id (provider-specific). For HF GGUF, try: unsloth/Qwen3.5-4B-GGUF",
    )
    args = ap.parse_args(argv)

    provider_name = args.provider
    model_name = args.model
    thinking = str(args.thinking or "").strip().lower()
    show_gen = args.show_gen == "on"
    show_raw = args.show_raw == "on"
    show_req = args.show_req == "on"

    llm = _build_provider(
        provider_name,
        model_name,
        lmstudio_base_url=args.lmstudio_base_url,
        ollama_base_url=args.ollama_base_url,
    )

    if args.load_session:
        session = BasicSession.load(args.load_session, provider=llm)
    else:
        session = BasicSession(provider=llm, system_prompt=args.system)

    last_provider_request: Optional[Dict[str, Any]] = None
    last_listed_entries: List[Tuple[str, str]] = []

    print(_c("Qwen Thinking REPL", Ansi.BOLD, Ansi.CYAN))
    print(_c(f"provider={provider_name}  model={model_name}  thinking={thinking}", Ansi.DIM))
    print(_c("Type :help for commands.\n", Ansi.DIM))

    while True:
        try:
            line = input(_c("> ", Ansi.BOLD)).rstrip("\n")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line.strip():
            continue

        if line.startswith(":"):
            cmd, *rest = line[1:].strip().split(" ", 1)
            arg = rest[0].strip() if rest else ""

            if cmd in {"q", "quit", "exit"}:
                break
            if cmd == "help":
                print(_provider_help())
                continue
            if cmd == "status":
                print(_c(f"provider={provider_name}  model={model_name}  thinking={thinking}", Ansi.CYAN))
                print(_c(f"show_gen={show_gen}  show_raw={show_raw}  show_req={show_req}", Ansi.DIM))
                print(_c(f"messages={len(session.messages)}", Ansi.DIM))
                if session.system_prompt:
                    print(_c("system_prompt=set", Ansi.DIM))
                continue
            if cmd == "history":
                hist = session.get_history(include_system=True)
                _print_block("Session Messages", _colorize_json(_safe_json_dumps(hist)))
                continue
            if cmd == "clear":
                session.clear_history(keep_system=True)
                print(_c("cleared session (kept system messages)", Ansi.CYAN))
                continue
            if cmd == "system":
                if not arg:
                    print(_c("usage: :system <TEXT>", Ansi.RED))
                    continue
                # Replace session-level system prompt and update/insert the system message in history.
                session.system_prompt = arg
                # Find the first system message (if any) and replace it; otherwise insert at top.
                replaced = False
                for m in session.messages:
                    if getattr(m, "role", None) == "system":
                        m.content = arg
                        replaced = True
                        break
                if not replaced:
                    session.messages.insert(0, Message(role="system", content=arg))
                print(_c("system_prompt updated", Ansi.CYAN))
                continue
            if cmd == "save":
                if not arg:
                    print(_c("usage: :save <PATH>", Ansi.RED))
                    continue
                path = Path(arg).expanduser()
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    session.save(path)
                    print(_c(f"saved session -> {path}", Ansi.CYAN))
                except Exception as e:  # noqa: BLE001
                    print(_c(f"save failed: {type(e).__name__}: {e}", Ansi.RED))
                continue
            if cmd == "load":
                if not arg:
                    print(_c("usage: :load <PATH>", Ansi.RED))
                    continue
                path = Path(arg).expanduser()
                try:
                    session = BasicSession.load(path, provider=llm)
                    print(_c(f"loaded session <- {path}", Ansi.CYAN))
                except Exception as e:  # noqa: BLE001
                    print(_c(f"load failed: {type(e).__name__}: {e}", Ansi.RED))
                continue
            if cmd == "lastreq":
                if last_provider_request is None:
                    _print_block("Last Provider Request", _c("(none yet)", Ansi.DIM))
                else:
                    _print_block("Last Provider Request", _colorize_json(_safe_json_dumps(last_provider_request)))
                continue
            if cmd == "provider":
                if not arg:
                    print(_c("usage: :provider <lmstudio|ollama|huggingface>", Ansi.RED))
                    continue
                new_provider = arg.strip().lower()
                if new_provider not in {"lmstudio", "ollama", "huggingface"}:
                    print(_c(f"unknown provider: {new_provider}", Ansi.RED))
                    continue
                # Unload HF GGUF model when switching away (best-effort).
                if provider_name == "huggingface":
                    try:
                        llm.unload_model(getattr(llm, "model", model_name))
                    except Exception:
                        pass
                provider_name = new_provider
                llm = _build_provider(
                    provider_name,
                    model_name,
                    lmstudio_base_url=args.lmstudio_base_url,
                    ollama_base_url=args.ollama_base_url,
                )
                session.provider = llm
                print(_c(f"provider={provider_name}", Ansi.CYAN))
                continue
            if cmd == "model":
                if not arg:
                    print(_c("usage: :model <MODEL_ID|N>", Ansi.RED))
                    continue
                raw = arg.strip()
                new_model = raw
                if raw.isdigit():
                    idx = int(raw)
                    if not last_listed_entries:
                        print(_c("no model list available; run :list first", Ansi.RED))
                        continue
                    if idx < 1 or idx > len(last_listed_entries):
                        print(_c(f"invalid index {idx}; pick 1..{len(last_listed_entries)}", Ansi.RED))
                        continue
                    selected_provider, selected_model = last_listed_entries[idx - 1]
                    if selected_provider != provider_name:
                        # Unload HF GGUF model when switching away (best-effort).
                        if provider_name == "huggingface":
                            try:
                                llm.unload_model(getattr(llm, "model", model_name))
                            except Exception:
                                pass
                        provider_name = selected_provider
                    new_model = selected_model
                if provider_name == "huggingface":
                    try:
                        llm.unload_model(getattr(llm, "model", model_name))
                    except Exception:
                        pass
                model_name = new_model
                llm = _build_provider(
                    provider_name,
                    model_name,
                    lmstudio_base_url=args.lmstudio_base_url,
                    ollama_base_url=args.ollama_base_url,
                )
                session.provider = llm
                print(_c(f"model={model_name}", Ansi.CYAN))
                continue
            if cmd == "thinking":
                if not arg:
                    print(_c("usage: :thinking <auto|on|off|none|low|medium|high>", Ansi.RED))
                    continue
                thinking = arg.strip().lower()
                print(_c(f"thinking={thinking}", Ansi.CYAN))
                continue
            if cmd == "showraw":
                try:
                    show_raw = _parse_on_off(arg)
                    print(_c(f"show_raw={show_raw}", Ansi.CYAN))
                except Exception:
                    print(_c("usage: :showraw <on|off>", Ansi.RED))
                continue
            if cmd == "showreq":
                try:
                    show_req = _parse_on_off(arg)
                    print(_c(f"show_req={show_req}", Ansi.CYAN))
                except Exception:
                    print(_c("usage: :showreq <on|off>", Ansi.RED))
                continue
            if cmd == "showgen":
                try:
                    show_gen = _parse_on_off(arg)
                    print(_c(f"show_gen={show_gen}", Ansi.CYAN))
                except Exception:
                    print(_c("usage: :showgen <on|off>", Ansi.RED))
                continue
            if cmd == "list":
                target = arg.strip().lower() if arg else ""
                if target in {"hf"}:
                    target = "huggingface"
                providers = [target] if target else ["lmstudio", "ollama", "huggingface"]
                if any(p not in {"lmstudio", "ollama", "huggingface"} for p in providers):
                    print(_c("usage: :list [lmstudio|ollama|huggingface]", Ansi.RED))
                    continue

                last_listed_entries = []
                index = 1

                for p in providers:
                    try:
                        if p == "lmstudio":
                            models = _lmstudio_list_models(args.lmstudio_base_url)
                        elif p == "ollama":
                            models = _ollama_list_models(args.ollama_base_url)
                        else:
                            models = _hf_list_gguf_models()
                    except Exception as e:  # noqa: BLE001
                        print(_c(f"({p}) list failed: {type(e).__name__}: {e}", Ansi.RED))
                        models = []

                    if not models:
                        print(_c(f"({p}) (no models found)", Ansi.DIM))
                        continue

                    for m in models:
                        last_listed_entries.append((p, m))
                        print(f"{p:>10} {index:>3}. {m}")
                        index += 1
                continue
            if cmd == "unload":
                try:
                    llm.unload_model(getattr(llm, "model", model_name))
                    print(_c(f"unloaded model={model_name} (best-effort)", Ansi.CYAN))
                except Exception as e:  # noqa: BLE001
                    print(_c(f"unload failed: {type(e).__name__}: {e}", Ansi.RED))
                continue

            print(_c(f"unknown command: :{cmd}", Ansi.RED))
            continue

        # Generate
        try:
            resp = session.generate(
                line,
                thinking=thinking,
                max_tokens=args.max_output_tokens,
                temperature=args.temperature,
            )
        except Exception as e:  # noqa: BLE001
            print(_c(f"error: {type(e).__name__}: {e}", Ansi.RED))
            continue

        if not isinstance(resp, GenerateResponse):
            # Defensive: structured outputs may return a model instance, etc.
            print(_c(f"(non-GenerateResponse) {type(resp).__name__}", Ansi.YELLOW))
            print(str(resp))
            continue

        print(_c("\n" + ("-" * 80), Ansi.DIM))
        print(_c(f"provider={provider_name}  model={model_name}  thinking={thinking}", Ansi.DIM))
        if resp.finish_reason:
            print(_c(f"finish_reason={resp.finish_reason}", Ansi.DIM))
        if isinstance(resp.usage, dict) and resp.usage:
            print(_c(f"usage={resp.usage}", Ansi.DIM))

        content = resp.content or ""
        reasoning = resp.reasoning or ""
        if reasoning:
            _print_block("Reasoning", reasoning)
        _print_block("Content", content if content else _c("(empty)", Ansi.DIM))

        if show_gen:
            _print_block("GenerateResponse", _colorize_json(_safe_json_dumps(_generate_response_to_dict(resp))))

        pr = _extract_provider_request(resp)
        last_provider_request = pr
        if show_req:
            if pr is None:
                _print_block("Provider Request", _c("(not available)", Ansi.DIM))
            else:
                _print_block("Provider Request", _colorize_json(_safe_json_dumps(pr)))

        if show_raw:
            _print_block("Raw Response", _colorize_json(_safe_json_dumps(resp.raw_response)))

    # Best-effort unload for HF GGUF models.
    if provider_name == "huggingface":
        try:
            llm.unload_model(getattr(llm, "model", model_name))
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
