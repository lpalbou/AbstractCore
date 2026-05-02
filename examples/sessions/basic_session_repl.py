#!/usr/bin/env python3
"""
Minimal multi-turn REPL using BasicSession.

What this demonstrates
- Creating a provider with `create_llm(...)`.
- Creating a `BasicSession` to hold system prompt + message history.
- Multi-turn chat in a loop.
- Saving/loading a session transcript.

Run
  python examples/sessions/basic_session_repl.py --provider ollama --model llama3.2:3b
  python examples/sessions/basic_session_repl.py --provider lmstudio --model openai/gpt-oss-20b --base-url http://localhost:1234/v1

Commands
  :help
  :history
  :clear
  :save <path.json>
  :load <path.json>
  :quit
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from abstractcore import BasicSession, create_llm


def _help() -> str:
    return (
        "Commands:\n"
        "  :help              show this help\n"
        "  :history           show message roles + sizes\n"
        "  :clear             clear history (keep system)\n"
        "  :save <path.json>  save session transcript\n"
        "  :load <path.json>  load session transcript\n"
        "  :quit              exit\n"
        "\n"
        "Anything else is sent as a user message.\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="ollama", help="e.g. ollama|lmstudio|openai|anthropic")
    ap.add_argument("--model", default="llama3.2:3b", help="provider model id")
    ap.add_argument("--base-url", default=None, help="Optional base URL (lmstudio/openai-compatible).")
    ap.add_argument(
        "--system",
        default="You are a helpful assistant. Be concise and correct.",
        help="System prompt for the session.",
    )
    ap.add_argument("--stream", action="store_true", help="Stream tokens as they arrive.")
    args = ap.parse_args()

    llm_kwargs: Dict[str, Any] = {"model": args.model}
    if args.base_url:
        llm_kwargs["base_url"] = args.base_url
    llm = create_llm(args.provider, **llm_kwargs)

    session = BasicSession(provider=llm, system_prompt=args.system)

    print("BasicSession REPL")
    print(f"provider={args.provider} model={args.model} stream={'on' if args.stream else 'off'}")
    print("Type :help for commands.\n")

    while True:
        try:
            line = input("> ").rstrip("\n")
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
                print(_help())
                continue
            if cmd == "history":
                for i, msg in enumerate(session.get_messages(), 1):
                    content = str(getattr(msg, "content", "") or "")
                    print(f"{i:>3}. role={msg.role:<9} chars={len(content):>6}")
                continue
            if cmd == "clear":
                session.clear_history(keep_system=True)
                print("cleared session (kept system messages)")
                continue
            if cmd == "save":
                if not arg:
                    print("usage: :save <path.json>")
                    continue
                path = Path(arg).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                session.save(path)
                print(f"saved -> {path}")
                continue
            if cmd == "load":
                if not arg:
                    print("usage: :load <path.json>")
                    continue
                path = Path(arg).expanduser()
                session = BasicSession.load(path, provider=llm)
                print(f"loaded <- {path}")
                continue

            print(f"unknown command: :{cmd} (try :help)")
            continue

        # Normal user turn.
        if args.stream:
            print("assistant: ", end="", flush=True)
            for chunk in session.generate(line, stream=True):
                text = str(getattr(chunk, "content", "") or "")
                if text:
                    print(text, end="", flush=True)
            print()
        else:
            resp = session.generate(line)
            print("assistant:", str(getattr(resp, "content", "") or "").strip())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

