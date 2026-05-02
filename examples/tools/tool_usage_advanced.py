#!/usr/bin/env python3
"""
Advanced tool calling patterns in AbstractCore (host-controlled execution).

What this demonstrates
- Tool definitions with `@tool`.
- Host-side tool execution policy (allowlist/denylist).
- Argument validation/sanitization before executing tools.
- Feeding tool results back to the model to produce a final answer.

Why this matters
- Tool calls are *untrusted input*: the model can request unsafe or nonsensical actions.
- Your application must decide which tools can run, with which args, and under what conditions.

Run
  python examples/tools/tool_usage_advanced.py --provider ollama --model llama3.2:3b
  python examples/tools/tool_usage_advanced.py --provider lmstudio --model openai/gpt-oss-20b --base-url http://localhost:1234/v1
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

from abstractcore import create_llm, tool
from abstractcore.tools import ToolCall, ToolResult
from abstractcore.tools.registry import ToolRegistry


@tool
def search_docs(query: str) -> str:
    """Search a document corpus (demo stub)."""
    q = " ".join(str(query or "").split()).strip()
    if not q:
        raise ValueError("empty query")
    # Deterministic stub so the example is runnable offline.
    return f"Top hit for '{q}': (stub) Use embeddings/RAG in a real app."


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (demo stub; intentionally dangerous for policy demo)."""
    return f"(stub) Email sent to={to!r} subject={subject!r} body_len={len(body or '')}"


def _parse_tool_calls(raw_calls: Optional[List[Dict[str, Any]]]) -> List[ToolCall]:
    out: List[ToolCall] = []
    for item in raw_calls or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        args = item.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        out.append(ToolCall(name=name, arguments=args))
    return out


def _sanitize_args(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal sanitizer for a demo: cap string sizes and drop unexpected keys.
    if name == "search_docs":
        q = args.get("query")
        q = " ".join(str(q or "").split()).strip()
        return {"query": q[:400]}
    if name == "send_email":
        to = " ".join(str(args.get("to") or "").split()).strip()[:200]
        subject = " ".join(str(args.get("subject") or "").split()).strip()[:200]
        body = str(args.get("body") or "")[:2000]
        return {"to": to, "subject": subject, "body": body}
    return {}


def _execute_with_policy(tool_calls: List[ToolCall]) -> List[ToolResult]:
    registry = ToolRegistry()
    registry.register(search_docs.tool_definition)
    registry.register(send_email.tool_definition)

    allowed = {"search_docs"}  # deny send_email by default
    results: List[ToolResult] = []
    for idx, call in enumerate(tool_calls):
        call_id = call.call_id or f"call:{idx}"
        if call.name not in allowed:
            results.append(
                ToolResult(
                    call_id=call_id,
                    output=None,
                    error=f"DENIED: tool '{call.name}' is not allowed by policy",
                    success=False,
                )
            )
            continue
        safe_args = _sanitize_args(call.name, dict(call.arguments or {}))
        results.append(registry.execute_tool(ToolCall(name=call.name, arguments=safe_args, call_id=call_id)))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="ollama", help="e.g. ollama|lmstudio|openai|anthropic")
    ap.add_argument("--model", default="llama3.2:3b", help="provider model id")
    ap.add_argument("--base-url", default=None, help="Optional base URL (lmstudio/openai-compatible).")
    args = ap.parse_args()

    llm_kwargs: Dict[str, Any] = {"model": args.model}
    if args.base_url:
        llm_kwargs["base_url"] = args.base_url
    llm = create_llm(args.provider, **llm_kwargs)

    system = (
        "You are a helpful assistant.\n"
        "You may call tools when needed.\n"
        "Important: you must ask before sending any email.\n"
        "If you need to look something up, call `search_docs`.\n"
    )
    question = (
        "Find a brief explanation of AbstractCore prompt caching, then email it to alice@example.com.\n"
        "Use tools."
    )

    print("=" * 80)
    print(f"provider={args.provider} model={args.model}")
    print("user:", question)

    resp = llm.generate(question, system_prompt=system, tools=[search_docs, send_email])
    print("\nassistant (phase 1):")
    print(str(getattr(resp, "content", "") or "").strip())

    tool_calls = _parse_tool_calls(getattr(resp, "tool_calls", None))
    if not tool_calls:
        print("\n(no tool calls requested)")
        return 0

    print("\nrequested tool calls:")
    for call in tool_calls:
        print(f"- {call.name} args={call.arguments}")

    tool_results = _execute_with_policy(tool_calls)
    print("\nexecuted tool results (policy applied):")
    for call, r in zip(tool_calls, tool_results):
        status = "ok" if r.success else "error"
        print(f"- {call.name} {status}: {r.output if r.success else r.error}")

    results_blob = "\n".join(
        [
            f"{c.name}({c.arguments}) -> {tr.output if tr.success else tr.error}"
            for c, tr in zip(tool_calls, tool_results)
        ]
    )
    followup = (
        "You previously requested tool calls.\n\n"
        f"User request:\n{question}\n\n"
        f"Tool results (policy applied):\n{results_blob}\n\n"
        "Now respond to the user.\n"
        "- If the email tool was denied, explain that you cannot send the email without explicit confirmation.\n"
        "- Still provide the explanation you found.\n"
    )
    resp2 = llm.generate(followup, system_prompt="You are a concise assistant.")
    print("\nassistant (final):")
    print(str(getattr(resp2, "content", "") or "").strip())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
