#!/usr/bin/env python3
"""
Basic tool calling (function calling) in AbstractCore.

What this demonstrates
- Defining tools using the `@tool` decorator.
- Asking the model to request tool calls (provider-native or prompted).
- Receiving structured tool calls, then executing them in the host in two ways:
  - using AbstractCore's `ToolRegistry` (a small built-in tool interpreter), or
  - using a tiny manual dispatcher (so you fully control execution).

Run
  python examples/tools/tool_usage_basic.py --provider ollama --model llama3.2:3b
  python examples/tools/tool_usage_basic.py --provider lmstudio --model openai/gpt-oss-20b --base-url http://localhost:1234/v1

Notes
- Many models will only request tools reliably if you explicitly instruct them to use tools.
- AbstractCore returns `response.tool_calls` but (by default) does not run tools for you.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

from abstractcore import create_llm, tool
from abstractcore.tools import ToolCall, ToolResult
from abstractcore.tools.registry import ToolRegistry


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city (demo stub)."""
    unit_norm = str(unit or "celsius").lower().strip()
    if unit_norm not in {"celsius", "fahrenheit"}:
        unit_norm = "celsius"
    # Deterministic stub output so you can test tool wiring without external APIs.
    if unit_norm == "fahrenheit":
        return f"{city}: 72°F, sunny"
    return f"{city}: 22°C, sunny"


@tool
def calculate(expression: str) -> float:
    """Compute a basic arithmetic expression (demo stub; do not use eval in production)."""
    expr = str(expression or "").strip()
    allowed = set("0123456789+-*/(). ")
    if not expr or any(ch not in allowed for ch in expr):
        raise ValueError("unsupported expression")
    # Safe-ish eval sandbox for a tiny demo (no builtins; still not recommended for real apps).
    return float(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307


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


def _execute_with_tool_registry(tool_calls: List[ToolCall]) -> List[ToolResult]:
    registry = ToolRegistry()
    # Register tool *definitions* to preserve the schema/metadata created by @tool.
    registry.register(get_weather.tool_definition)
    registry.register(calculate.tool_definition)
    return registry.execute_tools(tool_calls)


def _execute_manually(tool_calls: List[ToolCall]) -> List[ToolResult]:
    dispatch = {"get_weather": get_weather, "calculate": calculate}
    out: List[ToolResult] = []
    for idx, call in enumerate(tool_calls):
        call_id = call.call_id or f"call:{idx}"
        fn = dispatch.get(call.name)
        if fn is None:
            out.append(ToolResult(call_id=call_id, output="", error=f"unknown tool: {call.name}", success=False))
            continue
        try:
            result = fn(**dict(call.arguments or {}))
            out.append(ToolResult(call_id=call_id, output=result, success=True))
        except Exception as e:  # noqa: BLE001
            out.append(ToolResult(call_id=call_id, output="", error=str(e), success=False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="ollama", help="e.g. ollama|lmstudio|openai|anthropic")
    ap.add_argument("--model", default="llama3.2:3b", help="provider model id")
    ap.add_argument("--base-url", default=None, help="Optional base URL (lmstudio/openai-compatible).")
    ap.add_argument(
        "--executor",
        default="abstractcore",
        choices=["abstractcore", "manual"],
        help="How to execute tool calls in the host.",
    )
    args = ap.parse_args()

    llm_kwargs: Dict[str, Any] = {"model": args.model}
    if args.base_url:
        llm_kwargs["base_url"] = args.base_url
    llm = create_llm(args.provider, **llm_kwargs)

    system = (
        "You are a helpful assistant.\n"
        "If you need arithmetic, call the `calculate` tool.\n"
        "If you need weather, call the `get_weather` tool.\n"
        "Do not guess tool outputs.\n"
    )
    question = "What's the weather in Paris, and what is 15 * 23?"

    print("=" * 80)
    print(f"provider={args.provider} model={args.model}")
    print("user:", question)

    resp = llm.generate(question, system_prompt=system, tools=[get_weather, calculate])
    print("\nassistant (phase 1):")
    print(str(getattr(resp, "content", "") or "").strip())

    tool_calls = _parse_tool_calls(getattr(resp, "tool_calls", None))
    if not tool_calls:
        print("\n(no tool calls requested)")
        return 0

    print("\nrequested tool calls:")
    for call in tool_calls:
        print(f"- {call.name} args={call.arguments}")

    if args.executor == "manual":
        tool_results = _execute_manually(tool_calls)
    else:
        tool_results = _execute_with_tool_registry(tool_calls)
    print("\ntool results:")
    for call, result in zip(tool_calls, tool_results):
        status = "ok" if result.success else "error"
        print(f"- {call.name} {status}: {result.output if result.success else result.error}")

    # Provider-agnostic “follow-up”: send tool results back as plain text context.
    # (Keeps the example simple across providers with different tool-message formats.)
    results_blob = "\n".join(
        [
            f"{c.name}({c.arguments}) -> {tr.output if tr.success else tr.error}"
            for c, tr in zip(tool_calls, tool_results)
        ]
    )
    followup = (
        f"User question: {question}\n\n"
        f"Tool results:\n{results_blob}\n\n"
        "Now answer the user question using the tool results."
    )
    resp2 = llm.generate(followup, system_prompt="You are a concise assistant.")
    print("\nassistant (final):")
    print(str(getattr(resp2, "content", "") or "").strip())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
