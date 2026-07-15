"""Pins for the shared tool-placement helper `merge_tools_into_system`.

The "merge tools into ONE system turn" policy used to be copy-pasted across
every provider (9 sites, drifted 4 ways, one copy silently dropped the tool
prompt). It is now a single free function that duck-types on any handler with
`supports_prompted` + `format_tools_prompt`. These pin the byte-contract (so
prompt-cache byte-parity across providers is preserved) and the fixed
transformers no-template fallback that dropped tools.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from abstractcore.tools import merge_tools_into_system


class _Handler:
    supports_prompted = True

    def format_tools_prompt(self, tools, *, include_tool_list: bool = True, **_: Any) -> str:
        names = [
            (t.get("function", {}) or {}).get("name") or t.get("name")
            for t in (tools or [])
            if isinstance(t, dict)
        ]
        head = "## Tools (session)\n" if include_tool_list else ""
        return head + "\n".join(f"- {n}" for n in names if n)


class _NativeOnly:
    supports_prompted = False

    def format_tools_prompt(self, *a, **k) -> str:  # pragma: no cover - never called
        raise AssertionError("format_tools_prompt must not be called for a non-prompted handler")


TOOLS = [{"type": "function", "function": {"name": "read_file"}},
         {"type": "function", "function": {"name": "list_files"}}]


def test_merge_system_then_tools_one_block_bytes():
    out = merge_tools_into_system(_Handler(), "You are precise.", TOOLS)
    # Exact byte-contract: system, blank line, then the tool block.
    assert out == "You are precise.\n\n## Tools (session)\n- read_file\n- list_files"


def test_merge_tools_only_when_no_system():
    out = merge_tools_into_system(_Handler(), None, TOOLS)
    assert out == "## Tools (session)\n- read_file\n- list_files"


def test_no_tools_returns_system_unchanged():
    assert merge_tools_into_system(_Handler(), "SYS", None) == "SYS"
    assert merge_tools_into_system(_Handler(), "SYS", []) == "SYS"


def test_non_prompted_handler_returns_system_unchanged_and_never_formats():
    # Native-only handler: the prompted block must not be built at all.
    assert merge_tools_into_system(_NativeOnly(), "SYS", TOOLS) == "SYS"


def test_dedup_sentinel_suppresses_tool_list_when_already_present():
    # A system prompt that already carries the tool list must not get it twice.
    sys_with_list = "Persona.\n\n## Tools (session)\n- read_file"
    out = merge_tools_into_system(_Handler(), sys_with_list, TOOLS)
    assert out.count("## Tools (session)") == 1


def test_transformers_no_template_fallback_keeps_tools():
    """Regression: `_build_input_text_transformers`'s no-template fallback used
    the raw system_prompt and silently dropped the tool prompt. It now renders
    the merged system, so a template-less model still sees the tools."""
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider.model = "some/template-less-model"
    provider.architecture = "generic"
    provider.tool_handler = _Handler()

    class _TokNoTemplate:
        chat_template = None

    provider.tokenizer = _TokNoTemplate()

    text = provider._build_input_text_transformers(
        prompt="hi", messages=None, system_prompt="Persona.", tools=TOOLS
    )
    assert "## Tools (session)" in text  # tools survive the fallback
    assert "read_file" in text
    assert "Persona." in text
