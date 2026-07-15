"""Regression pins (2026-07-15): system_prompt + tools render ONE system block.

Chat templates (ChatML/Qwen, Gemma) are trained on exactly one system turn.
`_build_prompt_fragment` used to emit the user system prompt as one
`<|im_start|>system` block and the tool instructions as a SECOND consecutive
system block — out-of-distribution input that degrades tool-calling and
instruction-following (live find on Ornith-1.0-35B driven through a ReAct
tool loop). The fix merges the tool prompt into the single system turn.

Boundary semantics that must NOT change with the merge:
- tools-only (no user system prompt): one system block carrying the tools;
- system-only (no tools): unchanged;
- "system" module prefilled in the KV cache: its block is closed and cannot
  be reopened, so the tool prompt still enters as its own block (module-chain
  appends carry system and tools in separate calls — their bytes are
  byte-identical to the pre-fix render).
"""

from typing import Any, Dict, List, Optional

from abstractcore.providers.mlx_provider import MLXProvider


class _FakeToolHandler:
    supports_prompted = True

    def format_tools_prompt(
        self,
        tools: Optional[List[Dict[str, Any]]],
        *,
        include_tool_list: bool = True,
    ) -> str:
        names: List[str] = []
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            func = tool.get("function") if isinstance(tool.get("function"), dict) else None
            name = str((func or {}).get("name") or tool.get("name") or "").strip()
            if name:
                names.append(name)
        lines: List[str] = []
        if include_tool_list:
            lines.append("## Tools (session)")
        lines.extend(f"- {name}" for name in names)
        return "\n".join(lines)


class _FakeTokenizer:
    bos_token = "<bos>"


TOOLS = [
    {"type": "function", "function": {"name": "read_file"}},
    {"type": "function", "function": {"name": "list_files"}},
]


def _qwen_provider() -> MLXProvider:
    provider = MLXProvider.__new__(MLXProvider)
    provider.model = "mlx-community/Qwen3.5-4B"
    provider.tool_handler = _FakeToolHandler()
    return provider


def _gemma_provider() -> MLXProvider:
    provider = MLXProvider.__new__(MLXProvider)
    provider.model = "mlx-community/gemma-4-26b-a4b-4bit"
    provider.architecture_config = {"message_format": "gemma_turn"}
    provider.tokenizer = _FakeTokenizer()
    provider.tool_handler = _FakeToolHandler()
    return provider


def test_mlx_qwen_system_plus_tools_render_one_system_block() -> None:
    provider = _qwen_provider()

    fragment = provider._build_prompt_fragment(
        prompt="hi",
        system_prompt="You are a ReAct agent.",
        tools=TOOLS,
        add_generation_prompt=True,
    )

    assert fragment.count("<|im_start|>system") == 1
    start = fragment.index("<|im_start|>system")
    end = fragment.index("<|im_end|>", start)
    system_block = fragment[start:end]
    assert "You are a ReAct agent." in system_block
    assert "## Tools (session)" in system_block
    assert "read_file" in system_block


def test_mlx_gemma_system_plus_tools_render_one_system_turn() -> None:
    provider = _gemma_provider()

    fragment = provider._build_prompt_fragment(
        prompt="hi",
        system_prompt="You are a ReAct agent.",
        tools=TOOLS,
        add_generation_prompt=True,
    )

    assert fragment.count("<|turn>system") == 1
    start = fragment.index("<|turn>system")
    end = fragment.index("<turn|>", start)
    system_block = fragment[start:end]
    assert "You are a ReAct agent." in system_block
    assert "## Tools (session)" in system_block


def test_mlx_tools_only_render_one_system_block() -> None:
    provider = _qwen_provider()

    fragment = provider._build_prompt_fragment(prompt="hi", tools=TOOLS)

    assert fragment.count("<|im_start|>system") == 1
    assert "## Tools (session)" in fragment


def test_mlx_system_only_render_unchanged() -> None:
    provider = _qwen_provider()

    fragment = provider._build_prompt_fragment(prompt="hi", system_prompt="SYS")

    assert fragment.count("<|im_start|>system") == 1
    assert "<|im_start|>system\nSYS<|im_end|>\n" in fragment
    assert "## Tools" not in fragment


def test_mlx_prefilled_system_keeps_standalone_tool_block() -> None:
    """Module-append semantics: a prefilled system block lives in the KV cache
    and cannot be reopened — the tool fragment must stay its own block so the
    "tools" module of a module chain renders byte-identically to before."""
    provider = _qwen_provider()

    fragment = provider._build_prompt_fragment(
        prompt="hi",
        system_prompt="SYS",
        tools=TOOLS,
        prefilled_modules=["system"],
    )

    assert "SYS" not in fragment  # system module skipped (already in cache)
    assert fragment.count("<|im_start|>system") == 1  # the standalone tool block
    assert "## Tools (session)" in fragment


def test_mlx_prefilled_tools_skip_tool_prompt() -> None:
    provider = _qwen_provider()

    fragment = provider._build_prompt_fragment(
        prompt="hi",
        system_prompt="SYS",
        tools=TOOLS,
        prefilled_modules=["tools"],
    )

    assert fragment.count("<|im_start|>system") == 1
    assert "SYS" in fragment
    assert "## Tools" not in fragment


def test_mlx_module_chain_fragments_unchanged_by_merge() -> None:
    """Module-chain appends carry system and tools in SEPARATE calls (one per
    module); the merge only engages when one call carries both. These are the
    exact bytes `_prompt_cache_backend_append` feeds per module — pinning them
    proves prepared module caches keep their pre-fix composition."""
    provider = _qwen_provider()

    system_fragment = provider._build_prompt_fragment(system_prompt="SYS")
    tools_fragment = provider._build_prompt_fragment(tools=TOOLS)

    assert system_fragment == "<|im_start|>system\nSYS<|im_end|>\n"
    assert tools_fragment == (
        "<|im_start|>system\n## Tools (session)\n- read_file\n- list_files<|im_end|>\n"
    )


# --- ChatML rendering is registry-driven, not a model-name substring ---------
# Regression pin (2026-07-15): `_build_prompt_fragment` decided ChatML by
# `"qwen" in model_name`, so any ChatML model whose NAME lacks "qwen" — notably
# Ornith (arch qwen3_5_agentic, message_format "im_start_end") — rendered on the
# LIVE generate path as plain `role: content` text with ZERO ChatML markers.
# The decision now reads the registry's message_format.

def _ornith_like_provider() -> MLXProvider:
    """Name has no 'qwen'; registry arch is ChatML (im_start_end)."""
    provider = MLXProvider.__new__(MLXProvider)
    provider.model = "mlx-community/Ornith-1.0-9B-4bit"
    provider.architecture_config = {"message_format": "im_start_end"}
    provider.tool_handler = _FakeToolHandler()
    return provider


def test_mlx_chatml_by_message_format_not_name_substring() -> None:
    provider = _ornith_like_provider()

    fragment = provider._build_prompt_fragment(
        prompt="hello",
        system_prompt="You are precise.",
        add_generation_prompt=True,
    )

    # Renders ChatML from the registry arch, NOT the plain `user: hello` fallback.
    assert "<|im_start|>system\nYou are precise.<|im_end|>\n" in fragment
    assert "<|im_start|>user\nhello<|im_end|>\n" in fragment
    assert fragment.endswith("<|im_start|>assistant\n")
    assert "user: hello" not in fragment


def test_mlx_unknown_arch_still_uses_plain_fallback() -> None:
    """A genuinely unknown, non-ChatML arch keeps the existing plain fallback —
    the fix widens ChatML to registry-declared models, it does not force ChatML
    onto everything."""
    provider = MLXProvider.__new__(MLXProvider)
    provider.model = "some-vendor/mystery-model"
    provider.architecture_config = {"message_format": "basic"}
    provider.tool_handler = _FakeToolHandler()

    fragment = provider._build_prompt_fragment(prompt="hi", system_prompt="S")

    assert "<|im_start|>" not in fragment
    assert "user: hi" in fragment
