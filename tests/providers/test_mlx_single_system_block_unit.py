"""Regression pins (2026-07-15): system_prompt + tools render ONE system block.

Chat templates (ChatML/Qwen, Gemma) are trained on exactly one system turn.
`_build_prompt_fragment` used to emit the user system prompt as one
`<|im_start|>system` block and the tool instructions as a SECOND consecutive
system block — out-of-distribution input that degrades tool-calling and
instruction-following (live find on Ornith-1.0-35B driven through a ReAct
tool loop). The fix merges the tool prompt into the single system turn.

Boundary semantics that must NOT change with the merge:
- tools-only (no user system prompt): one system block carrying the tools;
- system-only (no tools): unchanged.

UPDATED 2026-08-03. Two tests in this file used to pin the module-chain's
per-module standalone render as CORRECT ("byte-identical to the pre-fix
render"). It was not correct — it was the bloc bug, and a green suite was
evidence the bug was present. Rendering each module as its own conversation
emits two consecutive `<|im_start|>system` blocks, so a bloc chain's KV
diverged from `generate()`'s prompt at the end of the system text. Those two
tests now pin the divergence AS a defect; the module chain no longer takes that
path (`BaseProvider.prompt_cache_plan_bloc_chain` cuts one cumulative render at
successor-independent token boundaries — see
tests/test_prompt_cache_bloc_composition.py).
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


def test_mlx_prefilled_system_opens_a_second_system_block() -> None:
    """`prefilled_modules` cannot express a MID-TURN continuation — pinned as a
    known limit, NOT as desired behaviour.

    Asking this renderer to skip an already-cached system module and then emit
    the tool prompt produces a SECOND `<|im_start|>system` block: bytes
    `generate()` never produces, because a closed system turn cannot be
    reopened from a fresh render. That is precisely why bloc chains no longer
    go through `prefilled_modules` — `BaseProvider.prompt_cache_plan_bloc_chain`
    cuts ONE cumulative render at token boundaries instead, so the `system`
    bloc ends BEFORE `<|im_end|>` and the `tools` bloc continues the same turn.
    See tests/test_prompt_cache_bloc_composition.py.

    This test exists so the limitation stays visible: any caller that still
    passes `prefilled_modules` for a mid-turn slot gets out-of-distribution
    bytes.
    """
    provider = _qwen_provider()

    fragment = provider._build_prompt_fragment(
        prompt="hi",
        system_prompt="SYS",
        tools=TOOLS,
        prefilled_modules=["system"],
    )

    assert "SYS" not in fragment  # system module skipped (already in cache)
    assert fragment.count("<|im_start|>system") == 1  # a second, reopened block
    assert "## Tools (session)" in fragment

    # The reason it is a limit and not a feature: the prefilled render is NOT a
    # continuation of the un-prefilled one.
    full = provider._build_prompt_fragment(prompt="hi", system_prompt="SYS", tools=TOOLS)
    head = provider._build_prompt_fragment(system_prompt="SYS")
    assert not full.startswith(head), "prefilled_modules would be sound if this held"


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


def test_mlx_standalone_per_module_fragments_do_not_concatenate() -> None:
    """The bloc bug, pinned as a bug.

    These two strings are what `_build_prompt_fragment` returns when each
    module is rendered as its own standalone conversation — the shape the
    module chain used to feed. Concatenated they are NOT what `generate()`
    builds: two consecutive `<|im_start|>system` blocks instead of one folded
    turn, so every token past the first block is unreachable KV (measured 618
    of 2148 on the live agent prefix).

    Pinned so the strings stay visible, and so the inequality below fails loudly
    if anyone reconnects the chain to this path. The chain now goes through
    `BaseProvider.prompt_cache_plan_bloc_chain`; see
    tests/test_prompt_cache_bloc_composition.py for the composition proof.
    """
    provider = _qwen_provider()

    system_fragment = provider._build_prompt_fragment(system_prompt="SYS")
    tools_fragment = provider._build_prompt_fragment(tools=TOOLS)

    assert system_fragment == "<|im_start|>system\nSYS<|im_end|>\n"
    assert tools_fragment == (
        "<|im_start|>system\n## Tools (session)\n- read_file\n- list_files<|im_end|>\n"
    )

    single_shot = provider._build_prompt_fragment(system_prompt="SYS", tools=TOOLS)
    assert single_shot == (
        "<|im_start|>system\nSYS\n\n## Tools (session)\n- read_file\n- list_files<|im_end|>\n"
    )
    assert system_fragment + tools_fragment != single_shot
    assert not single_shot.startswith(system_fragment)
    # The planner's cut, by contrast, IS a prefix of the single-shot render.
    assert single_shot.startswith("<|im_start|>system\nSYS")


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
