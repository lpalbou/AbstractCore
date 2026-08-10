"""Thinking-disable prefill on the hand-rendered HuggingFace lanes (2026-08-05).

CONTEXT — the defect these pin, measured on `deepreinforce-ai/Ornith-1.0-9B`
(bf16/MPS, `results/bench_b/hf_ornith9b_bf16_10000.json`): every KEYED-CACHE arm
returned EMPTY visible content (7/7 at 10k, 5/5 at 30k, `finish_reason="stop"`,
no exception) while the uncached arm was healthy at 9.81 s median.

The discriminator is the renderer, not the model:

  * UNCACHED  -> `_build_input_text_transformers` -> `tokenizer.apply_chat_template(
                 ..., enable_thinking=False)`. Ornith's own template turns that into
                 `<think>\\n\\n</think>\\n\\n`; without it the same template emits a
                 BARE `<think>\\n` opener, i.e. thinking is on by default.
  * CACHED    -> `_transformers_build_prompt_fragment`, a hand renderer that never
                 touches the chat template. It appended the disable marker only when
                 `self.architecture` was in the hardcoded set
                 `{"qwen3", "qwen3_5", "qwen3_6"}`. Ornith resolves to
                 `qwen3_5_agentic`, so the prompt ended at a bare
                 `<|im_start|>assistant\\n`, the model's first generated token was
                 `<think>` (id 248068, captured live), and `strip_thinking_tags`
                 turned that unterminated block into "".

So: a reasoning model + a hand-rendered generation prompt + no declared prefill =
silent empty completions on exactly one lane. The fix is to read the surface the
registries already declare (`thinking_control.assistant_prefill_disable`) instead
of an architecture allow-list, and to declare it for `qwen3_5_agentic`.

Every assertion below fails on the pre-fix code except the two that exist to prove
the fix did not change the families that already worked.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from abstractcore.architectures import get_architecture_format
from abstractcore.architectures.detection import get_model_capabilities
from abstractcore.architectures.response_postprocessing import strip_thinking_tags
from abstractcore.providers.huggingface_provider import HuggingFaceProvider

MARKER = "<think>\n\n</think>\n\n"
ORNITH = "deepreinforce-ai/Ornith-1.0-9B"

# Families that already rendered the marker before the fix. They must keep
# rendering byte-identically — this is the no-regression side of the change.
ALREADY_WORKING = ("qwen3", "qwen3_5", "qwen3_6")


def _provider(
    architecture: str,
    *,
    architecture_config: Optional[Dict[str, Any]] = None,
    model_capabilities: Optional[Dict[str, Any]] = None,
    model: str = ORNITH,
) -> HuggingFaceProvider:
    p = HuggingFaceProvider.__new__(HuggingFaceProvider)
    p.model = model
    p.model_type = "transformers"
    p.device = "cpu"
    p.tokenizer = None
    p.tool_handler = None
    p.architecture = architecture
    p.architecture_config = (
        architecture_config
        if architecture_config is not None
        else get_architecture_format(architecture)
    )
    p.model_capabilities = model_capabilities or {}
    return p


def _fragment(p: HuggingFaceProvider, enable_thinking: Optional[bool]) -> str:
    return p._transformers_build_prompt_fragment(
        prompt="",
        messages=[{"role": "user", "content": "Who is on shift? Answer briefly."}],
        system_prompt="A document with one planted fact.",
        tools=None,
        add_generation_prompt=True,
        prefilled_modules=None,
        enable_thinking=enable_thinking,
    )


# ------------------------------------------------------------------ registry
def test_ornith_family_declares_the_thinking_disable_prefill():
    """The registry gap itself. `qwen3_5_agentic` (Ornith 1.0, Qwen-AgentWorld,
    Agents-A1) declared only `template_kwarg`, so every local renderer that
    cannot pass a template variable had no marker to emit. The value is not
    invented here — it is the exact string Ornith's own chat_template.jinja
    emits for `enable_thinking is false`."""
    arch = get_architecture_format("qwen3_5_agentic")
    assert arch, "qwen3_5_agentic missing from architecture_formats.json"
    assert arch["thinking_control"]["assistant_prefill_disable"] == MARKER


def test_ornith_model_entry_routes_to_that_family():
    """The registry fix only reaches Ornith if the model still resolves to the
    family that carries it."""
    caps = get_model_capabilities(ORNITH) or {}
    assert caps.get("architecture") == "qwen3_5_agentic"
    assert caps.get("thinking_support") is True


# ------------------------------------------------------------------ renderer
def test_cached_renderer_emits_the_prefill_for_ornith():
    """THE PIN. On pre-fix code this fragment ends at `<|im_start|>assistant\\n`
    and the cached lane returns "" for every call."""
    frag = _fragment(_provider("qwen3_5_agentic"), False)

    assert frag.endswith(MARKER), (
        "cached-lane generation prompt has no thinking-disable prefill; the model "
        "will open a <think> block and the visible content will be stripped to ''"
    )


def test_cached_and_uncached_fallback_renderers_agree_on_the_prefill():
    """Both hand renderers in this provider feed the same model. A disable
    control that maps on one lane and silently no-ops on the other is the exact
    shape of the defect (uncached healthy, cached empty)."""
    p = _provider("qwen3_5_agentic")
    p.tokenizer = object()  # no chat_template attribute -> the fallback branch

    cached = _fragment(p, False)
    uncached = p._build_input_text_transformers(
        "", [{"role": "user", "content": "q"}], "doc", None, enable_thinking=False
    )

    assert cached.endswith(MARKER)
    assert uncached.endswith(MARKER)


def test_prefill_is_read_from_the_registry_not_an_architecture_allowlist():
    """Registry-driven, both directions: an unknown family that DECLARES the
    surface gets the marker, and one that declares nothing never does. This is
    what makes the next Qwen post-train work without a code edit — the whole
    point of the typed thinking-control surfaces."""
    declares = _provider(
        "some_future_reasoning_family",
        architecture_config={
            "assistant_prefix": "<|im_start|>assistant\n",
            "assistant_suffix": "<|im_end|>\n",
            "user_prefix": "<|im_start|>user\n",
            "user_suffix": "<|im_end|>\n",
            "system_prefix": "<|im_start|>system\n",
            "system_suffix": "<|im_end|>\n",
            "thinking_control": {"assistant_prefill_disable": MARKER},
        },
    )
    silent = _provider(
        "a_non_reasoning_family",
        architecture_config={
            "assistant_prefix": "<|im_start|>assistant\n",
            "assistant_suffix": "<|im_end|>\n",
            "user_prefix": "<|im_start|>user\n",
            "user_suffix": "<|im_end|>\n",
            "system_prefix": "<|im_start|>system\n",
            "system_suffix": "<|im_end|>\n",
        },
    )

    assert _fragment(declares, False).endswith(MARKER)
    assert not _fragment(silent, False).endswith(MARKER)
    assert _fragment(silent, False).endswith("<|im_start|>assistant\n")


@pytest.mark.parametrize("enable_thinking", [True, None])
def test_prefill_only_fires_when_thinking_is_explicitly_disabled(enable_thinking):
    """Thinking left on (or unspecified) must never be forced off by the
    renderer — the marker is a disable control, not a default."""
    assert MARKER not in _fragment(_provider("qwen3_5_agentic"), enable_thinking)


@pytest.mark.parametrize("architecture", ALREADY_WORKING)
def test_previously_allowlisted_families_render_byte_identically(architecture):
    """No-regression side. These three were the entire old allow-list and all
    three declare the same marker, so switching to the declared surface must be
    a no-op for them."""
    frag = _fragment(_provider(architecture), False)

    assert frag.endswith("<|im_start|>assistant\n" + MARKER)


# ----------------------------------------------------------------- mechanism
def test_a_bare_think_opener_becomes_empty_visible_content():
    """Why a missing prefill is silent rather than loud: at a small output
    budget the whole completion is the `<think>` opener, and the unterminated
    block is extracted, leaving "". No exception, finish_reason="stop" — which
    is why the benchmark recorded empty error lists and a 0.1 ms 'cache win'."""
    cleaned, reasoning = strip_thinking_tags(
        "<think>", architecture_format=get_architecture_format("qwen3_5_agentic")
    )

    assert cleaned == ""
    assert reasoning is None
