"""Regression pins (2026-09-17): a prepared prefix must be a prefix of the PROMPT.

Found on a live gateway session (AbstractAssistant, `mlx-community/Qwen3.8-27B-4bit`):
two consecutive turns over a ~5.5k-token context each took ~17 s of prefill. The
ledger said why, once read closely:

    turn 1   outcome=rebuilt      cached_tokens=0     fed_tokens=5554
    turn 2   outcome=hit_restore  cached_tokens=3     fed_tokens=5578
    turn 3   outcome=hit_restore  cached_tokens=5465  fed_tokens=299

The runtime prepares a (system, tools) bloc chain and forks it into the session
key. Qwen3.8 declares `thinking_control.effort_system_lines`, so under
`thinking="minimal"` (→ `low`) `generate()` opens the system block with
"Reasoning effort is set to low. …" — and the chain, planned without the thinking
request, opened it with the persona. They agreed on `<|im_start|>system\\n`: 3 tokens.

Two defects, pinned separately below:

1. `prompt_cache_prepare_modules` had no way to learn the thinking request, so the
   chain could not be a prefix of what `generate()` renders.
2. The snapshot lane read the diverged SEED as if it were the previous prompt, took
   their 3-token LCP as "the stable transcript", snapshotted there, and reported the
   next turn as a HIT. Nothing warned.

Tokenizer + renderer only. NO model weights are loaded anywhere in this file.
"""

from __future__ import annotations

import glob
import logging
import os
import tempfile
import threading
import uuid
import warnings
from typing import Any, Dict, List, Optional

import importlib.util

import pytest

from abstractcore.architectures import (
    detect_architecture,
    get_architecture_format,
    get_model_capabilities,
)
from abstractcore.providers.base import PromptCacheModule, PromptCacheStore
from abstractcore.providers.mlx_provider import MLXProvider
from abstractcore.tools import UniversalToolHandler


_requires_mlx_stack = pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("mlx", "mlx_lm", "mlx_vlm")),
    reason="requires the optional MLX stack (pip install \"abstractcore[mlx]\")",
)

QWEN38 = "mlx-community/Qwen3.8-27B-4bit"

SYSTEM = "## MY PERSONA\nYou are a careful desktop assistant.\n" + ("Stay concise. " * 30)
TOOLS: List[Dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a file from disk",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
]
MESSAGES = [{"role": "user", "content": "this is a test, answer with your name."}]


class _ToyTokenizer:
    """Deterministic char tokenizer with ChatML specials and one seam merge."""

    _MERGES = ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "e\n\n", "\n\n", ". ")
    bos_token = None

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        out: List[int] = []
        i = 0
        while i < len(text):
            for merge in self._MERGES:
                if text.startswith(merge, i):
                    out.append(self._vocab.setdefault(merge, len(self._vocab) + 1000))
                    i += len(merge)
                    break
            else:
                out.append(self._vocab.setdefault(text[i], len(self._vocab) + 1000))
                i += 1
        return out


def _real_qwen38_tokenizer():
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    hits = sorted(
        glob.glob(
            os.path.join(
                home, "hub", f"models--{QWEN38.replace('/', '--')}", "snapshots", "*", "tokenizer.json"
            )
        )
    )
    if not hits:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(os.path.dirname(hits[0]), local_files_only=True)
    except Exception:
        return None


def _tokenizer_cases():
    cases = [pytest.param(_ToyTokenizer, id="toy")]
    cases.append(
        pytest.param(
            _real_qwen38_tokenizer,
            id="qwen3.8-real",
            marks=pytest.mark.skipif(
                _real_qwen38_tokenizer() is None, reason=f"{QWEN38} tokenizer not installed locally"
            ),
        )
    )
    return cases


def _renderer(tokenizer, *, configured_reasoning: Any = None) -> MLXProvider:
    """An MLXProvider wired for RENDERING ONLY, on the REAL Qwen3.8 asset config.

    The effort sentences come from `assets/architecture_formats.json`, not from this
    file: if the asset stops declaring them these tests must notice.

    HERMETIC on the reasoning default: the config file is pointed at a path that does
    not exist, so the developer's own `reasoning:` setting cannot leak in (the machine
    this was written on is configured `minimal`), and `configured_reasoning` sets the
    text route's default the way a host's capability defaults do.
    """
    p = object.__new__(MLXProvider)
    p._abstractcore_config_file = os.path.join(
        tempfile.gettempdir(), f"acore-absent-config-{uuid.uuid4().hex}.json"
    )
    route = {
        "key": "input.text",
        "provider": "mlx",
        "model": QWEN38,
        "source": "abstractcore.capability_defaults",
    }
    if configured_reasoning is not None:
        route["reasoning"] = configured_reasoning
    p._abstractcore_capability_defaults = {"input.text": route}
    p.model = QWEN38
    p.provider = "mlx"
    p.architecture_config = get_architecture_format(detect_architecture(QWEN38))
    p.model_capabilities = get_model_capabilities(QWEN38)
    p.tool_handler = UniversalToolHandler(QWEN38)
    p.tokenizer = tokenizer
    p.logger = logging.getLogger("test.thinking_prefix")
    return p


def _generate_prompt_ids(p: MLXProvider, thinking: Any) -> List[int]:
    """Token ids of the prompt `generate(thinking=...)` builds — the same steps it takes.

    Including the FIRST one: `generate_with_telemetry` resolves the route from the real
    request and adopts its reasoning default when the caller named none. The planner
    resolves that default from its own probe request, so this is a genuine two-sided
    comparison — leave this step out and the `thinking=None` case checks nothing.
    """
    if thinking is None:
        request = p._build_generate_request(
            prompt="", request=None, text=None, messages=list(MESSAGES), media=None
        )
        route = p._resolve_generate_route(
            request=request,
            output={"modality": "text", "task": "text_generation"},
            thinking=None,
            kwargs={},
        )
        if route.reasoning is not None:
            thinking = route.reasoning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _prompt, messages, system_prompt, kwargs, _meta = p._apply_thinking_request(
            thinking=thinking,
            prompt="",
            messages=list(MESSAGES),
            system_prompt=SYSTEM,
            kwargs={"prompt_cache_key": "session:k"},
            request_shape={"has_response_model": False, "has_media": False, "messages": MESSAGES},
        )
    enable = kwargs.get("_acore_mlx_enable_thinking")
    effort = kwargs.get("_acore_mlx_reasoning_effort")
    full = p._build_prompt(
        "",
        messages,
        system_prompt,
        TOOLS,
        enable_thinking=enable if isinstance(enable, bool) else None,
        reasoning_effort=effort if isinstance(effort, str) else None,
    )
    ids = p._encode_prompt_token_ids(full)
    assert ids
    return ids


def _chain_ids(p: MLXProvider, thinking: Any) -> List[int]:
    modules = [
        PromptCacheModule(module_id="system", system_prompt=SYSTEM).normalized(),
        PromptCacheModule(module_id="tools", tools=TOOLS).normalized(),
    ]
    modules, _applied = p._prompt_cache_modules_with_thinking(modules, thinking)
    plan = p.prompt_cache_plan_bloc_chain(modules)
    assert plan is not None and plan.unsound_at is None
    return [t for frag in plan.fragments for t in frag]


def test_the_asset_still_declares_the_effort_lines_this_file_depends_on():
    p = _renderer(_ToyTokenizer())
    lines = p._thinking_control_surfaces().effort_system_lines or {}
    assert lines.get("low", "").strip(), "Qwen3.8 'low' effort line vanished from assets"
    assert lines.get("xhigh", "").strip()


@pytest.mark.parametrize("make_tokenizer", _tokenizer_cases())
def test_chain_planned_without_thinking_is_unreachable_under_an_effort_level(make_tokenizer):
    """THE DEFECT, pinned: this is what a caller that omits `thinking=` still gets."""
    p = _renderer(make_tokenizer())
    chain = _chain_ids(p, None)
    prompt = _generate_prompt_ids(p, "minimal")
    shared = p._token_lcp_len(chain, prompt)
    # `<|im_start|>`, `system`, `\n` on the real tokenizer; a few chars on the toy one.
    assert shared < len(chain) // 10
    assert shared < 20


@pytest.mark.parametrize("make_tokenizer", _tokenizer_cases())
@pytest.mark.parametrize("thinking", [None, "minimal", "low", "medium", "xhigh", "off", True, False])
def test_chain_planned_with_the_thinking_request_is_a_true_token_prefix(make_tokenizer, thinking):
    p = _renderer(make_tokenizer())
    chain = _chain_ids(p, thinking)
    prompt = _generate_prompt_ids(p, thinking)
    assert len(chain) > 100
    assert prompt[: len(chain)] == chain


@pytest.mark.parametrize("make_tokenizer", _tokenizer_cases())
@pytest.mark.parametrize("configured", ["minimal", "xhigh", "medium", None])
def test_an_unset_thinking_request_follows_the_configured_route_default(make_tokenizer, configured):
    """`generate()` does not stop at thinking=None — it adopts the reasoning effort
    configured on the text route. Found by the adversarial pass: a client built without
    capability defaults, on a machine configured `reasoning: minimal`, prepared under
    None, generated under `low`, and rebuilt every session (shared 3 of 4760)."""
    p = _renderer(make_tokenizer(), configured_reasoning=configured)
    chain = _chain_ids(p, None)
    prompt = _generate_prompt_ids(p, None)
    assert prompt[: len(chain)] == chain
    # Not vacuous: when a level with a sentence is configured, BOTH sides carry it.
    unaware = _chain_ids(_renderer(make_tokenizer()), None)
    assert (chain != unaware) == (configured in {"minimal", "xhigh"})


def test_the_default_is_resolved_from_a_request_that_carries_text():
    """An empty probe request skips the text route, so every configuration reads as
    "no reasoning default" and the planner silently plans without one."""
    p = _renderer(_ToyTokenizer(), configured_reasoning="xhigh")
    assert p._prompt_cache_default_thinking() == "xhigh"
    assert _renderer(_ToyTokenizer())._prompt_cache_default_thinking() is None


def test_effort_level_rewrites_the_system_bloc_and_only_the_system_bloc():
    p = _renderer(_ToyTokenizer())
    base = [
        PromptCacheModule(module_id="system", system_prompt=SYSTEM).normalized(),
        PromptCacheModule(module_id="tools", tools=TOOLS).normalized(),
    ]
    low, low_applied = p._prompt_cache_modules_with_thinking(list(base), "low")
    minimal, _ = p._prompt_cache_modules_with_thinking(list(base), "minimal")
    xhigh, _ = p._prompt_cache_modules_with_thinking(list(base), "xhigh")
    medium, medium_applied = p._prompt_cache_modules_with_thinking(list(base), "medium")

    assert low_applied is True
    assert low[0].system_prompt.startswith("Reasoning effort is set to low.")
    assert low[0].system_prompt.endswith(base[0].system_prompt)
    # These keys are SHARED ACROSS SESSIONS: one key for two levels would serve KV
    # built for the other level.
    assert low[0].fingerprint() != base[0].fingerprint()
    assert low[0].fingerprint() != xhigh[0].fingerprint()
    # `minimal` is not a Qwen3.8 level; generate() maps it to `low`, so must the chain.
    assert minimal[0].fingerprint() == low[0].fingerprint()
    # `medium` declares an EMPTY line: the template renders nothing, neither do we.
    assert medium_applied is False
    assert medium[0].fingerprint() == base[0].fingerprint()
    assert low[1].fingerprint() == base[1].fingerprint()


def test_a_tools_only_chain_gets_a_leading_module_for_the_effort_line():
    p = _renderer(_ToyTokenizer())
    tools_only = [PromptCacheModule(module_id="tools", tools=TOOLS).normalized()]
    out, applied = p._prompt_cache_modules_with_thinking(tools_only, "low")
    assert applied is True
    assert [m.module_id for m in out] == ["thinking", "tools"]
    assert out[0].system_prompt.startswith("Reasoning effort is set to low.")


def test_unresolvable_thinking_request_plans_without_it_and_says_so(caplog):
    p = _renderer(_ToyTokenizer())

    def _boom(**_kwargs):
        raise RuntimeError("ladder exploded")

    p._apply_thinking_request = _boom  # type: ignore[method-assign]
    base = [PromptCacheModule(module_id="system", system_prompt=SYSTEM).normalized()]
    with caplog.at_level(logging.WARNING):
        out, applied = p._prompt_cache_modules_with_thinking(list(base), "low")
    assert applied is False and out == base
    assert any("#FALLBACK" in r.getMessage() and "thinking" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------
# Defect 2: a diverged SEED is not a previous prompt.
# --------------------------------------------------------------------------


class _FakeLayer:
    def __init__(self, offset: int = 0):
        self.offset = int(offset)

    def empty(self) -> bool:
        return self.offset == 0

    def is_trimmable(self) -> bool:
        return False  # Gated-DeltaNet / SSM: never rewindable


class _WordTokenizer:
    def __init__(self):
        self._vocab: Dict[str, int] = {}

    def encode(self, text: str) -> List[int]:
        return [self._vocab.setdefault(w, len(self._vocab) + 1) for w in str(text).split()]


class _RecordingLogger:
    def __init__(self):
        self.warnings: List[str] = []

    def warning(self, msg, *a, **k):
        self.warnings.append(str(msg))

    def debug(self, *a, **k):
        pass


class _NoTools:
    supports_prompted = False


def _cache_provider(*, trimmable: bool = False) -> MLXProvider:
    p = MLXProvider.__new__(MLXProvider)
    p.tokenizer = _WordTokenizer()
    p.logger = _RecordingLogger()
    p.tool_handler = _NoTools()
    p.model = "vendor/testmodel-4b"
    p.architecture_config = {"message_format": "im_start_end"}
    p._prompt_cache_store = PromptCacheStore()
    p._delta_feed_warned_keys = set()
    p._append_stash_lock = threading.RLock()
    p._hybrid_snapshot_lock = threading.RLock()
    p._hybrid_snapshots = {}
    p._prompt_cache_backend_create = lambda: [_FakeLayer(0)]  # type: ignore[method-assign]

    def _fake_prefill(cache_value, token_ids):
        for layer in cache_value:
            layer.offset += len(token_ids)
        return True

    p._prefill_tokens_into_cache = _fake_prefill  # type: ignore[method-assign]
    if trimmable:
        p._cache_is_trimmable = lambda cache: True  # type: ignore[method-assign]

        def _fake_trim(cache_value, n):
            for layer in cache_value:
                layer.offset -= n
            return True

        p._trim_prompt_cache_tokens = _fake_trim  # type: ignore[method-assign]
    else:
        p._trim_prompt_cache_tokens = lambda cache, n: False  # type: ignore[method-assign]
    return p


def _fork_seed(p: MLXProvider, key: str, seed_text: str):
    """What `prompt_cache_fork` leaves behind: a cache holding EXACTLY its record."""
    ids = p.tokenizer.encode(seed_text)
    cache = [_FakeLayer(offset=len(ids))]
    p._prompt_cache_store.set(
        key,
        cache,
        meta={"backend": "mlx", "fed_token_ids": list(ids), "forked_from": "abstractcode:prefix"},
    )
    return cache


SEED = "SYS persona t1 t2 t3"                       # planned WITHOUT the effort line
HEAD = "SYS effort persona t1 t2 t3"                # what generate() really renders
TURN1 = f"{HEAD} U stamp-1 hello loop-1 GEN"
TURN2 = f"{HEAD} U hello A hi U stamp-2 again loop-1 GEN"   # final turn of TURN1 rewritten


@_requires_mlx_stack
def test_diverged_seed_is_reported_and_never_becomes_the_snapshot_boundary():
    p = _cache_provider()
    cache = _fork_seed(p, "k", SEED)
    head_ids = p.tokenizer.encode(HEAD)
    tel: Dict[str, Any] = {}

    p._prepare_cache_delta_feed(
        "k", cache, TURN1, full_context=True, telemetry=tel, stable_head=lambda: HEAD
    )

    assert tel["outcome"] == "rebuilt" and tel["cached_tokens"] == 0
    assert "prepared prefix is not a prefix of the prompt" in tel["degraded_reason"]
    assert "shared 1 of" in tel["degraded_reason"]
    assert len(p.logger.warnings) == 1 and "#FALLBACK" in p.logger.warnings[0]
    # The bug snapshotted the 1-token LCP of seed and prompt. The boundary is the
    # renderer-derived head: everything before this call's final turn.
    assert p._get_hybrid_snapshot("k")["ids"] == head_ids


@_requires_mlx_stack
def test_after_a_diverged_seed_the_next_turn_restores_the_head():
    p = _cache_provider()
    cache = _fork_seed(p, "k", SEED)
    p._prepare_cache_delta_feed("k", cache, TURN1, full_context=True, stable_head=lambda: HEAD)
    turn1_ids = p.tokenizer.encode(TURN1)
    # generate() ran: the live cache now holds the prompt plus a reply.
    live = [_FakeLayer(offset=len(turn1_ids) + 5)]
    p._prompt_cache_store.set(
        "k", live, meta={"backend": "mlx", "fed_token_ids": turn1_ids, "forked_from": "x"}
    )

    tel: Dict[str, Any] = {}
    p._prepare_cache_delta_feed("k", live, TURN2, full_context=True, telemetry=tel)

    assert tel["outcome"] == "hit_restore"
    assert tel["cached_tokens"] == len(p.tokenizer.encode(HEAD))
    assert len(p.logger.warnings) == 1  # once per key, not once per turn


def test_without_a_stable_head_a_diverged_seed_falls_back_to_the_turn_one_rule():
    p = _cache_provider()
    cache = _fork_seed(p, "k", SEED)
    prompt = f"{HEAD} U hello <|im_start|>assistant"
    ids = p.tokenizer.encode(prompt)

    p._prepare_cache_delta_feed("k", cache, prompt, full_context=True)

    snap = p._get_hybrid_snapshot("k")
    assert snap is not None and len(snap["ids"]) > 1          # never the seed/prompt LCP
    assert snap["ids"] == ids[: len(snap["ids"])]


def test_a_stable_head_that_is_not_a_text_prefix_is_ignored():
    p = _cache_provider()
    cache = _fork_seed(p, "k", SEED)
    p._prepare_cache_delta_feed(
        "k", cache, TURN1, full_context=True, stable_head=lambda: "SOMETHING ELSE ENTIRELY"
    )
    ids = p.tokenizer.encode(TURN1)
    assert p._get_hybrid_snapshot("k")["ids"] == ids[:-1]


def test_a_previous_PROMPT_record_keeps_the_lcp_holdback():
    """Guard against over-triggering: generated tokens past the record mean the record
    WAS a prompt, and the LCP of two consecutive prompts is the stable transcript."""
    p = _cache_provider()
    prev = p.tokenizer.encode(TURN1)
    live = [_FakeLayer(offset=len(prev) + 7)]
    p._prompt_cache_store.set("k", live, meta={"backend": "mlx", "fed_token_ids": prev})
    tel: Dict[str, Any] = {}

    p._prepare_cache_delta_feed(
        "k", live, TURN2, full_context=True, telemetry=tel, stable_head=lambda: "unused"
    )

    shared = p._token_lcp_len(prev, p.tokenizer.encode(TURN2))
    assert p._get_hybrid_snapshot("k")["ids"] == p.tokenizer.encode(TURN2)[:shared]
    assert "degraded_reason" not in tel
    assert p.logger.warnings == [] or all("prepared prefix" not in w for w in p.logger.warnings)


@_requires_mlx_stack
def test_a_healthy_seed_is_still_reused_silently():
    p = _cache_provider()
    cache = _fork_seed(p, "k", HEAD)
    tel: Dict[str, Any] = {}
    p._prepare_cache_delta_feed("k", cache, TURN1, full_context=True, telemetry=tel)
    assert tel["outcome"] == "hit_restore"
    assert tel["cached_tokens"] == len(p.tokenizer.encode(HEAD))
    assert p.logger.warnings == []


def test_trimmable_lane_reports_a_diverged_seed_instead_of_a_three_token_hit():
    p = _cache_provider(trimmable=True)
    cache = _fork_seed(p, "k", SEED)
    tel: Dict[str, Any] = {}
    p._prepare_cache_delta_feed("k", cache, TURN1, full_context=True, telemetry=tel)
    # Trimming keeps the call CORRECT (outcome unchanged) — which is why it was silent.
    assert tel["outcome"] == "hit_extend" and tel["cached_tokens"] == 1
    assert len(p.logger.warnings) == 1 and "NOT a prefix of the prompt" in p.logger.warnings[0]


# --------------------------------------------------------------------------
# The folding rule itself, per lane.
#
# `_prompt_cache_thinking_system_text` folds the effort sentence into the system text
# as f"{line}\n\n{system}" and plans the chain with NO `reasoning_effort`. That is only
# right if each lane's own renderer, handed `reasoning_effort=`, produces the same
# bytes. It was asserted in a docstring for all three lanes and tested for one
# (adversarial find, 2026-09-17).
# --------------------------------------------------------------------------

_HF_MODEL = "Qwen/Qwen3.8-27B"


def _hf_renderer(model_type: str):
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    p = object.__new__(HuggingFaceProvider)
    p.model = _HF_MODEL
    p.provider = "huggingface"
    p.model_type = model_type
    p.architecture_config = get_architecture_format(detect_architecture(_HF_MODEL))
    p.model_capabilities = get_model_capabilities(_HF_MODEL)
    p.tool_handler = UniversalToolHandler(_HF_MODEL)
    p.tokenizer = object()
    p.logger = logging.getLogger("test.thinking_prefix.hf")
    return p


def _render_pair(lane: str, level: str):
    """(renderer given reasoning_effort=level, renderer given the FOLDED system text)."""
    if lane == "mlx":
        p = _renderer(_ToyTokenizer())
        line = (p._thinking_control_surfaces().effort_system_lines or {}).get(level) or ""
        folded = f"{line}\n\n{SYSTEM.strip()}" if line.strip() else SYSTEM
        return (
            p._build_prompt_fragment(system_prompt=SYSTEM, tools=TOOLS, reasoning_effort=level),
            p._build_prompt_fragment(system_prompt=folded, tools=TOOLS),
        )
    p = _hf_renderer("transformers" if lane == "hf-transformers" else "gguf")
    line = (p._thinking_control_surfaces().effort_system_lines or {}).get(level) or ""
    folded = f"{line}\n\n{SYSTEM.strip()}" if line.strip() else SYSTEM
    if lane == "hf-transformers":
        return (
            p._transformers_build_prompt_fragment(system_prompt=SYSTEM, tools=TOOLS, reasoning_effort=level),
            p._transformers_build_prompt_fragment(system_prompt=folded, tools=TOOLS),
        )
    # GGUF: the ChatML hand renderer of the local control plane. (A GGUF that renders
    # through its EMBEDDED Jinja template writes the model's own sentence from the
    # kwarg; that branch needs a real template and is NOT covered here.)
    native = p._gguf_build_chat_messages(system_prompt=SYSTEM, tools=TOOLS, messages=None)
    pre_folded = p._gguf_build_chat_messages(system_prompt=folded, tools=TOOLS, messages=None)
    return (
        p._gguf_render_chatml_prompt(messages=native, add_generation_prompt=False, reasoning_effort=level),
        p._gguf_render_chatml_prompt(messages=pre_folded, add_generation_prompt=False),
    )


@pytest.mark.parametrize("lane", ["mlx", "hf-transformers", "gguf-chatml"])
@pytest.mark.parametrize("level", ["low", "xhigh", "medium"])
def test_folding_the_effort_line_matches_each_lanes_own_renderer(lane, level):
    native, folded = _render_pair(lane, level)
    assert native == folded
    assert ("Reasoning effort is set to" in native) == (level != "medium")
