"""Native MLX lane: the per-call prefix snapshot lands BEFORE trailing riders.

Operator report 2026-10-27: a gateway agent run on `Jundot/Qwen3.8-27B-oQ4e-mtp`
showed "prefill 16,384 / 24,552 tokens" at step 10. The ledger of that run has
10 of 10 tool-loop calls `outcome: cold` (`exact_stores: 1, exact_hits: 0`)
while the previous run of the same gateway, same model, restored every call
but the first. The difference: the runtime re-injects the session attachment
index as a TRAILING `system` message on every call (abstractruntime
`effect_handlers.py`, "rides the TAIL"). The model is hybrid (linear + full
attention), so mlx-vlm restores only an EXACT stored prefix, and the native lane
stores one snapshot per call `guard` (8) tokens before the end of the prompt —
inside the rider, which the next call renders after its new turns. No snapshot
was ever a prefix of a later prompt.

Contract pinned here:
- `_native_rider_tail_tokens` measures the rider tail in TOKENS (longest common
  token prefix of the prompt and the same request rendered without riders);
- the snapshot placed that many tokens (+ the manager's own guard) before the
  end is a token prefix of the NEXT call's prompt; the default placement is not;
- the per-call guard reaches mlx-vlm on a per-call view of the manager, never by
  mutating the shared manager, and only when riders trail;
- the mlx-vlm seam this relies on (`APCCoordinator.checkpoint_len` reads
  `manager.exact_cache_guard_tokens`) is checked against the installed mlx-vlm.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from abstractcore.providers.mlx_native_session import CHECKPOINT_GUARD_TOKENS, CheckpointGuardView
from tests.providers.test_mlx_chat_template_render_unit import (
    QWEN38,
    SYSTEM,
    _iterations,
    _JinjaTokenizer,
    _provider,
    _qwen38_provider,
    _hf_snapshot,
)

RIDER = {
    "role": "system",
    "content": (
        "Stored session attachments (most recent first; not necessarily active in this call). "
        "Do not mention this list:\n"
        "Open media via: open_attachment(artifact_id='…').\n"
        "- ask.wav (id=8e3d5731a0995afbc88c06f590ec9e31, sha=3586cf5b…, audio/wav, 467,754 bytes, "
        "added 2026-10-26T20:11:42.143534+00:00)"
    ),
    "metadata": {"kind": "attachment_index"},
}


def _with_rider(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(msgs) + [dict(RIDER)]


def _restored(prev_ids: List[int], snapshot_len: int, next_ids: List[int]) -> int:
    """mlx-vlm's non-trimmable rule (`_checkpoint_match_len`): the WHOLE stored
    snapshot must be a prefix of the new prompt (and shorter than it), else 0."""
    if 0 < snapshot_len < len(next_ids) and next_ids[:snapshot_len] == prev_ids[:snapshot_len]:
        return snapshot_len
    return 0


def _render(p, msgs, tools=None):
    text = p._build_prompt("", msgs, SYSTEM, tools)
    tail, unmeasured = p._native_rider_tail_tokens(
        full_prompt=text, processed_prompt="", messages=msgs, system_prompt=SYSTEM, tools=tools,
        prefilled_modules=None, enable_thinking=None, reasoning_effort=None,
    )
    assert unmeasured is None
    return text, p._encode_prompt_token_ids(text), tail


def _lcp(a: List[int], b: List[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def _assert_loop_restores(p) -> None:
    for k in (1, 2):
        _, ids1, tail1 = _render(p, _with_rider(_iterations(k)))
        _, ids2, _ = _render(p, _with_rider(_iterations(k + 1)))
        assert tail1 and tail1 > CHECKPOINT_GUARD_TOKENS, "the rider must be measured as a tail"
        # Measured exactly: the tail is what the rider-free render does not share.
        msgs1 = _iterations(k)
        head = p._build_prompt_fragment(prompt="", messages=msgs1, system_prompt=SYSTEM, add_generation_prompt=False)
        assert tail1 == len(ids1) - _lcp(ids1, p._encode_prompt_token_ids(head))
        # With NO guard at all, the snapshot at the measured boundary is still a prefix.
        assert _restored(ids1, len(ids1) - tail1, ids2) == len(ids1) - tail1
        default_snap = len(ids1) - CHECKPOINT_GUARD_TOKENS
        rider_snap = len(ids1) - (CHECKPOINT_GUARD_TOKENS + tail1)
        # The bug: the default end-of-prompt snapshot contains the rider.
        assert _restored(ids1, default_snap, ids2) == 0
        # The fix: everything but the rider (and the guard) is restored.
        assert _restored(ids1, rider_snap, ids2) == rider_snap
        assert len(ids2) - rider_snap < (len(ids2) - len(ids1)) + tail1 + 2 * CHECKPOINT_GUARD_TOKENS


def test_rider_snapshot_restores_across_tool_loop_iterations_hermetic():
    _assert_loop_restores(_provider(_JinjaTokenizer()))


@pytest.mark.real_home("reads the installed Qwen3.8 tokenizer (HF cache) read-only")
def test_golden_rider_snapshot_restores_across_tool_loop_iterations():
    snap = _hf_snapshot(QWEN38)
    if snap is None:
        pytest.skip(f"{QWEN38} is not in the local HF cache")
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
    _assert_loop_restores(_qwen38_provider(tok))


def test_rider_snapshot_restores_across_conversation_turns_hermetic():
    """The FIRST call of a turn stays restorable by the next turn's first call
    (durable history keeps user/assistant only; the rider re-rides the tail)."""
    p = _provider(_JinjaTokenizer())
    turn1 = [{"role": "user", "content": "Tell me about neural fields."}]
    turn2 = turn1 + [
        {"role": "assistant", "content": "Neural fields map coordinates to signals."},
        {"role": "user", "content": "And CRE?"},
    ]
    _, ids1, tail1 = _render(p, _with_rider(turn1))
    _, ids2, _ = _render(p, _with_rider(turn2))
    assert _restored(ids1, len(ids1) - CHECKPOINT_GUARD_TOKENS, ids2) == 0
    snap = len(ids1) - (CHECKPOINT_GUARD_TOKENS + tail1)
    assert _restored(ids1, snap, ids2) == snap


@pytest.mark.parametrize(
    "msgs, prompt",
    [
        (_iterations(2), ""),  # nothing trails: default placement
        (_with_rider(_iterations(2)), "a final user prompt"),  # riders do not trail the prompt
        ([dict(RIDER)], ""),  # nothing but riders: no head to keep
    ],
)
def test_no_rider_tail_keeps_the_default_placement(msgs, prompt):
    p = _provider(_JinjaTokenizer())
    text = p._build_prompt(prompt, msgs, SYSTEM, None)
    assert p._native_rider_tail_tokens(
        full_prompt=text, processed_prompt=prompt, messages=msgs, system_prompt=SYSTEM, tools=None,
        prefilled_modules=None, enable_thinking=None, reasoning_effort=None,
    ) == (None, None)


def test_an_unmeasurable_rider_is_reported_not_silent():
    p = _provider(_JinjaTokenizer())
    msgs = _with_rider(_iterations(2))
    text = p._build_prompt("", msgs, SYSTEM, None)

    def boom(**kwargs):
        raise RuntimeError("template exploded")

    p._build_prompt_fragment = boom
    tail, why = p._native_rider_tail_tokens(
        full_prompt=text, processed_prompt="", messages=msgs, system_prompt=SYSTEM, tools=None,
        prefilled_modules=None, enable_thinking=None, reasoning_effort=None,
    )
    assert tail is None and "RuntimeError" in why


def test_guard_view_overrides_only_the_guard_and_forwards_everything_else():
    class Manager:
        exact_cache_guard_tokens = CHECKPOINT_GUARD_TOKENS

        def __init__(self):
            self.lock = object()
            self.stores = 0

        def store_exact_cache(self, *a, **k):
            self.stores += 1
            return True

    m = Manager()
    view = CheckpointGuardView(m, 200)
    assert view.exact_cache_guard_tokens == 200
    assert m.exact_cache_guard_tokens == CHECKPOINT_GUARD_TOKENS  # the shared manager is untouched
    assert view.lock is m.lock
    assert view.store_exact_cache([1, 2]) is True and m.stores == 1
    view._prefill_reserve_bytes = 7  # writes land on the real manager
    assert m._prefill_reserve_bytes == 7
    with pytest.raises(AttributeError):
        view.exact_cache_guard_tokens = 99  # never through to the shared manager
    assert m.exact_cache_guard_tokens == CHECKPOINT_GUARD_TOKENS
    import copy

    clone = copy.copy(view)
    assert clone.exact_cache_guard_tokens == 200 and clone.lock is m.lock


# --- the wiring, end to end through the provider with a fake mlx-vlm --------


class _Manager:
    exact_cache_guard_tokens = CHECKPOINT_GUARD_TOKENS

    def stats_snapshot(self):
        return {"exact_stores": 1}


class _Session:
    apc = None

    def __init__(self):
        self.manager = _Manager()

    def prompt_cache(self, memory_max_gb=None):
        return self.manager


def _native_provider(monkeypatch):
    seen: Dict[str, Any] = {}
    fake = types.ModuleType("mlx_vlm")

    def generate(model, processor, text, **kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(text="ok", prompt_tokens=10, cached_tokens=0, generation_tokens=1,
                                     prompt_tps=1.0, generation_tps=1.0, peak_memory=0.1, finish_reason="stop")

    fake.generate = generate
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake)
    p = _provider(_JinjaTokenizer())
    p.logger = Mock()
    p.llm = object()
    p._mtp_processor = object()
    p._native_runtime = None
    p._native_session = _Session()
    p._mlx_cache_options = {}
    p._mlx_cache_scope = "local"
    p.max_tokens, p.max_output_tokens, p.max_input_tokens = 4096, 64, None
    p.temperature, p.seed, p.top_p = 0.0, None, 1.0
    p.generate_fn = p._mtp_generate_fn
    p.stream_generate_fn = p._mtp_stream_generate_fn
    return p, seen


def test_the_rider_guard_reaches_mlx_vlm_on_a_per_call_view(monkeypatch):
    p, seen = _native_provider(monkeypatch)
    msgs = _with_rider(_iterations(2))
    r = p._generate_internal_unlocked("", messages=msgs, system_prompt=SYSTEM, prompt_cache_key="session:x")
    manager = seen.get("apc_manager")
    _, _, tail = _render(p, msgs)
    assert r.finish_reason != "error", r.content
    assert r.metadata["prompt_cache"]["snapshot_excludes_tail_tokens"] == tail
    assert isinstance(manager, CheckpointGuardView)
    assert manager.exact_cache_guard_tokens == CHECKPOINT_GUARD_TOKENS + tail
    assert p._native_session.manager.exact_cache_guard_tokens == CHECKPOINT_GUARD_TOKENS


def test_without_riders_mlx_vlm_gets_the_shared_manager_itself(monkeypatch):
    p, seen = _native_provider(monkeypatch)
    p._generate_internal_unlocked("", messages=_iterations(2), system_prompt=SYSTEM, prompt_cache_key="session:x")
    assert seen.get("apc_manager") is p._native_session.manager


def test_the_rider_guard_is_consumed_by_its_call(monkeypatch):
    """An out-of-call user of the kwargs (cache warm-up, prefill) never gets a stale view."""
    p, seen = _native_provider(monkeypatch)
    p._generate_internal_unlocked("", messages=_with_rider(_iterations(2)), system_prompt=SYSTEM,
                                  prompt_cache_key="session:x")
    assert isinstance(seen.get("apc_manager"), CheckpointGuardView)
    monkeypatch.setattr(p, "_mtp_kwargs", lambda *a, **k: {})
    p._native_cache_key = "session:x"
    assert p._mtp_call_kwargs({})["apc_manager"] is p._native_session.manager


def test_unmeasured_riders_land_in_the_prompt_cache_record(monkeypatch):
    p, seen = _native_provider(monkeypatch)
    monkeypatch.setattr(p, "_encode_prompt_token_ids", lambda text: None)
    r = p._generate_internal_unlocked("", messages=_with_rider(_iterations(2)), system_prompt=SYSTEM,
                                      prompt_cache_key="session:x")
    assert r.finish_reason != "error", r.content
    record = (r.metadata or {}).get("prompt_cache") or {}
    assert record.get("snapshot_rider_tail", "").startswith("#FALLBACK unmeasured")
    assert seen.get("apc_manager") is p._native_session.manager


def test_the_rider_guard_does_not_leak_into_the_next_call(monkeypatch):
    p, seen = _native_provider(monkeypatch)
    p._generate_internal_unlocked("", messages=_with_rider(_iterations(2)), system_prompt=SYSTEM,
                                  prompt_cache_key="session:x")
    seen.clear()
    p._generate_internal_unlocked("", messages=_iterations(3), system_prompt=SYSTEM, prompt_cache_key="session:x")
    assert seen.get("apc_manager") is p._native_session.manager


# --- the mlx-vlm seam ------------------------------------------------------


def test_installed_mlx_vlm_places_the_checkpoint_by_the_views_guard():
    """FAILS LOUDLY if mlx-vlm stops reading `manager.exact_cache_guard_tokens`
    when placing the final checkpoint: the fix would then silently do nothing."""
    coord_mod = pytest.importorskip("mlx_vlm.apc_coordinator")

    class Coordinator(coord_mod.APCCoordinator):
        enabled = True
        is_checkpoint = True

        def __init__(self, manager):
            self.manager = manager

    class Manager:
        exact_cache_guard_tokens = CHECKPOINT_GUARD_TOKENS

    ids = list(range(1000))
    assert Coordinator(Manager()).checkpoint_len(ids, set()) == 1000 - CHECKPOINT_GUARD_TOKENS
    assert Coordinator(CheckpointGuardView(Manager(), 300)).checkpoint_len(ids, set()) == 700
