"""Opt-in proof that the native MTP lane reuses the prefix cache AND keeps MTP.

    ABSTRACTCORE_RUN_MLX_APC_LIVE=1 pytest -q tests/providers/test_mlx_native_apc_reuse_live.py

Needs the locally cached pair mlx-community/Qwen3.5-4B-4bit (target) and
mlx-community/Qwen3.5-4B-MTP-4bit (drafter, auto-resolved); never downloads.
Ordinary collection neither imports MLX nor loads weights.

The decisive case pushes ONE prefix snapshot past 512 MiB (about 15k prompt
tokens on this 4B: ~38 KB per token). mlx-vlm refuses to retain a snapshot
larger than its memory budget, so with the old forced 0.5 GiB budget turn 2
re-prefilled everything; with the machine-sized default it restores the prefix.
"""
import math
import os
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

TARGET = "mlx-community/Qwen3.5-4B-4bit"
DRAFTER = "mlx-community/Qwen3.5-4B-MTP-4bit"
HUB = Path.home() / ".cache/huggingface/hub"
PARA = (
    "The Antikythera mechanism is an ancient Greek hand-powered orrery, the oldest "
    "known analogue computer, used to predict astronomical positions and eclipses. "
)


def _cached(repo: str) -> bool:
    return (HUB / ("models--" + repo.replace("/", "--")) / "snapshots").is_dir()


def _require_local_pair():
    if os.getenv("ABSTRACTCORE_RUN_MLX_APC_LIVE") != "1":
        pytest.skip("Set ABSTRACTCORE_RUN_MLX_APC_LIVE=1 to run against local MLX weights")
    if not (_cached(TARGET) and _cached(DRAFTER)):
        pytest.skip("Local Qwen3.5-4B target + MTP pair required; no download attempted")


def _load(**provider_kwargs):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setenv("TRANSFORMERS_OFFLINE", "1")
        from abstractcore import create_llm

        return create_llm(
            "mlx", model=TARGET,
            speculation={"mode": "native_mtp", "num_draft_tokens": 3, "require_acceleration": True},
            **provider_kwargs,
        )


def _two_turns(llm, key: str, min_prompt_tokens: int):
    # Measure the paragraph in a run (a lone one tokenizes shorter than it does
    # repeated) and overshoot by 5% so the prompt lands past `min_prompt_tokens`.
    per_para = len(llm.tokenizer.encode(PARA * 10)) / 10
    repeat = math.ceil(min_prompt_tokens * 1.05 / per_para)
    system = "Reference material:\n\n" + PARA * repeat + "\n\nEnd of reference."
    messages = [{"role": "user", "content": "Answer in under 10 words. What is it?"}]
    rows = []
    for follow_up in (None, "Who built it?"):
        if follow_up:
            messages.append({"role": "user", "content": follow_up})
        started = time.perf_counter()
        response = llm.generate(messages=messages, system_prompt=system, prompt_cache_key=key,
                                max_output_tokens=24, temperature=0.0)
        rows.append((time.perf_counter() - started, response))
        messages.append({"role": "assistant", "content": response.content or "(empty)"})
    return rows


def test_default_budget_is_upstream_machine_sizing_on_the_real_library(monkeypatch):
    """Memory budget: upstream's machine-relative sizing. Checkpoint SHAPE: ours.

    Re-pinned 2026-09-22 (mission A3). This used to assert the checkpoint
    interval equals mlx-vlm's default (2048). It no longer should. mlx-vlm
    uses `checkpoint_entries` both as the number of snapshots ONE prompt emits
    and as the capacity of the WHOLE snapshot store, so any interval inside the
    prompt makes a single prompt fill the store. After a tool loop, no snapshot
    of the conversation survived and the next turn re-prefilled everything (the
    operator's live run 520d0e69: 8,407 tokens cold). The shipped shape is one
    snapshot per call (interval past any context window), 8 restorable prompts,
    placed 8 tokens before the end. The memory budget and reserve stay
    upstream's (a flat constant there once disabled reuse past ~10-19k tokens),
    and the operator's `APC_*` env vars still win.
    """
    pytest.importorskip("mlx_vlm.apc")
    from mlx_vlm import apc as upstream

    from abstractcore.providers.mlx_native_session import (
        CHECKPOINT_ENTRIES,
        CHECKPOINT_GUARD_TOKENS,
        CHECKPOINT_INTERVAL_TOKENS,
        NativeSession,
    )

    for env in ("APC_CHECKPOINT_INTERVAL_TOKENS", "APC_CHECKPOINT_ENTRIES", "APC_CHECKPOINT_GUARD_TOKENS"):
        monkeypatch.delenv(env, raising=False)
    manager = NativeSession().prompt_cache()
    reference = upstream.APCManager(overrides={})
    try:
        assert manager.memory_max_bytes == reference.memory_max_bytes
        assert manager.memory_reserve_bytes == reference.memory_reserve_bytes
        assert manager.checkpoint_interval_tokens == CHECKPOINT_INTERVAL_TOKENS >= 1_000_000
        assert manager._exact_cache_max == CHECKPOINT_ENTRIES == 8
        assert manager.exact_cache_guard_tokens == CHECKPOINT_GUARD_TOKENS == 8
    finally:
        manager.close()
        reference.close()

    # The operator's environment still wins over the shipped shape.
    monkeypatch.setenv("APC_CHECKPOINT_INTERVAL_TOKENS", "256")
    monkeypatch.setenv("APC_CHECKPOINT_ENTRIES", "4")
    monkeypatch.setenv("APC_CHECKPOINT_GUARD_TOKENS", "1")
    overridden = NativeSession().prompt_cache()
    try:
        assert overridden.checkpoint_interval_tokens == 256
        assert overridden._exact_cache_max == 4
        assert overridden.exact_cache_guard_tokens == 1
    finally:
        overridden.close()


def test_snapshot_past_half_a_gib_is_reused_and_mtp_still_runs():
    _require_local_pair()
    llm = _load()
    try:
        (d1, r1), (d2, r2) = _two_turns(llm, "apc-live:big", min_prompt_tokens=15000)
    finally:
        llm.unload_model(llm.model)
    pc1, pc2 = r1.metadata["prompt_cache"], r2.metadata["prompt_cache"]
    sp1, sp2 = r1.metadata["speculation"], r2.metadata["speculation"]
    assert r1.usage["input_tokens"] >= 15000, r1.usage
    assert sp1["used"] is True and sp2["used"] is True, (sp1, sp2)
    assert pc1["outcome"] == "cold" and pc1["cached_tokens"] == 0, pc1
    assert pc1["apc"]["exact_stores"] >= 1 and pc1["apc"]["memory_skips"] == 0, pc1
    assert "degraded_reason" not in pc1, pc1
    assert pc2["outcome"] == "hit_restore", pc2
    assert pc2["cached_tokens"] > 0.9 * pc1["fed_tokens"], (pc1, pc2)
    assert pc2["fed_tokens"] < 0.05 * pc1["fed_tokens"], (pc1, pc2)
    assert d2 < 0.35 * d1, (d1, d2)


def test_starved_explicit_budget_degrades_loudly_instead_of_silently():
    _require_local_pair()
    llm = _load(mlx_cache_memory_max_gb=0.06)
    try:
        (d1, r1), (d2, r2) = _two_turns(llm, "apc-live:starved", min_prompt_tokens=3000)
    finally:
        llm.unload_model(llm.model)
    pc1, pc2 = r1.metadata["prompt_cache"], r2.metadata["prompt_cache"]
    assert r1.metadata["speculation"]["used"] is True and r2.metadata["speculation"]["used"] is True
    assert pc1["apc"]["memory_max_bytes"] == int(0.06 * (1 << 30)), pc1
    assert pc1["apc"]["exact_stores"] == 0 and pc1["apc"]["memory_skips"] >= 1, pc1
    assert pc1["degraded_reason"].startswith("#FALLBACK native_apc_store_skipped"), pc1
    assert pc2["outcome"] == "cold" and pc2["cached_tokens"] == 0, pc2
    assert pc2["degraded_reason"].startswith("#FALLBACK native_apc_store_skipped"), pc2
