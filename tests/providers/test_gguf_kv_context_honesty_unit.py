"""GGUF KV honesty, device-fit context sizing, and the context-full failure.

Found on the Linux test server (Quadro RTX 5000 16 GB, gateway 0.17.x,
2026-10-29): the Models page showed `Qwen3.5-9B-Kimi-k3-Distilled-GGUF` as
"~5.4 GiB + 8.3 GiB cache" while the whole GPU held 9.1 GiB. The "cache" was
the KV estimate `n_ctx * block_count * 2 * kv_heads * head_dim * 2` at n_ctx
65,536 over all 33 blocks; qwen35 gives a KV cache to one block in four
(`full_attention_interval = 4`): 8 blocks, 2.0 GiB. The same log showed
`decode: failed to find a memory slot for batch of size 512`: a prompt longer
than the loaded context was evaluated anyway, retried cold, the engine was
rebuilt, and the request answered "Error: failed to prefill ..." as text.

Fakes and synthesized GGUF headers only: no weights.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, List, Sequence

import pytest

from abstractcore.exceptions import ContextWindowExceededError
from abstractcore.providers import hf_residency
from abstractcore.providers.huggingface_provider import HuggingFaceProvider, _LLAMA_CONTEXT_OVERFLOW_RE
from abstractcore.utils.kv_geometry import kv_cache_breakdown, kv_cache_bytes
from abstractcore.utils.model_cache import read_gguf_geometry

GIB = 1 << 30

# The server file's header, as read on 2026-10-29 (`qwen35.*`).
KIMI = dict(architecture="qwen35", block_count=33, head_count=16, head_count_kv=4, key_length=256,
            value_length=256, full_attention_interval=4, ssm_conv_kernel=4, ssm_state_size=128,
            ssm_group_count=16, ssm_inner_size=4096)


# -- a minimal GGUF header writer --------------------------------------------------------
def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv_u32(key: str, value: int) -> bytes:
    return _s(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_str(key: str, value: str) -> bytes:
    return _s(key) + struct.pack("<I", 8) + _s(value)


def _kv_arr(key: str, values: Sequence[int], elem_type: int, fmt: str) -> bytes:
    body = b"".join(struct.pack(fmt, v) for v in values)
    return _s(key) + struct.pack("<I", 9) + struct.pack("<IQ", elem_type, len(values)) + body


def _gguf(path: Path, kvs: List[bytes]) -> Path:
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + b"".join(kvs))
    return path


def _kimi_file(tmp_path: Path) -> Path:
    return _gguf(tmp_path / "kimi.gguf", [
        _kv_str("general.architecture", "qwen35"),
        _kv_u32("qwen35.block_count", 33),
        _kv_u32("qwen35.attention.head_count", 16),
        _kv_u32("qwen35.attention.head_count_kv", 4),
        _kv_u32("qwen35.attention.key_length", 256),
        _kv_u32("qwen35.attention.value_length", 256),
        _kv_u32("qwen35.full_attention_interval", 4),
        _kv_u32("qwen35.ssm.conv_kernel", 4),
        _kv_u32("qwen35.ssm.state_size", 128),
        _kv_u32("qwen35.ssm.group_count", 16),
        _kv_u32("qwen35.ssm.inner_size", 4096),
    ])


# -- 1. the KV estimate counts only the layers that hold a cache --------------------------
def test_hybrid_kv_counts_one_block_in_four_not_every_block():
    out = kv_cache_breakdown(KIMI, 65536)
    assert out["kv_layers"] == 8, "MUTANT: every block counted as attention -> 33 -> RED"
    assert out["kv_bytes"] == 65536 * 8 * 2 * 4 * 256 * 2 == 2 * GIB
    # the old figure the panel showed was 8.25 GiB (33 blocks)
    assert kv_cache_bytes(KIMI, 65536) < 2.1 * GIB


def test_dense_model_is_unchanged():
    dense = dict(block_count=36, head_count=32, head_count_kv=8, key_length=128, value_length=128)
    assert kv_cache_bytes(dense, 65536) == 65536 * 36 * 2 * 8 * 128 * 2


def test_per_layer_heads_and_sliding_window_layers():
    # gemma4-like: 6 layers, every 6th global; SWA layers use a narrower head
    geom = dict(block_count=6, head_count=16, head_count_kv=8, head_count_kv_layers=[8, 8, 8, 8, 8, 2],
                key_length=512, value_length=512, key_length_swa=256, value_length_swa=256,
                sliding_window=1024, sliding_window_pattern_layers=[1, 1, 1, 1, 1, 0])
    full = kv_cache_breakdown(geom, 32768, swa_full=True)
    assert full["kv_bytes"] == 32768 * (5 * 8 * 512 + 2 * 1024) * 2
    windowed = kv_cache_breakdown(geom, 32768, swa_full=False)
    assert windowed["kv_bytes"] == 1536 * 5 * 8 * 512 * 2 + 32768 * 2 * 1024 * 2


def test_header_reader_returns_the_layer_layout(tmp_path):
    path = _gguf(tmp_path / "g.gguf", [
        _kv_str("general.architecture", "gemma4"),
        _kv_u32("gemma4.block_count", 3),
        _kv_arr("gemma4.attention.head_count_kv", [8, 8, 2], 4, "<I"),
        _kv_arr("gemma4.attention.sliding_window_pattern", [1, 1, 0], 7, "<B"),
        _kv_u32("gemma4.attention.key_length", 512),
    ])
    geom = read_gguf_geometry(path)
    assert geom["head_count_kv_layers"] == [8, 8, 2] and geom["head_count_kv"] == 8
    assert geom["sliding_window_pattern_layers"] == [1, 1, 0]
    assert read_gguf_geometry(_kimi_file(tmp_path))["full_attention_interval"] == 4


class _FakeEngine:
    def __init__(self, path: Path, n_ctx: int):
        self.model_path = str(path)
        self._n_ctx = n_ctx
        self.metadata = {"general.architecture": "qwen35", "qwen35.block_count": "33",
                         "qwen35.attention.head_count": "16", "qwen35.attention.head_count_kv": "4",
                         "qwen35.attention.key_length": "256"}

    def n_ctx(self) -> int:
        return self._n_ctx


def test_residency_row_reports_the_layer_aware_kv_and_the_context(tmp_path, monkeypatch):
    engine = _FakeEngine(_kimi_file(tmp_path), 65536)
    info = hf_residency.llama_bytes(engine)
    assert info["n_ctx"] == 65536
    assert info["kv_alloc_bytes"] < 2.1 * GIB, (
        "the file header's full_attention_interval must be used (MUTANT: metadata only -> 8.25 GiB -> RED)")


# -- 2. the device rule sizes the context before allocating -------------------------------
def _provider(**attrs: Any) -> HuggingFaceProvider:
    p = HuggingFaceProvider.__new__(HuggingFaceProvider)
    import logging
    p.logger = logging.getLogger("test")
    p.model = "mradermacher/Qwen3.5-9B-Kimi-k3-Distilled-GGUF"
    p.n_gpu_layers = -1
    for k, v in attrs.items():
        setattr(p, k, v)
    return p


def _device(monkeypatch, *, backend="cuda", free=int(14.5 * GIB), total=int(14.56 * GIB), weights=int(5.6 * GIB)):
    import abstractcore.utils.memory as mem
    import abstractcore.providers.huggingface_provider as hp

    monkeypatch.setattr(mem, "get_memory_snapshot",
                        lambda: {"device": {"backend": backend, "free_bytes": free, "total_bytes": total}})
    monkeypatch.setattr(hp, "_gguf_total_weight_bytes", lambda _p: weights)


LADDER = [262144, 131072, 65536, 32768, 16384, 8192, 4096]


def test_kv_takes_at_most_half_of_what_is_free_after_weights(tmp_path, monkeypatch):
    _device(monkeypatch)
    p = _provider()
    rungs = p._gguf_device_fit_rungs(str(_kimi_file(tmp_path)), list(LADDER))
    # free 14.5 - weights 5.6 - reserve 1.46 = 7.44 GiB; half = 3.7 GiB -> 65,536 (2.0 GiB KV)
    assert rungs[0] == 65536, f"MUTANT: no rule -> starts at 262,144 -> RED (got {rungs[0]})"
    assert rungs[-1] == 4096 and p._gguf_context_rule["skipped_rungs"] == [262144, 131072]


def test_a_full_card_keeps_the_smallest_rung(tmp_path, monkeypatch):
    _device(monkeypatch, free=int(6 * GIB))
    assert _provider()._gguf_device_fit_rungs(str(_kimi_file(tmp_path)), list(LADDER)) == [4096]


def test_rule_does_not_apply_off_cuda_or_on_cpu(tmp_path, monkeypatch):
    path = str(_kimi_file(tmp_path))
    _device(monkeypatch, backend="metal")
    assert _provider()._gguf_device_fit_rungs(path, list(LADDER)) == LADDER
    _device(monkeypatch)
    assert _provider(n_gpu_layers=0)._gguf_device_fit_rungs(path, list(LADDER)) == LADDER


# -- 3. a prompt the context cannot hold fails clearly, before any eval --------------------
class _Llama:
    def __init__(self, n_ctx: int):
        self._n = n_ctx
        self.n_tokens = 0
        self._input_ids: List[int] = []
        self.evals: List[int] = []
        self.resets = 0

    def n_ctx(self) -> int:
        return self._n

    def reset(self) -> None:
        self.resets += 1
        self.n_tokens = 0

    def eval(self, tokens) -> None:
        self.evals.append(len(tokens))
        if self.n_tokens + len(tokens) > self._n:
            raise RuntimeError("llama_decode returned 1")
        self.n_tokens += len(tokens)


def test_prompt_longer_than_the_context_is_refused_before_eval():
    llm = _Llama(1024)
    p = _provider(llm=llm)
    with pytest.raises(ContextWindowExceededError) as err:
        p._gguf_prefill_prompt_cache(None, tuple(range(1, 1276)))
    assert llm.evals == [], "MUTANT: no pre-check -> eval, cold retry, engine rebuild -> RED"
    assert err.value.prompt_tokens == 1275 and err.value.context_length == 1024
    msg = str(err.value)
    assert "1,275 tokens" in msg and "1,024-token context" in msg and "Error" not in msg


def test_prompt_that_fits_still_prefills():
    llm = _Llama(1024)
    assert _provider(llm=llm)._gguf_prefill_prompt_cache(None, tuple(range(1, 500)))
    assert llm.evals == [499]


def test_llama_cpp_refusal_text_is_recognized():
    m = _LLAMA_CONTEXT_OVERFLOW_RE.search("Requested tokens (6402) exceed context window of 1024")
    assert m and (m.group(1), m.group(2)) == ("6402", "1024")
