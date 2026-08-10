"""A partial cache rollback must be refused, not accepted with a warning.

MEASURED (2026-08-03, `Qwen/Qwen3.5-4B` bf16, transformers 5.x, MPS): cropped-warm
arms returned EMPTY completions — `finish_reason='stop'`, no content — persisting
across retries until abstractcore's circuit breaker tripped. Cold arms were normal.

STRUCTURAL CAUSE, proven on real weights: `transformers.cache_utils` implements
`crop` as an explicit `pass` on the linear-attention layer classes ("We don't crop
the linear attention cache, so simply do nothing here"), while `Cache.crop` loops
every layer. Qwen3.5-4B is 24 linear / 8 full of 32 layers, so a crop leaves 24 of
32 layers BIT-IDENTICAL and 8 rolled back — an internally inconsistent cache, not
an approximation.

WHY IT SURVIVED, and what this file pins: the old verify read
`cache.get_seq_length()`, and `Cache.get_seq_length` deliberately resolves to the
first ATTENTION layer on exactly these hybrids. It therefore sampled a layer that
did crop, and always passed. **A test that only asserts `get_seq_length()` passes
on the broken code** — `test_seq_length_alone_cannot_detect_a_partial_rollback`
below demonstrates that directly, so the weaker assertion can never be mistaken
for coverage again.
"""

from __future__ import annotations

from typing import Any, List, Optional

import pytest

from abstractcore.providers.huggingface_provider import (
    HuggingFaceProvider,
    _TransformersPromptCacheValue,
)


class _FullAttentionLayer:
    """A `DynamicLayer` stand-in: `crop` really truncates, and it reports a length."""

    def __init__(self, seq_len: int):
        self.seq_len = int(seq_len)

    def crop(self, max_length: int) -> None:
        if self.seq_len > int(max_length):
            self.seq_len = int(max_length)

    def get_seq_length(self) -> int:
        return self.seq_len


class _StubbornAttentionLayer(_FullAttentionLayer):
    """An attention layer whose crop silently does nothing (the zamba-style class
    that inherits the no-op over its attention half)."""

    def crop(self, max_length: int) -> None:  # noqa: D102
        return None


class _LinearAttentionLayer:
    """A linear-attention/recurrent layer: `crop` is a no-op and there is NO
    sequence length to report — unverifiable by construction."""

    def __init__(self):
        self.recurrent_state = "carries every token ever seen"

    def crop(self, max_length: int) -> None:
        return None


class _FakeCache:
    """`Cache`-shaped, including the `get_seq_length` behaviour that hid the bug:
    when layer 0 is linear, resolve to the first attention layer instead."""

    def __init__(self, layers: List[Any]):
        self.layers = layers

    def crop(self, max_length: int) -> None:
        for layer in self.layers:
            layer.crop(max_length)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        for layer in self.layers[layer_idx:]:
            fn = getattr(layer, "get_seq_length", None)
            if callable(fn):
                return int(fn())
        return 0


def _provider() -> HuggingFaceProvider:
    p = HuggingFaceProvider.__new__(HuggingFaceProvider)

    class _Log:
        def __init__(self):
            self.warnings: List[str] = []

        def warning(self, msg: str, *a: Any, **k: Any) -> None:
            self.warnings.append(str(msg))

        def debug(self, *a: Any, **k: Any) -> None:
            pass

    p.logger = _Log()
    return p


def _state(cache: Any) -> _TransformersPromptCacheValue:
    s = _TransformersPromptCacheValue.__new__(_TransformersPromptCacheValue)
    s.cache = cache
    return s


def _hybrid(n_full: int = 8, n_linear: int = 24, seq_len: int = 10_000) -> _FakeCache:
    """Qwen3.5-4B's shape: 24 linear / 8 full of 32, linear first (so
    `get_seq_length` must skip forward, exactly as in transformers)."""
    layers: List[Any] = []
    for i in range(n_full + n_linear):
        if i % 4 == 3:
            layers.append(_FullAttentionLayer(seq_len))
        else:
            layers.append(_LinearAttentionLayer())
    return _FakeCache(layers)


def test_hybrid_cache_crop_is_refused():
    """THE PIN. A cache with layers that cannot roll back must refuse to crop."""
    p = _provider()
    cache = _hybrid()

    assert p._transformers_crop_cache(_state(cache), 9_000) is False


def test_hybrid_cache_is_left_untouched_by_the_refusal():
    """Refusal must come BEFORE mutation. `Cache.crop` rolls back the attention
    layers even while no-opping the linear ones, so refusing afterwards would hand
    back the very inconsistent cache the guard exists to prevent."""
    p = _provider()
    cache = _hybrid(seq_len=10_000)
    before = [l.get_seq_length() for l in cache.layers if hasattr(l, "get_seq_length")]

    p._transformers_crop_cache(_state(cache), 9_000)

    after = [l.get_seq_length() for l in cache.layers if hasattr(l, "get_seq_length")]
    assert after == before, "the attention layers were cropped before the refusal"


def test_refusal_is_labelled_once_and_names_the_layers():
    p = _provider()
    cache = _hybrid()

    p._transformers_crop_cache(_state(cache), 9_000)
    p._transformers_crop_cache(_state(cache), 8_000)

    assert len(p.logger.warnings) == 1                      # once per provider
    msg = p.logger.warnings[0]
    assert "#FALLBACK" in msg
    assert "empty completions" in msg                        # the observed symptom
    assert "24 layer(s)" in msg                              # counted, not hand-waved


def test_seq_length_alone_cannot_detect_a_partial_rollback():
    """The reason this survived, made explicit.

    After a hybrid crop the cache's own `get_seq_length()` reports the CROPPED
    length, because it resolves to an attention layer. A verify built on it — and
    a test asserting only it — passes on a cache where 24 of 32 layers still carry
    the removed tokens.
    """
    cache = _hybrid(seq_len=10_000)
    cache.crop(9_000)                                        # raw transformers behaviour

    assert cache.get_seq_length() == 9_000                   # looks perfectly fine...
    linear = [l for l in cache.layers if isinstance(l, _LinearAttentionLayer)]
    assert len(linear) == 24                                 # ...while 24 layers never moved

    # The real predicate sees it.
    assert _provider()._transformers_uncroppable_layers(cache)


def test_pure_attention_cache_still_crops():
    """Do not break the lane that works: an all-attention cache must still crop,
    and every layer must land on the requested length."""
    p = _provider()
    cache = _FakeCache([_FullAttentionLayer(10_000) for _ in range(8)])

    assert p._transformers_crop_cache(_state(cache), 9_000) is True
    assert [l.get_seq_length() for l in cache.layers] == [9_000] * 8
    assert p.logger.warnings == []


def test_partial_rollback_among_attention_layers_is_refused_by_the_verify():
    """The per-layer verify's own job: one attention layer that silently keeps its
    state must be caught even though `get_seq_length()` reads a layer that moved."""
    p = _provider()
    layers: List[Any] = [_FullAttentionLayer(10_000) for _ in range(7)]
    layers.append(_StubbornAttentionLayer(10_000))           # last layer never rolls back
    cache = _FakeCache(layers)

    assert cache.get_seq_length() == 10_000                  # layer 0 pre-crop
    assert p._transformers_crop_cache(_state(cache), 9_000) is False


def test_layer_seq_lengths_reports_every_measurable_layer():
    p = _provider()
    cache = _FakeCache([_FullAttentionLayer(5), _LinearAttentionLayer(), _FullAttentionLayer(7)])

    assert p._transformers_layer_seq_lengths(cache) == [5, 7]   # linear contributes nothing


def test_uncroppable_detection_prefers_type_over_name():
    """The predicate now decides correctness, so it must not rest on a substring.
    A linear layer whose class name says nothing must still be caught by type."""

    try:
        from transformers.cache_utils import LinearAttentionCacheLayerMixin
    except Exception:  # pragma: no cover - exercised wherever transformers ships it
        pytest.skip("transformers does not export LinearAttentionCacheLayerMixin")

    class Innocuous(LinearAttentionCacheLayerMixin):          # no telltale substring
        def __init__(self):
            pass

        def lazy_initialization(self, *a, **k):
            pass

        def update_conv_state(self, *a, **k):
            pass

        def update_recurrent_state(self, *a, **k):
            pass

    cache = _FakeCache([_FullAttentionLayer(10), Innocuous()])
    found = _provider()._transformers_uncroppable_layers(cache)

    assert [i for i, _ in found] == [1]
    assert "linear" not in found[0][1].lower()               # caught by type, not by name
