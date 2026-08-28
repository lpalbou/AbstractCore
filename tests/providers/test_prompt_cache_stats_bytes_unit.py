"""Best-effort byte visibility in `get_prompt_cache_stats()`.

`meta_by_key` rows gain `bytes` where the provider can measure the stored
cache value; MLX stats additionally expose hybrid-snapshot presence under a
top-level `snapshots` section. Stats must never raise. CPU-only, model-free.
"""

from __future__ import annotations

from types import SimpleNamespace

from abstractcore.providers.base import PromptCacheStore
from abstractcore.providers.huggingface_provider import HuggingFaceProvider
from abstractcore.providers.mlx_provider import MLXProvider


class _FakeArray:
    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes


class _FakeLayer:
    def __init__(self, *nbytes: int) -> None:
        self.state = tuple(_FakeArray(n) for n in nbytes)


def _mlx_provider() -> MLXProvider:
    p = MLXProvider.__new__(MLXProvider)
    p.provider = "mlx"
    p.model = "mlx-test-model"
    p._default_prompt_cache_key = None
    p._prompt_cache_store = PromptCacheStore(max_entries=8)
    p._ensure_hybrid_snapshot_state()
    return p


def test_mlx_stats_report_store_entry_bytes_and_snapshots() -> None:
    p = _mlx_provider()
    p._prompt_cache_store.set("k1", [_FakeLayer(100, 50), _FakeLayer(25)])
    p._store_hybrid_snapshot("k1", [_FakeLayer(7)], [1, 2])

    stats = p.get_prompt_cache_stats()

    assert stats["entries"] == 1
    assert stats["meta_by_key"]["k1"]["bytes"] == 175
    assert stats["snapshots"] == {"count": 1, "bytes": 7}


def test_mlx_stats_unmeasurable_values_stay_unknown_and_never_raise() -> None:
    p = _mlx_provider()
    p._prompt_cache_store.set("opaque", object())

    stats = p.get_prompt_cache_stats()

    assert "bytes" not in stats["meta_by_key"]["opaque"]
    assert stats["snapshots"] == {"count": 0, "bytes": None}


def test_mlx_stats_do_not_disturb_lru_order() -> None:
    p = _mlx_provider()
    p._prompt_cache_store.set("old", [_FakeLayer(1)])
    p._prompt_cache_store.set("new", [_FakeLayer(2)])

    p.get_prompt_cache_stats()

    # `peek` must not refresh recency: "old" is still first in eviction order.
    assert p._prompt_cache_store.keys() == ["old", "new"]


def test_huggingface_transformers_value_bytes_best_effort() -> None:
    class _FakeTensor:
        def __init__(self, numel: int, element_size: int) -> None:
            self._numel = numel
            self._element_size = element_size

        def numel(self) -> int:
            return self._numel

        def element_size(self) -> int:
            return self._element_size

    p = object.__new__(HuggingFaceProvider)
    cache = SimpleNamespace(
        key_cache=[_FakeTensor(10, 2)],
        value_cache=[_FakeTensor(10, 2), _FakeTensor(5, 4)],
    )

    assert p._prompt_cache_value_bytes(SimpleNamespace(cache=cache)) == 60
    assert p._prompt_cache_value_bytes(None) is None
    assert p._prompt_cache_value_bytes(SimpleNamespace(cache=object())) is None
