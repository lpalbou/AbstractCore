"""CPU contract for Qwen4 MTP cohort compaction; no MLX import/evaluation.

Installed upstream method bodies are compiled in a NumPy-only namespace to
reproduce the real inherited filter and indexer concatenation, not a guessed
copy of their behavior. Missing optional mlx-vlm source skips these probes.
"""

import ast
from importlib import metadata
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from abstractcore.providers.mlx_qwen4 import _qwen4_mtp_drafter_class


class CPUArrayType(type):
    def __instancecheck__(cls, value):
        return isinstance(value, np.ndarray)

    def __call__(cls, *args, **kwargs):
        return np.array(*args, **kwargs)


class CPUArray(metaclass=CPUArrayType):
    pass


CPU_MX = types.SimpleNamespace(array=CPUArray, int32=np.int32, concatenate=np.concatenate,
                               arange=np.arange, argmax=np.argmax, broadcast_to=np.broadcast_to)


def upstream_method(relative, class_name, method_name):
    try:
        source = Path(metadata.distribution("mlx-vlm").locate_file("mlx_vlm")) / relative
    except metadata.PackageNotFoundError:
        pytest.skip("CPU source-contract test requires installed mlx-vlm metadata")
    if not source.is_file():
        pytest.skip("Installed mlx-vlm source is unavailable")
    tree = ast.parse(source.read_text())
    parent = (next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
              if class_name else tree)
    method = next(node for node in parent.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    method.decorator_list = []
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    ast.fix_missing_locations(unit)
    namespace = {"mx": CPU_MX}
    exec(compile(unit, str(source), "exec"), namespace)
    return namespace[method_name]


@pytest.fixture
def backend(monkeypatch):
    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    core.array, core.int32 = CPUArray, np.int32
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    base_filter = upstream_method("speculative/drafters/mtp_base.py", "AutoregressiveMTPDraftModel", "filter_batch")
    update = upstream_method("models/qwen4_exp/language.py", "QSAKVCache", "update_indexer")
    update.__globals__["_append_indexer_positions"] = upstream_method(
        "models/qwen4_exp/language.py", None, "_append_indexer_positions")
    cache_filter = upstream_method("models/qwen4_exp/language.py", "QSAKVCache", "filter")
    base = type("InstalledGenericMTPFilter", (), {"filter_batch": base_filter})
    cache_type = type("InstalledQSACacheMethods", (), {"update_indexer": update, "filter": cache_filter})
    return types.SimpleNamespace(base=base, cache_type=cache_type)


def draft_fixture(base, cache_type, *, mrope=False):
    draft = base()
    cache = cache_type()
    cache.keys = np.arange(4 * 2 * 4 * 8).reshape(4, 2, 4, 8)
    cache.values = cache.keys + 5000
    cache.index_keys = np.arange(4 * 4 * 128).reshape(4, 4, 128)
    cache.index_position_ids = (np.arange(3 * 4 * 4).reshape(3, 4, 4) if mrope
                                else np.arange(4 * 4).reshape(4, 4))
    cache.index_block_keys = np.arange(4 * 2 * 128).reshape(4, 2, 128)
    cache.offset, cache.index_block_ratio = 4, 2
    draft._cache = [cache]
    draft._seed_token = np.arange(4).reshape(4, 1)
    draft._seed_hidden = np.arange(4 * 8).reshape(4, 1, 8)
    draft._next_position = np.array([11, 13, 17, 19])
    return draft, cache


def test_installed_generic_filter_reproduces_qsa_row_mismatch(backend):
    draft, cache = draft_fixture(backend.base, backend.cache_type)
    draft.filter_batch(np.array([1, 3]))
    assert cache.keys.shape[0] == 2
    if cache.index_keys.shape[0] != 4:
        pytest.skip("Installed upstream now filters QSA auxiliary rows natively")
    with pytest.raises(ValueError, match="dimensions.*match"):
        cache.update_indexer(np.zeros((2, 1, 128)), np.zeros((2, 1)))


@pytest.mark.parametrize("mrope", [False, True], ids=["text-positions", "mrope-positions"])
def test_compaction_preserves_nonprefix_rows_and_repeated_four_two_one_decode(backend, mrope):
    draft, cache = draft_fixture(_qwen4_mtp_drafter_class(backend.base), backend.cache_type, mrope=mrope)
    original = {name: getattr(cache, name).copy() for name in (
        "keys", "values", "index_keys", "index_position_ids", "index_block_keys")}
    seed_token, seed_hidden, position = draft._seed_token.copy(), draft._seed_hidden.copy(), draft._next_position.copy()
    keep = np.array([3, 1])  # Both reorder and non-prefix selection catch double-filtering.
    draft.filter_batch(keep)
    for name in ("keys", "values", "index_keys", "index_block_keys"):
        np.testing.assert_array_equal(getattr(cache, name), original[name][keep])
    positions = original["index_position_ids"][:, keep] if mrope else original["index_position_ids"][keep]
    np.testing.assert_array_equal(cache.index_position_ids, positions)
    np.testing.assert_array_equal(draft._seed_token, seed_token[keep])
    np.testing.assert_array_equal(draft._seed_hidden, seed_hidden[keep])
    np.testing.assert_array_equal(draft._next_position, position[keep])
    assert cache.offset == 4 and cache.index_block_ratio == 2
    assert cache.update_indexer(np.zeros((2, 1, 128)), np.zeros((3, 2, 1) if mrope else (2, 1)))[0].shape == (2, 5, 128)

    draft.filter_batch(np.array([1]))
    np.testing.assert_array_equal(cache.keys, original["keys"][[1]])
    np.testing.assert_array_equal(cache.index_keys[:, :4], original["index_keys"][[1]])
    np.testing.assert_array_equal(cache.index_block_keys, original["index_block_keys"][[1]])
    np.testing.assert_array_equal(draft._seed_token, seed_token[[1]])
    np.testing.assert_array_equal(draft._seed_hidden, seed_hidden[[1]])
    np.testing.assert_array_equal(draft._next_position, position[[1]])
    assert cache.update_indexer(np.zeros((1, 1, 128)), np.zeros((3, 1, 1) if mrope else (1, 1)))[0].shape == (1, 6, 128)


def test_future_upstream_auxiliary_filter_is_not_applied_twice(backend):
    class FixedUpstream(backend.base):
        def filter_batch(self, keep):
            # Simulate upstream adopting cache.filter while retaining seed and
            # position updates. Avoid duplicating the original generic KV slice.
            caches, self._cache = self._cache, []
            try:
                super().filter_batch(keep)
            finally:
                self._cache = caches
            for cache in caches:
                cache.filter(keep)

    draft, cache = draft_fixture(_qwen4_mtp_drafter_class(FixedUpstream), backend.cache_type, mrope=True)
    original = cache.index_keys.copy()
    draft.filter_batch(np.array([1, 3]))
    np.testing.assert_array_equal(cache.index_keys, original[[1, 3]])
    assert cache.keys.shape[0] == cache.index_position_ids.shape[1] == cache.index_block_keys.shape[0] == 2


def test_upstream_invalidation_and_empty_auxiliary_caches_are_preserved(backend):
    class InvalidatingUpstream(backend.base):
        def filter_batch(self, keep):
            super().filter_batch(keep)
            for cache in self._cache:
                cache.index_block_keys = None

    draft, cache = draft_fixture(_qwen4_mtp_drafter_class(InvalidatingUpstream), backend.cache_type)
    cache.index_keys = cache.index_position_ids = None
    draft._seed_token = draft._seed_hidden = None
    draft._next_position = 19
    draft.filter_batch(np.array([1, 3]))
    assert cache.keys.shape[0] == 2
    assert cache.index_keys is cache.index_position_ids is cache.index_block_keys is None
    assert draft._seed_token is draft._seed_hidden is None
    assert draft._next_position == 19 and cache.offset == 4 and cache.index_block_ratio == 2


def test_adapter_is_local_subclass_and_does_not_mutate_upstream_class(backend):
    original = backend.base.filter_batch
    cls = _qwen4_mtp_drafter_class(backend.base)
    assert issubclass(cls, backend.base)
    assert cls.filter_batch is not original
    assert backend.base.filter_batch is original


def forward_drafter(backend):
    """Actual upstream lifecycle/position methods; only neural math is fake."""
    methods = {name: upstream_method("speculative/drafters/mtp_base.py", "AutoregressiveMTPDraftModel", name)
               for name in ("filter_batch", "_position_ids", "_forward_tokens", "_set_seed_from_hidden",
                            "prefill_from_target_hidden", "draft_block", "set_shared_kv",
                            "accept_verified_tokens", "accept_verified_tokens_batch")}
    forward = upstream_method("speculative/drafters/qwen4_exp_mtp/qwen4_exp_mtp.py",
                              "Qwen4ExpMTPDraftModel", "_forward_hidden")
    forward.__globals__["_create_qwen3_5_attention_mask"] = lambda *args: None
    methods["_forward_hidden"] = forward
    base = type("InstalledQwen4ForwardAndMTPLifecycle", (), methods)
    draft, cache = draft_fixture(_qwen4_mtp_drafter_class(base), backend.cache_type)
    cache.keys = cache.values = cache.index_keys = cache.index_position_ids = cache.index_block_keys = None
    cache.offset = 0
    draft._seed_token = draft._seed_hidden = None
    draft._next_position = 0
    draft._draft_round = 0
    draft._input_embed = lambda tokens: np.zeros((*tokens.shape, 2))
    draft._lm_head_fn = lambda hidden: np.broadcast_to(np.array([1., 2., 3.]), (*hidden.shape[:2], 3))
    draft.fuse_inputs = lambda embeddings, hidden: hidden
    draft.hyper_connection_mixer = lambda hidden: hidden

    def layer(hidden, tokens, *, mask, cache, position_ids):
        # Stand in only for learned projections/attention. Real upstream QSA
        # cache concatenation consumes positions produced by real _position_ids.
        keys = np.zeros((tokens.shape[0], 2, tokens.shape[1], 8))
        cache.keys = keys if cache.keys is None else np.concatenate([cache.keys, keys], axis=2)
        cache.values = cache.keys.copy()
        cache.update_indexer(np.zeros((*tokens.shape, 128)), position_ids)
        cache.offset += tokens.shape[1]
        return hidden

    draft.layers = [layer]
    return draft, cache


@pytest.mark.parametrize("mrope", [False, True], ids=["broadcast-text", "broadcast-mrope"])
def test_actual_prefill_draft_forward_and_acceptance_survive_cohort_compaction(backend, mrope):
    draft, cache = forward_drafter(backend)
    sampler = lambda logits: np.argmax(logits, axis=-1)
    draft.prefill_from_target_hidden(np.ones((4, 3), dtype=np.int32), np.zeros((4, 3, 8)),
                                    np.ones(4, dtype=np.int32), sampler, np.int32, greedy=True)
    assert isinstance(draft._next_position, int) and draft._next_position == 3
    assert cache.index_keys.shape == (4, 3, 128)
    assert cache.index_position_ids.shape == (1, 3), "Real prefill uses shared broadcast positions"
    if mrope:
        cache.index_position_ids = np.broadcast_to(cache.index_position_ids[None], (3, 1, 3))
    shared_positions = cache.index_position_ids
    draft.filter_batch(np.array([3, 1]))
    assert cache.index_position_ids is shared_positions
    assert cache.index_keys.shape == (2, 3, 128)
    # The actual lifecycle deliberately retains its scalar while an existing
    # cache is nonempty, even when the server supplies per-row target positions.
    draft.set_shared_kv({}, kv_offset=3, kv_valid_len=np.array([3, 3]))
    assert draft._next_position == 3
    proposals = draft.draft_block(np.ones(2, dtype=np.int32), np.zeros((2, 1, 8)),
                                 None, 3, sampler, np.int32, greedy=True)
    assert proposals.shape == (2, 2)
    assert cache.index_keys.shape == (2, 4, 128)
    assert cache.index_position_ids.shape == ((3, 1, 4) if mrope else (1, 4))
    draft.accept_verified_tokens_batch(np.zeros((2, 3, 8)), proposals, [1, 1], [[2, 3], [2, 3]],
                                       sampler, np.int32, greedy=True)
    assert draft._next_position == 5
    assert cache.index_keys.shape == (2, 5, 128)

    draft.filter_batch(np.array([1]))
    proposals = draft.draft_block(np.ones(1, dtype=np.int32), np.zeros((1, 1, 8)),
                                 None, 3, sampler, np.int32, greedy=True)
    assert proposals.shape == (1, 2)
    assert cache.keys.shape[0] == cache.index_keys.shape[0] == 1
    assert cache.index_position_ids.shape == ((3, 1, 6) if mrope else (1, 6))
    np.testing.assert_array_equal(cache.index_position_ids,
                                  np.broadcast_to(np.arange(6), cache.index_position_ids.shape))
    assert draft._next_position == 6


@pytest.mark.parametrize("mrope", [False, True])
def test_actual_forward_keeps_genuine_per_row_positions_after_compaction(backend, mrope):
    draft, cache = forward_drafter(backend)
    draft._next_position = np.array([11, 13, 17, 19])
    draft._forward_tokens(np.ones((4, 1), dtype=np.int32), np.zeros((4, 1, 8)), np.int32)
    assert cache.index_position_ids.shape == (4, 1)
    if mrope:
        cache.index_position_ids = np.broadcast_to(cache.index_position_ids[None], (3, 4, 1))
    draft.filter_batch([3, 1])  # Public list input must normalize exactly once.
    draft.draft_block(np.ones(2, dtype=np.int32), np.zeros((2, 1, 8)), None, 3,
                      lambda logits: np.argmax(logits, axis=-1), np.int32, greedy=True)
    positions = np.array([[19, 20, 21], [13, 14, 15]])
    np.testing.assert_array_equal(cache.index_position_ids,
                                  np.broadcast_to(positions[None], (3, 2, 3)) if mrope else positions)
    np.testing.assert_array_equal(draft._next_position, [22, 16])


@pytest.mark.parametrize("cached_mrope", [False, True])
def test_actual_upstream_append_preserves_shared_positions_across_rank_promotion(backend, cached_mrope):
    append = upstream_method("models/qwen4_exp/language.py", None, "_append_indexer_positions")
    cached = np.arange(3)[None]
    current = np.array([[3]])
    if cached_mrope:
        cached = np.broadcast_to(cached[None], (3, 1, 3))
    else:
        current = np.broadcast_to(current[None], (3, 1, 1))
    result = append(cached, current)
    np.testing.assert_array_equal(result, np.broadcast_to(np.arange(4), (3, 1, 4)))
