"""GGUF weight size must cover the WHOLE quant, not just the first shard.

A split GGUF is loaded by handing llama.cpp `<stem>-00001-of-000NN.gguf`; it
opens the siblings itself. `Llama.model_path` therefore names one shard, and
`stat().st_size` on it reports a fraction of the model. Live regression that
motivated this: `unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q3_K_XL` reported
10,946,624 B for a 89,986,353,824 B three-shard set — off by 8,220x — and that
number was promoted into every UI as the model's memory footprint.

Sparse files keep the fixtures free: `truncate` allocates no blocks.
"""

import os

import pytest

from abstractcore.providers.huggingface_provider import _gguf_total_weight_bytes


def _sparse(path, size: int) -> None:
    """Create a sparse file of `size` bytes (no disk blocks consumed)."""
    with open(path, "wb") as handle:
        handle.truncate(size)


def test_single_file_gguf_is_its_own_size(tmp_path):
    target = tmp_path / "Model-Q4_K_M.gguf"
    _sparse(target, 1_234_567)

    assert _gguf_total_weight_bytes(target) == 1_234_567


def test_sharded_gguf_sums_every_shard(tmp_path):
    """The live shape: first shard tiny, payload in the later shards."""
    sizes = [10_946_624, 49_983_253_824, 39_992_153_376]
    for index, size in enumerate(sizes, start=1):
        _sparse(tmp_path / f"Qwen3.8-Flash-Next-UD-Q3_K_XL-{index:05d}-of-00003.gguf", size)

    first = tmp_path / "Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf"

    # The defect: the first shard alone.
    assert first.stat().st_size == 10_946_624
    # The fix: the whole set.
    assert _gguf_total_weight_bytes(first) == 89_986_353_824


def test_sharded_gguf_is_stable_whichever_shard_is_named(tmp_path):
    for index in range(1, 4):
        _sparse(tmp_path / f"M-UD-Q3_K_XL-{index:05d}-of-00003.gguf", 1000 * index)

    total = 1000 + 2000 + 3000
    for index in range(1, 4):
        shard = tmp_path / f"M-UD-Q3_K_XL-{index:05d}-of-00003.gguf"
        assert _gguf_total_weight_bytes(shard) == total


def test_missing_shard_reports_what_is_there_and_does_not_fake_completeness(tmp_path):
    """A partially-fetched set under-reports; it never invents the absent shard."""
    _sparse(tmp_path / "M-UD-Q3_K_XL-00001-of-00003.gguf", 1_000)
    _sparse(tmp_path / "M-UD-Q3_K_XL-00003-of-00003.gguf", 3_000)
    # shard 2 deliberately absent

    first = tmp_path / "M-UD-Q3_K_XL-00001-of-00003.gguf"
    assert _gguf_total_weight_bytes(first) == 4_000


def test_other_quants_in_the_same_directory_are_not_mixed_in(tmp_path):
    """Sibling scan keys on the exact stem: a second quant must not leak in."""
    _sparse(tmp_path / "M-UD-Q3_K_XL-00001-of-00002.gguf", 100)
    _sparse(tmp_path / "M-UD-Q3_K_XL-00002-of-00002.gguf", 200)
    _sparse(tmp_path / "M-UD-IQ4_XS-00001-of-00002.gguf", 9_000)
    _sparse(tmp_path / "M-UD-IQ4_XS-00002-of-00002.gguf", 9_000)

    assert _gguf_total_weight_bytes(tmp_path / "M-UD-Q3_K_XL-00001-of-00002.gguf") == 300
    assert _gguf_total_weight_bytes(tmp_path / "M-UD-IQ4_XS-00001-of-00002.gguf") == 18_000


def test_shards_declaring_a_different_of_count_are_excluded(tmp_path):
    """A stale re-split left behind must not be double-counted into the total."""
    _sparse(tmp_path / "M-00001-of-00002.gguf", 100)
    _sparse(tmp_path / "M-00002-of-00002.gguf", 200)
    _sparse(tmp_path / "M-00001-of-00004.gguf", 50_000)

    assert _gguf_total_weight_bytes(tmp_path / "M-00001-of-00002.gguf") == 300


def test_non_gguf_siblings_are_ignored(tmp_path):
    _sparse(tmp_path / "M-00001-of-00002.gguf", 100)
    _sparse(tmp_path / "M-00002-of-00002.gguf", 200)
    _sparse(tmp_path / "M-00001-of-00002.gguf.tmp", 999_999)
    _sparse(tmp_path / "mmproj-F16.gguf", 777_777)

    assert _gguf_total_weight_bytes(tmp_path / "M-00001-of-00002.gguf") == 300


def test_symlinked_shards_are_measured_through_to_the_blob(tmp_path):
    """The HuggingFace cache links snapshot names at `blobs/`; follow them."""
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshot"
    blobs.mkdir()
    snapshot.mkdir()
    for index, size in enumerate((1_000, 2_000), start=1):
        blob = blobs / f"blob{index}"
        _sparse(blob, size)
        os.symlink(blob, snapshot / f"M-{index:05d}-of-00002.gguf")

    assert _gguf_total_weight_bytes(snapshot / "M-00001-of-00002.gguf") == 3_000


def test_missing_path_is_none_not_a_raise(tmp_path):
    assert _gguf_total_weight_bytes(tmp_path / "absent.gguf") is None


def test_accepts_str_paths(tmp_path):
    _sparse(tmp_path / "M-00001-of-00002.gguf", 100)
    _sparse(tmp_path / "M-00002-of-00002.gguf", 200)

    assert _gguf_total_weight_bytes(str(tmp_path / "M-00001-of-00002.gguf")) == 300


def test_provider_est_weights_bytes_uses_the_sharded_total(tmp_path):
    """The provider hook, not just the helper, must report the whole quant."""
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    for index, size in enumerate((10_946_624, 49_983_253_824, 39_992_153_376), start=1):
        _sparse(tmp_path / f"Q-UD-Q3_K_XL-{index:05d}-of-00003.gguf", size)

    class _FakeLlama:
        model_path = str(tmp_path / "Q-UD-Q3_K_XL-00001-of-00003.gguf")

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider.llm = _FakeLlama()

    assert provider._est_weights_bytes() == 89_986_353_824
