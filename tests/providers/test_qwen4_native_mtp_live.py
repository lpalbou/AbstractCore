"""Opt-in native Qwen4 MTP proof against an already-downloaded local checkpoint.

Run in a compatible Apple Silicon environment (approximately 76 GB allocation):

    ABSTRACTCORE_RUN_QWEN4_MTP_TESTS=1 pytest -q \
        tests/providers/test_qwen4_native_mtp_live.py

ABSTRACTCORE_QWEN4_MTP_TEST_MODEL may override the pinned local snapshot path.
It must be a complete local directory; this suite never downloads a model.
Ordinary collection neither imports MLX nor loads weights.
"""

import json
import os
from pathlib import Path
import sys

import pytest


REVISION = "2615fc0e976e65c2f3b55daca3a948f1cdc5b9f8"
DEFAULT_MODEL = (
    Path.home() / ".cache/huggingface/hub"
    / "models--Jundot--Qwen3.8-Flash-Next-oQ4e-mtp" / "snapshots" / REVISION
)
pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="session")
def native_qwen4_mtp():
    if os.getenv("ABSTRACTCORE_RUN_QWEN4_MTP_TESTS") != "1":
        pytest.skip("Set ABSTRACTCORE_RUN_QWEN4_MTP_TESTS=1 for native Qwen4 weights")
    model = Path(os.getenv("ABSTRACTCORE_QWEN4_MTP_TEST_MODEL", str(DEFAULT_MODEL))).expanduser()
    index = model / "model.safetensors.index.json"
    if not model.is_dir() or not (model / "config.json").is_file() or not index.is_file():
        pytest.skip(f"Complete local Qwen4 checkpoint required; no download attempted: {model}")
    weights = json.loads(index.read_text())["weight_map"]
    assert weights and any(key.startswith("mtp.") for key in weights), "No indexed embedded MTP head"
    for filename in set(weights.values()):
        assert isinstance(filename, str) and Path(filename).name == filename
        assert (model / filename).is_file(), f"Missing local shard: {filename}"

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setenv("TRANSFORMERS_OFFLINE", "1")
        from abstractcore import create_llm

        llm = create_llm(
            "mlx", model=str(model.resolve()),
            speculation={"mode": "native_mtp", "num_draft_tokens": 3,
                         "require_acceleration": True},
        )
        try:
            yield llm
        finally:
            llm.unload_model(llm.model)
            assert not llm.validate_config()
            assert not any(name == "omlx" or name.startswith("omlx.") for name in sys.modules)


def _generate(llm, prompt, *, depth, **kwargs):
    response = llm.generate(
        prompt, thinking=False, max_output_tokens=64,
        speculation=False if depth is False else {"num_draft_tokens": depth}, **kwargs,
    )
    assert response.finish_reason != "error", response.content
    assert response.content and response.usage["output_tokens"] > 0
    evidence = response.metadata["speculation"]
    assert evidence["used"] is (depth is not False), evidence
    if depth is not False:
        assert evidence["num_draft_tokens"] == depth
        assert evidence["rounds"] > 0 and evidence["drafted_tokens"] > 0
        assert 0 <= evidence["accepted_tokens"] <= evidence["drafted_tokens"]
    return response.content


def test_native_qwen4_greedy_depths_match_same_runtime_baseline(native_qwen4_mtp):
    llm = native_qwen4_mtp
    prompt = "Write a small Python function that returns the Fibonacci sequence up to n."
    baseline = _generate(llm, prompt, depth=False, temperature=0)
    for depth in (1, 3, 5):
        assert _generate(llm, prompt, depth=depth, temperature=0) == baseline


def test_native_qwen4_seeded_depths_are_repeatable(native_qwen4_mtp):
    llm = native_qwen4_mtp
    prompt = "Explain why hash tables sometimes resize in three concise sentences."
    sampling = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.05, seed=813)
    baseline = _generate(llm, prompt, depth=False, **sampling)
    for depth in (1, 3, 5, 3):
        assert _generate(llm, prompt, depth=depth, **sampling) == baseline


def test_native_qwen4_stream_close_restores_following_call(native_qwen4_mtp):
    llm = native_qwen4_mtp
    stream = llm.generate(
        "Explain hash tables in detail.", thinking=False, temperature=0,
        max_output_tokens=64, stream=True, speculation={"num_draft_tokens": 5},
    )
    try:
        chunks = [next(stream) for _ in range(3)]
        assert any(chunk.content for chunk in chunks)
    finally:
        stream.close()
    assert _generate(llm, "What is 17 multiplied by 23? Number only.", depth=1, temperature=0).strip() == "391"
    response = llm.generate("Reply READY", thinking=False, temperature=0, max_output_tokens=16)
    assert "READY" in response.content
    assert response.metadata["speculation"]["num_draft_tokens"] == 3

