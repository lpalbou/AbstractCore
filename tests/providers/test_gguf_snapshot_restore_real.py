"""Real weights: an answer after a restore from a COMPACTED snapshot (one zero scores row) is token
for token the cold answer, greedy. Two restores: the whole prompt (the zero-feed guard rolls back one
token and decodes it) and a prefix of a grown prompt (the tail is decoded). Opt-in:
`ABSTRACTCORE_RUN_GGUF_TESTS=1`; model `ABSTRACTCORE_GGUF_TEST_MODEL` (default
`unsloth/Qwen3.5-4B-GGUF`, a 248k vocabulary: 508 MB of scores per uncompacted snapshot), from the
local cache."""
from __future__ import annotations

import os

import pytest

if os.getenv("ABSTRACTCORE_RUN_GGUF_TESTS") != "1":
    pytest.skip("real-weights GGUF test; set ABSTRACTCORE_RUN_GGUF_TESTS=1", allow_module_level=True)

llama_cpp = pytest.importorskip("llama_cpp")

from abstractcore import create_llm  # noqa: E402

pytestmark = pytest.mark.real_home("loads a cached GGUF read-only from the operator's Hugging Face hub cache")

MODEL = os.getenv("ABSTRACTCORE_GGUF_TEST_MODEL", "unsloth/Qwen3.5-4B-GGUF")
OPTS = dict(max_tokens=4096, max_output_tokens=24, temperature=0.0, seed=7, n_gpu_layers=int(os.getenv("ABSTRACTCORE_GGUF_TEST_GPU_LAYERS", "-1")))
SYSTEM = "You are a careful assistant. " + " ".join(f"Fact {i}: the code for item {i} is {i * 7919 % 1000}." for i in range(120))
Q1 = "What is the code for item 42? Answer with the number only, then the code for item 7."
Q2 = "Now the code for item 99 and for item 3, numbers only."


def _llm():
    hub = os.path.join(os.environ["ABSTRACT_TEST_REAL_HOME"], ".cache", "huggingface", "hub")
    os.environ["HF_HUB_CACHE"] = hub  # restored by the fixture below
    try:
        return create_llm("huggingface", model=MODEL, **OPTS)
    except Exception as exc:  # noqa: BLE001 - not cached / no engine on this machine
        pytest.skip(f"{MODEL} unavailable: {type(exc).__name__}: {exc}")


@pytest.fixture(autouse=True)
def _restore_hub_env(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", os.environ.get("HF_HUB_CACHE", ""))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # read the cache, never download
    yield


def test_a_restore_from_a_compacted_snapshot_answers_like_a_cold_prefill(monkeypatch):
    loads = []
    real_load = llama_cpp.Llama.load_state

    def _counted(self, state):
        loads.append(tuple(state.scores.shape))
        return real_load(self, state)

    monkeypatch.setattr(llama_cpp.Llama, "load_state", _counted)
    cold = _llm()
    cold_1 = cold.generate(Q1, system_prompt=SYSTEM).content
    cold_2 = cold.generate(Q2, system_prompt=SYSTEM, messages=[{"role": "user", "content": Q1},
                                                              {"role": "assistant", "content": cold_1}]).content
    llm = _llm()
    llm.prompt_cache_set("a", make_default=False)
    first = llm.generate(Q1, system_prompt=SYSTEM, prompt_cache_key="a").content
    vocab = int(llm.llm.n_vocab())

    def _elsewhere():
        # Another prompt takes the live context, so the next call on "a" must load its snapshot.
        llm.prompt_cache_set("b", make_default=False)
        llm.generate("Say hi.", system_prompt="Be terse.", prompt_cache_key="b")

    _elsewhere()
    loads.clear()
    again = llm.generate(Q1, system_prompt=SYSTEM, prompt_cache_key="a").content
    assert loads and all(shape == (1, vocab) for shape in loads), f"no compacted restore: {loads}"
    assert first == cold_1 and again == cold_1, (cold_1, first, again)

    _elsewhere()
    loads.clear()
    grown = llm.generate(Q2, system_prompt=SYSTEM, prompt_cache_key="a",
                         messages=[{"role": "user", "content": Q1}, {"role": "assistant", "content": first}]).content
    assert loads and all(shape == (1, vocab) for shape in loads), f"no compacted restore: {loads}"
    assert grown == cold_2, (cold_2, grown)
