"""Both HuggingFace lanes must FEED the planner's bloc cut, not re-render.

`BaseProvider.prompt_cache_prepare_modules` hands each module the exact token
ids that make the cache a true token prefix of the prompt `generate()` builds
(`bloc_token_ids`). MLX has honoured that since the bloc work landed; the
transformers and GGUF lanes ignored it until 2026-08-07, and the transformers
lane's `tools is not None` branch forced a full rebuild — so a tools bloc threw
away the system bloc's KV and re-prefilled everything.

Measured on Qwen3-4B-Instruct-2507 (bf16/MPS), 702-token system bloc + 661-token
tools bloc:

    before   cold build of [system, tools] .... 2068 tokens prefilled
             edit ONE tool description ........ 1367 tokens re-prefilled
    after    cold build ....................... 1363 tokens prefilled
             edit ONE tool description ........  665 tokens re-prefilled

No weights are loaded here: the cache backends are exercised directly.
"""

from __future__ import annotations

import threading
import warnings
from typing import Any, Dict, List, Optional

import pytest

np = pytest.importorskip("numpy")
llama_cpp = pytest.importorskip("llama_cpp")

from llama_cpp.llama import LlamaState

from abstractcore.providers.base import BaseProvider
from abstractcore.providers.huggingface_provider import (
    HuggingFaceProvider,
    _GGUFPromptCacheValue,
    _TransformersPromptCacheValue,
)


class _FakeToolHandler:
    supports_prompted = True

    def format_tools_prompt(self, tools, *, include_tool_list: bool = True) -> str:
        names = [str(t.get("name") or "") for t in (tools or []) if isinstance(t, dict)]
        return "## Tools (session)\n" + "\n".join(f"- {n}" for n in names if n)


# --------------------------------------------------------------------------
# transformers lane
# --------------------------------------------------------------------------


def _transformers_provider() -> HuggingFaceProvider:
    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    BaseProvider.__init__(provider, "Qwen/Qwen3-4B-Instruct-2507")
    provider.provider = "huggingface"
    provider.model_type = "transformers"
    provider.tool_handler = _FakeToolHandler()
    return provider


def test_transformers_feeds_the_planned_fragment_and_keeps_the_predecessor_bloc():
    provider = _transformers_provider()
    fed: List[List[int]] = []

    def _record(state, token_ids):
        fed.append(list(token_ids))
        state.prompt_tokens = tuple(state.prompt_tokens) + tuple(int(t) for t in token_ids)
        return True

    provider._transformers_prefill_cache = _record  # type: ignore[assignment]

    # A warm cache already holding the system bloc.
    state = _TransformersPromptCacheValue(cache=object(), prompt_tokens=(11, 12, 13))
    state.system_prompt_parts.append("SYS")
    sentinel_cache = state.cache

    planned = [90, 91, 92, 93]
    assert provider._prompt_cache_backend_append(
        state,
        tools=[{"name": "read_file"}],
        bloc_token_ids=planned,
    ) is True

    assert fed == [planned], "the tools bloc must feed exactly the planned ids"
    assert state.prompt_tokens == (11, 12, 13, 90, 91, 92, 93), "the system bloc's tokens were discarded"
    assert state.cache is sentinel_cache, "the cache was rebuilt instead of extended"
    # The logical state still advances, so a later UNPLANNED append renders the
    # correct cumulative text.
    assert state.tools == [{"name": "read_file"}]


def test_transformers_without_a_plan_still_rebuilds():
    """The legacy path is unchanged when no plan is supplied (older callers)."""
    provider = _transformers_provider()
    fed: List[List[int]] = []

    def _record(state, token_ids):
        fed.append(list(token_ids))
        state.prompt_tokens = tuple(state.prompt_tokens) + tuple(int(t) for t in token_ids)
        return True

    provider._transformers_prefill_cache = _record  # type: ignore[assignment]
    provider._transformers_empty_native_cache = lambda: "fresh"  # type: ignore[assignment]
    provider._transformers_build_prompt_fragment = lambda **kw: "REBUILT"  # type: ignore[assignment]
    provider._transformers_tokenize_fragment = lambda text, add_bos_if_empty: [7, 7, 7]  # type: ignore[assignment]

    state = _TransformersPromptCacheValue(cache=object(), prompt_tokens=(11, 12, 13))
    assert provider._prompt_cache_backend_append(state, tools=[{"name": "read_file"}]) is True

    assert fed == [[7, 7, 7]]
    assert state.cache == "fresh", "the no-plan path must still rebuild"


# --------------------------------------------------------------------------
# GGUF lane
# --------------------------------------------------------------------------


class _FakeLlamaModelMeta:
    def add_bos_token(self) -> bool:
        return False

    def add_eos_token(self) -> bool:
        return False

    def token_cls(self) -> int:
        return -1

    def token_sep(self) -> int:
        return -1

    def token_get_text(self, token_id: int) -> str:
        return ""


class _FakeLlama:
    """Byte-level tokenizer with an EXACT detokenizer — the round-trip the
    GGUF planned path verifies before it trusts a text/token pair."""

    def __init__(self) -> None:
        self.chat_format = "chatml-function-calling"
        self.metadata: Dict[str, str] = {}
        self._model = _FakeLlamaModelMeta()
        self._tokens: List[int] = []
        self.n_tokens = 0
        self.cache = None
        self.eval_calls: List[List[int]] = []

    def token_bos(self) -> int:
        return 1

    def token_eos(self) -> int:
        return 2

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> List[int]:
        toks = [int(b) + 3 for b in text]
        return ([self.token_bos()] + toks) if add_bos else toks

    def detokenize(self, tokens: List[int], prev_tokens: Optional[List[int]] = None, special: bool = False) -> bytes:
        return bytes(int(t) - 3 for t in tokens if int(t) >= 3)

    def reset(self) -> None:
        self._tokens = []
        self.n_tokens = 0

    def load_state(self, state: LlamaState) -> None:
        self._tokens = [int(t) for t in state.input_ids[: state.n_tokens].tolist()]
        self.n_tokens = len(self._tokens)

    def eval(self, tokens: List[int]) -> None:
        ints = [int(t) for t in tokens]
        self.eval_calls.append(list(ints))
        self._tokens.extend(ints)
        self.n_tokens = len(self._tokens)

    def save_state(self) -> LlamaState:
        rows = max(len(self._tokens), 1)
        return LlamaState(
            input_ids=np.asarray(self._tokens, dtype=np.intc).copy(),
            scores=np.zeros((rows, 4), dtype=np.single),
            n_tokens=len(self._tokens),
            llama_state=bytes((t % 251 for t in self._tokens)),
            llama_state_size=len(self._tokens),
            seed=0,
        )

    def set_cache(self, cache: Any) -> None:
        self.cache = cache


def _gguf_provider() -> HuggingFaceProvider:
    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    BaseProvider.__init__(provider, "unsloth/Qwen3-4B-Instruct-2507-GGUF")
    provider.provider = "huggingface"
    provider.model_type = "gguf"
    provider.tool_handler = _FakeToolHandler()
    provider.llm = _FakeLlama()
    provider._gguf_prompt_cache_lock = threading.Lock()
    provider._gguf_prompt_cache_default_capacity_bytes = 512 << 20
    provider._gguf_prompt_cache_pending_capacity_bytes = None
    return provider


def test_gguf_feeds_the_planned_fragment_with_text_and_tokens_in_sync():
    from llama_cpp.llama_cache import LlamaRAMCache

    provider = _gguf_provider()
    prev_text = "SYSTEM"
    prev_tokens = tuple(provider.llm.tokenize(prev_text.encode("utf-8"), add_bos=False))
    state = _GGUFPromptCacheValue(
        cache=LlamaRAMCache(capacity_bytes=1 << 20),
        capacity_bytes=1 << 20,
        prompt_text=prev_text,
        prompt_tokens=prev_tokens,
    )
    state.system_prompt_parts.append("SYSTEM")

    planned = list(provider.llm.tokenize(b"TOOLS", add_bos=False))
    assert provider._prompt_cache_backend_append(
        state,
        tools=[{"name": "read_file"}],
        bloc_token_ids=planned,
        bloc_stable_text="SYSTEMTOOLS",
    ) is True

    assert state.prompt_tokens == prev_tokens + tuple(planned)
    # The recorded text must tokenize back to EXACTLY the recorded ids —
    # `_gguf_compose_cached_prompt_tokens` concatenates it with the live suffix,
    # so a one-token disagreement corrupts every warm turn after it.
    assert tuple(provider._gguf_tokenize_rendered_prompt(state.prompt_text)) == state.prompt_tokens
    # Only the delta was evaluated; the predecessor bloc was not re-fed.
    assert sum(len(c) for c in provider.llm.eval_calls) <= len(prev_tokens) + len(planned)


def test_gguf_falls_back_loudly_when_the_text_cannot_be_verified():
    """No silent text/token mismatch: an unverifiable pair rebuilds, out loud."""
    provider = _gguf_provider()
    provider.llm.detokenize = lambda *a, **k: b"\xff\xfe"  # type: ignore[assignment]
    provider._gguf_render_prompt_tokens = lambda **kw: ("REBUILT", (4, 5, 6))  # type: ignore[assignment]
    provider._gguf_prefill_prompt_cache = lambda *a, **k: True  # type: ignore[assignment]

    state = _GGUFPromptCacheValue(cache=None, capacity_bytes=1 << 20, prompt_text="S", prompt_tokens=(86,))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok = provider._prompt_cache_backend_append(
            state,
            tools=[{"name": "read_file"}],
            bloc_token_ids=[9, 9],
            bloc_stable_text="NOT THE PLANNED TEXT",
        )

    assert ok is True
    assert any("could not verify the planned fragment" in str(w.message) for w in caught)
    assert state.prompt_text == "REBUILT" and state.prompt_tokens == (4, 5, 6)
