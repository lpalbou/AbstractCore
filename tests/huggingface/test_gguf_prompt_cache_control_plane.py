from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import pytest

llama_cpp = pytest.importorskip("llama_cpp")
np = pytest.importorskip("numpy")

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider, PromptCacheUnsupportedError
from abstractcore.providers.huggingface_provider import HuggingFaceProvider, _GGUFPromptCacheValue

from llama_cpp.llama import LlamaState


class _FakeToolHandler:
    supports_prompted = True

    def format_tools_prompt(
        self,
        tools: Optional[List[Dict[str, Any]]],
        *,
        include_tool_list: bool = True,
    ) -> str:
        names: List[str] = []
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            func = tool.get("function") if isinstance(tool.get("function"), dict) else None
            name = ""
            if func:
                name = str(func.get("name") or "").strip()
            if not name:
                name = str(tool.get("name") or "").strip()
            if name:
                names.append(name)
        lines: List[str] = []
        if include_tool_list:
            lines.append("## Tools (session)")
        for name in names:
            lines.append(f"- {name}")
        return "\n".join(lines)


class _FakeLlamaModelMeta:
    def add_bos_token(self) -> bool:
        return True

    def add_eos_token(self) -> bool:
        return True

    def token_cls(self) -> int:
        return -1

    def token_sep(self) -> int:
        return -1


class _FakeLlama:
    def __init__(self, *, chat_format: str = "chatml-function-calling") -> None:
        self.chat_format = chat_format
        self.metadata = {"tokenizer.ggml.add_space_prefix": "true"}
        self._model = _FakeLlamaModelMeta()
        self._tokens: List[int] = []
        self.n_tokens = 0
        self.cache = None
        self.eval_calls: List[List[int]] = []
        self.set_cache_calls: List[Any] = []

    def token_bos(self) -> int:
        return 1

    def token_eos(self) -> int:
        return 2

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> List[int]:
        _ = special
        toks = [int(b) + 3 for b in text]
        if add_bos:
            return [self.token_bos()] + toks
        return toks

    def reset(self) -> None:
        self._tokens = []
        self.n_tokens = 0

    def load_state(self, state: LlamaState) -> None:
        self._tokens = [int(tok) for tok in state.input_ids[: state.n_tokens].tolist()]
        self.n_tokens = len(self._tokens)

    def eval(self, tokens: List[int]) -> None:
        ints = [int(tok) for tok in tokens]
        self.eval_calls.append(list(ints))
        self._tokens.extend(ints)
        self.n_tokens = len(self._tokens)

    def save_state(self) -> LlamaState:
        rows = max(len(self._tokens), 1)
        return LlamaState(
            input_ids=np.asarray(self._tokens, dtype=np.intc).copy(),
            scores=np.zeros((rows, 4), dtype=np.single),
            n_tokens=len(self._tokens),
            llama_state=bytes((tok % 251 for tok in self._tokens)),
            llama_state_size=len(self._tokens),
            seed=0,
        )

    def set_cache(self, cache: Any) -> None:
        self.cache = cache
        self.set_cache_calls.append(cache)


def _new_provider(*, chat_format: str = "chatml-function-calling") -> HuggingFaceProvider:
    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    BaseProvider.__init__(provider, "unsloth/Qwen3.5-2B-GGUF")
    provider.provider = "huggingface"
    provider.model_type = "gguf"
    provider.temperature = 0.2
    provider.tool_handler = _FakeToolHandler()
    provider.llm = _FakeLlama(chat_format=chat_format)
    provider._gguf_prompt_cache_lock = threading.Lock()
    provider._gguf_prompt_cache_default_capacity_bytes = 512 << 20
    provider._gguf_prompt_cache_pending_capacity_bytes = None
    return provider


def test_gguf_prompt_cache_capabilities_are_local_control_plane_for_qwen_chatml() -> None:
    provider = _new_provider(chat_format="chatml-function-calling")

    caps = provider.get_prompt_cache_capabilities()

    assert caps.supported is True
    assert caps.mode == "local_control_plane"
    assert caps.supports_update is True
    assert caps.supports_fork is True
    assert caps.supports_prepare_modules is True


def test_gguf_prompt_cache_capabilities_are_local_control_plane_for_llama3() -> None:
    provider = _new_provider(chat_format="llama-3")

    caps = provider.get_prompt_cache_capabilities()

    assert caps.supported is True
    assert caps.mode == "local_control_plane"
    assert caps.supports_update is True
    assert caps.supports_fork is True
    assert caps.supports_prepare_modules is True


def test_gguf_prompt_cache_capabilities_downshift_for_unsupported_chat_format() -> None:
    provider = _new_provider(chat_format="functionary-v2")

    caps = provider.get_prompt_cache_capabilities()

    assert caps.supported is True
    assert caps.mode == "keyed"
    assert caps.supports_prepare_modules is False

    with pytest.raises(PromptCacheUnsupportedError):
        provider.prompt_cache_prepare_modules(
            namespace="tenant:model",
            modules=[{"module_id": "system", "system_prompt": "SYSTEM"}],
        )


@pytest.mark.parametrize("chat_format", ["chatml-function-calling", "llama-3"])
def test_gguf_prompt_cache_prepare_modules_fork_and_update_reuse_prefix(chat_format: str) -> None:
    provider = _new_provider(chat_format=chat_format)

    prepared = provider.prompt_cache_prepare_modules(
        namespace="tenant:model",
        modules=[
            {"module_id": "system", "system_prompt": "You are helpful."},
            {"module_id": "tools", "tools": [{"type": "function", "function": {"name": "shell"}}]},
        ],
        make_default=False,
    )
    assert prepared["supported"] is True
    prefix_key = prepared["final_cache_key"]
    prefix_state = provider._prompt_cache_store.get(prefix_key)
    assert isinstance(prefix_state, _GGUFPromptCacheValue)

    eval_lengths = [len(call) for call in provider.llm.eval_calls]
    assert len(eval_lengths) == 2
    assert eval_lengths[0] > 0
    assert 0 < eval_lengths[1] < len(prefix_state.prompt_tokens)

    assert provider.prompt_cache_fork(prefix_key, "sess", make_default=False) is True

    provider.llm.eval_calls.clear()
    assert provider.prompt_cache_update("sess", messages=[{"role": "user", "content": "hi"}]) is True

    session_state = provider._prompt_cache_store.get("sess")
    assert isinstance(session_state, _GGUFPromptCacheValue)
    assert session_state.messages[-1]["content"] == "hi"
    assert session_state.prompt_tokens
    assert session_state.prompt_tokens[: len(prefix_state.prompt_tokens)] == prefix_state.prompt_tokens
    assert len(provider.llm.eval_calls) == 1
    assert 0 < len(provider.llm.eval_calls[0]) < len(session_state.prompt_tokens)

    stats = provider.get_prompt_cache_stats()
    assert stats["capabilities"]["mode"] == "local_control_plane"
    assert stats["meta_by_key"]["sess"]["token_count"] == len(session_state.prompt_tokens)


def test_gguf_prompt_cache_tracks_assistant_tool_call_history_in_prompt_text() -> None:
    provider = _new_provider(chat_format="chatml-function-calling")
    assert provider.prompt_cache_set("sess", make_default=False) is True

    assert provider.prompt_cache_update(
        "sess",
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\":\"README.md\"}",
                        },
                    }
                ],
            }
        ],
    ) is True

    session_state = provider._prompt_cache_store.get("sess")
    assert isinstance(session_state, _GGUFPromptCacheValue)
    assert "functions.read_file:" in session_state.prompt_text
    assert "{\"path\":\"README.md\"}" in session_state.prompt_text


def test_generate_gguf_attaches_underlying_cache_object() -> None:
    provider = _new_provider(chat_format="chatml-function-calling")
    assert provider.prompt_cache_set("sess", make_default=False) is True
    cache_value = provider._prompt_cache_store.get("sess")
    assert isinstance(cache_value, _GGUFPromptCacheValue)

    def _fake_single(kwargs: Dict[str, Any]) -> GenerateResponse:
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    provider._single_generate_gguf = _fake_single  # type: ignore[method-assign]

    response = provider._generate_gguf(
        prompt="hello",
        messages=[],
        system_prompt="SYSTEM",
        tools=None,
        media=None,
        stream=False,
        prompt_cache_key="sess",
    )

    assert isinstance(response, GenerateResponse)
    assert provider.llm.set_cache_calls
    assert provider.llm.set_cache_calls[-1] is cache_value.cache
