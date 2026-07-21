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

    def token_get_text(self, token_id: int) -> str:
        if int(token_id) == 1:
            return "<bos>"
        if int(token_id) == 2:
            return "<turn|>"
        return ""


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


_GEMMA_TURN_TEMPLATE = """{{ bos_token }}{% for message in messages %}<|turn>{{ 'model' if message['role'] == 'assistant' else message['role'] }}
{{ message['content'] }}<turn|>
{% endfor %}{% if add_generation_prompt %}<|turn>model
{% endif %}"""


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


def _ornith_template() -> str:
    from pathlib import Path

    return (Path(__file__).resolve().parent.parent / "fixtures" / "ornith_chat_template.jinja").read_text(
        encoding="utf-8"
    )


def test_gguf_embedded_chatml_template_reaches_local_control_plane() -> None:
    """0821: an embedded ChatML Jinja template (Ornith 1.0 — a Qwen3.5
    post-train whose NAME lacks 'qwen' and whose chat_format reports the
    generic 'chat_template.default') must reach the control plane by template
    CONTENT, rendered through the model's OWN template."""
    provider = _new_provider(chat_format="chat_template.default")
    provider.model = "deepreinforce-ai/Ornith-1.0-35B-GGUF"
    provider.architecture = "qwen3_5"
    provider.architecture_config = {"message_format": "im_start_end"}
    provider.llm.metadata["tokenizer.chat_template"] = _ornith_template()

    assert provider._gguf_prompt_cache_control_plane_chat_format() == "llama-cpp-chat-template"
    caps = provider.get_prompt_cache_capabilities()
    assert caps.supported is True
    assert caps.mode == "local_control_plane"

    prompt_text, prompt_tokens = provider._gguf_render_prompt_tokens(
        messages=[
            {"role": "system", "content": "SYS-FACT"},
            {"role": "user", "content": "USER-TURN"},
        ],
        add_generation_prompt=True,
    )
    assert "<|im_start|>system" in prompt_text and "SYS-FACT" in prompt_text
    assert "<|im_start|>user" in prompt_text and "USER-TURN" in prompt_text
    # The model's OWN template appends its think-opening after the assistant
    # header — the fidelity detail the plain-ChatML renderer would have
    # dropped, and why the embedded-template renderer is the right lane.
    assert "<|im_start|>assistant" in prompt_text
    assert prompt_text.rstrip().endswith(("<|im_start|>assistant", "<think>"))
    assert prompt_tokens, "control-plane tokenization must produce tokens"

    # Growing-prefix property: the two-turn render extends the one-turn render
    # byte-for-byte (what the snapshot lane rides on).
    longer_text, _ = provider._gguf_render_prompt_tokens(
        messages=[
            {"role": "system", "content": "SYS-FACT"},
            {"role": "user", "content": "USER-TURN"},
            {"role": "assistant", "content": "ANSWER-ONE"},
            {"role": "user", "content": "USER-TWO"},
        ],
        add_generation_prompt=True,
    )
    shared = prompt_text[: prompt_text.rfind("<|im_start|>assistant")]
    assert longer_text.startswith(shared)


def test_gguf_embedded_non_chatml_template_stays_keyed() -> None:
    """A generic embedded template WITHOUT ChatML markers keeps the honest
    keyed fallback (never claim a renderer we cannot serve)."""
    provider = _new_provider(chat_format="chat_template.default")
    provider.architecture = "some_arch"
    provider.architecture_config = {"message_format": "im_start_end"}
    provider.llm.metadata["tokenizer.chat_template"] = (
        "{% for m in messages %}[{{ m['role'] }}]: {{ m['content'] }}\n{% endfor %}"
    )

    assert provider._gguf_prompt_cache_control_plane_chat_format() == ""
    assert provider.get_prompt_cache_capabilities().mode == "keyed"


def test_gguf_embedded_chatml_template_that_cannot_render_stays_keyed() -> None:
    """ChatML markers present but the template REFUSES to render (Jinja
    raise_exception on plain string content): the probe must catch it and
    keep keyed — claiming the control plane would crash every cached turn."""
    provider = _new_provider(chat_format="chat_template.default")
    provider.architecture = "qwen3_5"
    provider.architecture_config = {"message_format": "im_start_end"}
    provider.llm.metadata["tokenizer.chat_template"] = (
        "{{ raise_exception('nope') }}<|im_start|>{{ messages }}<|im_end|>"
    )

    assert provider._gguf_prompt_cache_control_plane_chat_format() == ""
    assert provider.get_prompt_cache_capabilities().mode == "keyed"


def test_gguf_embedded_template_probe_is_cached_per_template() -> None:
    provider = _new_provider(chat_format="chat_template.default")
    provider.architecture = "qwen3_5"
    provider.architecture_config = {"message_format": "im_start_end"}
    provider.llm.metadata["tokenizer.chat_template"] = _ornith_template()

    calls = {"n": 0}
    original = provider._gguf_render_llama_cpp_chat_template_prompt

    def _counting(**kwargs):
        calls["n"] += 1
        return original(**kwargs)

    provider._gguf_render_llama_cpp_chat_template_prompt = _counting  # type: ignore[method-assign]
    assert provider._gguf_prompt_cache_control_plane_chat_format() == "llama-cpp-chat-template"
    assert provider._gguf_prompt_cache_control_plane_chat_format() == "llama-cpp-chat-template"
    assert calls["n"] == 1, "probe render must run once per template identity"


def test_gguf_probe_rejects_marker_mentioning_non_chatml_templates() -> None:
    """Adversary F2 (2026-07-19): templates that MENTION the ChatML markers
    without being ChatML-shaped must not be admitted. Two reproduced shapes:
    a llama-2-wire template whose preamble mentions the markers, and one that
    ChatML-wraps only the SYSTEM turn."""
    provider = _new_provider(chat_format="chat_template.default")
    provider.architecture = "some_arch"
    provider.architecture_config = {"message_format": "other"}

    mentioning = (
        "{{ 'This model does not use <|im_start|> or <|im_end|> markers.\\n' }}"
        "{% for m in messages %}[INST] {{ m['content'] }} [/INST]{% endfor %}"
    )
    provider.llm.metadata["tokenizer.chat_template"] = mentioning
    assert provider._gguf_prompt_cache_control_plane_chat_format() == ""

    system_only = (
        "{% for m in messages %}{% if m['role'] == 'system' %}"
        "<|im_start|>system\n{{ m['content'] }}<|im_end|>\n"
        "{% else %}[{{ m['role'] }}] {{ m['content'] }}\n{% endif %}{% endfor %}"
    )
    provider.llm.metadata["tokenizer.chat_template"] = system_only
    provider._gguf_embedded_template_probe_cache = {}
    assert provider._gguf_prompt_cache_control_plane_chat_format() == ""


def test_gguf_prefix_state_reuses_what_the_state_holds_never_the_key() -> None:
    """Adversary F1 (2026-07-19, the silently-wrong-cache class, reproduced):
    fallback-lane writers (llama.cpp's own save after create_chat_completion)
    key states by prompt+completion while the last sampled token was never
    eval'd — the state HOLDS len(key)-1 tokens. A reader trusting the key
    skipped eval'ing that token: one mid-prompt token missing from KV, every
    later position shifted, wrong output, zero errors. The reader must reuse
    exactly what the state holds and eval the remainder."""
    provider = _new_provider(chat_format="chatml-function-calling")
    assert provider.prompt_cache_set("sess", make_default=False) is True
    cache_value = provider._prompt_cache_store.get("sess")
    cache_obj = cache_value.cache

    prompt_tokens = tuple(range(100, 120))
    # Simulate the fallback-lane writer: state holds one token FEWER than its key.
    provider.llm.reset()
    provider.llm.eval(list(prompt_tokens[:-1]))
    short_state = provider.llm.save_state()
    cache_obj[prompt_tokens] = short_state

    prefix_len, prefix_state = provider._gguf_prompt_cache_prefix_state(cache_obj, prompt_tokens)
    assert prefix_len == len(prompt_tokens) - 1, "reuse must stop at what the state HOLDS"
    assert prefix_state is short_state

    provider.llm.reset()
    provider.llm.eval_calls.clear()
    assert provider._gguf_prefill_prompt_cache(cache_obj, prompt_tokens, save_state=False, set_cache=False)
    # The missing token MUST be eval'd — this is the token the old reader dropped.
    assert provider.llm.eval_calls, "the held/key gap must be eval'd"
    assert provider.llm.eval_calls[-1] == [prompt_tokens[-1]]
    assert provider.llm.n_tokens == len(prompt_tokens)


def test_gguf_prefix_state_refuses_state_disagreeing_with_its_key() -> None:
    """A state whose held tokens DISAGREE with its map key is foreign/corrupt:
    refuse it entirely (never splice mismatched KV)."""
    provider = _new_provider(chat_format="chatml-function-calling")
    assert provider.prompt_cache_set("sess", make_default=False) is True
    cache_obj = provider._prompt_cache_store.get("sess").cache

    prompt_tokens = tuple(range(200, 210))
    provider.llm.reset()
    provider.llm.eval([999] * (len(prompt_tokens) - 1))  # different tokens than the key claims
    foreign_state = provider.llm.save_state()
    cache_obj[prompt_tokens] = foreign_state

    prefix_len, prefix_state = provider._gguf_prompt_cache_prefix_state(cache_obj, prompt_tokens)
    assert prefix_len == 0
    assert prefix_state is None


def test_gguf_control_plane_stream_render_failure_degrades_not_raises() -> None:
    """Adversary F3 (2026-07-19): a template that REFUSES a conversation shape
    (Ornith raises on a mid-history system message) used to escape as a raw
    ValueError at the consumer's first next() on the streaming lane. It must
    degrade to a finish_reason='error' chunk like the non-stream path."""
    provider = _new_provider(chat_format="chat_template.default")
    provider.model = "deepreinforce-ai/Ornith-1.0-35B-GGUF"
    provider.architecture = "qwen3_5"
    provider.architecture_config = {"message_format": "im_start_end"}
    provider.llm.metadata["tokenizer.chat_template"] = _ornith_template()
    assert provider._gguf_prompt_cache_control_plane_chat_format() == "llama-cpp-chat-template"

    assert provider.prompt_cache_set("sess", make_default=False) is True
    cache_value = provider._prompt_cache_store.get("sess")

    stream = provider._gguf_control_plane_stream_generate(
        chat_messages=[
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "MID-HISTORY SYSTEM"},  # the template refuses this
            {"role": "user", "content": "again"},
        ],
        cache_obj=cache_value.cache,
        max_output_tokens=16,
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        min_p=0.05,
        typical_p=1.0,
        repeat_penalty=1.1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        tfs_z=1.0,
        mirostat_mode=0,
        mirostat_tau=5.0,
        mirostat_eta=0.1,
        seed=None,
        enable_thinking=None,
        cache_state=cache_value,
    )
    first = next(stream)
    assert isinstance(first, GenerateResponse)
    assert first.finish_reason == "error"
    assert "System message must be at the beginning" in str(first.content)


def test_gguf_prompt_cache_capabilities_are_local_control_plane_for_gemma4_template() -> None:
    provider = _new_provider(chat_format="chat_template.default")
    provider.architecture = "gemma4"
    provider.architecture_config = {
        "message_format": "gemma_turn",
        "assistant_suffix": "<turn|>\n",
    }
    provider.llm.metadata["tokenizer.chat_template"] = _GEMMA_TURN_TEMPLATE

    caps = provider.get_prompt_cache_capabilities()

    assert caps.supported is True
    assert caps.mode == "local_control_plane"
    assert provider._gguf_prompt_cache_control_plane_chat_format() == "llama-cpp-chat-template"

    prompt_text, prompt_tokens = provider._gguf_render_prompt_tokens(
        messages=[{"role": "user", "content": "FILEBOX"}],
        add_generation_prompt=True,
    )
    assert prompt_text.startswith("<bos><|turn>user\nFILEBOX<turn|>")
    assert prompt_text.endswith("<|turn>model\n")
    assert prompt_tokens


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
    # Single-system-turn fix (2026-07-15): tools now MERGE into the system
    # message instead of opening a second consecutive system block, so the
    # system-only module render is no longer a token-prefix of system+tools
    # (the closing tag precedes the tool text). The tools-module append is a
    # one-time full re-prefill of the merged prompt; session-time reuse
    # (fork + message appends, asserted below) is what must stay incremental.
    assert eval_lengths[1] == len(prefix_state.prompt_tokens)
    final_messages = provider._gguf_build_chat_messages(
        system_prompt="You are helpful.",
        tools=[{"type": "function", "function": {"name": "shell"}}],
    )
    assert [m["role"] for m in final_messages] == ["system"]

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


def test_gguf_qwen_thinking_off_control_plane_marker_is_generation_prompt() -> None:
    provider = _new_provider(chat_format="chatml-function-calling")

    prompt, messages, system_prompt, kwargs, thinking_meta = provider._apply_thinking_request(
        thinking="off",
        prompt="What is the answer?",
        messages=[{"role": "user", "content": "Cached memory bloc."}],
        system_prompt=None,
        kwargs={},
    )

    assert prompt == "What is the answer?"
    assert system_prompt is None
    assert kwargs["_acore_gguf_enable_thinking"] is False
    assert thinking_meta["thinking_effective"] == "off"
    assert messages == [{"role": "user", "content": "Cached memory bloc."}]

    chat_messages = provider._gguf_build_chat_messages(messages=messages, user_message_content=prompt)
    prompt_text, _ = provider._gguf_render_prompt_tokens(
        messages=chat_messages,
        add_generation_prompt=True,
        enable_thinking=kwargs["_acore_gguf_enable_thinking"],
    )

    assert prompt_text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "<think>\n\n</think>\n\n<|im_end|>\n<|im_start|>assistant" not in prompt_text


def test_gguf_cached_generation_prompt_extends_loaded_bloc_prefix() -> None:
    provider = _new_provider(chat_format="chatml-function-calling")
    assert provider.prompt_cache_set("sess", make_default=False) is True
    assert provider.prompt_cache_update("sess", messages=[{"role": "user", "content": "FILEBOX"}]) is True
    cache_state = provider._prompt_cache_store.get("sess")
    assert isinstance(cache_state, _GGUFPromptCacheValue)

    live_messages = provider._gguf_build_chat_messages(user_message_content="QUESTION")
    live_text, live_tokens = provider._gguf_render_prompt_tokens(
        messages=live_messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    composed_text, composed_tokens, meta = provider._gguf_compose_cached_prompt_tokens(
        cache_state=cache_state,
        live_prompt_text=live_text,
        live_prompt_tokens=live_tokens,
    )

    assert meta["prompt_cache_prefix_source"] == "loaded_cache"
    assert meta["prompt_cache_composed"] is True
    assert composed_tokens[: len(cache_state.prompt_tokens)] == cache_state.prompt_tokens
    assert len(composed_tokens) > len(cache_state.prompt_tokens)
    assert composed_text.startswith(cache_state.prompt_text)
    assert "FILEBOX" in composed_text
    assert "QUESTION" in composed_text


def test_gguf_gemma4_cached_generation_prompt_strips_duplicate_template_bos() -> None:
    provider = _new_provider(chat_format="chat_template.default")
    provider.architecture = "gemma4"
    provider.architecture_config = {
        "message_format": "gemma_turn",
        "assistant_suffix": "<turn|>\n",
    }
    provider.llm.metadata["tokenizer.chat_template"] = _GEMMA_TURN_TEMPLATE

    assert provider.prompt_cache_set("sess", make_default=False) is True
    assert provider.prompt_cache_update("sess", messages=[{"role": "user", "content": "FILEBOX"}]) is True
    cache_state = provider._prompt_cache_store.get("sess")
    assert isinstance(cache_state, _GGUFPromptCacheValue)

    live_messages = provider._gguf_build_chat_messages(user_message_content="QUESTION")
    live_text, live_tokens = provider._gguf_render_prompt_tokens(
        messages=live_messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    composed_text, composed_tokens, meta = provider._gguf_compose_cached_prompt_tokens(
        cache_state=cache_state,
        live_prompt_text=live_text,
        live_prompt_tokens=live_tokens,
    )

    assert meta["prompt_cache_prefix_source"] == "loaded_cache"
    assert meta["prompt_cache_composed"] is True
    assert composed_tokens[: len(cache_state.prompt_tokens)] == cache_state.prompt_tokens
    assert composed_text.count("<bos>") == 1
    assert composed_text.startswith(cache_state.prompt_text)
    assert "FILEBOX" in composed_text
    assert "QUESTION" in composed_text


def test_generate_gguf_control_plane_receives_thinking_flag() -> None:
    provider = _new_provider(chat_format="chatml-function-calling")
    assert provider.prompt_cache_set("sess", make_default=False) is True

    captured: Dict[str, Any] = {}

    def _fake_control_plane_generate(**kwargs: Any) -> GenerateResponse:
        captured.update(kwargs)
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    provider._gguf_control_plane_generate = _fake_control_plane_generate  # type: ignore[method-assign]

    response = provider._generate_gguf(
        prompt="What is the answer?",
        messages=[{"role": "user", "content": "Cached memory bloc."}],
        system_prompt=None,
        tools=None,
        media=None,
        stream=False,
        prompt_cache_key="sess",
        _acore_gguf_enable_thinking=False,
    )

    assert isinstance(response, GenerateResponse)
    assert captured["enable_thinking"] is False


@pytest.mark.parametrize("chat_format", ["chatml-function-calling", "llama-3"])
def test_gguf_system_plus_tools_render_one_system_message(chat_format: str) -> None:
    """Regression pin (2026-07-15): system_prompt + tools must produce ONE
    system message containing both, never two consecutive system blocks.

    Chat templates (ChatML/Qwen, Gemma, Llama-3) are trained on exactly one
    system turn; the control-plane lane used to insert the tool prompt as a
    SECOND system message, which is out-of-distribution and degrades
    tool-calling (live find on Ornith-1.0-35B GGUF through a ReAct loop).
    """
    provider = _new_provider(chat_format=chat_format)
    tools = [
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "list_files"}},
    ]

    msgs = provider._gguf_build_chat_messages(
        system_prompt="You are a ReAct agent.",
        tools=tools,
        user_message_content="hi",
    )

    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "You are a ReAct agent." in msgs[0]["content"]
    assert "## Tools (session)" in msgs[0]["content"]
    assert "read_file" in msgs[0]["content"]

    prompt_text, _ = provider._gguf_render_prompt_tokens(messages=msgs, add_generation_prompt=True)
    if chat_format == "llama-3":
        assert prompt_text.count("<|start_header_id|>system<|end_header_id|>") == 1
    else:
        assert prompt_text.count("<|im_start|>system") == 1

    # Tools-only (no user system prompt): still exactly one system message.
    tools_only = provider._gguf_build_chat_messages(tools=tools, user_message_content="hi")
    assert [m["role"] for m in tools_only] == ["system", "user"]
    assert "## Tools (session)" in tools_only[0]["content"]

    # System-only (no tools): unchanged.
    sys_only = provider._gguf_build_chat_messages(system_prompt="SYS", user_message_content="hi")
    assert [m["role"] for m in sys_only] == ["system", "user"]
    assert sys_only[0]["content"] == "SYS"


def test_gguf_system_plus_tools_bytes_identical_across_build_paths() -> None:
    """Cache byte-prefix consistency: the control-plane rebuild
    (_prompt_cache_backend_append via prompt_cache_update) and the direct
    build (_gguf_build_chat_messages, what _generate_gguf renders) must
    produce IDENTICAL bytes for the same logical (system_prompt, tools,
    messages) — a cache prepared one way must not miss when generated the
    other way (the PromptCacheModule.normalized() tool-sorting incident is
    the precedent for this exact failure class).
    """
    provider = _new_provider(chat_format="chatml-function-calling")
    tools = [{"type": "function", "function": {"name": "shell"}}]
    messages = [{"role": "user", "content": "FILEBOX"}]

    # Control-plane path: set + one update carrying system+tools+messages.
    assert provider.prompt_cache_set("sess", make_default=False) is True
    assert provider.prompt_cache_update(
        "sess",
        system_prompt="You are helpful.",
        tools=tools,
        messages=messages,
        add_generation_prompt=False,
    ) is True
    state = provider._prompt_cache_store.get("sess")
    assert isinstance(state, _GGUFPromptCacheValue)

    # Direct path: the same logical inputs through the single-shot builder.
    direct_messages = provider._gguf_build_chat_messages(
        system_prompt="You are helpful.",
        tools=tools,
        messages=messages,
        user_message_content=None,
    )
    direct_text, direct_tokens = provider._gguf_render_prompt_tokens(
        messages=direct_messages,
        add_generation_prompt=False,
    )

    assert state.prompt_text == direct_text
    assert state.prompt_tokens == tuple(direct_tokens)
    assert direct_text.count("<|im_start|>system") == 1


def test_gguf_prompt_cache_update_uses_unified_thinking_control() -> None:
    provider = _new_provider(chat_format="chatml-function-calling")
    captured: List[Any] = []
    original = provider._gguf_render_prompt_tokens

    def _capture_tokens(**kwargs: Any):
        captured.append(kwargs.get("enable_thinking"))
        return original(**kwargs)

    provider._gguf_render_prompt_tokens = _capture_tokens  # type: ignore[method-assign]
    assert provider.prompt_cache_set("thinking", make_default=False) is True
    assert provider.prompt_cache_update(
        "thinking",
        prompt="hi",
        add_generation_prompt=True,
        thinking="off",
    ) is True

    assert False in captured
