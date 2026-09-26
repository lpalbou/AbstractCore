"""HuggingFace transformers + GGUF lanes flag a prompt that opened `<think>`.

Same contract as the MLX lane (`test_stream_prompt_opened_thinking.py`): when the
chat template rendered IN-PROCESS ends inside an opened thinking block, the
stream leads with an empty `THINKING_OPENED_BY_PROMPT` chunk, so BaseProvider
streams the reasoning from the first token instead of holding it until
`</think>`. Unflagged prompts are unchanged; the flag never reaches a
non-streamed result. (OpenAI-compatible servers render server-side: no flag.)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pytest

from abstractcore.architectures import detect_architecture, get_architecture_format, get_model_capabilities
from abstractcore.architectures.response_postprocessing import THINKING_OPENED_BY_PROMPT
from abstractcore.core.types import GenerateResponse
from abstractcore.providers.huggingface_provider import HuggingFaceProvider

OPENED = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n"
PLAIN = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"
RAW = "Counting letters.\n</think>\n\nThree."


def _provider(monkeypatch, model: str = "qwen3-4b") -> HuggingFaceProvider:
    p = object.__new__(HuggingFaceProvider)
    p.model = model
    p.architecture_config = get_architecture_format(detect_architecture(model))
    p.model_capabilities = get_model_capabilities(model)
    p.logger = logging.getLogger("test")
    return p


def _flagged(chunks: List[GenerateResponse]) -> List[bool]:
    return [bool((c.metadata or {}).get(THINKING_OPENED_BY_PROMPT)) for c in chunks]


# --- transformers lane ---------------------------------------------------------


@pytest.mark.parametrize("rendered, flagged", [(OPENED, True), (PLAIN, False)])
def test_transformers_stream_leads_with_the_flag_only_when_the_prompt_opened_thinking(monkeypatch, rendered, flagged):
    p = _provider(monkeypatch)
    p.pipeline = object()
    seen: Dict[str, Any] = {}

    def fake_stream(input_text, *a, **k):
        seen["input_text"] = input_text
        yield GenerateResponse(content=RAW, model=p.model)
        yield GenerateResponse(content="", model=p.model, finish_reason="stop")

    for name, fn in {
        "_build_input_text_transformers": lambda *a, **k: rendered,
        "_prepare_generation_kwargs": lambda **k: {},
        "_get_provider_max_tokens_param": lambda k: 16,
        "_stream_generate_transformers_with_tools": fake_stream,
        "_transformers_prompt_cache_supported": lambda: False,
    }.items():
        monkeypatch.setattr(p, name, fn, raising=False)
    p.temperature, p.top_p, p.structured_output_method = 0.0, 1.0, "prompted"
    chunks = list(p._generate_transformers("q", stream=True))
    assert seen["input_text"] == rendered
    assert _flagged(chunks) == ([True] if flagged else []) + [False, False]
    assert "".join(c.content for c in chunks) == RAW


# --- GGUF control-plane lane -----------------------------------------------------


def _control_plane(monkeypatch, rendered: str) -> HuggingFaceProvider:
    p = _provider(monkeypatch)
    text = list(RAW)

    class FakeLlama:
        def tokenize(self, b, add_bos=False, special=True):
            return [99999]

        def detokenize(self, toks):
            return "".join(text[t - 1] for t in toks).encode("utf-8")

        def set_seed(self, s):
            pass

        def token_eos(self):
            return 10_000

        def generate(self, tokens, **kw):
            yield from range(1, len(text) + 1)

    p.llm = FakeLlama()
    monkeypatch.setattr(p, "_gguf_control_plane_stop_strings", lambda: [], raising=False)
    monkeypatch.setattr(p, "_gguf_render_prompt_tokens", lambda **k: (rendered, (1, 2, 3)), raising=False)
    monkeypatch.setattr(p, "_gguf_compose_cached_prompt_tokens",
                        lambda **k: (k["live_prompt_text"], k["live_prompt_tokens"], {}), raising=False)
    monkeypatch.setattr(p, "_gguf_generation_prompt_boundary", lambda **k: None, raising=False)
    monkeypatch.setattr(p, "_gguf_prefill_prompt_cache", lambda *a, **k: True, raising=False)
    return p


_CP_ARGS = dict(
    chat_messages=[{"role": "user", "content": "q"}], cache_obj=None, max_output_tokens=10_000,
    temperature=0.0, top_p=1.0, top_k=1, min_p=0.0, typical_p=1.0, repeat_penalty=1.0,
    presence_penalty=0.0, frequency_penalty=0.0, tfs_z=1.0, mirostat_mode=0, mirostat_tau=5.0,
    mirostat_eta=0.1, seed=None,
)


@pytest.mark.parametrize("rendered, flagged", [(OPENED, True), (PLAIN, False)])
def test_gguf_control_plane_stream_flags_and_non_stream_never_carries_the_flag(monkeypatch, rendered, flagged):
    p = _control_plane(monkeypatch, rendered)
    chunks = list(p._gguf_control_plane_stream_generate(**_CP_ARGS))
    assert _flagged(chunks)[0] is flagged
    assert sum(_flagged(chunks)) == (1 if flagged else 0)
    streamed_text = "".join(c.content or "" for c in chunks)

    p = _control_plane(monkeypatch, rendered)
    args = {k: v for k, v in _CP_ARGS.items()}
    whole = p._gguf_control_plane_generate(stream=False, **args)
    assert THINKING_OPENED_BY_PROMPT not in (whole.metadata or {})
    assert whole.content == streamed_text


# --- GGUF fallback lane (create_chat_completion) ----------------------------------

TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{{ opener }}{% endif %}"
)


@pytest.mark.parametrize("opener, chat_format, flagged", [
    ("<think>\n", "chat_template.default", True),
    ("", "chat_template.default", False),
    ("<think>\n", "chatml", False),  # built-in formats never open a thinking block
])
def test_gguf_fallback_stream_flags_from_the_embedded_template(monkeypatch, opener, chat_format, flagged):
    p = _provider(monkeypatch)

    class FakeLlama:
        metadata = {"tokenizer.chat_template": TEMPLATE.replace("{{ opener }}", opener)}

        def __init__(self):
            self.chat_format = chat_format

        def token_eos(self):
            return 2

        def create_chat_completion(self, **kw):
            assert kw["stream"] is True
            for part in (RAW[:10], RAW[10:]):
                yield {"choices": [{"delta": {"content": part}, "finish_reason": None}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

    p.llm = FakeLlama()
    for name, fn in {
        "_gguf_build_chat_messages": lambda **k: [{"role": "user", "content": "q"}],
        "_prepare_generation_kwargs": lambda **k: {},
        "_get_provider_max_tokens_param": lambda k: 16,
        "_gguf_prompt_cache_supports_local_control_plane": lambda: False,
        "_thinking_disable_prefill": lambda x: "",
        "_gguf_normalize_tool_call_arguments_for_template": lambda m: m,
        "_gguf_template_bos_text": lambda: "",
        "_gguf_model_token_text": lambda t: "",
    }.items():
        monkeypatch.setattr(p, name, fn, raising=False)
    p.temperature = 0.0
    chunks = list(p._generate_gguf("q", None, None, None, None, True, None))
    assert _flagged(chunks)[0] is flagged and sum(_flagged(chunks)) == (1 if flagged else 0)
    assert "".join(c.content or "" for c in chunks) == RAW
