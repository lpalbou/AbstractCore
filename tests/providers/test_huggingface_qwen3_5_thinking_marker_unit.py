import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider
from abstractcore.providers.huggingface_provider import HuggingFaceProvider


class _DummyProvider(BaseProvider):
    def _generate_internal(self, prompt: str, *args, **kwargs):  # pragma: no cover
        return GenerateResponse(content="ok", model=self.model, finish_reason="stop")

    def get_capabilities(self):  # pragma: no cover
        return []

    def list_available_models(self):  # pragma: no cover
        return []

    def unload_model(self, model_name: str) -> None:  # pragma: no cover
        return None


def test_qwen3_5_thinking_off_no_base_marker_for_huggingface_gguf() -> None:
    """Base must NOT append a message-form no-think marker for hf-gguf shapes.

    Live-disproven 2026-08-19: a trailing assistant marker renders as a CLOSED
    turn through llama-cpp-python's create_chat_completion and does not disable
    thinking (113-1359 reasoning chars behind an "off" claim on three Qwen
    GGUFs). The real HuggingFaceProvider places the marker itself at the true
    generation boundary via the control-plane render; a provider whose hook
    declines (like this dummy) must degrade HONESTLY — warning, no fabricated
    marker, no "off" claim.
    """
    provider = _DummyProvider(model="unsloth/Qwen3.5-0.8B-GGUF")
    provider.provider = "huggingface"
    provider.model_type = "gguf"

    with pytest.warns(RuntimeWarning):
        prompt, messages, system_prompt, kwargs, meta = provider._apply_thinking_request(
            thinking="off",
            prompt="hi",
            messages=None,
            system_prompt=None,
            kwargs={},
        )

    assert system_prompt is None
    assert kwargs == {}
    assert prompt == "hi"
    assert messages is None
    assert isinstance(meta, dict)
    assert meta.get("thinking_effective") != "off"
    assert meta.get("thinking_handled_enable_disable") is False


def test_qwen3_5_thinking_off_does_not_apply_marker_for_huggingface_transformers() -> None:
    provider = _DummyProvider(model="unsloth/Qwen3.5-0.8B")
    provider.provider = "huggingface"
    provider.model_type = "transformers"

    with pytest.warns(RuntimeWarning):
        prompt, messages, system_prompt, kwargs, meta = provider._apply_thinking_request(
            thinking="off",
            prompt="hi",
            messages=None,
            system_prompt=None,
            kwargs={},
        )

    assert prompt == "hi"
    assert messages is None
    assert system_prompt is None
    assert kwargs == {}
    assert isinstance(meta, dict)


def test_huggingface_gguf_chat_messages_skips_empty_user_prompt() -> None:
    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)

    marker = "<think>\n\n</think>\n\n"
    out = provider._gguf_build_chat_messages(
        system_prompt=None,
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": marker}],
        tools=None,
        user_message_content="",
    )

    assert out[-1] == {"role": "assistant", "content": marker}
    assert len(out) == 2
