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


def test_qwen3_5_thinking_off_inserts_empty_think_marker_for_huggingface_gguf() -> None:
    provider = _DummyProvider(model="unsloth/Qwen3.5-0.8B-GGUF")
    provider.provider = "huggingface"
    provider.model_type = "gguf"

    prompt, messages, system_prompt, kwargs = provider._apply_thinking_request(
        thinking="off",
        prompt="hi",
        messages=None,
        system_prompt=None,
        kwargs={},
    )

    assert system_prompt is None
    assert kwargs == {}
    assert prompt == ""
    assert isinstance(messages, list)
    assert messages[-2] == {"role": "user", "content": "hi"}
    assert messages[-1] == {"role": "assistant", "content": "<think>\n\n</think>\n\n"}


def test_qwen3_5_thinking_off_does_not_apply_marker_for_huggingface_transformers() -> None:
    provider = _DummyProvider(model="unsloth/Qwen3.5-0.8B")
    provider.provider = "huggingface"
    provider.model_type = "transformers"

    with pytest.warns(RuntimeWarning):
        prompt, messages, system_prompt, kwargs = provider._apply_thinking_request(
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
