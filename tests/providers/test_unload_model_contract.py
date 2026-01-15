from __future__ import annotations

import pytest


@pytest.mark.basic
def test_baseprovider_exposes_only_unload_model() -> None:
    from abstractcore.providers.base import BaseProvider
    from abstractcore.core.types import GenerateResponse

    class _Provider(BaseProvider):
        def __init__(self):
            super().__init__(model="test-model")
            self.unload_model_calls: list[str] = []

        def unload_model(self, model_name: str) -> None:
            self.unload_model_calls.append(model_name)

        def generate(self, *args, **kwargs):
            return GenerateResponse(content="ok", model=self.model)

        def get_capabilities(self):
            return ["chat"]

        def list_available_models(self, **kwargs):
            return ["test-model"]

    p = _Provider()
    assert hasattr(p, "unload") is False

    p.unload_model(p.model)
    assert p.unload_model_calls == ["test-model"]


@pytest.mark.basic
def test_baseprovider_forbids_unload_override() -> None:
    import textwrap

    with pytest.raises(TypeError):
        exec(
            textwrap.dedent(
                """
                from abstractcore.providers.base import BaseProvider
                from abstractcore.core.types import GenerateResponse

                class BadProvider(BaseProvider):
                    def unload(self) -> None:
                        pass

                    def unload_model(self, model_name: str) -> None:
                        pass

                    def generate(self, *args, **kwargs):
                        return GenerateResponse(content="ok", model=self.model)

                    def get_capabilities(self):
                        return []
                """
            ),
            {},
            {},
        )


@pytest.mark.basic
def test_provider_classes_do_not_override_unload() -> None:
    from abstractcore.providers.anthropic_provider import AnthropicProvider
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider
    from abstractcore.providers.mlx_provider import MLXProvider
    from abstractcore.providers.ollama_provider import OllamaProvider
    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider
    from abstractcore.providers.openai_provider import OpenAIProvider

    for cls in [
        AnthropicProvider,
        HuggingFaceProvider,
        MLXProvider,
        OllamaProvider,
        OpenAICompatibleProvider,
        OpenAIProvider,
    ]:
        assert "unload" not in cls.__dict__
