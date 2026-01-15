import pytest

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
def test_empty_tools_does_not_trigger_hybrid_structured_flow() -> None:
    class Out(BaseModel):
        x: int

    class DummyProvider(BaseProvider):
        def __init__(self) -> None:
            super().__init__(model="dummy-structured-model")
            # Force native structured-output path for deterministic call counts.
            self.model_capabilities = {"structured_output": "native"}
            self.calls = []

        def get_capabilities(self):
            return []

        def list_available_models(self, **kwargs):
            return [self.model]

        def unload_model(self, model_name: str) -> None:
            return None

        def _generate_internal(self, prompt: str, **kwargs):
            self.calls.append(dict(kwargs))
            response_model = kwargs.get("response_model")
            if response_model is not None:
                return GenerateResponse(content={"x": 1})
            return GenerateResponse(content='{"x": 1}')

    provider = DummyProvider()
    result = provider.generate(prompt="hi", tools=[], response_model=Out)

    assert result.x == 1
    # Empty tools list must not trigger hybrid mode (which would call twice).
    assert len(provider.calls) == 1
    assert provider.calls[0].get("response_model") is Out
