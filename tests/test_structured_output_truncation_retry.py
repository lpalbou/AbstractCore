from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from abstractcore.structured.handler import StructuredOutputHandler
from abstractcore.structured.retry import FeedbackRetry


class _Out(BaseModel):
    assertions: List[Dict[str, Any]]


class _Resp:
    def __init__(self, *, content: str, finish_reason: Optional[str]):
        self.content = content
        self.finish_reason = finish_reason


class _DummyProvider:
    def __init__(self):
        # Intentionally low "default" to simulate the historical bug where provider defaults
        # accidentally acted as a hard cap on retry bumping.
        self.max_output_tokens = 700
        self.model = "qwen/qwen3-next-80b"
        self.calls: list[dict[str, Any]] = []

    def _generate_internal(self, *, prompt: str, response_model: Any = None, **kwargs: Any) -> _Resp:  # noqa: ARG002
        self.calls.append(dict(kwargs))
        if len(self.calls) == 1:
            # Simulate provider truncation even though JSON is valid: finish_reason=length must not be accepted.
            return _Resp(content='{"assertions":[]}', finish_reason="length")
        return _Resp(content='{"assertions":[]}', finish_reason="stop")


def test_structured_output_truncation_retry_bumps_budget() -> None:
    provider = _DummyProvider()
    handler = StructuredOutputHandler(retry_strategy=FeedbackRetry(max_attempts=2))

    out = handler.generate_structured(provider, prompt="x", response_model=_Out)
    assert isinstance(out, _Out)
    assert out.assertions == []

    assert len(provider.calls) == 2
    # First call: no explicit max_output_tokens passed in.
    assert provider.calls[0].get("max_output_tokens") is None
    # Second call: retry must bump beyond the provider's low default using model capabilities.
    assert int(provider.calls[1].get("max_output_tokens") or 0) > provider.max_output_tokens
