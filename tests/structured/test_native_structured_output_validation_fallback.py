from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

from abstractcore.structured.handler import StructuredOutputHandler


class _AssertionModel(BaseModel):
    subject: str
    predicate: str
    object: str


class _ResponseModel(BaseModel):
    assertions: list[_AssertionModel]


def test_native_structured_output_dict_validation_error_falls_back_to_prompted() -> None:
    """Regression: some servers return dicts that don't match the schema.

    In that case, native mode must fall back to prompted structured output so callers
    still receive a validated model instance.
    """

    class LMStudioProvider:  # noqa: N801 - intentional name to trigger native detection
        def __init__(self) -> None:
            self.model = "dummy"
            self.calls: list[dict] = []

        def _generate_internal(self, *, prompt: str, **kwargs):  # noqa: ANN001 - test double
            self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
            # First call (native) returns an invalid dict that can't be coerced.
            if len(self.calls) == 1:
                return SimpleNamespace(content={"foo": "bar"})
            # Second call (prompted fallback) returns valid JSON.
            return SimpleNamespace(content='{"assertions":[{"subject":"x","predicate":"schema:mentions","object":"y"}]}')

    provider = LMStudioProvider()
    handler = StructuredOutputHandler()

    out = handler.generate_structured(provider=provider, prompt="extract", response_model=_ResponseModel)
    assert isinstance(out, _ResponseModel)
    assert out.assertions and out.assertions[0].subject == "x"
    assert len(provider.calls) >= 2


def test_native_structured_output_repairs_single_item_object_into_wrapper() -> None:
    """Some servers emit a single list-item object instead of the wrapper."""

    class LMStudioProvider:  # noqa: N801 - trigger native detection
        def __init__(self) -> None:
            self.model = "dummy"

        def _generate_internal(self, *, prompt: str, **kwargs):  # noqa: ANN001 - test double
            return SimpleNamespace(content='{"subject":"x","predicate":"schema:mentions","object":"y"}')

    provider = LMStudioProvider()
    handler = StructuredOutputHandler()
    out = handler.generate_structured(provider=provider, prompt="extract", response_model=_ResponseModel)
    assert out.assertions and out.assertions[0].subject == "x"


def test_native_structured_output_repairs_list_root_into_wrapper() -> None:
    """Some servers emit the list itself instead of the wrapper object."""

    class LMStudioProvider:  # noqa: N801 - trigger native detection
        def __init__(self) -> None:
            self.model = "dummy"

        def _generate_internal(self, *, prompt: str, **kwargs):  # noqa: ANN001 - test double
            return SimpleNamespace(content='[{"subject":"x","predicate":"schema:mentions","object":"y"}]')

    provider = LMStudioProvider()
    handler = StructuredOutputHandler()
    out = handler.generate_structured(provider=provider, prompt="extract", response_model=_ResponseModel)
    assert out.assertions and out.assertions[0].predicate == "schema:mentions"
