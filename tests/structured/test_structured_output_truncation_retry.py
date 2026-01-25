import json

from pydantic import BaseModel

from abstractcore.core.types import GenerateResponse
from abstractcore.structured.handler import StructuredOutputHandler


class _Attributes(BaseModel):
    evidence_quote: str


class _Assertion(BaseModel):
    subject: str
    predicate: str
    object: str
    attributes: _Attributes


class _Output(BaseModel):
    assertions: list[_Assertion]


class LMStudioProvider:
    """Deterministic fake provider used to validate truncation retries.

    StructuredOutputHandler treats `LMStudioProvider` as natively supporting structured outputs.
    """

    model = "unit-test-model"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def _generate_internal(self, *, prompt: str, response_model=None, **kwargs):  # noqa: ANN001
        del prompt, response_model
        raw_max = kwargs.get("max_output_tokens")
        if raw_max is None:
            raw_max = kwargs.get("max_tokens")
        max_out = int(raw_max) if raw_max is not None else 0
        if max_out <= 0:
            max_out = 700
        self.calls.append(max_out)

        if max_out < 1500:
            # Truncated JSON (missing closing brackets/braces) — equivalent to hitting max_tokens.
            return GenerateResponse(
                content=(
                    '{"assertions":[{"subject":"ex:a","predicate":"rdf:type","object":"schema:Thing",'
                    '"attributes":{"evidence_quote":"x"}}'
                ),
                finish_reason="length",
                usage={"output_tokens": max_out, "completion_tokens": max_out},
            )

        assertions = []
        for i in range(6):
            assertions.append(
                {
                    "subject": f"ex:s{i}",
                    "predicate": "rdf:type",
                    "object": "schema:Thing",
                    "attributes": {"evidence_quote": "x"},
                }
            )
        return GenerateResponse(
            content=json.dumps({"assertions": assertions}),
            finish_reason="stop",
            usage={"output_tokens": 200, "completion_tokens": 200},
        )


def test_structured_output_retries_increase_max_output_tokens_on_truncation() -> None:
    provider = LMStudioProvider()
    handler = StructuredOutputHandler()

    result = handler.generate_structured(
        provider=provider,
        prompt="extract",
        response_model=_Output,
        max_output_tokens=700,
    )

    assert isinstance(result, _Output)
    assert len(result.assertions) == 6
    assert provider.calls[0] == 700
    assert provider.calls == sorted(provider.calls)
    assert provider.calls[-1] >= 1500

