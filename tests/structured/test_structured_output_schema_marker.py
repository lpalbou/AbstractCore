from __future__ import annotations

from pydantic import BaseModel

from abstractcore.structured.handler import StructuredOutputHandler


class _DemoModel(BaseModel):
    assertions: list[str]


def test_schema_prompt_inserts_at_marker_when_present() -> None:
    handler = StructuredOutputHandler()
    prompt = "HEADER\n<<STRUCTURED_OUTPUT_SCHEMA>>\nFOOTER\nTEXT:\nhello"
    out = handler._create_schema_prompt(prompt, _DemoModel)

    assert "<<STRUCTURED_OUTPUT_SCHEMA>>" not in out
    assert "Please respond with valid JSON" in out
    assert "FOOTER\nTEXT:\nhello" in out
    assert out.index("Please respond with valid JSON") < out.index("FOOTER\nTEXT:\nhello")


def test_schema_prompt_appends_when_marker_absent() -> None:
    handler = StructuredOutputHandler()
    prompt = "TEXT:\nhello"
    out = handler._create_schema_prompt(prompt, _DemoModel)
    assert out.startswith(prompt)
    assert "Please respond with valid JSON" in out
    assert out.rstrip().endswith("Important: Return ONLY the JSON object, no additional text or formatting.")

