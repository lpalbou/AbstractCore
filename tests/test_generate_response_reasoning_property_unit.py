from abstractcore.core.types import GenerateResponse


def test_generate_response_reasoning_property_prefers_canonical_reasoning_key():
    resp = GenerateResponse(
        content="final",
        metadata={"reasoning": "r1", "reasoning_content": "r2"},
    )
    assert resp.reasoning == "r1"


def test_generate_response_reasoning_property_falls_back_to_reasoning_content():
    resp = GenerateResponse(content="final", metadata={"reasoning_content": "r2"})
    assert resp.reasoning == "r2"


def test_generate_response_reasoning_property_falls_back_to_thinking():
    resp = GenerateResponse(content="final", metadata={"thinking": "r3"})
    assert resp.reasoning == "r3"


def test_generate_response_reasoning_setter_sets_metadata():
    resp = GenerateResponse(content="final")
    resp.reasoning = "r1"
    assert isinstance(resp.metadata, dict)
    assert resp.metadata["reasoning"] == "r1"


def test_generate_response_reasoning_setter_clears_canonical_key_only():
    resp = GenerateResponse(content="final", metadata={"reasoning": "r1", "reasoning_content": "r2"})
    resp.reasoning = None
    assert isinstance(resp.metadata, dict)
    assert resp.metadata.get("reasoning") is None
    assert resp.metadata.get("reasoning_content") == "r2"

