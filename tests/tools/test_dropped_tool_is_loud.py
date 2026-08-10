"""A tool that does not reach the model must SAY so (ADR 0001).

Found 2026-08-07. `UniversalToolHandler._convert_to_tool_definitions` swallowed
every conversion failure into `logger.warning`, which is DEAD in a default
AbstractCore process (root logger at ERROR, all `abstractcore.*` loggers
NOTSET). The practical consequence, measured on all three local lanes (MLX, HF
transformers, HF GGUF): a tool set whose descriptions exceeded `ToolDefinition`'s
200-char cap converted to ZERO definitions, `format_tools_prompt` returned "",
and the rendered prompt contained no tools at all — with no output anywhere.
`generate()` and the prompt-cache bloc planner both saw a tool-less prompt and
reported nothing wrong.

`warnings.warn` is the channel a caller actually receives.

UPDATED 2026-08-07 (TOOL-desc). The over-long-description cases in here asserted
that such a tool is DROPPED-but-loud. That was the stopgap. The policy is now:
a description we cannot edit (a dict tool — the shape MCP servers and gateway
payloads arrive in) is ADAPTED to the cap on a sentence boundary and reported
for correction; the tool still reaches the model. Those cases were rewritten to
assert the new contract, not deleted — see `test_overlong_external_description_*`.
The drop path itself is unchanged and still tested here for tools that genuinely
cannot be converted (a dict of the wrong shape, a non-tool).
"""

from __future__ import annotations

import warnings

import pytest

from abstractcore.tools.handler import UniversalToolHandler


MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

GOOD_TOOL = {
    "name": "read_file",
    "description": "Read a file from disk and return its text.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
}
OVERLONG_TOOL = {
    "name": "write_file",
    "description": ("This description is deliberately far past the metadata cap. " * 5).strip(),
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
}


def _warn_texts(caught) -> list:
    return [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]


@pytest.mark.parametrize(
    "bad,expect",
    [
        ({"nonsense": True}, "OpenAI-style"),
        (42, "unsupported tool format"),
    ],
    ids=["unrecognized-dict", "not-a-tool"],
)
def test_a_dropped_tool_warns_the_caller(bad, expect):
    handler = UniversalToolHandler(MODEL)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rendered = handler.format_tools_prompt([GOOD_TOOL, bad])

    texts = _warn_texts(caught)
    assert any("DROPPED" in t and expect in t for t in texts), texts
    # The surviving tool still renders — a drop is per-tool, not per-request.
    assert "read_file" in rendered


def test_the_native_lane_warns_too():
    """`prepare_tools_for_native` shares the converter; a wire drop is as silent."""
    handler = UniversalToolHandler(MODEL)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        native = handler.prepare_tools_for_native([GOOD_TOOL, {"nonsense": True}])

    assert [t["function"]["name"] for t in native] == ["read_file"]
    assert any("DROPPED" in t for t in _warn_texts(caught))


def test_healthy_tools_are_silent():
    handler = UniversalToolHandler(MODEL)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rendered = handler.format_tools_prompt([GOOD_TOOL])

    assert not _warn_texts(caught)
    assert "read_file" in rendered


def test_the_warning_names_the_tool_and_the_model():
    handler = UniversalToolHandler(MODEL)
    unconvertible = {"name": "write_file", "parameters": {}}  # no description key

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        handler.format_tools_prompt([unconvertible])

    text = "\n".join(_warn_texts(caught))
    assert "write_file" in text and MODEL in text


# --- the new contract: an over-long EXTERNAL description costs nobody the tool ---


def test_overlong_external_description_keeps_the_tool():
    """The whole point. A wordy MCP tool is still a tool."""
    handler = UniversalToolHandler(MODEL)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        native = handler.prepare_tools_for_native([GOOD_TOOL, OVERLONG_TOOL])

    assert [t["function"]["name"] for t in native] == ["read_file", "write_file"]
    assert all(len(t["function"]["description"]) <= 200 for t in native)
    # Adapted, not dropped.
    assert not any("DROPPED" in t for t in _warn_texts(caught))


def test_overlong_external_description_is_reported_actionably():
    """The caller must learn WHICH tool, HOW long, WHAT was lost, and the rule."""
    handler = UniversalToolHandler(MODEL)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        handler.format_tools_prompt([OVERLONG_TOOL])

    text = "\n".join(_warn_texts(caught))
    assert "write_file" in text
    assert str(len(OVERLONG_TOOL["description"])) in text  # the actual length
    assert "kept" in text and "dropped" in text  # the quoted evidence
    assert "max 200 chars" in text  # what good looks like


def test_adaptation_never_cuts_mid_sentence():
    """A half-sentence can invert the tool's meaning; whole sentences cannot."""
    from abstractcore.tools.core import adapt_external_description

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        source_text = (
            "Deletes the record permanently. "
            "Do not use this unless the user has explicitly confirmed the deletion, "
            "because there is no undo, the audit trail is removed along with it, and "
            "downstream replicas will drop their copies on the next sync pass."
        )
        assert len(source_text) > 200, "fixture must actually exceed the cap"
        out = adapt_external_description(source_text, tool_name="purge", source="test")

    # The whole first sentence, and nothing of the second — a cut at 200 chars
    # would have landed inside "Do not use this unless…" and inverted the advice.
    assert out == "Deletes the record permanently."


def test_our_own_tools_still_fail_hard():
    """The cap must stay a forcing function where we CAN edit the source."""
    from abstractcore.tools.core import tool

    with pytest.raises(ValueError, match="description is too long"):

        @tool(description="x" * 250)
        def my_tool(a: str) -> str:
            return a
