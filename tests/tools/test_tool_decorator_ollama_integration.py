"""
Real-LLM integration test for the consolidated @tool decorator.

Backlog reference:
- docs/backlog/completed/013-abstractcore-consolidate-tool-decorator.md

This verifies that:
- `abstractcore.tools.core.tool` attaches a `_tool_definition` usable by providers
- Passing decorated callables via `generate(..., tools=[...])` results in tool calls
  with a real Ollama-backed model (qwen3 tool format).

The test is intentionally narrow: one prompted tool-call roundtrip and parsing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from abstractcore import create_llm
from abstractcore.tools.core import tool


def _skip_if_ollama_unavailable(exc: Exception) -> None:
    msg = str(exc).lower()
    if any(
        keyword in msg
        for keyword in [
            "connection",
            "refused",
            "timeout",
            "not running",
            "operation not permitted",
            "no such host",
            "not found",
            "model not found",
            "pull",
        ]
    ):
        pytest.skip(f"Ollama/model not available: {exc}")


@pytest.mark.integration
def test_ollama_tool_calling_with_core_tool_decorator(tmp_path: Path) -> None:
    model = os.getenv("ABSTRACTCORE_OLLAMA_TOOL_MODEL", "qwen3:4b-instruct")
    base_url = os.getenv("ABSTRACTCORE_OLLAMA_BASE_URL", "http://localhost:11434")

    target_file = tmp_path / "tool_call_sentinel.txt"
    target_file.write_text("sentinel\n", encoding="utf-8")

    @tool
    def read_file(path: str) -> str:
        """Read a UTF-8 text file at the given path and return its content."""
        return Path(path).read_text(encoding="utf-8")

    # Sanity check: decorator attached metadata used by provider conversion.
    assert hasattr(read_file, "_tool_definition")
    assert read_file._tool_definition.name == "read_file"

    prompt = (
        "You do not know the content of the file. "
        "Use the read_file tool to read it.\n"
        f"File path: {target_file}\n"
        "Return ONLY a single tool call in the required format."
    )

    try:
        llm = create_llm(
            "ollama",
            model=model,
            base_url=base_url,
            temperature=0,
            seed=42,
        )
        response = llm.generate(prompt, tools=[read_file], execute_tools=False)
    except Exception as e:
        _skip_if_ollama_unavailable(e)
        raise

    assert response is not None
    content = response.content or ""

    # Some providers return errors as content instead of raising.
    if (response.finish_reason or "").lower() == "error" and content.lower().startswith("error:"):
        _skip_if_ollama_unavailable(RuntimeError(content))
        pytest.fail(f"Ollama provider returned error content: {content!r}")

    tool_calls = response.tool_calls or []
    assert tool_calls, f"No tool calls detected. Response content: {content!r}"

    call = next((tc for tc in tool_calls if tc.get("name") == "read_file"), None)
    assert call is not None, f"Expected read_file tool call. Tool calls: {tool_calls}. Content: {content!r}"

    args = call.get("arguments", {})
    if isinstance(args, str) and args.strip():
        import json

        args = json.loads(args)

    assert isinstance(args, dict)
    asserted_path = args.get("path")
    assert isinstance(asserted_path, str)
    assert asserted_path.strip() == str(target_file)
