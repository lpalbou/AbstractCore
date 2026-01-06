from __future__ import annotations

from pathlib import Path


def _extract_optional_dependency_block(text: str, *, key: str) -> str:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"{key} = ["):
            start = i
            break
    assert start is not None, f"Missing optional-dependencies entry: {key}"

    block: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() == "]":
            break
        block.append(line)
    return "\n".join(block)


def test_tools_extra_includes_bs4_and_tool_alias_exists() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    tools_block = _extract_optional_dependency_block(text, key="tools")
    tool_block = _extract_optional_dependency_block(text, key="tool")

    assert "beautifulsoup4" in tools_block
    assert "beautifulsoup4" in tool_block

