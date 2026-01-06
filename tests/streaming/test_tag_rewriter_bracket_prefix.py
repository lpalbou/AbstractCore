from __future__ import annotations

from abstractcore.tools.tag_rewriter import ToolCallTagRewriter, ToolCallTags


def test_rewrite_text_bracket_prefix_into_tags() -> None:
    rewriter = ToolCallTagRewriter(ToolCallTags("<|tool_call|>", "</|tool_call|>"))
    content = "tool: [list_files]: {'directory_path': 'rtype', 'recursive': True}\n"
    out = rewriter.rewrite_text(content)
    assert "<|tool_call|>" in out
    assert "</|tool_call|>" in out
    assert '"name": "list_files"' in out
    assert '"directory_path": "rtype"' in out
    assert '"recursive": true' in out

