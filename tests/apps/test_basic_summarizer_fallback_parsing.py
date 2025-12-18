from __future__ import annotations

from abstractcore.processing import BasicSummarizer


def test_parse_fallback_response_marker_format() -> None:
    content = """SUMMARY:
This is the summary paragraph.

KEY POINTS:
- First point
- Second point
- Third point

CONFIDENCE: 0.8
FOCUS ALIGNMENT: 0.9
"""

    summary, key_points, confidence, focus_alignment = BasicSummarizer._parse_fallback_response(content)
    assert summary == "This is the summary paragraph."
    assert key_points == ["First point", "Second point", "Third point"]
    assert confidence == 0.8
    assert focus_alignment == 0.9


def test_parse_fallback_response_percent_scores() -> None:
    content = """SUMMARY:
Short summary.

KEY POINTS:
• Alpha
• Beta

CONFIDENCE: 98%
FOCUS_ALIGNMENT: 75%
"""

    _, key_points, confidence, focus_alignment = BasicSummarizer._parse_fallback_response(content)
    assert key_points == ["Alpha", "Beta"]
    assert confidence == 0.98
    assert focus_alignment == 0.75

