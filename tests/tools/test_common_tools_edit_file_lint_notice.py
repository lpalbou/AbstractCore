from __future__ import annotations

from abstractcore.tools import common_tools


def test_edit_file_appends_lint_notice_when_ruff_reports_issues(tmp_path, monkeypatch) -> None:
    path = tmp_path / "demo.py"
    path.write_text("x = 1\n", encoding="utf-8")

    def fake_ruff(content: str, filename, *, max_messages: int = 20, timeout_s: int = 10):  # type: ignore[no-untyped-def]
        return {
            "available": True,
            "error": None,
            "total": 1,
            "fixable": 1,
            "codes": ["F401"],
            "messages": ["  - 1:1 F401: fake issue (fixable)"],
        }

    monkeypatch.setattr(common_tools, "_run_ruff_check_content", fake_ruff)

    out = common_tools.edit_file(file_path=str(path), pattern="x = 1", replacement="x = 2", use_regex=False)
    assert "Notice: lint (python/ruff) found 1 issue(s) (1 fixable)" in out
    assert "F401" in out


def test_edit_file_does_not_append_lint_notice_when_clean(tmp_path, monkeypatch) -> None:
    path = tmp_path / "demo.py"
    path.write_text("x = 1\n", encoding="utf-8")

    def fake_ruff(content: str, filename, *, max_messages: int = 20, timeout_s: int = 10):  # type: ignore[no-untyped-def]
        return {"available": True, "error": None, "total": 0, "fixable": 0, "codes": [], "messages": []}

    monkeypatch.setattr(common_tools, "_run_ruff_check_content", fake_ruff)

    out = common_tools.edit_file(file_path=str(path), pattern="x = 1", replacement="x = 2", use_regex=False)
    assert "Notice: lint (python/ruff)" not in out

