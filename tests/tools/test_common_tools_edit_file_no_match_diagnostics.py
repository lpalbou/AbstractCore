from __future__ import annotations

from abstractcore.tools.common_tools import edit_file


def test_edit_file_regex_no_match_includes_actionable_diagnostics(tmp_path) -> None:
    path = tmp_path / "demo.py"
    path.write_text(
        "import pygame\n"
        "pygame.draw.polygon(self.game.screen, self.game.GREEN, [])\n",
        encoding="utf-8",
    )

    out = edit_file(
        file_path=str(path),
        pattern=r"pygame.draw.polygon\(self\.image, BLUE",
        replacement="x",
        use_regex=True,
    )

    assert out.startswith("❌ No matches found for regex pattern"), out
    assert "Closest lines (token match:" in out
    assert "pygame.draw.polygon" in out
    assert "2: pygame.draw.polygon" in out
    assert path.read_text(encoding="utf-8").startswith("import pygame")


def test_edit_file_rejects_empty_pattern(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("hello\n", encoding="utf-8")

    out = edit_file(file_path=str(path), pattern="", replacement="x", use_regex=False)

    assert out.startswith("❌ Invalid pattern:"), out
    assert path.read_text(encoding="utf-8") == "hello\n"
