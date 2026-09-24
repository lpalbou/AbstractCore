"""llms-full.txt inlines every user page the docs index links, and stays fresh."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_updater():
    spec = importlib.util.spec_from_file_location("update_llms", ROOT / "scripts" / "update_llms.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_top_level_docs_page_is_a_source():
    sources = set(_load_updater().indexed_sources())
    pages = {"docs/" + p.name for p in (ROOT / "docs").glob("*.md") if p.name != "README.md"}
    assert pages, "no docs/*.md pages found"
    assert pages - sources == set()


@pytest.mark.parametrize(
    "page",
    [
        "docs/web-tools.md",
        "docs/prompt-caching.md",
        "docs/huggingface-model-compatibility.md",
        "docs/fallbacks.md",
        "docs/memory-management.md",
        "docs/reasoning-control.md",
    ],
)
def test_topic_pages_are_inlined(page):
    assert "### Inlined: `%s`" % page in (ROOT / "llms-full.txt").read_text(encoding="utf-8")


def test_no_planning_or_history_pages_are_sources():
    for source in _load_updater().indexed_sources()[1:]:
        assert source.split("/")[1] not in {"adr", "archive", "backlog", "known_bugs", "reports", "research"}


def test_a_page_missing_from_the_index_fails(tmp_path, monkeypatch):
    updater = _load_updater()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "README.md").write_text("- [A](a.md)\n", encoding="utf-8")
    (tmp_path / "docs" / "a.md").write_text("# A\n", encoding="utf-8")
    (tmp_path / "docs" / "b.md").write_text("# B\n", encoding="utf-8")
    monkeypatch.setattr(updater, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="docs/b.md"):
        updater.indexed_sources()


def test_llms_full_is_current():
    updater = _load_updater()
    current = (ROOT / "llms-full.txt").read_text(encoding="utf-8")
    assert current == updater.render(current), "llms-full.txt is stale; run python scripts/update_llms.py"
