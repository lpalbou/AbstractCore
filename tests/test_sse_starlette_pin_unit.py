"""sse-starlette pin: AbstractCore never imports it itself (the server streams with
Starlette's StreamingResponse); the pin only has to stay compatible with what shares
the environment. mcp 2.3 requires sse-starlette>=3, and abstractcore pinned <2.0.0, so
every environment holding both ended in a pip conflict (operator machine, 2026-10-29)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sse_requirement() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'"(sse-starlette[^"]*)"', text)
    assert m, "sse-starlette is a declared dependency"
    return m.group(1)


def test_pin_admits_the_versions_mcp_needs():
    from packaging.requirements import Requirement

    spec = Requirement(_sse_requirement()).specifier
    for v in ("1.8.2", "2.1.0", "3.0.0", "3.6.0"):
        assert spec.contains(v), f"MUTANT: <2.0.0 -> {v} refused -> pip conflict with mcp>=2.3 -> RED"
    assert not spec.contains("4.0.0"), "the next major is not admitted blind"


def test_core_does_not_import_sse_starlette_itself():
    hits = [p for p in (ROOT / "abstractcore").rglob("*.py")
            if "sse_starlette" in p.read_text(encoding="utf-8", errors="ignore")]
    assert hits == [], f"a new import of sse_starlette must be checked against 1.8..3.x: {hits}"
