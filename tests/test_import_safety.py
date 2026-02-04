from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.basic


def test_import_abstractcore_does_not_eagerly_import_optional_dependencies() -> None:
    code = r"""
import json
import sys

import abstractcore  # noqa: F401

targets = [
    # Tools extra
    "requests",
    "bs4",
    "lxml",
    "ddgs",
    # Embeddings extra
    "sentence_transformers",
    # Media extra (PDF tooling)
    "pymupdf",
    "fitz",
    "pymupdf4llm",
]

print(json.dumps({m: (m in sys.modules) for m in targets}))
"""

    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    data = json.loads(out)

    # `pip install abstractcore` should stay import-safe and avoid eager optional deps.
    assert data["requests"] is False
    assert data["bs4"] is False
    assert data["lxml"] is False
    assert data["ddgs"] is False
    assert data["sentence_transformers"] is False
    assert data["pymupdf"] is False
    assert data["fitz"] is False
    assert data["pymupdf4llm"] is False
