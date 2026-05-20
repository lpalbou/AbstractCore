from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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

    repo_root = Path(__file__).resolve().parents[1]
    out = subprocess.check_output([sys.executable, "-c", code], text=True, cwd=str(repo_root)).strip()
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


def test_public_vision_catalog_helper_does_not_import_server_stack() -> None:
    code = r"""
import json
import sys

from abstractcore.capabilities import get_local_vision_cache_catalog  # noqa: F401

targets = [
    "fastapi",
    "multipart",
    "abstractcore.server.vision_endpoints",
]

print(json.dumps({m: (m in sys.modules) for m in targets}))
"""

    repo_root = Path(__file__).resolve().parents[1]
    out = subprocess.check_output([sys.executable, "-c", code], text=True, cwd=str(repo_root)).strip()
    data = json.loads(out)

    assert data["fastapi"] is False
    assert data["multipart"] is False
    assert data["abstractcore.server.vision_endpoints"] is False
