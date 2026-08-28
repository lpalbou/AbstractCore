"""Server-suite hermeticity fixtures.

`/acore/models/loaded` now merges a best-effort residency sweep of REAL local
model servers (Ollama / LM Studio). On a developer machine with models
resident that would leak host state into unit assertions, so the sweep is
stubbed empty by default; sweep-specific tests monkeypatch their own fake
records on top.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _stub_provider_server_sweep(monkeypatch):
    from abstractcore.server import app as server_app

    monkeypatch.setattr(server_app, "sweep_loaded_models", lambda timeout_s=2.0: [])
