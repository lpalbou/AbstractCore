"""Served-model cross-check: rogue-embedder-label defense (incident 2026-07-11).

The OpenAI-compatible /v1/embeddings response carries a `model` field naming
what the server ACTUALLY served — the only server-side label truth in the
stack, previously discarded. These tests pin the warn-only SIGNAL layer:
- the served label is recorded and surfaced (introspection + `served_model`);
- a genuine mismatch warns ONCE (never raises — the pin is the authority);
- label formatting variance (org/ prefix, :tag) does NOT warn;
- the HuggingFace-local path (no server label) is unaffected.

Offline: a stub provider returns a controlled OpenAI-shaped response; no model
downloads, no network.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pytest

from abstractcore.embeddings.manager import EmbeddingManager


class _StubEmbedProvider:
    """Minimal OpenAI-compatible embeddings provider double."""

    def __init__(self, served_model: str, dim: int = 8) -> None:
        self._served = served_model
        self._dim = dim

    def embed(self, input_text: Any, **kwargs: Any) -> Dict[str, Any]:
        texts: List[str] = input_text if isinstance(input_text, list) else [input_text]
        return {
            "model": self._served,
            "data": [{"embedding": [0.1] * self._dim} for _ in texts],
        }


def _manager_with_stub(requested_model: str, served_model: str) -> EmbeddingManager:
    """Build a manager on the served (lmstudio) route without touching config
    or the network, then swap in the stub provider."""
    mgr = EmbeddingManager.__new__(EmbeddingManager)
    # Minimal state the cross-check + introspection touch.
    mgr.provider = "lmstudio"
    mgr.model_id = requested_model
    mgr.output_dims = None
    mgr.served_model = None
    mgr._served_model_mismatch_warned = set()
    mgr._persistent_cache = {}
    mgr._provider_instance = _StubEmbedProvider(served_model)
    return mgr


def test_served_label_recorded_and_mismatch_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    mgr = _manager_with_stub(
        requested_model="text-embedding-qwen3-embedding-0.6b",
        served_model="mlx-community/all-minilm-l6-v2",  # the exact rogue label
    )
    with caplog.at_level(logging.WARNING):
        mgr._record_served_model(mgr._provider_instance.embed("hello"))
        mgr._record_served_model(mgr._provider_instance.embed("again"))  # same mismatch

    assert mgr.served_model == "mlx-community/all-minilm-l6-v2"
    warnings = [r for r in caplog.records if "served-model mismatch" in r.getMessage()]
    assert len(warnings) == 1, "must warn exactly once per distinct mismatch, not per call"
    msg = warnings[0].getMessage()
    assert "text-embedding-qwen3-embedding-0.6b" in msg and "all-minilm-l6-v2" in msg
    assert "#FALLBACK" in msg


def test_formatting_variance_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    # Same model, decorated with an org prefix + quant tag — NOT a real divergence.
    mgr = _manager_with_stub(
        requested_model="qwen3-embedding-0.6b",
        served_model="mlx-community/qwen3-embedding-0.6b:q8",
    )
    with caplog.at_level(logging.WARNING):
        mgr._record_served_model(mgr._provider_instance.embed("hello"))
    assert mgr.served_model == "mlx-community/qwen3-embedding-0.6b:q8"
    assert not [r for r in caplog.records if "served-model mismatch" in r.getMessage()]


def test_missing_served_label_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    mgr = _manager_with_stub("some-model", served_model="")
    with caplog.at_level(logging.WARNING):
        mgr._record_served_model({"data": [{"embedding": [0.0] * 4}]})  # no model key
    assert mgr.served_model is None
    assert not [r for r in caplog.records if "served-model mismatch" in r.getMessage()]


def test_cross_check_never_raises() -> None:
    mgr = _manager_with_stub("x", served_model="y")
    # Malformed inputs must be swallowed (warn-only means never break a fetch).
    mgr._record_served_model(None)
    mgr._record_served_model("not a dict")
    mgr._record_served_model({"model": 12345, "data": []})
