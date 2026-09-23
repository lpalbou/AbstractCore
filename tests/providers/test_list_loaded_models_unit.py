"""Unit tests for provider loaded-model enumeration (no live servers).

Covers the base `list_loaded_models()` default (derived from provider-owned
residency truth, ADR 0008), the Ollama/LM Studio server-wide classmethods with
faked HTTP, and the host-level `sweep_loaded_models()` fan-out.
"""

from __future__ import annotations

import importlib.util

import pytest

from typing import Any, Dict, List

import httpx

from abstractcore.providers.lmstudio_provider import LMStudioProvider
from abstractcore.providers.mlx_provider import MLXProvider
from abstractcore.providers.ollama_provider import OllamaProvider
from abstractcore.utils.residency import (
    SWEEP_PROVIDERS,
    normalize_sweep_model,
    sweep_loaded_models,
    sweep_models_match,
)


_requires_mlx_stack = pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("mlx", "mlx_lm", "mlx_vlm")),
    reason="requires the optional MLX stack (pip install \"abstractcore[mlx]\")",
)


def _fake_httpx_get(monkeypatch, url_to_payload: Dict[str, Any]) -> List[str]:
    requested: List[str] = []

    def fake_get(url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        requested.append(url)
        payload = url_to_payload.get(url)
        if payload is None:
            raise httpx.ConnectError("connection refused", request=httpx.Request("GET", url))
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    return requested


# ---------------------------------------------------------------------------
# Base default: derived from get_model_residency()
# ---------------------------------------------------------------------------

def test_base_list_loaded_models_from_residency_loaded() -> None:
    provider = object.__new__(MLXProvider)
    provider.provider = "mlx"
    provider.model = "mlx-test-model"
    provider.llm = None  # keep est_weights_bytes silent for the base shape check
    provider.tokenizer = object()

    assert provider.list_loaded_models() == []

    provider.llm = object()
    records = provider.list_loaded_models()
    assert len(records) == 1
    assert records[0]["provider"] == "mlx"
    assert records[0]["model"] == "mlx-test-model"
    assert records[0]["loaded"] is True


@_requires_mlx_stack
def test_mlx_list_loaded_models_estimates_weight_bytes() -> None:
    class _FakeArray:
        def __init__(self, nbytes: int) -> None:
            self.nbytes = nbytes

    class _FakeModel:
        def parameters(self) -> Dict[str, Any]:
            return {"layers": {"w": _FakeArray(100), "b": _FakeArray(24)}}

    provider = object.__new__(MLXProvider)
    provider.provider = "mlx"
    provider.model = "mlx-test-model"
    provider.llm = _FakeModel()
    provider.tokenizer = object()

    records = provider.list_loaded_models()
    assert len(records) == 1
    assert records[0]["est_weights_bytes"] == 124


# ---------------------------------------------------------------------------
# Ollama: server-wide enumeration from /api/ps
# ---------------------------------------------------------------------------

def test_ollama_list_server_loaded_models_normalizes_api_ps(monkeypatch) -> None:
    requested = _fake_httpx_get(
        monkeypatch,
        {
            "http://fake-ollama:11434/api/ps": {
                "models": [
                    {
                        "name": "gemma3:1b",
                        "model": "gemma3:1b",
                        "size": 1000,
                        "size_vram": 900,
                        "expires_at": "2318-08-31T12:29:48+02:00",
                        "context_length": 32768,
                        "digest": "abc123",
                    },
                    {"name": "qwen3:4b", "size": 4000},
                    "not-a-dict",
                ]
            }
        },
    )

    records = OllamaProvider.list_server_loaded_models(base_url="http://fake-ollama:11434")

    assert requested == ["http://fake-ollama:11434/api/ps"]
    assert len(records) == 2
    first = records[0]
    assert first == {
        "provider": "ollama",
        "model": "gemma3:1b",
        "resident": True,
        "loaded": True,
        "source": "abstractcore.provider.ollama.native_rest",
        "size_bytes": 1000,
        "size_vram_bytes": 900,
        "expires_at": "2318-08-31T12:29:48+02:00",
        "context_length": 32768,
        "digest": "abc123",
    }
    assert records[1]["model"] == "qwen3:4b"
    assert records[1]["size_bytes"] == 4000
    assert "size_vram_bytes" not in records[1]


def test_ollama_instance_list_loaded_models_delegates_with_instance_base_url(monkeypatch) -> None:
    requested = _fake_httpx_get(
        monkeypatch,
        {"http://instance-ollama:11434/api/ps": {"models": [{"name": "gemma3:1b"}]}},
    )

    provider = object.__new__(OllamaProvider)
    provider.provider = "ollama"
    provider.model = "gemma3:1b"
    provider.base_url = "http://instance-ollama:11434"

    records = provider.list_loaded_models()

    assert requested == ["http://instance-ollama:11434/api/ps"]
    assert [record["model"] for record in records] == ["gemma3:1b"]


# ---------------------------------------------------------------------------
# LM Studio: server-wide enumeration from /api/v1/models loaded_instances
# ---------------------------------------------------------------------------

def test_lmstudio_list_server_loaded_models_walks_loaded_instances(monkeypatch) -> None:
    requested = _fake_httpx_get(
        monkeypatch,
        {
            "http://fake-lmstudio:1234/api/v1/models": {
                "models": [
                    {
                        "key": "qwen/qwen3-4b",
                        "loaded_instances": [{"id": "inst-1"}, {"id": "inst-2"}],
                        "size_bytes": 4321,
                        "max_context_length": 32768,
                    },
                    {"key": "unloaded-model", "loaded_instances": []},
                    {"key": "no-instances-field"},
                ]
            }
        },
    )

    records = LMStudioProvider.list_server_loaded_models(base_url="http://fake-lmstudio:1234/v1")

    assert requested == ["http://fake-lmstudio:1234/api/v1/models"]
    assert len(records) == 1
    assert records[0] == {
        "provider": "lmstudio",
        "model": "qwen/qwen3-4b",
        "provider_instance_ids": ["inst-1", "inst-2"],
        "resident": True,
        "loaded": True,
        "source": "abstractcore.provider.lmstudio.native_rest",
        "size_bytes": 4321,
        "context_length": 32768,
    }


def test_lmstudio_instance_list_loaded_models_delegates_with_instance_base_url(monkeypatch) -> None:
    requested = _fake_httpx_get(
        monkeypatch,
        {
            "http://instance-lmstudio:1234/api/v1/models": {
                "models": [{"key": "m", "loaded_instances": [{"id": "i-1"}]}]
            }
        },
    )

    provider = object.__new__(LMStudioProvider)
    provider.provider = "lmstudio"
    provider.model = "m"
    provider.base_url = "http://instance-lmstudio:1234/v1"

    records = provider.list_loaded_models()

    assert requested == ["http://instance-lmstudio:1234/api/v1/models"]
    assert [record["provider_instance_ids"] for record in records] == [["i-1"]]


def test_lmstudio_residency_probe_still_matches_after_refactor(monkeypatch) -> None:
    """`_native_rest_loaded_instance_ids_for_model` keeps its matching semantics
    on top of the shared item-walking helpers."""
    _fake_httpx_get(
        monkeypatch,
        {
            "http://instance-lmstudio:1234/api/v1/models": {
                "models": [
                    {"key": "qwen/qwen3-4b", "loaded_instances": [{"id": "i-1"}]},
                    {"key": "other-model", "loaded_instances": [{"id": "i-2"}]},
                ]
            }
        },
    )

    provider = object.__new__(LMStudioProvider)
    provider.provider = "lmstudio"
    provider.model = "qwen/qwen3-4b"
    provider.base_url = "http://instance-lmstudio:1234/v1"
    provider._timeout = 5.0
    provider.api_key = None

    assert provider._native_rest_loaded_instance_ids_for_model("qwen/qwen3-4b") == ["i-1"]
    assert provider._native_rest_loaded_instance_ids_for_model("missing") == []


# ---------------------------------------------------------------------------
# Host-level sweep
# ---------------------------------------------------------------------------

def test_sweep_loaded_models_tags_source_and_skips_erroring_providers(monkeypatch) -> None:
    monkeypatch.setattr(
        OllamaProvider,
        "list_server_loaded_models",
        staticmethod(
            lambda base_url=None, timeout_s=2.0: [
                {"provider": "ollama", "model": "gemma3:1b", "resident": True, "loaded": True, "size_bytes": 7}
            ]
        ),
    )

    def _boom(base_url=None, timeout_s=2.0):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(LMStudioProvider, "list_server_loaded_models", staticmethod(_boom))

    records = sweep_loaded_models(timeout_s=0.5)

    assert len(records) == 1
    assert records[0]["provider"] == "ollama"
    assert records[0]["model"] == "gemma3:1b"
    assert records[0]["source"] == "provider_server"
    assert records[0]["size_bytes"] == 7
    # Errors are skipped silently — no error markers leak into the result.
    assert all("sweep_errors" not in record for record in records)


def test_sweep_loaded_models_never_raises_when_all_providers_fail(monkeypatch) -> None:
    def _boom(base_url=None, timeout_s=2.0):
        raise RuntimeError("no server")

    monkeypatch.setattr(OllamaProvider, "list_server_loaded_models", staticmethod(_boom))
    monkeypatch.setattr(LMStudioProvider, "list_server_loaded_models", staticmethod(_boom))

    assert sweep_loaded_models() == []


# ---------------------------------------------------------------------------
# Public dedup/matching helpers (single source of truth for server + runtime)
# ---------------------------------------------------------------------------

def test_sweep_providers_tuple() -> None:
    assert SWEEP_PROVIDERS == ("ollama", "lmstudio")


def test_normalize_sweep_model() -> None:
    assert normalize_sweep_model("Qwen3:LATEST") == "qwen3"
    assert normalize_sweep_model(" gemma3:1b ") == "gemma3:1b"
    assert normalize_sweep_model("qwen3") == "qwen3"
    assert normalize_sweep_model(None) == ""
    assert normalize_sweep_model("") == ""


def test_sweep_models_match_ollama_latest_alias() -> None:
    record = {"provider": "ollama", "model": "qwen3:latest"}
    assert sweep_models_match("ollama", "qwen3", record) is True
    assert sweep_models_match("ollama", "QWEN3:latest", record) is True
    assert sweep_models_match("ollama", "gemma3", record) is False
    # Substring is NOT a match rule outside lmstudio.
    assert sweep_models_match("ollama", "qwen", {"model": "qwen3"}) is False


def test_sweep_models_match_lmstudio_substring_and_instance_ids() -> None:
    record = {
        "provider": "lmstudio",
        "model": "qwen/qwen3-vl-4b",
        "provider_instance_ids": ["qwen/qwen3-vl-4b-instance-2"],
    }
    assert sweep_models_match("lmstudio", "qwen3-vl-4b", record) is True
    assert sweep_models_match("lmstudio", "qwen/qwen3-vl-4b", record) is True
    assert sweep_models_match("lmstudio", "other-model", record) is False
    # Explicit instance_ids override the record's own list.
    assert sweep_models_match("lmstudio", "variant-9", {"model": "server-key"}, ["big-variant-9"]) is True
    assert sweep_models_match("lmstudio", "variant-9", record, []) is False


def test_sweep_models_match_empty_inputs_never_match() -> None:
    assert sweep_models_match("ollama", "", {"model": "qwen3"}) is False
    assert sweep_models_match("ollama", "qwen3", {"model": ""}) is False
    assert sweep_models_match("ollama", "qwen3", None) is False
