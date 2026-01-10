#!/usr/bin/env python3
"""
Lightweight smoke tests for provider availability.

These tests are intentionally opt-in to avoid:
- accidental cloud usage (cost)
- flaky CI environments without network/localhost access
"""

import pytest
import os
from abstractcore import create_llm


def _is_connectivity_error(err: Exception) -> bool:
    msg = str(err).lower()
    return any(
        keyword in msg
        for keyword in (
            "connection error",
            "connecterror",
            "connection refused",
            "operation not permitted",
            "network is unreachable",
            "nodename nor servname provided",
            "timeout",
        )
    )


def test_anthropic_generation_smoke():
    """Test Anthropic provider with a current Haiku model."""
    if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
        pytest.skip("Live API test; set ABSTRACTCORE_RUN_LIVE_API_TESTS=1 to run")
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    llm = create_llm("anthropic", model="claude-haiku-4-5", timeout=30.0)
    try:
        resp = llm.generate("Say 'ok' and nothing else.", max_output_tokens=10)
    except Exception as e:
        if _is_connectivity_error(e):
            pytest.skip(f"Anthropic not reachable in this environment: {e}")
        raise

    assert resp is not None
    assert isinstance(resp.content, str) and resp.content.strip()


def test_openai_generation_smoke():
    """Test OpenAI provider with a current GPT-5 mini model."""
    if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
        pytest.skip("Live API test; set ABSTRACTCORE_RUN_LIVE_API_TESTS=1 to run")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    llm = create_llm("openai", model="gpt-5-mini", timeout=30.0)
    try:
        resp = llm.generate("Say 'ok' and nothing else.", max_output_tokens=64)
        # Reasoning models can consume small token budgets entirely in hidden reasoning tokens and return
        # empty visible content. Retry once with a larger budget to keep this smoke test stable.
        if not (isinstance(resp.content, str) and resp.content.strip()):
            resp = llm.generate("Say 'ok' and nothing else.", max_output_tokens=512)
    except Exception as e:
        if _is_connectivity_error(e):
            pytest.skip(f"OpenAI not reachable in this environment: {e}")
        raise

    assert resp is not None
    assert isinstance(resp.content, str) and resp.content.strip()


def test_openrouter_generation_smoke():
    """Test OpenRouter provider via OpenAI-compatible API (opt-in)."""
    if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
        pytest.skip("Live API test; set ABSTRACTCORE_RUN_LIVE_API_TESTS=1 to run")
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    llm = create_llm("openrouter", model="openai/gpt-4o-mini", timeout=30.0)
    try:
        resp = llm.generate("Say 'ok' and nothing else.", max_output_tokens=16)
    except Exception as e:
        if _is_connectivity_error(e):
            pytest.skip(f"OpenRouter not reachable in this environment: {e}")
        raise

    assert resp is not None
    assert isinstance(resp.content, str) and resp.content.strip()


def test_ollama_generation_smoke():
    """Test Ollama provider with a lightweight local model."""
    if os.getenv("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS") != "1":
        pytest.skip("Local provider tests disabled (set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1)")
    try:
        llm = create_llm("ollama", model="qwen3:4b-instruct", base_url="http://localhost:11434", timeout=10.0)
        resp = llm.generate("Say 'ok' and nothing else.", max_output_tokens=10)
    except Exception as e:
        if _is_connectivity_error(e):
            pytest.skip("Ollama not reachable in this environment")
        raise

    assert resp is not None
    assert isinstance(resp.content, str) and resp.content.strip()


def test_mlx_generation_smoke():
    """Test MLX provider with a small local model."""
    if os.getenv("ABSTRACTCORE_RUN_MLX_TESTS") != "1":
        pytest.skip("MLX test is heavy; set ABSTRACTCORE_RUN_MLX_TESTS=1 to run")
    try:
        llm = create_llm("mlx", model="mlx-community/Qwen3-4B-4bit", timeout=30.0)
        resp = llm.generate("Say 'ok' and nothing else.", max_output_tokens=10)
    except Exception as e:
        if any(keyword in str(e).lower() for keyword in ["mlx", "import", "not installed", "not found", "failed to load"]):
            pytest.skip(f"MLX not available: {e}")
        raise

    assert resp is not None
    assert isinstance(resp.content, str) and resp.content.strip()


if __name__ == "__main__":
    print("🧪 Running provider smoke tests (opt-in)...")
    print()

    test_anthropic_generation_smoke()
    test_openai_generation_smoke()
    test_ollama_generation_smoke()
    test_mlx_generation_smoke()

    print()
    print("✅ All provider smoke tests completed!")
