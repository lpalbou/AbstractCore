from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from abstractcore.server.app import app


pytestmark = pytest.mark.skipif(
    os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1",
    reason="Live API smoke tests disabled; set ABSTRACTCORE_RUN_LIVE_API_TESTS=1 to run.",
)


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)
    return TestClient(app)


def test_openai_embedding_endpoint_live(client: TestClient) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    resp = client.post(
        "/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "openai/text-embedding-3-small",
            "input": "AbstractCore remote embeddings smoke test.",
            "dimensions": 8,
        },
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert len(data) == 1
    assert len(data[0]["embedding"]) == 8


def test_openai_tts_then_stt_live(client: TestClient) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    speech = client.post(
        "/v1/audio/speech",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "openai/gpt-4o-mini-tts",
            "input": "AbstractCore remote audio smoke test.",
            "voice": "alloy",
            "response_format": "mp3",
        },
    )
    assert speech.status_code == 200, speech.text
    assert speech.content

    transcription = client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("speech.mp3", speech.content, "audio/mpeg")},
        data={"model": "openai/gpt-4o-mini-transcribe"},
    )
    assert transcription.status_code == 200, transcription.text
    assert "abstractcore" in transcription.json()["text"].replace(" ", "").lower()


def test_openrouter_embedding_endpoint_live(client: TestClient) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")

    resp = client.post(
        "/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "openrouter/openai/text-embedding-3-small",
            "input": "AbstractCore OpenRouter embeddings smoke test.",
        },
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert len(data) == 1
    assert len(data[0]["embedding"]) > 0
