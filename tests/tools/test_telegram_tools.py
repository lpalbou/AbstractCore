from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_send_telegram_message_bot_api_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.telegram_tools import send_telegram_message

    monkeypatch.setenv("ABSTRACT_TELEGRAM_TRANSPORT", "bot_api")
    monkeypatch.delenv("ABSTRACT_TELEGRAM_BOT_TOKEN", raising=False)

    out = send_telegram_message(chat_id=1, text="hi")
    assert out["success"] is False
    assert out["transport"] == "bot_api"
    assert "Missing env var" in str(out.get("error") or "")


def test_send_telegram_message_bot_api_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.telegram_tools import send_telegram_message

    monkeypatch.setenv("ABSTRACT_TELEGRAM_TRANSPORT", "bot_api")
    monkeypatch.setenv("ABSTRACT_TELEGRAM_BOT_TOKEN", "123:token")

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"ok": True, "result": {"message_id": 1}}

    with patch("requests.post", return_value=resp) as post:
        out = send_telegram_message(chat_id=42, text="hello", parse_mode="Markdown")

    assert out["success"] is True
    assert out["transport"] == "bot_api"
    post.assert_called_once()
    args, kwargs = post.call_args
    assert "bot123:token/sendMessage" in str(args[0])
    assert kwargs["json"]["chat_id"] == 42
    assert kwargs["json"]["text"] == "hello"
    assert kwargs["json"]["parse_mode"] == "Markdown"


def test_send_telegram_message_tdlib_missing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.telegram_tools import send_telegram_message

    monkeypatch.setenv("ABSTRACT_TELEGRAM_TRANSPORT", "tdlib")
    monkeypatch.delenv("ABSTRACT_TELEGRAM_API_ID", raising=False)
    monkeypatch.delenv("ABSTRACT_TELEGRAM_API_HASH", raising=False)
    monkeypatch.delenv("ABSTRACT_TELEGRAM_PHONE_NUMBER", raising=False)
    monkeypatch.delenv("ABSTRACT_TELEGRAM_DB_DIR", raising=False)

    out = send_telegram_message(chat_id=1, text="hi")
    assert out["success"] is False
    assert out["transport"] == "tdlib"
    assert "Missing env var" in str(out.get("error") or "")


def test_send_telegram_artifact_missing_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from abstractcore.tools.telegram_tools import send_telegram_artifact

    base = tmp_path / "artifacts_base"
    base.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TEST_ART_BASE", str(base))

    out = send_telegram_artifact(chat_id=1, artifact_id="abc123", artifact_base_dir_env_var="TEST_ART_BASE")
    assert out["success"] is False
    assert "not found" in str(out.get("error") or "").lower()


def test_send_telegram_artifact_bot_api_success(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from abstractcore.tools.telegram_tools import send_telegram_artifact

    base = tmp_path / "artifacts_base"
    base.mkdir(parents=True, exist_ok=True)
    artifacts_dir = base / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "abc123.bin").write_bytes(b"hello")

    monkeypatch.setenv("ABSTRACT_TELEGRAM_TRANSPORT", "bot_api")
    monkeypatch.setenv("ABSTRACT_TELEGRAM_BOT_TOKEN", "123:token")
    monkeypatch.setenv("TEST_ART_BASE", str(base))

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"ok": True, "result": {"message_id": 2}}

    with patch("requests.post", return_value=resp) as post:
        out = send_telegram_artifact(
            chat_id=42,
            artifact_id="abc123",
            caption="file",
            artifact_base_dir_env_var="TEST_ART_BASE",
        )

    assert out["success"] is True
    assert out["transport"] == "bot_api"
    post.assert_called_once()
