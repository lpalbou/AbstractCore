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


def test_send_telegram_message_bot_api_splits_long_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.telegram_tools import send_telegram_message

    monkeypatch.setenv("ABSTRACT_TELEGRAM_TRANSPORT", "bot_api")
    monkeypatch.setenv("ABSTRACT_TELEGRAM_BOT_TOKEN", "123:token")

    def _resp(mid: int):
        r = MagicMock()
        r.status_code = 200
        r.json.return_value = {"ok": True, "result": {"message_id": int(mid)}}
        return r

    long_text = "a" * 8000
    with patch("requests.post", side_effect=[_resp(1), _resp(2), _resp(3)]) as post:
        out = send_telegram_message(chat_id=42, text=long_text)

    assert out["success"] is True
    assert out["transport"] == "bot_api"
    assert out.get("parts") == 3
    assert out.get("message_ids") == [1, 2, 3]
    assert post.call_count == 3
    for call in post.call_args_list:
        payload = call.kwargs.get("json") or {}
        assert payload.get("chat_id") == 42
        assert isinstance(payload.get("text"), str)
        assert 0 < len(payload["text"]) <= 3800


def test_send_telegram_message_bot_api_retries_without_parse_mode_on_entity_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.telegram_tools import send_telegram_message

    monkeypatch.setenv("ABSTRACT_TELEGRAM_TRANSPORT", "bot_api")
    monkeypatch.setenv("ABSTRACT_TELEGRAM_BOT_TOKEN", "123:token")

    resp_err = MagicMock()
    resp_err.status_code = 400
    resp_err.json.return_value = {
        "ok": False,
        "description": "Bad Request: can't parse entities: Can't find end of the entity starting at byte offset 10",
    }

    resp_ok = MagicMock()
    resp_ok.status_code = 200
    resp_ok.json.return_value = {"ok": True, "result": {"message_id": 1}}

    with patch("requests.post", side_effect=[resp_err, resp_ok]) as post:
        out = send_telegram_message(chat_id=42, text="hello", parse_mode="Markdown")

    assert out["success"] is True
    assert post.call_count == 2
    first_payload = post.call_args_list[0].kwargs["json"]
    second_payload = post.call_args_list[1].kwargs["json"]
    assert first_payload.get("parse_mode") == "Markdown"
    assert "parse_mode" not in second_payload


def test_send_telegram_message_bot_api_empty_text_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.telegram_tools import send_telegram_message

    monkeypatch.setenv("ABSTRACT_TELEGRAM_TRANSPORT", "bot_api")
    monkeypatch.setenv("ABSTRACT_TELEGRAM_BOT_TOKEN", "123:token")

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"ok": True, "result": {"message_id": 1}}

    with patch("requests.post", return_value=resp) as post:
        out = send_telegram_message(chat_id=42, text="   ")

    assert out["success"] is True
    assert out.get("was_empty_input") is True
    payload = post.call_args.kwargs["json"]
    assert isinstance(payload.get("text"), str)
    assert payload["text"].strip()


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
