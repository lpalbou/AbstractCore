"""Telegram tools (send message / send artifact).

These tools are designed to be executed via AbstractRuntime's durable TOOL_CALLS boundary.

Security model:
- Telegram Bot API is *not* end-to-end encrypted (cloud chat; Telegram can decrypt).
- For true E2EE, use TDLib + Secret Chats (see docs/guide/telegram-integration.md).

Dependency policy:
- Bot API transport uses `requests` (install with: pip install "abstractcore[tools]").
- TDLib transport uses stdlib `ctypes` and an externally installed TDLib (tdjson).
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Dict, Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]
    REQUESTS_AVAILABLE = False

from abstractcore.tools.core import tool
from abstractcore.tools.telegram_tdlib import TdlibNotAvailable, get_global_tdlib_client


_ARTIFACT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def _telegram_transport() -> str:
    raw = str(os.getenv("ABSTRACT_TELEGRAM_TRANSPORT", "") or "").strip().lower()
    if raw in {"tdlib", "bot", "bot_api", "botapi"}:
        return "tdlib" if raw == "tdlib" else "bot_api"
    # Default to TDLib to match the "E2EE permanent contact" goal.
    return "tdlib"


def _resolve_required_env(env_var: str, *, label: str) -> tuple[Optional[str], Optional[str]]:
    name = str(env_var or "").strip()
    if not name:
        return None, f"Missing {label} env var name"
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return None, f"Missing env var {name} for {label}"
    return str(value), None


def _artifact_path(artifact_id: str, *, base_dir_env_var: str) -> tuple[Optional[Path], Optional[str]]:
    aid = str(artifact_id or "").strip()
    if not aid:
        return None, "Missing artifact_id"
    if not _ARTIFACT_ID_PATTERN.match(aid):
        return None, "Invalid artifact_id (expected [a-zA-Z0-9_-]+)"

    base = str(os.getenv(base_dir_env_var, "") or "").strip()
    if not base:
        # Common fallback used across hosts.
        base = str(os.getenv("ABSTRACTFLOW_RUNTIME_DIR", "") or "").strip()
    if not base:
        return None, f"Missing artifact store base dir env var ({base_dir_env_var} or ABSTRACTFLOW_RUNTIME_DIR)"

    p = Path(base).expanduser().resolve() / "artifacts" / f"{aid}.bin"
    if not p.exists():
        return None, f"Artifact content not found at {p}"
    return p, None


@tool(
    name="send_telegram_message",
    description="Send a Telegram message to a chat_id. Uses TDLib (Secret Chats) when configured; falls back to Bot API when enabled.",
)
def send_telegram_message(
    *,
    chat_id: int,
    text: str,
    parse_mode: str = "",
    disable_web_page_preview: bool = False,
    timeout_s: float = 20.0,
    bot_token_env_var: str = "ABSTRACT_TELEGRAM_BOT_TOKEN",
) -> Dict[str, Any]:
    transport = _telegram_transport()

    if transport == "bot_api":
        if not REQUESTS_AVAILABLE:
            return {
                "success": False,
                "transport": "bot_api",
                "error": "requests is required for Telegram Bot API transport. Install with: pip install \"abstractcore[tools]\"",
            }

        token, err = _resolve_required_env(bot_token_env_var, label="Telegram bot token")
        if err:
            return {"success": False, "transport": "bot_api", "error": err}

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": str(text or ""),
        }
        if isinstance(parse_mode, str) and parse_mode.strip():
            payload["parse_mode"] = parse_mode.strip()
        if disable_web_page_preview:
            payload["disable_web_page_preview"] = True

        try:
            resp = requests.post(url, json=payload, timeout=float(timeout_s))  # type: ignore[union-attr]
        except Exception as e:
            return {"success": False, "transport": "bot_api", "error": str(e)}

        try:
            body = resp.json()
        except Exception:
            body = None

        if resp.status_code >= 400:
            return {"success": False, "transport": "bot_api", "error": f"HTTP {resp.status_code}", "response": body}

        if isinstance(body, dict) and body.get("ok") is False:
            return {"success": False, "transport": "bot_api", "error": str(body.get("description") or "Telegram error"), "response": body}

        return {"success": True, "transport": "bot_api", "response": body}

    # TDLib (preferred for E2EE Secret Chats)
    try:
        client = get_global_tdlib_client(start=True)
    except (TdlibNotAvailable, ValueError) as e:
        return {"success": False, "transport": "tdlib", "error": str(e)}

    if not client.wait_until_ready(timeout_s=10.0):
        err = client.last_error or "TDLib client not ready (authorization incomplete)"
        return {"success": False, "transport": "tdlib", "error": err}

    req: Dict[str, Any] = {
        "@type": "sendMessage",
        "chat_id": int(chat_id),
        "input_message_content": {
            "@type": "inputMessageText",
            "text": {"@type": "formattedText", "text": str(text or "")},
            "disable_web_page_preview": bool(disable_web_page_preview),
        },
    }
    try:
        out = client.request(req, timeout_s=float(timeout_s))
    except TimeoutError:
        # Best-effort: TDLib may still send later; keep tool durable.
        return {"success": True, "transport": "tdlib", "queued": True}
    except Exception as e:
        return {"success": False, "transport": "tdlib", "error": str(e)}

    if isinstance(out, dict) and out.get("@type") == "error":
        return {"success": False, "transport": "tdlib", "error": str(out.get("message") or "TDLib error"), "response": out}
    return {"success": True, "transport": "tdlib", "response": out}


@tool(
    name="send_telegram_artifact",
    description="Send an artifact (stored under <artifact_store>/artifacts/<artifact_id>.bin) to a Telegram chat_id as a document/photo.",
)
def send_telegram_artifact(
    *,
    chat_id: int,
    artifact_id: str,
    caption: str = "",
    filename: str = "",
    as_photo: bool = False,
    artifact_base_dir_env_var: str = "ABSTRACTGATEWAY_DATA_DIR",
    timeout_s: float = 60.0,
    bot_token_env_var: str = "ABSTRACT_TELEGRAM_BOT_TOKEN",
) -> Dict[str, Any]:
    transport = _telegram_transport()

    path, err = _artifact_path(artifact_id, base_dir_env_var=artifact_base_dir_env_var)
    if err:
        return {"success": False, "error": err}

    if transport == "bot_api":
        if not REQUESTS_AVAILABLE:
            return {
                "success": False,
                "transport": "bot_api",
                "error": "requests is required for Telegram Bot API transport. Install with: pip install \"abstractcore[tools]\"",
            }

        token, err2 = _resolve_required_env(bot_token_env_var, label="Telegram bot token")
        if err2:
            return {"success": False, "transport": "bot_api", "error": err2}

        endpoint = "sendPhoto" if as_photo else "sendDocument"
        url = f"https://api.telegram.org/bot{token}/{endpoint}"

        name = str(filename or "").strip() or path.name
        field = "photo" if as_photo else "document"

        try:
            with open(path, "rb") as f:
                files = {field: (name, f)}
                data: Dict[str, Any] = {"chat_id": str(int(chat_id))}
                if caption:
                    data["caption"] = str(caption)
                resp = requests.post(url, data=data, files=files, timeout=float(timeout_s))  # type: ignore[union-attr]
        except Exception as e:
            return {"success": False, "transport": "bot_api", "error": str(e)}

        try:
            body = resp.json()
        except Exception:
            body = None

        if resp.status_code >= 400:
            return {"success": False, "transport": "bot_api", "error": f"HTTP {resp.status_code}", "response": body}
        if isinstance(body, dict) and body.get("ok") is False:
            return {"success": False, "transport": "bot_api", "error": str(body.get("description") or "Telegram error"), "response": body}
        return {"success": True, "transport": "bot_api", "response": body}

    # TDLib send (Secret Chat compatible)
    try:
        client = get_global_tdlib_client(start=True)
    except (TdlibNotAvailable, ValueError) as e:
        return {"success": False, "transport": "tdlib", "error": str(e)}

    if not client.wait_until_ready(timeout_s=10.0):
        err3 = client.last_error or "TDLib client not ready (authorization incomplete)"
        return {"success": False, "transport": "tdlib", "error": err3}

    caption_text = str(caption or "")
    caption_obj = {"@type": "formattedText", "text": caption_text} if caption_text else {"@type": "formattedText", "text": ""}

    if as_photo:
        input_content = {
            "@type": "inputMessagePhoto",
            "photo": {"@type": "inputFileLocal", "path": str(path)},
            "caption": caption_obj,
        }
    else:
        input_content = {
            "@type": "inputMessageDocument",
            "document": {"@type": "inputFileLocal", "path": str(path)},
            "caption": caption_obj,
        }

    req2: Dict[str, Any] = {
        "@type": "sendMessage",
        "chat_id": int(chat_id),
        "input_message_content": input_content,
    }

    try:
        out = client.request(req2, timeout_s=float(timeout_s))
    except TimeoutError:
        return {"success": True, "transport": "tdlib", "queued": True}
    except Exception as e:
        return {"success": False, "transport": "tdlib", "error": str(e)}

    if isinstance(out, dict) and out.get("@type") == "error":
        return {"success": False, "transport": "tdlib", "error": str(out.get("message") or "TDLib error"), "response": out}
    return {"success": True, "transport": "tdlib", "response": out}
