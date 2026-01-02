from __future__ import annotations

from typing import Any, Dict, Optional

from .client import McpClient
from .stdio_client import McpStdioClient


def create_mcp_client(*, config: Dict[str, Any], timeout_s: Optional[float] = 30.0) -> Any:
    """Create an MCP client from a persisted config entry.

    Supported config shapes (backwards compatible):
    - HTTP (Streamable HTTP):
        {"url": "https://...", "headers": {...}, "transport": "streamable_http" (optional)}
    - stdio:
        {"transport": "stdio", "command": ["ssh", "host", "..."], "cwd": "...", "env": {...}}
    """
    if not isinstance(config, dict):
        raise ValueError("MCP client config must be a dict")

    transport = str(config.get("transport") or "").strip().lower()
    if not transport:
        transport = "stdio" if "command" in config else "streamable_http"

    if transport in ("stdio",):
        command_raw = config.get("command")
        if isinstance(command_raw, str):
            # Allow power-users to store a single shell string (best-effort).
            command = [command_raw]
        elif isinstance(command_raw, list):
            command = [str(c) for c in command_raw if str(c).strip()]
        else:
            command = []
        if not command:
            raise ValueError("stdio MCP config requires a non-empty 'command' list")

        cwd = config.get("cwd")
        cwd_s = str(cwd).strip() if isinstance(cwd, str) and cwd.strip() else None

        env_raw = config.get("env")
        if isinstance(env_raw, dict):
            env = {
                str(k): str(v)
                for k, v in env_raw.items()
                if isinstance(k, str) and str(k).strip() and v is not None
            }
        else:
            env = None

        return McpStdioClient(command=command, cwd=cwd_s, env=env, timeout_s=timeout_s)

    url = config.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("HTTP MCP config requires a non-empty 'url'")

    headers_raw = config.get("headers")
    if headers_raw is None:
        headers = None
    elif isinstance(headers_raw, dict):
        headers = {str(k): str(v) for k, v in headers_raw.items() if isinstance(k, str) and str(k).strip()}
    else:
        headers = None

    return McpClient(url=url.strip(), headers=headers, timeout_s=timeout_s)
