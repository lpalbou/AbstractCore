"""Light-install smoke check, run INSIDE a fresh venv that holds only `pip install abstractcore`.

tests/install/test_install_settings_resolution.py builds the wheel, creates the venv, installs
the light setting and runs this file with that venv's interpreter. It must not import anything
outside the standard library and abstractcore's own base dependencies.

It proves the light install is complete for remote use: every remote provider is constructed
and answers one request against a local fake server (no real keys, no network), and the built-in
tools, media inputs, HTTP server and capability plugins import.

Exit code 0 and a final line `LIGHT_OK` mean every check passed; any failure is printed and the
exit code is 1.
"""

from __future__ import annotations

import json
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "fake-model"
REPLY = "pong"


class _FakeInferenceServer(BaseHTTPRequestHandler):
    """Answers the OpenAI, Anthropic and Ollama wire shapes the remote providers use."""

    def log_message(self, *args):  # silence
        pass

    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path.endswith("/models"):
            self._send({"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "fake",
                                                    "type": "model", "display_name": MODEL,
                                                    "created_at": "2026-01-01T00:00:00Z"}],
                        "has_more": False, "first_id": MODEL, "last_id": MODEL})
        elif path.endswith("/api/tags"):
            self._send({"models": [{"name": MODEL, "model": MODEL}]})
        elif path.endswith("/api/ps"):
            self._send({"models": []})
        else:
            self._send({})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        path = self.path.split("?", 1)[0]
        if request.get("stream"):
            self.send_response(400)
            self.end_headers()
            return
        if path.endswith("/chat/completions"):
            self._send({
                "id": "chatcmpl-fake", "object": "chat.completion", "created": 0, "model": MODEL,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": REPLY}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })
        elif path.endswith("/messages"):
            self._send({
                "id": "msg_fake", "type": "message", "role": "assistant", "model": MODEL,
                "content": [{"type": "text", "text": REPLY}], "stop_reason": "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
            })
        elif path.endswith("/api/chat"):
            self._send({"model": MODEL, "message": {"role": "assistant", "content": REPLY}, "done": True,
                        "prompt_eval_count": 1, "eval_count": 1})
        elif path.endswith("/api/generate"):
            self._send({"model": MODEL, "response": REPLY, "done": True, "prompt_eval_count": 1, "eval_count": 1})
        else:
            self.send_response(404)
            self.end_headers()


def _providers(root: str) -> dict[str, dict]:
    v1 = f"{root}/v1"
    return {
        "openai": {"base_url": v1, "api_key": "sk-fake"},
        "anthropic": {"base_url": root, "api_key": "sk-ant-fake"},
        "openrouter": {"base_url": v1, "api_key": "sk-or-fake"},
        "portkey": {"base_url": v1, "api_key": "pk-fake", "config_id": "pc-fake"},
        "openai-compatible": {"base_url": v1},
        "lmstudio": {"base_url": v1},
        "vllm": {"base_url": v1},
        "ollama": {"base_url": root},
    }


def main() -> int:
    failures: list[str] = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeInferenceServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    root = f"http://127.0.0.1:{server.server_address[1]}"

    from abstractcore import create_llm

    for name, kwargs in _providers(root).items():
        try:
            llm = create_llm(name, model=MODEL, **kwargs)
            response = llm.generate("ping", max_output_tokens=8)
            content = getattr(response, "content", None)
            if REPLY not in str(content):
                failures.append(f"provider {name}: unexpected reply {content!r}")
            else:
                print(f"provider {name}: ok")
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            failures.append(f"provider {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    # Everything else a light install promises imports without an extra.
    imports = {
        "tools (fetch_url / web_search)": ["requests", "bs4", "lxml", "psutil", "abstractcore.tools.common_tools"],
        "token counting": ["tiktoken"],
        "media inputs": ["PIL.Image", "pypdf", "pandas", "unstructured.partition.docx",
                         "unstructured.partition.xlsx", "unstructured.partition.pptx"],
        "http server": ["fastapi", "uvicorn", "sse_starlette", "multipart", "abstractcore.server.app"],
        "capability plugins": ["abstractvoice", "abstractvision"]
        + (["abstractmusic", "abstract3d"] if sys.version_info >= (3, 10) else []),
    }
    import importlib

    for label, modules in imports.items():
        for module in modules:
            try:
                importlib.import_module(module)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{label}: import {module} failed: {type(exc).__name__}: {exc}")
        print(f"{label}: checked")

    server.shutdown()
    if failures:
        print("LIGHT_FAILED")
        for failure in failures:
            print(" -", failure)
        return 1
    print("LIGHT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
