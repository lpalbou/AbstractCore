from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


@pytest.mark.integration
def test_fetch_url_local_html_normalization_strips_tags_and_scripts() -> None:
    from abstractcore.tools.common_tools import fetch_url

    html_doc = (
        "<html><head><title>Local</title></head><body>"
        "<nav>Home • About</nav>"
        "<main><h1>Hello</h1><p>Force pod, wave cannon.</p></main>"
        "<script>console.log('noise')</script>"
        "</body></html>"
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/html":
                body = html_doc.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/plain":
                body = html_doc.encode("utf-8")
                self.send_response(200)
                # Intentionally misleading content type.
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            # Silence noisy test output.
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        out_html = fetch_url(f"{base_url}/html", timeout=10, include_full_content=False)
        assert out_html.get("success") is True
        assert "🌐 HTML Document Analysis" in str(out_html.get("rendered") or "")
        norm_html = str(out_html.get("normalized_text") or "")
        assert "<html" not in norm_html.lower()
        assert "console.log" not in norm_html
        assert "Force pod, wave cannon." in norm_html

        out_plain = fetch_url(f"{base_url}/plain", timeout=10, include_full_content=False)
        assert out_plain.get("success") is True
        assert "🌐 HTML Document Analysis" in str(out_plain.get("rendered") or "")
        norm_plain = str(out_plain.get("normalized_text") or "")
        assert "<html" not in norm_plain.lower()
        assert "console.log" not in norm_plain
        assert "Force pod, wave cannon." in norm_plain
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

