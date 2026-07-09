from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.basic


def test_native_pdf_config_uses_openai_env_without_enable_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractcore.media.pdf_routing as pdf_routing

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_FETCH_URL_PDF_NATIVE_MODEL", raising=False)

    config = pdf_routing._native_pdf_config_from_env()

    assert config is not None
    assert config.model == "gpt-4.1-mini"
    assert config.api_key == "sk-test"


def test_route_pdf_bytes_auto_prefers_native_summary_and_local_text(monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractcore.media.pdf_routing as pdf_routing

    config = pdf_routing.NativePDFConfig(
        model="gpt-4.1-mini",
        api_key="test-key",
        base_url=None,
        max_pages=4,
        max_bytes=1_000_000,
        timeout_s=60,
    )

    monkeypatch.setattr(pdf_routing, "_peek_pdf_metadata", lambda _b: {"page_count": 2, "title": "Quarterly Metrics", "warnings": []})
    monkeypatch.setattr(pdf_routing, "_native_pdf_config_from_env", lambda: config)
    monkeypatch.setattr(
        pdf_routing,
        "_call_native_pdf_model",
        lambda *args, **kwargs: {
            "backend": "native_llm",
            "title": "Quarterly Metrics",
            "summary": "Native summary",
            "text_preview": "Native preview",
            "page_count_detected": 2,
            "key_facts": ["Region A revenue: $1.2M"],
            "warnings": [],
            "model": "gpt-4.1-mini",
            "base_url": "",
        },
    )
    monkeypatch.setattr(pdf_routing, "_local_backend_available", lambda backend: backend == "pymupdf")
    monkeypatch.setattr(
        pdf_routing,
        "_extract_local_pdf_bytes",
        lambda *args, **kwargs: {
            "backend": "pymupdf",
            "content": "Structured local text",
            "metadata": {"page_count": 2},
        },
    )

    out = pdf_routing.route_pdf_bytes(b"%PDF-1.4\nfake", source_url="https://example.com/report.pdf")

    assert out["summary_backend"] == "native_llm"
    assert out["text_backend"] == "pymupdf"
    assert out["native_used"] is True
    assert out["raw_text"] == "Structured local text"
    assert any(item.get("backend") == "native_llm" and item.get("status") == "used" for item in out["backend_attempts"])


def test_route_pdf_bytes_uses_native_preview_when_local_text_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractcore.media.pdf_routing as pdf_routing

    config = pdf_routing.NativePDFConfig(
        model="gpt-4.1-mini",
        api_key="test-key",
        base_url=None,
        max_pages=4,
        max_bytes=1_000_000,
        timeout_s=60,
    )

    monkeypatch.setattr(pdf_routing, "_peek_pdf_metadata", lambda _b: {"page_count": 2, "title": "", "warnings": []})
    monkeypatch.setattr(pdf_routing, "_native_pdf_config_from_env", lambda: config)
    monkeypatch.setattr(
        pdf_routing,
        "_call_native_pdf_model",
        lambda *args, **kwargs: {
            "backend": "native_llm",
            "title": "Scanned Invoice",
            "summary": "OCR-like native summary",
            "text_preview": "Invoice Number: INV-2048\nAmount Due: $482.17",
            "page_count_detected": 2,
            "key_facts": [],
            "warnings": [],
            "model": "gpt-4.1-mini",
            "base_url": "",
        },
    )
    monkeypatch.setattr(pdf_routing, "_local_backend_available", lambda backend: backend == "pypdf")
    monkeypatch.setattr(
        pdf_routing,
        "_extract_local_pdf_bytes",
        lambda *args, **kwargs: {
            "backend": "pypdf",
            "content": "![](scan-0-full.png)\n![](scan-1-full.png)",
            "metadata": {"page_count": 2},
        },
    )

    out = pdf_routing.route_pdf_bytes(b"%PDF-1.4\nfake", source_url="https://example.com/scan.pdf")

    assert out["text_backend"] == "native_llm"
    assert out["raw_text"] == "Invoice Number: INV-2048\nAmount Due: $482.17"
    assert out["degraded"] is True
    assert "native PDF preview text as fallback evidence" in str(out["rendered"])


def test_route_pdf_bytes_skips_native_when_pdf_is_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractcore.media.pdf_routing as pdf_routing

    config = pdf_routing.NativePDFConfig(
        model="gpt-4.1-mini",
        api_key="test-key",
        base_url=None,
        max_pages=1,
        max_bytes=100,
        timeout_s=60,
    )

    monkeypatch.setattr(pdf_routing, "_peek_pdf_metadata", lambda _b: {"page_count": 3, "title": "", "warnings": []})
    monkeypatch.setattr(pdf_routing, "_native_pdf_config_from_env", lambda: config)
    monkeypatch.setattr(pdf_routing, "_local_backend_available", lambda backend: backend == "pypdf")
    monkeypatch.setattr(
        pdf_routing,
        "_extract_local_pdf_bytes",
        lambda *args, **kwargs: {
            "backend": "pypdf",
            "content": "Fallback text",
            "metadata": {"page_count": 3},
        },
    )

    out = pdf_routing.route_pdf_bytes(b"%PDF-1.4\n" + (b"x" * 500), source_url="https://example.com/report.pdf")

    assert out["native_used"] is False
    assert any(item.get("backend") == "native_llm" and item.get("status") == "skipped" for item in out["backend_attempts"])
    assert out["raw_text"] == "Fallback text"


def test_call_native_pdf_model_falls_back_to_data_url_when_file_id_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import abstractcore.media.pdf_routing as pdf_routing

    class FakeFilesAPI:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        def create(self, *, file, purpose: str):
            assert purpose == "user_data"
            assert file.read().startswith(b"%PDF-1.4")
            file.seek(0)
            return SimpleNamespace(id="file_123")

        def delete(self, file_id: str) -> None:
            self.deleted.append(file_id)

    class FakeResponsesAPI:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, *, model: str, input):
            self.calls.append({"model": model, "input": input})
            content = input[0]["content"]
            file_part = content[1] if content and content[0].get("type") == "input_text" else content[0]
            if "file_id" in file_part:
                raise RuntimeError("File `tiny.pdf` with content type `application/pdf` is not supported by the subscription backend.")
            assert file_part["file_data"].startswith("data:application/pdf;base64,")
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "title": "Tiny PDF Test",
                        "summary": "Summary from data URL fallback",
                        "text_preview": "Tiny PDF Test",
                        "page_count_detected": 1,
                        "key_facts": ["Tiny PDF Test"],
                        "warnings": [],
                    }
                )
            )

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.files = FakeFilesAPI()
            self.responses = FakeResponsesAPI()

    monkeypatch.setattr(pdf_routing, "_module_available", lambda name: name == "openai")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeClient))

    config = pdf_routing.NativePDFConfig(
        model="gpt-5.4",
        api_key="EMPTY",
        base_url="http://localhost:8090/v1",
        max_pages=4,
        max_bytes=1_000_000,
        timeout_s=60,
    )

    result = pdf_routing._call_native_pdf_model(
        b"%PDF-1.4\nfake\n%%EOF\n",
        source_url="https://example.com/tiny.pdf",
        source_name="tiny.pdf",
        include_full_content=False,
        config=config,
    )

    assert result["backend"] == "native_llm"
    assert result["transport"] == "data_url"
    assert result["title"] == "Tiny PDF Test"
