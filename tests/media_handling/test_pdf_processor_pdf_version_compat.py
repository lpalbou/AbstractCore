from __future__ import annotations

from abstractcore.media.processors.pdf_processor import _safe_pdf_version


def test_safe_pdf_version_missing_attribute_returns_none() -> None:
    class DummyDoc:
        pass

    assert _safe_pdf_version(DummyDoc()) is None


def test_safe_pdf_version_callable_returns_string() -> None:
    class DummyDoc:
        def pdf_version(self):  # type: ignore[no-untyped-def]
            return "1.7"

    assert _safe_pdf_version(DummyDoc()) == "1.7"


def test_safe_pdf_version_property_returns_string() -> None:
    class DummyDoc:
        pdf_version = "1.4"

    assert _safe_pdf_version(DummyDoc()) == "1.4"


def test_safe_pdf_version_falls_back_to_metadata_format() -> None:
    class DummyDoc:
        metadata = {"format": "PDF 1.5"}

    assert _safe_pdf_version(DummyDoc()) == "1.5"
