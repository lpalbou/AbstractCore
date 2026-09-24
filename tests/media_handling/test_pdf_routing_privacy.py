"""Remote PDF extraction is an explicit operator opt-in (mission EE, 2026-09-24).

Before: `route_pdf_bytes(..., preferred_backend="auto")` uploaded the document
to OpenAI whenever OPENAI_API_KEY was set. Now only the setting
`offline.allow_remote_pdf_extraction` (`abstractcore --allow-remote-pdf-extraction`)
lets a PDF leave the machine.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.basic


def _scratch_config(monkeypatch: pytest.MonkeyPatch, tmp_path, *, allow: bool):
    import abstractcore.config.manager as manager_module
    from abstractcore.config.manager import ConfigurationManager

    cfg = ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)
    if allow:
        assert cfg.set_allow_remote_pdf_extraction(True)
    monkeypatch.setattr(manager_module, "_config_manager", cfg)
    return cfg


def test_setting_defaults_off_and_round_trips(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from abstractcore.config.manager import ConfigurationManager

    cfg = _scratch_config(monkeypatch, tmp_path, allow=False)
    assert cfg.is_remote_pdf_extraction_allowed() is False
    assert cfg.set_allow_remote_pdf_extraction(True)
    reread = ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)
    assert reread.is_remote_pdf_extraction_allowed() is True


@pytest.mark.parametrize("backend", ["auto", "native_llm"])
def test_key_without_opt_in_never_calls_the_remote_model(monkeypatch: pytest.MonkeyPatch, tmp_path, backend) -> None:
    import abstractcore.media.pdf_routing as pdf_routing

    _scratch_config(monkeypatch, tmp_path, allow=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    called = []
    monkeypatch.setattr(pdf_routing, "_call_native_pdf_model", lambda *a, **k: called.append(1) or {})
    monkeypatch.setattr(pdf_routing, "_local_backend_available", lambda b: b == "pypdf")
    monkeypatch.setattr(
        pdf_routing,
        "_extract_local_pdf_bytes",
        lambda *a, **k: {"backend": "pypdf", "content": "Local text of the private document", "metadata": {}},
    )

    out = pdf_routing.route_pdf_bytes(b"%PDF-1.4\nfake", preferred_backend=backend)

    assert called == []
    assert out["native_used"] is False
    assert out["native_available"] is False
    assert out["remote_extraction_enabled"] is False
    assert out["text_backend"] == "pypdf"
    assert {"backend": "native_llm", "status": "skipped", "reason": "remote_extraction_disabled"} in out["backend_attempts"]


def test_opt_in_setting_enables_the_remote_model(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import abstractcore.media.pdf_routing as pdf_routing

    _scratch_config(monkeypatch, tmp_path, allow=True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(pdf_routing, "_peek_pdf_metadata", lambda _b: {"page_count": 1, "title": "", "warnings": []})
    monkeypatch.setattr(
        pdf_routing,
        "_call_native_pdf_model",
        lambda *a, **k: {"backend": "native_llm", "summary": "S", "text_preview": "", "model": "m", "base_url": ""},
    )
    monkeypatch.setattr(pdf_routing, "_local_backend_available", lambda b: b == "pypdf")
    monkeypatch.setattr(
        pdf_routing,
        "_extract_local_pdf_bytes",
        lambda *a, **k: {"backend": "pypdf", "content": "Local text of the document", "metadata": {}},
    )

    out = pdf_routing.route_pdf_bytes(b"%PDF-1.4\nfake")

    assert out["remote_extraction_enabled"] is True
    assert out["native_used"] is True
    assert out["summary_backend"] == "native_llm"
    assert out["text_backend"] == "pypdf"


def test_config_failure_means_local_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractcore.config as config_pkg
    import abstractcore.media.pdf_routing as pdf_routing

    def boom():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr(config_pkg, "get_config_manager", boom)
    assert pdf_routing._remote_pdf_extraction_opted_in() is False
