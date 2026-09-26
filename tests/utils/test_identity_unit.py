"""The vendored identity descriptor renders complete About facts for every app."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from abstractcore.utils import identity


def test_framework_identity_facts():
    fw = identity.framework_identity()
    assert fw.name == "AbstractFramework"
    assert fw.website == "https://abstractframework.ai"
    assert fw.author == "Laurent-Philippe Albou, PhD"
    assert fw.years == "2023-2026"
    assert fw.license == "MIT"
    assert "2023-2026" in fw.copyright and "MIT" in fw.copyright


@pytest.mark.parametrize("app_id", identity.known_app_ids())
def test_every_known_app_has_every_link(app_id):
    app = identity.app_identity(app_id, version="1.2.3")
    assert app.id == app_id
    assert app.version == "1.2.3"
    for url in (app.website, app.repo, app.docs, app.issues, app.feedback):
        assert url.startswith("https://"), (app_id, url)
    assert app.repo.startswith("https://github.com/lpalbou/")


def test_missing_distribution_is_an_error_not_unknown(monkeypatch):
    import importlib.metadata as md

    def _absent(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(identity.metadata, "version", _absent)
    with pytest.raises(md.PackageNotFoundError):
        identity.app_identity("abstractassistant")


def test_gateway_version_rows_match_the_ui_kit_contract():
    rows = identity.gateway_version_rows(
        {"abstractgateway": "0.5.0", "abstractframework": "0.4.0", "packages": {"abstractruntime": "0.5.0", "abstractcore": "2.16.0", "abstractgateway": "0.5.0", "abstractvoice": None}}
    )
    assert rows == [
        ("Gateway", "AbstractGateway 0.5.0"),
        ("Gateway framework", "AbstractFramework 0.4.0"),
        ("Gateway package abstractcore", "2.16.0"),
        ("Gateway package abstractruntime", "0.5.0"),
    ]
    assert identity.gateway_version_rows({"abstractgateway": "0.5.0"}) == [
        ("Gateway", "AbstractGateway 0.5.0"),
        ("Gateway framework", "not installed on the gateway host"),
    ]
    assert identity.gateway_version_rows(None, error="HTTP 404") == [("Gateway", "unavailable (HTTP 404)")]
    assert identity.gateway_version_rows({"packages": {}}) == [("Gateway", "unavailable (the gateway did not report its version)")]
    assert identity.gateway_version_rows(None, error="  ") == [("Gateway", "unavailable (unknown error)")]
    # Only strings are versions (parity with the ui-kit helper): numbers/booleans are "not reported".
    assert identity.gateway_version_rows({"abstractgateway": 1.0}) == [("Gateway", "unavailable (the gateway did not report its version)")]
    assert identity.gateway_version_rows({"abstractgateway": "0.5.0", "abstractframework": 3, "packages": {"x": True, "y": 2}}) == [
        ("Gateway", "AbstractGateway 0.5.0"),
        ("Gateway framework", "not installed on the gateway host"),
    ]


def test_unknown_app_is_refused():
    with pytest.raises(KeyError):
        identity.app_identity("not-an-app")


def test_about_rows_cover_the_required_facts_and_extra_rows():
    app = identity.app_identity("abstractassistant", version="0.6.0")
    lines = identity.about_lines(app, {"Gateway": "abstractgateway 0.5.0"})
    text = "\n".join(lines)
    assert "Application: AbstractAssistant 0.6.0" in text
    assert "Part of: AbstractFramework — https://abstractframework.ai" in text
    assert "Author: Laurent-Philippe Albou, PhD (2023-2026)" in text
    assert "Copyright: © 2023-2026 Laurent-Philippe Albou, PhD. Released under the MIT License." in text
    assert "Source: https://github.com/lpalbou/AbstractAssistant" in text
    assert "Report an issue: https://github.com/lpalbou/AbstractAssistant/issues" in text
    assert "Give feedback: https://github.com/lpalbou/AbstractAssistant/issues/new?labels=feedback" in text
    assert lines[-1] == "Gateway: abstractgateway 0.5.0"


def test_about_html_links_urls_and_escapes():
    app = identity.app_identity("abstractgateway", version="0.5.0")
    html = identity.about_html(app, {"Note": "<unsafe>"})
    assert '<a href="https://www.lpalbou.info/AbstractGateway/">' in html
    assert 'href="mailto:contact@abstractframework.ai"' in html
    assert "&lt;unsafe&gt;" in html and "<unsafe>" not in html
    # A workflow id is not an e-mail address; the framework URL inside "Part of" is a link.
    html2 = identity.about_html(app, {"Workflow": "basic-agent@0.1.0:main"})
    assert "mailto:basic-agent" not in html2 and "basic-agent@0.1.0:main" in html2
    assert '<a href="https://abstractframework.ai">https://abstractframework.ai</a>' in html2


VENDORED_SHA256 = "2ee5dba4cd15b0f90fe4d71be25b7cbf333496e2838b1520964a649f5e4158f9"


def test_vendored_copy_is_the_reviewed_descriptor():
    """The vendored copy is pinned by content: a change to identity facts must be deliberate (update the pin)."""
    import hashlib
    from importlib import resources

    data = resources.files("abstractcore.assets").joinpath("abstractframework_identity.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == VENDORED_SHA256


def test_vendored_copy_matches_the_canonical_descriptor_when_present():
    canonical = Path(__file__).resolve().parents[3] / "identity" / "abstractframework.json"
    if not canonical.exists():
        pytest.skip("canonical descriptor lives in the AbstractFramework root repo (the sha256 pin above covers CI)")
    assert canonical.read_bytes() == (Path(__file__).resolve().parents[2] / "abstractcore" / "assets" / "abstractframework_identity.json").read_bytes()
