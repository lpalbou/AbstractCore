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


def test_vendored_copy_matches_the_canonical_descriptor_when_present():
    canonical = Path(__file__).resolve().parents[3] / "identity" / "abstractframework.json"
    if not canonical.exists():
        pytest.skip("canonical descriptor lives in the AbstractFramework root repo")
    assert json.loads(canonical.read_text(encoding="utf-8")) == identity._descriptor()
