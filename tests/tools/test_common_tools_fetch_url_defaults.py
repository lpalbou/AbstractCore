from __future__ import annotations

import inspect

import pytest


@pytest.mark.basic
def test_fetch_url_defaults_are_full_content_and_keep_links() -> None:
    from abstractcore.tools.common_tools import fetch_url

    sig = inspect.signature(fetch_url)
    assert sig.parameters["include_full_content"].default is True
    assert sig.parameters["keep_links"].default is True
    assert "max_content_length" not in sig.parameters
    assert "follow_redirects" not in sig.parameters
    assert "extract_links" not in sig.parameters
    assert "convert_html_to_markdown" not in sig.parameters
