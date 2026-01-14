from __future__ import annotations

import inspect

import pytest


@pytest.mark.basic
def test_fetch_url_defaults_are_full_content_and_no_link_noise() -> None:
    from abstractcore.tools.common_tools import fetch_url

    sig = inspect.signature(fetch_url)
    assert sig.parameters["include_full_content"].default is True
    assert sig.parameters["extract_links"].default is False
    assert sig.parameters["convert_html_to_markdown"].default is True
    assert sig.parameters["keep_links"].default is False
