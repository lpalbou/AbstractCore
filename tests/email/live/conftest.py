"""Fixtures and report scrubbing for the opt-in live email tests (see live_email_support.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from live_email_support import LiveMailbox, SmtpSelfOnlyGuard, load_live_mailbox, scrub, skip_reason  # noqa: E402

_MAILBOX, _MISSING = load_live_mailbox()


@pytest.fixture(scope="session")
def live_mailbox() -> LiveMailbox:
    if _MAILBOX is None:
        pytest.skip(skip_reason(_MISSING))
    return _MAILBOX


@pytest.fixture(scope="session")
def live_state() -> Dict[str, Any]:
    """Plain data shared between the tests of one run (no connections are opened here)."""
    return {}


@pytest.fixture(autouse=True)
def smtp_self_only(request, monkeypatch) -> Any:
    """Every live test: SMTP may only reach the test account's own address."""
    if _MAILBOX is None:
        yield None
        return
    yield SmtpSelfOnlyGuard(_MAILBOX).install(monkeypatch)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Last line of defence: no credential value survives into a report or captured output."""
    outcome = yield
    report = outcome.get_result()
    if _MAILBOX is None:
        return
    if report.failed and report.longrepr is not None:
        report.longrepr = scrub(report.longrepr, _MAILBOX)
    report.sections = [(name, scrub(content, _MAILBOX)) for name, content in report.sections]
