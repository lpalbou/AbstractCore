"""`tests/timing.py`: a wall-clock budget is reported by default and asserted only on request."""
from __future__ import annotations

import pytest

from tests import timing
from tests.timing import PATIENCE, STRICT_ENV, patient, timing_budget


@pytest.fixture(autouse=True)
def _own_report(monkeypatch):
    """These synthetic measures stay out of the session's "timing budgets" summary."""
    monkeypatch.setattr(timing, "REPORT", [])


def test_an_over_budget_measure_is_reported_not_failed_by_default(monkeypatch, capsys):
    monkeypatch.delenv(STRICT_ENV, raising=False)
    assert timing_budget(2.0, 1.0, "slow step") is False
    assert "[timing] slow step: 2.00 s vs limit 1.00 s (OVER)" in capsys.readouterr().err
    assert patient(2.0) == 2.0 * PATIENCE


def test_strict_timing_asserts_the_budget(monkeypatch):
    monkeypatch.setenv(STRICT_ENV, "1")
    assert timing_budget(0.5, 1.0, "fast step") is True
    with pytest.raises(AssertionError, match="slow step"):
        timing_budget(2.0, 1.0, "slow step")
    assert patient(2.0) == 2.0


def test_a_ceiling_is_asserted_even_by_default_and_every_measure_reaches_the_report(monkeypatch):
    monkeypatch.delenv(STRICT_ENV, raising=False)
    timing_budget(2.0, 1.0, "slow but fine", ceiling=10.0)
    assert timing.REPORT == [(False, "[timing] slow but fine: 2.00 s vs limit 1.00 s (OVER)")]
    with pytest.raises(AssertionError, match="past the ceiling"):
        timing_budget(20.0, 1.0, "a blowup", ceiling=10.0)
