"""Debug output must let you TELL TWO RUNS APART.

Written from an operator report: two `abstractcore-chat` sessions, one with
`--speculation native_mtp` and one without, produced visually identical debug
output (`Response in 8128ms` vs `Response in 4063ms`) with nothing saying
whether the drafter ran. Worse, the slower-looking line was the accelerated
one -- because the two runs generated different numbers of tokens, and raw
milliseconds have no denominator.

These pin the two properties that fix that:
  1. timing carries tokens and tok/s, not just elapsed wall clock;
  2. every turn says whether the drafter actually ran, in BOTH the sync and
     streaming lanes (streaming yields bare content chunks with no usage or
     metadata, so it has to ask the provider).

Hermetic: no models, no MLX, no network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from abstractcore.utils.cli import SimpleCLI


def _cli(provider: Any = None) -> SimpleCLI:
    """A SimpleCLI with only the fields the debug helpers read."""
    cli = SimpleCLI.__new__(SimpleCLI)
    cli.provider = provider
    cli._last_stream_text = None
    return cli


def _response(usage: Optional[Dict[str, Any]] = None, metadata=None) -> Any:
    return SimpleNamespace(usage=usage, metadata=metadata, content="x")


# ---------------------------------------------------------------------------
# Timing needs a denominator
# ---------------------------------------------------------------------------


def test_timing_reports_tokens_and_rate_not_just_milliseconds():
    line = _cli()._debug_timing_line(
        _response({"output_tokens": 400, "input_tokens": 47}), 10_000.0
    )
    assert "400 out" in line and "47 in" in line
    assert "40.0 tok/s" in line, (
        f"no rate in {line!r} -- raw milliseconds cannot be compared across "
        "runs that generated different numbers of tokens, which is exactly how "
        "an accelerated run looked slower than an unaccelerated one"
    )


def test_timing_falls_back_to_counting_streamed_text():
    """Streamed chunks carry no usage; the text is still countable."""
    provider = SimpleNamespace(_count_tokens=lambda text: 123)
    cli = _cli(provider)
    cli._last_stream_text = "some streamed answer"
    line = cli._debug_timing_line(_response(usage=None), 1_000.0)
    assert "123 out" in line and "123.0 tok/s" in line


def test_timing_degrades_to_bare_latency_when_nothing_can_be_counted():
    line = _cli()._debug_timing_line(_response(usage=None), 250.0)
    assert line.strip().endswith("250ms")
    assert "tok/s" not in line, "must not invent a rate it cannot compute"


# ---------------------------------------------------------------------------
# Every turn says whether the drafter ran
# ---------------------------------------------------------------------------


def test_metadata_used_is_reported_with_the_drafter():
    line = _cli()._debug_speculation_line(
        _response(metadata={"speculation": {"used": True, "drafter": "org/d", "num_draft_tokens": 3}})
    )
    assert "USED" in line and "org/d" in line and "draft_tokens=3" in line


def test_metadata_requested_but_unused_names_the_reason():
    line = _cli()._debug_speculation_line(
        _response(metadata={"speculation": {"used": False, "reason": "mlx_vlm_missing"}})
    )
    assert "NOT used" in line and "mlx_vlm_missing" in line


def test_output_preserving_false_is_surfaced():
    """A drafter that changes the text must say so where the user is looking."""
    line = _cli()._debug_speculation_line(
        _response(metadata={"speculation": {"used": True, "drafter": "org/d",
                                            "output_preserving": False}})
    )
    assert "output_preserving=False" in line


def test_streaming_asks_the_provider_rather_than_claiming_unused():
    """The regression this file exists for.

    With no metadata (the streaming lane), an earlier version printed
    "lane loaded, not used on this call" even when the drafter HAD run.
    """
    provider = SimpleNamespace(
        speculation_status=lambda: {
            "lane_loaded": True, "drafter": "org/d", "draft_tokens": 2,
            "output_preserving": None, "last_call_used": True,
            "unavailable_reason": None,
        }
    )
    line = _cli(provider)._debug_speculation_line(_response(metadata=None))
    assert "USED" in line and "org/d" in line, (
        f"streaming turn reported {line!r} while the drafter actually ran"
    )


def test_streaming_reports_a_genuinely_skipped_drafter_as_skipped():
    provider = SimpleNamespace(
        speculation_status=lambda: {
            "lane_loaded": True, "drafter": "org/d", "draft_tokens": 2,
            "output_preserving": None, "last_call_used": False,
            "unavailable_reason": None,
        }
    )
    line = _cli(provider)._debug_speculation_line(_response(metadata=None))
    assert "SKIPPED" in line


def test_nothing_is_claimed_when_no_lane_and_nothing_requested():
    provider = SimpleNamespace(
        speculation_status=lambda: {
            "lane_loaded": False, "unavailable_reason": None,
        }
    )
    assert _cli(provider)._debug_speculation_line(_response(metadata=None)) is None
    assert _cli(None)._debug_speculation_line(_response(metadata=None)) is None


def test_a_failed_lane_is_reported_even_with_no_metadata():
    provider = SimpleNamespace(
        speculation_status=lambda: {
            "lane_loaded": False, "unavailable_reason": "mlx_vlm_missing",
        }
    )
    line = _cli(provider)._debug_speculation_line(_response(metadata=None))
    assert "NOT loaded" in line and "mlx_vlm_missing" in line


# ---------------------------------------------------------------------------
# The diagnosis must name the interpreter
# ---------------------------------------------------------------------------


def test_mlx_vlm_missing_message_names_the_interpreter():
    """The operator's actual failure: mlx-vlm WAS installed -- in another venv.

    `abstractcore-chat` resolves through whichever environment is active, and
    this repo carries a `.venv` whose console scripts shadow the pyenv ones. A
    bare "install mlx-vlm" sends someone to reinstall a package they already
    have. The message has to name `sys.executable`.
    """
    import sys

    from abstractcore.providers.mlx_provider import MLXProvider
    from abstractcore.providers.speculation import SpeculationRequest

    provider = MLXProvider.__new__(MLXProvider)
    provider.logger = SimpleNamespace(
        warning=lambda *a, **k: None, info=lambda *a, **k: None,
        debug=lambda *a, **k: None, error=lambda *a, **k: None,
    )
    provider.model = "some/model"
    provider.model_capabilities = {
        "speculation": {"native_mtp": True,
                        "runtimes": {"mlx": {"drafter": "org/drafter"}}}
    }
    provider._speculation_request = SpeculationRequest(mode="native_mtp")
    provider._mtp_outcome_at_load = None

    import builtins

    real_import = builtins.__import__

    def _no_mlx_vlm(name, *args, **kwargs):
        if name == "mlx_vlm" or name.startswith("mlx_vlm."):
            raise ImportError("No module named 'mlx_vlm'")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = _no_mlx_vlm
    try:
        assert provider._plan_mtp_lane("/tmp/x") is None
    finally:
        builtins.__import__ = real_import

    outcome = provider._mtp_outcome_at_load
    assert outcome.reason == "mlx_vlm_missing"
    message = outcome.details["message"]
    assert sys.executable in message, (
        "the message must name the interpreter that could not import mlx-vlm; "
        f"got {message!r}"
    )
    assert "which -a abstractcore-chat" in message
