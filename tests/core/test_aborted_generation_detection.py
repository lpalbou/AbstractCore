"""Aborted generation (operator 2026-08-02): a cut tool call comes back as a turn.

LM Studio 0.3.x + qwen/qwen3.6-35b-a3b. When a tool call is cut mid-generation
the server drops it and says so ONLY in its own log
(`~/.lmstudio/server-logs/2026-08/2026-08-02.1.log`)::

    [ERROR][qwen/qwen3.6-35b-a3b] Failed to generate a tool call
    (this tool call will be omitted from the response)

The HTTP body is a clean 200 carrying the tool call's PREFACE and nothing else::

    "message": {"role":"assistant",
                "content":"Good, I have the context. Let me continue building…",
                "tool_calls":[]},
    "finish_reason": "stop",
    "usage": {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}

No error field, and `finish_reason` says "stop" — so every consumer read lost
work as an assistant turn. The one surviving tell is the usage block: text
cannot come from a completion that consumed zero prompt tokens and produced
zero completion tokens.

Pins:
- that exact shape is annotated `truncation_kind="aborted_generation"` and warns;
- ABSENT usage stays UNKNOWN — a missing counter is never a fault verdict;
- a fully-accounted turn with no tool calls is untouched (no land-grab);
- the output-cap lane keeps its own kind and its own wording;
- a length-truncated EMPTY completion no longer buys three full resamples.
"""

from __future__ import annotations

import warnings

import pytest

from abstractcore.core.retry import RetryConfig
from abstractcore.core.types import GenerateResponse

from tests.core.test_empty_completion_retry import ScriptedProvider


def _provider(responses) -> ScriptedProvider:
    return ScriptedProvider(
        "stub-model",
        responses,
        retry_config=RetryConfig(max_attempts=3, initial_delay=0.0, use_jitter=False),
    )


ABORTED_KWARGS = dict(
    content="Good, I have the context. Let me continue building the game files. "
    "I need to create the game loop and main entry point.\n\n",
    model="stub",
    finish_reason="stop",
    tool_calls=None,
    usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
)


def test_zero_usage_completion_with_content_is_annotated_aborted():
    provider = _provider([GenerateResponse(**ABORTED_KWARGS)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response = provider.generate(prompt="build it")

    assert provider.calls == 1  # detection, not a blind resample of a long generation
    assert response.metadata["generation_aborted"] is True
    assert response.metadata["truncation_kind"] == "aborted_generation"
    assert response.metadata["output_truncated"] is True
    messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("Aborted generation" in m and "lost work" in m for m in messages)


@pytest.mark.parametrize("usage", [None, {}])
def test_absent_usage_is_unknown_never_a_verdict(usage):
    """Evidence law: missing evidence is UNKNOWN, not a clean result either way.

    (An all-None usage dict would say the same thing, but `_track_generation`
    crashes on `total_tokens=None` — a separate, pre-existing telemetry bug,
    not this lane's to assert around.)
    """
    provider = _provider([GenerateResponse(**{**ABORTED_KWARGS, "usage": usage})])
    response = provider.generate(prompt="build it")
    meta = response.metadata or {}
    assert "generation_aborted" not in meta
    assert "truncation_kind" not in meta


def test_accounted_completion_is_never_flagged():
    provider = _provider(
        [
            GenerateResponse(
                **{**ABORTED_KWARGS, "usage": {"input_tokens": 5462, "output_tokens": 40, "total_tokens": 5502}}
            )
        ]
    )
    response = provider.generate(prompt="build it")
    assert "generation_aborted" not in (response.metadata or {})


def test_tool_calls_are_progress_not_lost_work():
    provider = _provider(
        [
            GenerateResponse(
                **{**ABORTED_KWARGS, "tool_calls": [{"name": "write_file", "arguments": {}}]}
            )
        ]
    )
    response = provider.generate(prompt="build it")
    assert "generation_aborted" not in (response.metadata or {})


def test_output_cap_keeps_its_own_kind():
    provider = _provider(
        [
            GenerateResponse(
                content="Long plan…",
                model="stub",
                finish_reason="length",
                usage={"input_tokens": 10, "output_tokens": 120, "total_tokens": 130},
            )
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response = provider.generate(prompt="build it")
    assert response.metadata["truncation_kind"] == "output_cap"
    assert "generation_aborted" not in response.metadata
    assert any("Output truncated" in str(w.message) for w in caught)


def test_length_truncated_empty_completion_is_not_resampled_three_times():
    """A tool call cut by the output cap returns content="" + tool_calls=[].

    It used to read as the all-empty transient and buy three full resamples,
    each hitting the identical cap. Resampling cannot widen a budget.
    """
    provider = _provider(
        [
            GenerateResponse(
                content="",
                model="stub",
                finish_reason="length",
                usage={"input_tokens": 335, "output_tokens": 120, "total_tokens": 455},
            )
        ]
    )
    response = provider.generate(prompt="write a big file with a tool call")
    assert provider.calls == 1
    assert response.metadata["truncation_kind"] == "output_cap"
