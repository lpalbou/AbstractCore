"""Pin: a TERMINAL server-side chat-template render 400 emits an actionable WARNING.

LM Studio's bundled Jinja engine cannot render the `safe` filter that the
Qwen3-Coder XML tool convention (Qwen3-Coder / Ornith / Step-3.5) uses in its
chat template, so a well-formed multi-turn tool request 400s server-side:
`Error rendering prompt with jinja template: "Unknown StringValue filter: safe"`.
That surfaced in abstractassistant as a mystery HTTP 400.

Division of labor (2026-07-15): the ONE repairable class (LM Studio + non-string
tool-call history args) is retried reactively at the request sites with its own
#FALLBACK warning (`LMStudioProvider._render_400_repaired_payload`, pinned in
test_lmstudio_reactive_stringify_unit.py). Every render-400 that still reaches
`_raise_for_status` is TERMINAL and gets this warning: name the cause; do NOT
switch provider/model, do NOT suggest switching. The request is well-formed —
the incompatibility is between the SERVER's template engine and the MODEL's
embedded template.
"""
from __future__ import annotations

import pytest

from abstractcore.exceptions import InvalidRequestError
from abstractcore.providers.lmstudio_provider import LMStudioProvider


class _CapLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, *a, **k):
        self.warnings.append(str(msg))

    def debug(self, *a, **k):
        pass


class _Resp:
    def __init__(self, body):
        self.status_code = 400
        self._body = body
        self.text = ""

    def json(self):
        return self._body


def _provider():
    p = LMStudioProvider.__new__(LMStudioProvider)
    p.model = "ornith-1.0-35b"
    p.PROVIDER_DISPLAY_NAME = "LMStudio"
    p.logger = _CapLogger()
    return p


def test_template_render_400_warns_with_cause_and_still_raises():
    p = _provider()
    body = {"error": 'Error rendering prompt with jinja template: "Unknown StringValue filter: safe".'}
    with pytest.raises(InvalidRequestError):
        p._raise_for_status(_Resp(body))
    assert p.logger.warnings, "a warning must be emitted"
    w = p.logger.warnings[0].lower()
    assert "template" in w and "safe" in w          # names the cause
    assert "well-formed" in w or "not a malformed" in w


def test_warning_does_not_suggest_changing_provider_or_model():
    p = _provider()
    body = {"error": 'Error rendering prompt with jinja template: "Unknown StringValue filter: safe".'}
    with pytest.raises(InvalidRequestError):
        p._raise_for_status(_Resp(body))
    joined = " ".join(p.logger.warnings).lower()
    # Operator ruling: warning only — never route/suggest a different model or provider.
    assert "--provider" not in joined
    assert "huggingface" not in joined
    assert "switch" not in joined and "instead" not in joined


def test_unrelated_400_does_not_warn():
    p = _provider()
    # model-not-found style 400 must not trip the template-render warning
    with pytest.raises(Exception):
        p._raise_for_status(_Resp({"error": "The model does not exist"}))
    assert not any("chat template" in w.lower() for w in p.logger.warnings)
