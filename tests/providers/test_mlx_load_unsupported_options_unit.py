"""`MLXProvider.load_model(ttl_s=..., keep_alive=...)` used to accept the
options and drop them (`_ = kwargs`): the model then stayed resident forever
while the caller believed it would expire (M1, 2026-09-25). MLX has no
idle/TTL unload, so the response must SAY the options were not applied.
`pin` is the caller's residency lock, not a provider option: no warning."""
from __future__ import annotations

from types import SimpleNamespace

from abstractcore.providers.mlx_provider import MLXProvider


def _provider(loaded: bool):
    p = MLXProvider.__new__(MLXProvider)
    p.model = "fake/model"
    p.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    p.llm = object() if loaded else None
    p.tokenizer = object() if loaded else None
    calls = []

    def _load():
        calls.append(1)
        p.llm, p.tokenizer = object(), object()

    p._load_model = _load
    return p, calls


def test_ttl_and_keep_alive_are_reported_as_not_applied():
    p, calls = _provider(loaded=False)
    out = p.load_model("fake/model", ttl_s=20, keep_alive="20s", pin=True)
    assert calls == [1] and out["action"] == "loaded"
    assert out["unsupported_options"] == ["keep_alive", "ttl_s"]
    assert any("no idle/TTL unload" in w and "ttl_s" in w for w in out["warnings"])


def test_already_loaded_still_reports_the_option_as_not_applied():
    p, _ = _provider(loaded=True)
    out = p.load_model(ttl_s=5)
    assert out["action"] == "already_loaded" and out["unsupported_options"] == ["ttl_s"]


def test_other_options_are_named_and_pin_alone_is_silent():
    p, _ = _provider(loaded=False)
    out = p.load_model(pin=True, context_length=4096)
    assert out["unsupported_options"] == ["context_length"]
    assert any("context_length" in w for w in out["warnings"])
    p2, _ = _provider(loaded=False)
    clean = p2.load_model(pin=True)
    assert "warnings" not in clean and "unsupported_options" not in clean
