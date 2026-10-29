"""Per-model context (core ADR 0010 point 8): set by a person, loaded as is, explicit on the claim."""
from __future__ import annotations

import json

from abstractcore.config import model_contexts as mc


def test_set_read_and_clear_a_model_context(tmp_path):
    cfg = tmp_path / "abstractcore.json"
    cfg.write_text(json.dumps({"keep": 1}))
    mc.set_model_context("HuggingFace", "org/m-GGUF", 32768, config_file=cfg)
    assert mc.configured_model_context("huggingface", "org/m-GGUF", config_file=cfg) == 32768
    assert json.loads(cfg.read_text())["keep"] == 1, "other keys untouched"
    mc.set_model_context("huggingface", "org/m-GGUF", None, config_file=cfg)
    assert mc.configured_model_context("huggingface", "org/m-GGUF", config_file=cfg) is None


def test_a_set_context_is_a_pin_for_the_provider(tmp_path, monkeypatch):
    """MUTANT: the configured context ignored -> the fit rule/ladder shrinks it -> RED."""
    cfg = tmp_path / "abstractcore.json"
    mc.set_model_context("huggingface", "org/m-GGUF", 32768, config_file=cfg)
    import abstractcore.providers.huggingface_provider as hp

    seen = {}

    def fake_base_init(self, model, **kwargs):
        seen.update(kwargs)
        self.model = model

    monkeypatch.setattr(hp.BaseProvider, "__init__", fake_base_init)
    monkeypatch.setattr(hp.HuggingFaceProvider, "_coerce_requested_model_type", lambda self, v: None)
    p = hp.HuggingFaceProvider.__new__(hp.HuggingFaceProvider)
    try:
        hp.HuggingFaceProvider.__init__(p, "org/m-GGUF", _abstractcore_config_file=str(cfg))
    except Exception:
        pass  # the rest of init needs a real model; the context decision happened before
    assert seen.get("max_tokens") == 32768 and p._context_source == "configured" and p._user_provided_max_tokens
