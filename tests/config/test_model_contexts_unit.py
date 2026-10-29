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


def test_kimi_9b_reserves_the_maximum_that_fits_with_a_5_percent_buffer(tmp_path, monkeypatch):
    """Operator ruling 2026-10-29: the smaller of the advertised context and the largest that fits
    the free memory minus 5 %. Kimi-9B (qwen35, 33 blocks, one in four with KV) on the 16 GB card:
    15.24 GB free, 5.6 GB weights -> about 135k of 262,144. MUTANT: the old 'half of free' rule
    (65,536) or fixed rungs (131,072) -> RED."""
    import sys
    from types import SimpleNamespace
    import logging

    import abstractcore.providers.huggingface_provider as hp
    import abstractcore.providers.hf_residency as hr

    geom = dict(architecture="qwen35", block_count=33, head_count=16, head_count_kv=4, key_length=256,
                value_length=256, full_attention_interval=4, ssm_conv_kernel=4, ssm_state_size=128,
                ssm_group_count=16, ssm_inner_size=4096)
    free = 15_240_000_000
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1, mem_get_info=lambda: (free, 16_106_127_360))))
    monkeypatch.setattr(hr, "gguf_file_geometry", lambda p: geom)
    monkeypatch.setattr(hp, "_gguf_total_weight_bytes", lambda p: 5_600_000_000)
    p = hp.HuggingFaceProvider.__new__(hp.HuggingFaceProvider)
    p.logger = logging.getLogger("t"); p.model = "mradermacher/Qwen3.5-9B-Kimi-k3-Distilled-GGUF"; p.n_gpu_layers = -1
    p._abstractcore_config_file = str(tmp_path / "none.json")
    fit = p._gguf_max_context_that_fits("x.gguf", 262144)
    budget = int(free * 0.95) - 5_600_000_000 - 54_886_400 // 33 * 0  # recurrent state is small
    assert fit["reason"] == "limited by memory" and fit["buffer"] == 0.05
    assert 128_000 < fit["reserved"] < 140_000 and fit["reserved"] % 256 == 0, fit
    # a custom buffer is read from the AbstractCore config
    (tmp_path / "abc.json").write_text('{"model_memory_buffer": 0.2}')
    p._abstractcore_config_file = str(tmp_path / "abc.json")
    assert p._gguf_max_context_that_fits("x.gguf", 262144)["reserved"] < fit["reserved"]
    # the advertised context wins when it fits
    assert p._gguf_max_context_that_fits("x.gguf", 32768)["reserved"] == 32768
