from __future__ import annotations

from typing import Any, Dict

import pytest

from abstractcore.providers import huggingface_provider
from abstractcore.providers.huggingface_provider import HuggingFaceProvider


class _DummyLlama:
    def __init__(self, **kwargs: Any):
        self._kwargs: Dict[str, Any] = dict(kwargs)

    def n_ctx(self) -> int:
        return int(self._kwargs.get("n_ctx") or 0)

class _FailOnLargeCtxDummyLlama(_DummyLlama):
    def __init__(self, **kwargs: Any):
        n_ctx = int(kwargs.get("n_ctx") or 0)
        if n_ctx > 8192:
            raise RuntimeError(f"simulated OOM for n_ctx={n_ctx}")
        super().__init__(**kwargs)


def test_huggingface_gguf_uses_capabilities_max_tokens_as_default_n_ctx(tmp_path, monkeypatch) -> None:
    # Create a tiny placeholder file so HuggingFaceProvider treats it as a direct GGUF path.
    gguf_path = tmp_path / "dummy.gguf"
    gguf_path.write_bytes(b"GGUF")

    monkeypatch.setattr(huggingface_provider, "LLAMACPP_AVAILABLE", True, raising=False)
    monkeypatch.setattr(huggingface_provider, "Llama", _DummyLlama, raising=False)

    llm = HuggingFaceProvider(
        model=str(gguf_path),
        device="cpu",
    )

    assert llm.model_type == "gguf"
    # Unknown models use the default capabilities context window (16384) unless overridden.
    assert llm.llm.n_ctx() == 16384
    assert llm.max_tokens == 16384


def test_huggingface_gguf_falls_back_to_smaller_n_ctx_on_load_failure(tmp_path, monkeypatch) -> None:
    gguf_path = tmp_path / "dummy.gguf"
    gguf_path.write_bytes(b"GGUF")

    monkeypatch.setattr(huggingface_provider, "LLAMACPP_AVAILABLE", True, raising=False)
    monkeypatch.setattr(huggingface_provider, "Llama", _FailOnLargeCtxDummyLlama, raising=False)

    llm = HuggingFaceProvider(
        model=str(gguf_path),
        device="cpu",
    )

    assert llm.model_type == "gguf"
    assert llm.llm.n_ctx() == 8192
    assert llm.max_tokens == 8192


def test_huggingface_gguf_respects_explicit_max_tokens_as_runtime_n_ctx(tmp_path, monkeypatch) -> None:
    gguf_path = tmp_path / "dummy.gguf"
    gguf_path.write_bytes(b"GGUF")

    monkeypatch.setattr(huggingface_provider, "LLAMACPP_AVAILABLE", True, raising=False)
    monkeypatch.setattr(huggingface_provider, "Llama", _DummyLlama, raising=False)

    with pytest.warns(UserWarning):
        llm = HuggingFaceProvider(
            model=str(gguf_path),
            device="cpu",
            max_tokens=4096,
            max_output_tokens=5000,
        )

    assert llm.model_type == "gguf"
    assert llm.llm.n_ctx() == 4096
    assert llm.max_tokens == 4096
    assert llm.max_output_tokens == 4096


# ---------------------------------------------------------------------------
# Probe-gated ladder + context calibration
# ---------------------------------------------------------------------------

class _ProbeGatedDummyLlama(_DummyLlama):
    """Allocation always succeeds; the probe decode fails above a threshold —
    the Metal-overcommit failure mode the probe exists to catch."""

    probe_pass_threshold = 8192
    constructed_n_ctx: list[int] = []
    eval_probed_n_ctx: list[int] = []

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        type(self).constructed_n_ctx.append(int(kwargs.get("n_ctx") or 0))
        self.n_batch = int(kwargs.get("n_batch") or 512)

    def reset(self) -> None:
        return None

    def token_bos(self) -> int:
        return 1

    def eval(self, tokens: list[int]) -> None:
        type(self).eval_probed_n_ctx.append(self.n_ctx())
        if self.n_ctx() > type(self).probe_pass_threshold:
            raise RuntimeError(f"simulated decode -3 at n_ctx={self.n_ctx()}")


@pytest.fixture()
def _probe_gated_llama(monkeypatch):
    class _Llama(_ProbeGatedDummyLlama):
        constructed_n_ctx: list[int] = []
        eval_probed_n_ctx: list[int] = []

    monkeypatch.setattr(huggingface_provider, "LLAMACPP_AVAILABLE", True, raising=False)
    monkeypatch.setattr(huggingface_provider, "Llama", _Llama, raising=False)
    return _Llama


@pytest.fixture()
def _calibration_spy(monkeypatch):
    from abstractcore.utils import context_calibration as cal_mod

    recorded: list[Dict[str, Any]] = []
    monkeypatch.setattr(cal_mod, "record_context_calibration", lambda entry: recorded.append(dict(entry)))
    monkeypatch.setattr(cal_mod, "lookup_context_calibration", lambda *a, **k: None)
    return recorded


def test_huggingface_gguf_ladder_settles_at_largest_probe_passing_rung(
    tmp_path, _probe_gated_llama, _calibration_spy
) -> None:
    gguf_path = tmp_path / "dummy.gguf"
    gguf_path.write_bytes(b"GGUF")

    llm = HuggingFaceProvider(model=str(gguf_path), device="cpu")

    assert llm.model_type == "gguf"
    # 16384 allocated but failed its probe; 8192 is the largest usable rung.
    assert llm.llm.n_ctx() == 8192
    assert llm.max_tokens == 8192
    assert _probe_gated_llama.constructed_n_ctx == [16384, 8192]
    # The probe ran on EVERY allocated rung (it is the gate, never skipped).
    assert _probe_gated_llama.eval_probed_n_ctx[:2] == [16384, 8192]


def test_huggingface_gguf_evalless_engine_is_not_probed() -> None:
    provider = object.__new__(HuggingFaceProvider)

    class _EvalLess:
        def reset(self) -> None:
            raise AssertionError("probe must not touch an engine without eval")

    # No eval attribute -> the probe declines to run and never raises.
    provider._gguf_probe_decode(_EvalLess())


def test_huggingface_gguf_records_calibration_on_settle(
    tmp_path, _probe_gated_llama, _calibration_spy
) -> None:
    gguf_path = tmp_path / "dummy.gguf"
    gguf_path.write_bytes(b"GGUF")

    llm = HuggingFaceProvider(model=str(gguf_path), device="cpu")

    assert len(_calibration_spy) == 1
    entry = _calibration_spy[0]
    assert entry["provider"] == "huggingface"
    assert entry["model_id"] == "dummy.gguf"
    assert entry["requested_context"] == 16384
    assert entry["settled_context"] == 8192
    assert entry["rungs_tried"] == [16384, 8192]
    assert "device_total_bytes" in entry and "ram_total_bytes" in entry

    # The provider exposes the settle as residency truth for the gateway record.
    assert llm._gguf_context_calibrated is True
    assert llm._gguf_calibrated_context == 8192
    residency = llm.get_model_residency()
    assert residency["context_calibrated"] is True
    assert residency["calibrated_context_length"] == 8192


def test_huggingface_gguf_calibration_seeds_candidate_order(
    tmp_path, _probe_gated_llama, monkeypatch
) -> None:
    from abstractcore.utils import context_calibration as cal_mod

    recorded: list[Dict[str, Any]] = []
    monkeypatch.setattr(cal_mod, "record_context_calibration", lambda entry: recorded.append(dict(entry)))
    monkeypatch.setattr(
        cal_mod,
        "lookup_context_calibration",
        lambda provider, model_id, device_total, ram_total: {
            "provider": provider,
            "model_id": model_id,
            "settled_context": 4096,
        },
    )
    _probe_gated_llama.probe_pass_threshold = 4096

    gguf_path = tmp_path / "dummy.gguf"
    gguf_path.write_bytes(b"GGUF")

    llm = HuggingFaceProvider(model=str(gguf_path), device="cpu")

    # Seeded ladder: requested first, then the calibrated rung — 8192 skipped.
    assert _probe_gated_llama.constructed_n_ctx == [16384, 4096]
    # The cached rung was still probed (memory conditions change).
    assert _probe_gated_llama.eval_probed_n_ctx[:2] == [16384, 4096]
    assert llm.max_tokens == 4096
    assert recorded and recorded[0]["rungs_tried"] == [16384, 4096]


def test_huggingface_gguf_seeded_ladder_stays_descending(
    tmp_path, _probe_gated_llama, monkeypatch
) -> None:
    """Rungs LARGER than the calibrated seed must never follow it: they
    already failed on this hardware, and a seeded ladder [16384, 4096, 8192]
    would re-pay the 8192 allocation the seed exists to skip."""
    from abstractcore.utils import context_calibration as cal_mod

    monkeypatch.setattr(cal_mod, "record_context_calibration", lambda entry: None)
    monkeypatch.setattr(
        cal_mod,
        "lookup_context_calibration",
        lambda *a, **k: {"settled_context": 4096},
    )
    # Every rung fails its probe, so the FULL candidate walk becomes visible.
    _probe_gated_llama.probe_pass_threshold = 0

    gguf_path = tmp_path / "dummy.gguf"
    gguf_path.write_bytes(b"GGUF")

    with pytest.raises(RuntimeError, match="probe decode at every context"):
        HuggingFaceProvider(model=str(gguf_path), device="cpu")

    # Strictly descending: nothing larger than the 4096 seed after it.
    assert _probe_gated_llama.constructed_n_ctx == [16384, 4096]
    assert 8192 not in _probe_gated_llama.constructed_n_ctx


def test_huggingface_gguf_user_max_tokens_skips_calibration(
    tmp_path, monkeypatch, _calibration_spy
) -> None:
    gguf_path = tmp_path / "dummy.gguf"
    gguf_path.write_bytes(b"GGUF")

    monkeypatch.setattr(huggingface_provider, "LLAMACPP_AVAILABLE", True, raising=False)
    monkeypatch.setattr(huggingface_provider, "Llama", _DummyLlama, raising=False)

    llm = HuggingFaceProvider(model=str(gguf_path), device="cpu", max_tokens=4096)

    # A user-provided max_tokens is a pin, not a measurement: nothing recorded,
    # and the residency claim carries no calibration fields.
    assert _calibration_spy == []
    assert not getattr(llm, "_gguf_context_calibrated", False)
    assert "context_calibrated" not in llm.get_model_residency()
