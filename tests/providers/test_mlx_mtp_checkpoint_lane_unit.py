"""An MTP-preserving MLX checkpoint never loads through mlx-lm, in any lane.

mlx-lm <= 0.31.3 `qwen3_5.Model.sanitize` reads ANY `mtp.` tensor as "raw HF
checkpoint" and adds +1.0 to every RMSNorm weight; an MTP-preserving checkpoint
that is already MLX-converted (`mlx-works/Qwen3.5-9B-oQ4e-mtp`,
`Jundot/Qwen3.8-27B-oQ4e-mtp`) is shifted twice and generates garbage (mission
W2, 2026-09-24). mlx-vlm strips `mtp.` before that decision. These tests drive
the REAL `MLXProvider._load_model` over a fake checkpoint directory with fake
`mlx_lm` / native-session loaders injected -- no weights, no GPU, no network --
and assert which loader each lane reaches.
"""

from __future__ import annotations

import json
import struct
import sys
import types
from pathlib import Path
from typing import Any, List

import pytest

from abstractcore.exceptions import ModelNotFoundError, ProviderAPIError
from abstractcore.providers import mlx_native_session as ns
from abstractcore.providers.mlx_provider import MLXProvider
from abstractcore.providers.speculation import SpeculationRequest, SpeculationUnavailableError


class _Logger:
    def __init__(self) -> None:
        self.warnings: List[str] = []

    def warning(self, msg: str, *a: Any, **k: Any) -> None:
        self.warnings.append(str(msg))

    def info(self, *a: Any, **k: Any) -> None:
        pass

    debug = error = info


def _checkpoint(tmp_path: Path, *, mtp: bool, model_type: str = "qwen3_5", sharded: bool = True) -> Path:
    root = tmp_path / ("ckpt-mtp" if mtp else "ckpt-plain")
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": model_type}))
    keys = {"model.layers.0.input_layernorm.weight": "model-00001-of-00001.safetensors"}
    if mtp:
        keys["mtp.layers.0.input_layernorm.weight"] = "model-00001-of-00001.safetensors"
    if sharded:
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": keys}))
        (root / "model-00001-of-00001.safetensors").write_bytes(b"not read by the router")
    else:
        header = json.dumps({k: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for k in keys}).encode()
        (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")
    return root


class _Loaders:
    """Records which loader ran: `mlx_lm`, or `vlm(<drafter>)`."""

    def __init__(self, monkeypatch, *, drafter_missing: bool = False, vlm_fails: bool = False) -> None:
        self.calls: List[str] = []
        loaders = self

        mlx_lm = types.ModuleType("mlx_lm")

        def load(target, *a, **k):
            loaders.calls.append("mlx_lm")
            return object(), object()

        mlx_lm.load = load
        mlx_lm.generate = lambda *a, **k: ""
        mlx_lm.stream_generate = lambda *a, **k: iter(())
        monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
        mlx = types.ModuleType("mlx")
        core = types.ModuleType("mlx.core")
        mlx.core = core
        monkeypatch.setitem(sys.modules, "mlx", mlx)
        monkeypatch.setitem(sys.modules, "mlx.core", core)
        monkeypatch.setitem(sys.modules, "mlx_vlm", types.ModuleType("mlx_vlm"))

        def load_native_session(path, drafter_path=None):
            if drafter_path and drafter_missing:
                raise ModelNotFoundError(f"MLX drafter {drafter_path!r} is not in the local Hugging Face cache")
            if vlm_fails:
                raise RuntimeError("mlx-vlm has no model class for this checkpoint")
            loaders.calls.append(f"vlm({drafter_path})")
            processor = types.SimpleNamespace(tokenizer=object())
            return types.SimpleNamespace(
                model=object(), processor=processor, drafter=object() if drafter_path else None,
                draft_kind="mtp" if drafter_path else None,
            )

        monkeypatch.setattr(ns, "load_native_session", load_native_session)
        monkeypatch.setattr(MLXProvider, "_bind_native_session", lambda self, session: None)
        monkeypatch.setattr(MLXProvider, "_load_or_adopt_shared_model", lambda self, key, loader: loader())


def _provider(model: Path, **attrs: Any) -> MLXProvider:
    p = MLXProvider.__new__(MLXProvider)
    p.logger = _Logger()
    p.model = str(model)
    p.model_capabilities = {}
    p._speculation_request = None
    p._speculation_inherits_config = False
    p._mlx_batching = False
    p._mlx_ple_offload = False
    p._mtp_drafter = None
    p._mtp_processor = None
    p._mtp_outcome_at_load = None
    for key, value in attrs.items():
        setattr(p, key, value)
    return p


_ON = SpeculationRequest(mode="native_mtp", drafter="org/Some-MTP-4bit")


# --- detection -------------------------------------------------------------


def test_detection_reads_the_index_weight_map(tmp_path):
    assert ns.mtp_weight_keys(_checkpoint(tmp_path, mtp=True)) == ["mtp.layers.0.input_layernorm.weight"]


def test_detection_is_empty_for_a_checkpoint_without_mtp_tensors(tmp_path):
    assert ns.mtp_weight_keys(_checkpoint(tmp_path, mtp=False)) == []


def test_detection_reads_a_single_file_header_when_there_is_no_index(tmp_path):
    assert ns.mtp_weight_keys(_checkpoint(tmp_path, mtp=True, sharded=False)) == ["mtp.layers.0.input_layernorm.weight"]


def test_detection_refuses_an_unreadable_index(tmp_path):
    root = _checkpoint(tmp_path, mtp=True)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": ["not", "a", "map"]}))
    with pytest.raises(ValueError):
        ns.mtp_weight_keys(root)


# --- every lane -------------------------------------------------------------


@pytest.mark.parametrize(
    "label, attrs, expected",
    [
        ("speculation unset", {}, "vlm(None)"),
        ("speculation off", {"_speculation_request": SpeculationRequest(mode="off")}, "vlm(None)"),
        ("speculation on, drafter present", {"_speculation_request": _ON}, "vlm(org/Some-MTP-4bit)"),
        ("batching", {"_mlx_batching": True}, "vlm(None)"),
        ("batching + speculation on", {"_mlx_batching": True, "_speculation_request": _ON}, "vlm(org/Some-MTP-4bit)"),
    ],
)
def test_an_mtp_checkpoint_loads_through_mlx_vlm_in_every_lane(tmp_path, monkeypatch, label, attrs, expected):
    loaders = _Loaders(monkeypatch)
    provider = _provider(_checkpoint(tmp_path, mtp=True), **attrs)
    provider._load_model()
    assert loaders.calls == [expected], f"{label}: loaders were {loaders.calls}"
    assert "mlx_lm" not in loaders.calls
    assert provider._mtp_preserving_checkpoint is True


def test_an_mtp_checkpoint_single_file_also_avoids_mlx_lm(tmp_path, monkeypatch):
    loaders = _Loaders(monkeypatch)
    _provider(_checkpoint(tmp_path, mtp=True, sharded=False))._load_model()
    assert loaders.calls == ["vlm(None)"]


def test_missing_companion_keeps_the_mlx_vlm_lane_and_says_so_in_words(tmp_path, monkeypatch):
    loaders = _Loaders(monkeypatch, drafter_missing=True)
    provider = _provider(_checkpoint(tmp_path, mtp=True), _speculation_request=_ON)
    provider._load_model()
    assert loaders.calls == ["vlm(None)"], "a missing drafter must not fall back to mlx-lm"
    outcome = provider._mtp_outcome_at_load
    assert outcome is not None and outcome.used is False
    message = outcome.details["message"]
    assert message.startswith("MTP acceleration off: companion org/Some-MTP-4bit")
    assert "not downloaded" in message
    assert "abstractcore models download mlx org/Some-MTP-4bit" in message
    assert outcome.to_metadata()["message"] == message
    assert any("MTP acceleration off" in w for w in provider.logger.warnings)


def test_inherited_default_without_a_cached_head_still_avoids_mlx_lm(tmp_path, monkeypatch):
    loaders = _Loaders(monkeypatch)
    import abstractcore.providers.speculation as spec

    monkeypatch.setattr(spec, "configured_speculation_default", lambda **k: {"mode": "native_mtp", "num_draft_tokens": 2})
    monkeypatch.setattr(spec, "describe_speculation_capabilities", lambda *a, **k: {"supported": True})
    monkeypatch.setattr(spec, "_local_model_directory", lambda model: None)
    monkeypatch.setattr(spec, "mlx_speculation_artifact", lambda model: {"drafter": "org/Some-MTP-4bit", "block_size": 3})
    provider = _provider(_checkpoint(tmp_path, mtp=True), _speculation_inherits_config=True)
    provider._load_model()
    assert loaders.calls == ["vlm(None)"]
    outcome = provider._mtp_outcome_at_load
    assert outcome.reason == "mtp_head_not_cached"
    assert "MTP acceleration off: companion org/Some-MTP-4bit" in outcome.details["message"]
    assert "abstractcore models download mlx org/Some-MTP-4bit" in outcome.details["message"]


def test_require_acceleration_with_a_missing_companion_raises_and_loads_nothing(tmp_path, monkeypatch):
    loaders = _Loaders(monkeypatch, drafter_missing=True)
    strict = SpeculationRequest(mode="native_mtp", drafter="org/Some-MTP-4bit", require_acceleration=True)
    provider = _provider(_checkpoint(tmp_path, mtp=True), _speculation_request=strict)
    with pytest.raises(SpeculationUnavailableError):
        provider._load_model()
    assert loaders.calls == []


def test_mlx_vlm_failure_raises_provider_error_never_mlx_lm(tmp_path, monkeypatch):
    loaders = _Loaders(monkeypatch, vlm_fails=True)
    root = _checkpoint(tmp_path, mtp=True)
    provider = _provider(root)
    with pytest.raises(ProviderAPIError) as excinfo:
        provider._load_model()
    text = str(excinfo.value)
    assert str(root) in text, "the error must name the model"
    assert "MTP tensor" in text and "mlx-vlm" in text, "the error must name the reason"
    assert "not loaded through mlx-lm" in text
    assert loaders.calls == []


def test_a_plain_checkpoint_still_loads_through_mlx_lm(tmp_path, monkeypatch):
    loaders = _Loaders(monkeypatch)
    provider = _provider(_checkpoint(tmp_path, mtp=False))
    provider._load_model()
    assert loaders.calls == ["mlx_lm"]
    assert provider._mtp_preserving_checkpoint is False


# --- error text names the actual family --------------------------------------


def test_native_lane_errors_name_the_checkpoint_family_not_qwen4():
    provider = _provider(Path("mlx-works/Qwen3.5-9B-oQ4e-mtp"), _mtp_processor=object(), _native_model_type="qwen3_5")
    with pytest.raises(ProviderAPIError) as excinfo:
        provider._prompt_cache_backend_create()
    text = str(excinfo.value)
    assert "Qwen4" not in text
    assert "mlx-works/Qwen3.5-9B-oQ4e-mtp (model_type qwen3_5)" in text


def test_no_native_lane_message_hardcodes_qwen4():
    source = Path(sys.modules[MLXProvider.__module__].__file__).read_text(encoding="utf-8")
    assert "Native Qwen4 prompt_cache_key" not in source
    assert '"Native Qwen4 ' not in source
