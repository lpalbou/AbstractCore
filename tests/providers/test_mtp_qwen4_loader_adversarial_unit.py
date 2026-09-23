"""Adversarial Qwen4 checkpoint tests; all device/backend modules are fakes.

The artifact index, source immutability, per-module quantization choices and
cleanup are real.  No model/GPU/network operation occurs in this file.
"""

import json
import sys
import types
from pathlib import Path

import pytest

from abstractcore.providers import mlx_qwen4 as loader


def _checkpoint(tmp_path, weight_map=None, config=None):
    root = tmp_path / "source"
    root.mkdir()
    config = config or {"model_type": "qwen4_exp", "text_config": {"mtp_num_hidden_layers": 1}}
    (root / "config.json").write_text(json.dumps(config))
    if weight_map is not None:
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        for filename in set(weight_map.values()):
            if isinstance(filename, str) and Path(filename).name == filename:
                (root / filename).write_bytes(b"fake tensor payload")
    return root


def _fake_backend(monkeypatch, *, tensors=None, required=None, fail_target=False):
    state = types.SimpleNamespace(
        tensors=tensors or {}, required=set(required or ()), reads=[], loads=[],
        quantization={}, compatibility=[], target_path=None, drafter=None,
    )

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__path__ = []
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    module("mlx")
    module("mlx.core", eval=lambda *args: None)

    def quantize(model, *, class_predicate):
        for key in state.tensors:
            if key.startswith("mtp.") and key.endswith(".scales"):
                path = key[4:-len(".scales")]
                state.quantization[path] = class_predicate(path, object())
        state.quantization["unquantized"] = class_predicate("unquantized", object())

    module("mlx.nn", quantize=quantize, Module=type("Module", (), {}))

    class SafeOpen:
        def __init__(self, path, framework):
            assert framework == "mlx"
            self.path = Path(path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get_tensor(self, key):
            state.reads.append((self.path.name, key))
            return state.tensors[key]

    module("safetensors", safe_open=SafeOpen)

    def load(path, **kwargs):
        state.target_path = Path(path)
        state.loads.append((path, kwargs, json.loads((Path(path) / "config.json").read_text())))
        assert all(item.exists() for item in Path(path).iterdir()), "broken overlay symlink"
        if fail_target:
            raise RuntimeError("target rejected malformed layer layout")
        return object(), object()

    module("mlx_vlm", load=load)
    monkeypatch.setattr(loader, "_load_target", load)
    module("mlx_vlm.models")
    module("mlx_vlm.models.qwen4_exp")

    def manifest(source, destination, *, cache_rows):
        assert cache_rows >= 0
        Path(destination).write_text(json.dumps({"source": str(source)}))

    module("mlx_vlm.models.qwen4_exp.ple_storage", build_quantized_ple_manifest=manifest)
    module("mlx_vlm.speculative")
    module("mlx_vlm.speculative.drafters", validate_drafter_compatibility=lambda *args: state.compatibility.append(args))
    module("mlx_vlm.speculative.drafters.qwen4_exp_mtp")
    module("mlx_vlm.speculative.drafters.qwen4_exp_mtp.config", Qwen4ExpMTPConfig=types.SimpleNamespace(from_dict=lambda config: config))

    class Draft:
        prefer_requested_block_size = False

        def __init__(self, config):
            self.config = config
            state.drafter = self

        def sanitize(self, weights):
            return {key.removeprefix("mtp."): value for key, value in weights.items()}

        def load_weights(self, weights, *, strict):
            assert strict is True
            actual = dict(weights)
            if not state.required.issubset(actual):
                raise ValueError("missing embedded head weights")
            self.weights = actual

        def eval(self):
            pass

        def parameters(self):
            return {}

    state.drafter_class = Draft
    module("mlx_vlm.speculative.drafters.qwen4_exp_mtp.qwen4_exp_mtp", Qwen4ExpMTPDraftModel=Draft)
    return state


def test_config_claim_alone_is_not_embedded_mtp_evidence(tmp_path):
    root = _checkpoint(tmp_path)
    assert loader.is_qwen4_checkpoint(str(root)) is True
    assert loader.embedded_mtp_keys(str(root)) == {}


def test_plain_target_index_is_not_mtp_evidence(tmp_path):
    root = _checkpoint(tmp_path, {"language_model.layers.0.weight": "model.safetensors"})
    assert loader.embedded_mtp_keys(str(root)) == {}


@pytest.mark.parametrize("filename", ["../outside.safetensors", "/tmp/outside.safetensors", "nested/model.safetensors"])
def test_embedded_index_cannot_select_a_path_outside_checkpoint_members(tmp_path, filename):
    root = _checkpoint(tmp_path, {"mtp.fc_hidden.weight": filename})
    with pytest.raises(ValueError, match="shard"):
        loader.embedded_mtp_keys(str(root))


def test_missing_shard_is_not_accepted_as_weight_evidence(tmp_path):
    root = _checkpoint(tmp_path, {"mtp.fc_hidden.weight": "missing.safetensors"})
    (root / "missing.safetensors").unlink()
    with pytest.raises(ValueError, match="shard"):
        loader.embedded_mtp_keys(str(root))


@pytest.mark.parametrize("filename", [None, 3, []])
def test_malformed_index_filename_has_actionable_validation_error(tmp_path, filename):
    root = _checkpoint(tmp_path)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"mtp.fc_hidden.weight": filename}}))
    with pytest.raises(ValueError):
        loader.embedded_mtp_keys(str(root))


@pytest.mark.parametrize("mapping", [None, [], "bad index"])
def test_malformed_weight_map_has_actionable_validation_error(tmp_path, mapping):
    root = _checkpoint(tmp_path)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(ValueError):
        loader.embedded_mtp_keys(str(root))


def test_incomplete_head_fails_before_expensive_target_load(tmp_path, monkeypatch):
    root = _checkpoint(tmp_path, {"mtp.fc_hidden.weight": "head.safetensors"})
    state = _fake_backend(monkeypatch, tensors={"mtp.fc_hidden.weight": object()}, required={"fc_hidden.weight", "fc_embedding.weight"})
    with pytest.raises(ValueError, match="missing embedded"):
        loader.load_qwen4_session(str(root), mtp=True)
    assert state.loads == []


def test_index_claim_missing_from_shard_fails_before_target_load(tmp_path, monkeypatch):
    root = _checkpoint(tmp_path, {"mtp.fc_hidden.weight": "head.safetensors"})
    state = _fake_backend(monkeypatch)
    with pytest.raises((ValueError, KeyError)):
        loader.load_qwen4_session(str(root), mtp=True)
    assert state.loads == []


def test_mixed_head_quantization_is_preserved_and_unrelated_tensors_not_loaded(tmp_path, monkeypatch):
    quantization = {"bits": 4, "group_size": 64, "mode": "affine"}
    weights = {}
    for name, bits in (("fc_hidden", 5), ("fc_embedding", 6), ("layers.0.self_attn.q_proj", 8), ("default_proj", 4)):
        weights[f"mtp.{name}.weight"] = object()
        weights[f"mtp.{name}.scales"] = object()
        if bits != 4:
            quantization[f"mtp.{name}"] = {"bits": bits, "group_size": 32, "mode": "affine"}
    index = {key: "head.safetensors" for key in weights}
    index["language_model.huge_experts.weight"] = "target.safetensors"
    config = {"model_type": "qwen4_exp", "text_config": {}, "quantization": quantization}
    root = _checkpoint(tmp_path, index, config)
    state = _fake_backend(monkeypatch, tensors=weights)
    before = (root / "config.json").read_bytes()

    session = loader.load_qwen4_session(str(root), mtp=True, ple_offload=False)

    assert state.quantization["fc_hidden"] == quantization["mtp.fc_hidden"]
    assert state.quantization["fc_embedding"] == quantization["mtp.fc_embedding"]
    assert state.quantization["layers.0.self_attn.q_proj"] == quantization["mtp.layers.0.self_attn.q_proj"]
    assert state.quantization["default_proj"] == {"bits": 4, "group_size": 64, "mode": "affine"}
    assert state.quantization["unquantized"] is False
    assert set(state.reads) == {("head.safetensors", key) for key in weights}
    assert (root / "config.json").read_bytes() == before
    assert session.drafter.prefer_requested_block_size is True
    assert state.drafter_class.prefer_requested_block_size is False, "global class was mutated"
    assert state.compatibility and state.compatibility[0][2] == "mtp"


def test_quantized_head_without_metadata_does_not_guess_q4(tmp_path, monkeypatch):
    tensors = {"mtp.fc_hidden.weight": object(), "mtp.fc_hidden.scales": object()}
    root = _checkpoint(tmp_path, {key: "head.safetensors" for key in tensors})
    state = _fake_backend(monkeypatch, tensors=tensors)
    with pytest.raises(ValueError, match="quantization metadata"):
        loader.load_qwen4_session(str(root), mtp=True)
    assert state.loads == []


def test_offloaded_overlay_is_temporary_and_never_changes_original_checkpoint(tmp_path, monkeypatch):
    root = _checkpoint(tmp_path, {"language_model.weight": "target.safetensors"})
    # Reproduce HF's symlink snapshots, not only ordinary directory files.
    blob = tmp_path / "blob.safetensors"
    blob.write_bytes((root / "target.safetensors").read_bytes())
    (root / "target.safetensors").unlink()
    (root / "target.safetensors").symlink_to("../blob.safetensors")
    before = {entry.name: entry.read_bytes() for entry in root.iterdir()}
    state = _fake_backend(monkeypatch)

    session = loader.load_qwen4_session(str(root), mtp=False, ple_offload=True)
    try:
        assert state.target_path != root
        assert (state.target_path / "target.safetensors").resolve() == blob.resolve()
        assert session.ple_offload is True
        assert "ple_storage" in state.loads[0][2]["text_config"]
        assert {entry.name: entry.read_bytes() for entry in root.iterdir()} == before
    finally:
        session.storage.cleanup()
    assert not state.target_path.exists()
    assert blob.read_bytes() == b"fake tensor payload"


def test_failed_target_load_cleans_overlay_even_while_exception_traceback_is_retained(tmp_path, monkeypatch):
    root = _checkpoint(tmp_path, {"language_model.weight": "target.safetensors"})
    state = _fake_backend(monkeypatch, fail_target=True)
    with pytest.raises(RuntimeError, match="target rejected") as caught:
        loader.load_qwen4_session(str(root), mtp=False, ple_offload=True)
    assert caught.value.__traceback__ is not None  # Keep failed frame alive.
    assert state.target_path is not None  # Non-vacuity: target load was attempted.
    assert not state.target_path.exists(), "failed load leaked an overlay until GC"


def test_repeated_sessions_and_plain_to_mtp_upgrade_reuse_one_target(tmp_path, monkeypatch):
    tensors = {"mtp.fc_hidden.weight": object()}
    root = _checkpoint(tmp_path, {key: "head.safetensors" for key in tensors})
    state = _fake_backend(monkeypatch, tensors=tensors)
    plain = loader.load_qwen4_session(str(root), mtp=False, ple_offload=False)
    upgraded = loader.load_qwen4_session(str(root), mtp=True, ple_offload=False)
    again = loader.load_qwen4_session(str(root), mtp=True, ple_offload=False)
    assert plain is upgraded is again
    assert len(state.loads) == 1, "loading the head duplicated the ~70GiB target"
    assert upgraded.drafter is state.drafter
    assert len(state.reads) == 1


def _fake_target_backend(monkeypatch, tensors):
    state = types.SimpleNamespace(sanitized=[], quantized={}, processor_kwargs=None, image_kwargs=None)

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__path__ = []
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    module("mlx")
    module("mlx.core", load=lambda path: dict(tensors), eval=lambda *args: None, bfloat16="bfloat16")

    def quantize(model, **kwargs):
        state.quantization_defaults = {k: v for k, v in kwargs.items() if k != "class_predicate"}
        for name in ("dense", "mixed", "unquantized"):
            state.quantized[name] = kwargs["class_predicate"](name, object())

    module("mlx.nn", quantize=quantize, Module=type("Module", (), {}))
    module("mlx_vlm")
    module("mlx_vlm.models")
    module("mlx_vlm.models.qwen3_5")
    module("mlx_vlm.models.qwen3_5.qwen3_5", sanitize_key=lambda key: key.replace("model.language_model", "language_model.model", 1))

    class FakeArray:
        def astype(self, dtype):
            assert dtype == "bfloat16"
            state.casts = getattr(state, "casts", 0) + 1
            return self

        def __mul__(self, scale):
            return 2.0 * scale

    class Embedding:
        row_count = 64

        def __call__(self, indices):
            state.embedding_calls = getattr(state, "embedding_calls", 0) + 1
            return FakeArray()

    state.original_embedding = Embedding()

    class Model:
        def __init__(self, config):
            self.config = config
            self.language_model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[
                types.SimpleNamespace(), types.SimpleNamespace(),
                types.SimpleNamespace(ple=types.SimpleNamespace(ple_embedding=types.SimpleNamespace(
                    ngram_embedding=state.original_embedding))),
            ]))

        def load_weights(self, weights, *, strict):
            assert strict is True
            self.weights = dict(weights)

        def parameters(self):
            return {}

        def eval(self):
            pass

    class VisionModel:
        pass

    class LanguageModel:
        pass

    def model_config(config):
        return types.SimpleNamespace(
            quantization=config.get("quantization"), text_config=config["text_config"],
            vision_config={}, eos_token_id=[2],
        )

    module("mlx_vlm.models.qwen4_exp", Model=Model, VisionModel=VisionModel, LanguageModel=LanguageModel,
           ModelConfig=types.SimpleNamespace(from_dict=model_config))

    def sanitize(model, weights, *args):
        state.sanitized.append(set(weights))
        return {k: v for k, v in weights.items() if not k.startswith("mtp.")}

    def processor(path, *args, **kwargs):
        state.processor_kwargs = kwargs
        return types.SimpleNamespace()

    def image_processor(path, **kwargs):
        state.image_kwargs = kwargs
        return None

    module("mlx_vlm.utils", load_config=lambda path: json.loads((Path(path) / "config.json").read_text()),
           load_processor=processor, load_image_processor=image_processor, sanitize_weights=sanitize)
    return state


def test_target_offload_filters_ple_before_any_upstream_fp8_sanitizer(tmp_path, monkeypatch):
    prefix = "model.language_model.layers.2.ple.ple_embedding.ngram_embedding."
    tensors = {prefix + "weight_scale": object(), prefix + "shards.0.weight": object(),
               prefix + "shards.0.scales": object(), "dense.weight": object(), "mtp.fc_hidden.weight": object()}
    config = {"model_type": "qwen4_exp", "text_config": {"ple_storage": {"manifest": "ple-store.json"}}}
    root = _checkpoint(tmp_path, {key: "weights.safetensors" for key in tensors}, config)
    state = _fake_target_backend(monkeypatch, tensors)
    model, _ = loader._load_target(root)
    assert state.sanitized, "sanitizer was not exercised"
    assert all(not any(prefix in key for key in seen) for seen in state.sanitized)
    assert model.weights == {"dense.weight": tensors["dense.weight"]}
    assert model.config.text_config["ple_storage"]["manifest"] == str(root / "ple-store.json")
    assert state.processor_kwargs["trust_remote_code"] is False
    assert state.image_kwargs["trust_remote_code"] is False


@pytest.mark.parametrize("affine_evidence", [True, False])
def test_target_shared_scale_extracted_from_fp8_sanitization_only_with_affine_evidence(tmp_path, monkeypatch, affine_evidence):
    prefix = "model.language_model.layers.2.ple.ple_embedding.ngram_embedding."
    tensors = {prefix + "weight_scale": object(), "dense.weight": object()}
    if affine_evidence:
        tensors[prefix + "shards.0.scales"] = object()
    root = _checkpoint(tmp_path, {key: "weights.safetensors" for key in tensors})
    state = _fake_target_backend(monkeypatch, tensors)
    loader._load_target(root)
    assert ((prefix + "weight_scale") in state.sanitized[0]) is not affine_evidence


def test_target_quantizer_preserves_mixed_modules_and_dense_unquantized_weights(tmp_path, monkeypatch):
    tensors = {"dense.weight": object(), "dense.scales": object(), "mixed.weight": object(),
               "mixed.scales": object(), "unquantized.weight": object()}
    override = {"bits": 6, "group_size": 128, "mode": "affine"}
    config = {"model_type": "qwen4_exp", "text_config": {},
              "quantization": {"bits": 4, "group_size": 64, "mixed": override}}
    root = _checkpoint(tmp_path, {key: "weights.safetensors" for key in tensors}, config)
    state = _fake_target_backend(monkeypatch, tensors)
    model, _ = loader._load_target(root)
    assert state.quantization_defaults == {"bits": 4, "group_size": 64, "mode": "affine"}
    assert state.quantized == {"dense": True, "mixed": override, "unquantized": False}
    assert model.weights == tensors


@pytest.mark.parametrize("offload", [True, False])
@pytest.mark.parametrize("has_shared_scale", [True, False])
def test_affine_ple_shared_scale_applied_once_to_rows_in_both_storage_modes(tmp_path, monkeypatch, offload, has_shared_scale):
    prefix = "model.language_model.layers.2.ple.ple_embedding.ngram_embedding."
    tensors = {prefix + "shards.0.weight": object(), prefix + "shards.0.scales": object()}
    if has_shared_scale:
        tensors[prefix + "weight_scale"] = 0.125
    config = {"model_type": "qwen4_exp", "text_config": {}}
    if offload:
        config["text_config"]["ple_storage"] = {"manifest": "ple-store.json"}
    root = _checkpoint(tmp_path, {key: "weights.safetensors" for key in tensors}, config)
    state = _fake_target_backend(monkeypatch, tensors)
    model, _ = loader._load_target(root)
    embedding = model.language_model.model.layers[2].ple.ple_embedding.ngram_embedding
    if has_shared_scale:
        assert embedding is not state.original_embedding
        assert embedding([7]) == 0.25, "shared scale was lost or multiplied twice"
        assert state.embedding_calls == 1
        assert state.casts == 1
        assert embedding.embedding.row_count == 64
    else:
        assert embedding is state.original_embedding, "native affine rows without shared scale were changed"


def test_native_prompt_cache_is_lazy_bounded_and_shared_per_session(monkeypatch):
    from unittest.mock import Mock

    package = types.ModuleType("mlx_vlm")
    package.__path__ = []
    module = types.ModuleType("mlx_vlm.apc")
    factory = Mock(return_value=object())
    module.APCManager = factory
    monkeypatch.setitem(sys.modules, "mlx_vlm", package)
    monkeypatch.setitem(sys.modules, "mlx_vlm.apc", module)
    session = loader.Qwen4Session()
    factory.assert_not_called()
    assert session.prompt_cache() is session.prompt_cache()
    assert factory.call_count == 1
    kwargs = factory.call_args.kwargs
    # The memory budget caps EACH retained snapshot, and mlx-vlm silently skips
    # a snapshot that does not fit. The flat 0.5 GiB once forced here disabled
    # reuse past ~10-19k prompt tokens on hybrid models; absent an operator
    # value the budget is mlx-vlm's machine-relative default, so no constant
    # may be forced for it (nor for the reserve or checkpoint schedule).
    # The checkpoint SHAPE is forced on purpose (measured; mission A3 — one
    # snapshot per call, 8 restorable prompts; pinned in
    # test_mlx_native_apc_budget_unit / test_mlx_native_apc_lineage_unit), but
    # the memory budget and reserve never are.
    for forced in ("memory_max_gb", "memory_reserve_gb"):
        assert forced not in kwargs["overrides"], forced
    assert kwargs["num_blocks"] * kwargs["block_size"] <= 65536
    explicit = loader.Qwen4Session()
    explicit.prompt_cache(memory_max_gb=3)
    assert factory.call_args.kwargs["overrides"]["memory_max_gb"] == 3.0
    assert "memory_reserve_gb" not in factory.call_args.kwargs["overrides"]
