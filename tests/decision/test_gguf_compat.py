"""The llama.cpp compatibility shim: a decision GGUF loses its head tensors and metadata,
keeps every other tensor byte for byte, and the head round-trips. (Synthetic file; no model.)"""

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from abstractcore.decision.gguf_compat import derive_backbone, gguf_has_pointer_head, llamacpp_has_decision_support
from abstractcore.decision.head import PointerHead


def _write_fake_decision_gguf(path, d=16, dp=4):
    from gguf import GGUFWriter, GGUFValueType

    w = GGUFWriter(str(path), "qwen35")
    w.add_key_value("qwen35.embedding_length", d, GGUFValueType.UINT32)
    w.add_key_value("qwen35.embedding_length_out", 2 * dp, GGUFValueType.UINT32)
    w.add_key_value("qwen35.decision.type", "kev", GGUFValueType.STRING)
    w.add_key_value("qwen35.decision.temperature.choice", 2.5, GGUFValueType.FLOAT32)
    w.add_key_value("tokenizer.ggml.tokens", ["a", "b"], GGUFValueType.ARRAY, sub_type=GGUFValueType.STRING)
    w.add_key_value("tokenizer.chat_template.systemone", "{{ state }}", GGUFValueType.STRING)
    rng = np.random.default_rng(0)
    body = rng.standard_normal((d, d)).astype(np.float32)
    head_w = rng.standard_normal((2 * dp, d)).astype(np.float32)
    head_b = rng.standard_normal(2 * dp).astype(np.float32)
    w.add_tensor("blk.0.attn_q.weight", body)
    w.add_tensor("cls.output.weight", head_w)
    w.add_tensor("cls.output.bias", head_b)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(progress=False); w.close()
    return body, head_w, head_b


def test_derive_backbone_strips_head_and_keeps_weights(tmp_path):
    src = tmp_path / "Fake-Q4.gguf"
    body, head_w, head_b = _write_fake_decision_gguf(src)
    assert gguf_has_pointer_head(src)
    backbone, head_dir = derive_backbone(src, out_dir=tmp_path / "derived")
    reader = gguf.GGUFReader(str(backbone))
    names = {t.name for t in reader.tensors}
    assert "cls.output.weight" not in names and "cls.output.bias" not in names and "blk.0.attn_q.weight" in names
    assert "qwen35.decision.type" not in reader.fields and "qwen35.embedding_length_out" not in reader.fields
    assert reader.fields["tokenizer.ggml.tokens"].contents() == ["a", "b"]
    copied = np.asarray([t for t in reader.tensors if t.name == "blk.0.attn_q.weight"][0].data, dtype=np.float32)
    assert np.array_equal(copied.reshape(body.shape), body)
    head = PointerHead.load(str(head_dir))
    assert head.temperature == pytest.approx(2.5) and head.head_dim == 4
    assert np.array_equal(head.q_weight, head_w[:4]) and np.array_equal(head.k_bias, head_b[4:])
    # a second call reuses the cache
    again, _ = derive_backbone(src, out_dir=tmp_path / "derived")
    assert again == backbone


def test_support_probe_answers_a_bool_or_none():
    assert llamacpp_has_decision_support() in (True, False, None)
