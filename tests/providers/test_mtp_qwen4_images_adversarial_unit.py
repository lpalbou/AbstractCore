"""Native Qwen4 pixel-input and delivery-honesty tests without MLX/GPU."""

import base64
import io
import sys
import threading
import types
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from abstractcore.core.types import GenerateResponse
from abstractcore.media.delivery import MediaReport
from abstractcore.providers import mlx_qwen4
from abstractcore.providers.mlx_provider import MLXProvider


def _png():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def _inputs(monkeypatch, *, grids=None, token_count=4, pixels=True):
    values = {
        "input_ids": np.array([[99] * token_count + [1]]),
        "pixel_values": np.zeros((16, 12)) if pixels else None,
        "image_grid_thw": np.array([[1, 4, 4]] if grids is None else grids),
        "attention_mask": np.ones((1, token_count + 1)),
    }
    root = types.ModuleType("mlx")
    root.__path__ = []
    core = types.ModuleType("mlx.core")
    core.sum = np.sum
    vlm = types.ModuleType("mlx_vlm")
    vlm.__path__ = []
    utils = types.ModuleType("mlx_vlm.utils")
    utils.prepare_inputs = Mock(return_value=values)
    for name, module in (("mlx", root), ("mlx.core", core), ("mlx_vlm", vlm), ("mlx_vlm.utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)
    return values, utils.prepare_inputs


def _model():
    return types.SimpleNamespace(config=types.SimpleNamespace(
        image_token_id=99, vision_config=types.SimpleNamespace(spatial_merge_size=2)))


@pytest.mark.parametrize("source", ["bytes", "base64", "file"])
def test_image_sources_are_decoded_and_delivery_counts_match_expanded_tokens(tmp_path, monkeypatch, source):
    data = _png()
    part = types.SimpleNamespace(content=data, file_path=None)
    if source == "base64":
        part.content = base64.b64encode(data).decode()
    elif source == "file":
        path = tmp_path / "image.png"
        path.write_bytes(data)
        part.file_path = str(path)
        part.content = b"ignored because the actual file is supplied"
    _, prepare = _inputs(monkeypatch)
    result, records = mlx_qwen4.prepare_images(_model(), object(), "rendered prompt", [(2, part)])
    assert records == [{"index": 2, "kind": "image", "content": data, "transport": "mlx_vlm_native", "tokens": 4}]
    assert "mask" in result and "attention_mask" not in result
    assert prepare.call_args.kwargs["prompts"] == "rendered prompt"
    image = prepare.call_args.kwargs["images"][0]
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize("change", ["missing_grid", "wrong_grid_count", "no_pixels", "unexpanded", "wrong_token_count", "empty_pixels"])
def test_incomplete_or_inconsistent_image_inputs_fail_closed(monkeypatch, change):
    values, _ = _inputs(monkeypatch)
    if change == "missing_grid":
        values.pop("image_grid_thw")
    elif change == "wrong_grid_count":
        values["image_grid_thw"] = np.array([[1, 4, 4], [1, 4, 4]])
    elif change == "no_pixels":
        values["pixel_values"] = None
    elif change == "unexpanded":
        values["input_ids"] = np.array([[99]])
    elif change == "wrong_token_count":
        values["input_ids"] = np.array([[99, 99, 99]])
    elif change == "empty_pixels":
        values["pixel_values"] = np.empty((0, 12))
    part = types.SimpleNamespace(content=_png(), file_path=None)
    with pytest.raises(ValueError, match="pixel|grid|placeholder"):
        mlx_qwen4.prepare_images(_model(), object(), "rendered", [(0, part)])


def _provider():
    p = MLXProvider.__new__(MLXProvider)
    p.model = "native/qwen4"
    p.logger = Mock()
    p._native_qwen4 = types.SimpleNamespace()
    p._mtp_processor = object()
    p._mtp_drafter = None
    p._mtp_generation_lock = threading.Lock()
    p._speculation_request = None
    p._mtp_prompt_cache_warned = False
    return p


@pytest.mark.parametrize("error_chunk", [False, True])
def test_stream_delivery_is_stamped_after_lazy_forward_not_before(error_chunk):
    p = _provider()

    def core(prompt, *, report, **kwargs):
        p._native_media_report = report

        def stream():
            report.deliver(index=0, kind="image", content=b"pixels", tokens=4, transport="mlx_vlm_native")
            yield GenerateResponse(content="red", model=p.model, finish_reason="error" if error_chunk else None)

        return stream()

    p._generate_core = core
    stream = p._generate_internal("color?", media=[object()], stream=True)
    try:
        chunk = next(stream)
        delivered = (chunk.metadata or {}).get("media_delivered")
        if error_chunk:
            assert not delivered, "failed generation claimed visual delivery"
        else:
            assert delivered and delivered[0]["tokens"] == 4
    finally:
        stream.close()


def test_failed_forward_does_not_claim_delivery_and_next_call_has_no_image_state(monkeypatch):
    p = _provider()
    report = MediaReport(provider="mlx", model=p.model, requested=1)
    p._native_media_report = report
    p._native_media_records = [{"index": 0, "kind": "image", "content": b"pixels", "tokens": 4, "transport": "mlx_vlm_native"}]
    p._native_media = [(0, object())]
    p._native_image_kwargs = Mock(return_value={"input_ids": object(), "pixel_values": object()})
    module = types.ModuleType("mlx_vlm")
    module.generate = Mock(side_effect=RuntimeError("vision forward failed"))
    monkeypatch.setitem(sys.modules, "mlx_vlm", module)
    with pytest.raises(RuntimeError, match="vision forward failed"):
        p._mtp_generate_fn(object(), object(), prompt="color?")
    assert report.delivered == []
    p._apply_per_call_speculation(None)
    assert p._native_media == []
    assert p._native_media_records == []
    assert p._native_media_report is None
