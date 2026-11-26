"""`llm.vision` and `generate(output=...)` hand the vision plugin every typed option unchanged.

The typed options are AbstractVision's request fields: `memory_mode`, `quantize` and `cancel_event`
on every call, and the soundtrack fields on video calls. Discovery returns what the plugin publishes.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

import pytest

from abstractcore.capabilities.registry import CapabilityRegistry
from abstractcore.server.capability_generation import create_capability_generation_core

_SOUND = {"soundscape": "Paws in dry snow.", "music": "Sparse piano.", "generate_audio": False, "audio_shift": 3.0}


class _RecordingVisionPlugin:
    backend_id = "abstractvision:fake"

    def __init__(self) -> None:
        self.calls: List[tuple[str, Dict[str, Any]]] = []

    def _record(self, name: str, kwargs: Dict[str, Any], result: Any) -> Any:
        self.calls.append((name, dict(kwargs)))
        return result

    def t2i(self, prompt: str, **kwargs: Any) -> bytes:
        return self._record("t2i", kwargs, b"png")

    def t2i_batch(self, prompt: str, **kwargs: Any) -> list:
        return self._record("t2i_batch", kwargs, [b"png"] * int(kwargs.get("count") or 1))

    def i2i(self, prompt: str, image: Any, **kwargs: Any) -> bytes:
        return self._record("i2i", kwargs, b"png")

    def i2i_batch(self, prompt: str, image: Any, **kwargs: Any) -> list:
        return self._record("i2i_batch", kwargs, [b"png"] * int(kwargs.get("count") or 1))

    def upscale_image(self, image: Any, **kwargs: Any) -> bytes:
        return self._record("upscale_image", kwargs, b"png")

    def t2v(self, prompt: str, **kwargs: Any) -> bytes:
        return self._record("t2v", kwargs, b"mp4")

    def t2v_batch(self, prompt: str, **kwargs: Any) -> list:
        return self._record("t2v_batch", kwargs, [b"mp4"] * int(kwargs.get("count") or 1))

    def i2v(self, image: Any, **kwargs: Any) -> bytes:
        return self._record("i2v", kwargs, b"mp4")

    def i2v_batch(self, image: Any, **kwargs: Any) -> list:
        return self._record("i2v_batch", kwargs, [b"mp4"] * int(kwargs.get("count") or 1))

    def available_providers(self, *, task: Any = None) -> Dict[str, Any]:
        return {
            "providers": ["mlx-gen"],
            "available_providers": ["mlx-gen"],
            "details": {"mlx-gen": {"memory_modes": ["default", "low"], "local": True}},
        }

    def list_provider_models(self, *, task: Any = None) -> List[Dict[str, Any]]:
        return [
            {
                "id": "AbstractFramework/minimax-h3-8bit",
                "provider": "mlx-gen",
                "parameter_constraints": {
                    "memory_mode": {"enum": ["default", "low", "lowest"]},
                    "audio": {"channels": 2, "sample_rate": 32000},
                },
            }
        ]


@pytest.fixture()
def plugin_and_facade():
    registry = CapabilityRegistry(object())
    registry._plugins_loaded = True
    plugin = _RecordingVisionPlugin()
    registry.register_vision_backend(backend_id="fake-vision", factory=lambda _owner: plugin)
    return plugin, registry.vision


def test_every_facade_method_forwards_memory_mode_quantize_and_cancel_event(plugin_and_facade):
    plugin, vision = plugin_and_facade
    cancel = threading.Event()
    common = {"memory_mode": "low", "quantize": 8, "cancel_event": cancel}
    vision.t2i("cube", **common)
    vision.t2i_batch("cube", count=2, **common)
    vision.i2i("blue", b"img", **common)
    vision.i2i_batch("blue", b"img", count=2, **common)
    vision.upscale_image(b"img", **common)
    vision.t2v("fox", **common, **_SOUND)
    vision.t2v_batch("fox", count=2, **common, **_SOUND)
    vision.i2v(b"img", prompt="turn", **common, **_SOUND)
    vision.i2v_batch(b"img", prompt="turn", count=2, **common, **_SOUND)

    names = [name for name, _ in plugin.calls]
    assert names == ["t2i", "t2i_batch", "i2i", "i2i_batch", "upscale_image", "t2v", "t2v_batch", "i2v", "i2v_batch"]
    for name, kwargs in plugin.calls:
        assert kwargs["memory_mode"] == "low" and kwargs["quantize"] == 8, name
        assert kwargs["cancel_event"] is cancel, name
        if name.startswith(("t2v", "i2v")):
            assert {key: kwargs[key] for key in _SOUND} == _SOUND, name


def test_discovery_returns_what_the_plugin_publishes(plugin_and_facade):
    _plugin, vision = plugin_and_facade
    details = vision.available_providers(task="text_to_video")["details"]
    assert details["mlx-gen"]["memory_modes"] == ["default", "low"]
    entry = vision.list_provider_models(task="text_to_video")[0]
    assert entry["parameter_constraints"]["memory_mode"]["enum"] == ["default", "low", "lowest"]
    assert entry["parameter_constraints"]["audio"] == {"channels": 2, "sample_rate": 32000}


def test_generate_output_specs_forward_the_typed_options_to_the_plugin():
    plugin = _RecordingVisionPlugin()
    core = create_capability_generation_core(vision_facade=plugin)

    result = core.generate(
        "fox",
        output={"modality": "video", "task": "text_to_video", "memory_mode": "lowest", **_SOUND},
    )
    assert result.outputs["video"][0].data == b"mp4"
    name, kwargs = plugin.calls[-1]
    assert name == "t2v"
    assert kwargs["memory_mode"] == "lowest"
    assert {key: kwargs[key] for key in _SOUND} == _SOUND

    core.generate("cube", output={"modality": "image", "count": 2, "memory_mode": "low", "quantize": 4})
    name, kwargs = plugin.calls[-1]
    assert name == "t2i_batch" and kwargs["count"] == 2
    assert (kwargs["memory_mode"], kwargs["quantize"]) == ("low", 4)

    core.generate(
        "",
        media=[{"type": "image", "content": b"img", "role": "source"}],
        output={"modality": "image", "task": "image_upscale", "memory_mode": "low"},
    )
    name, kwargs = plugin.calls[-1]
    assert name == "upscale_image" and kwargs["memory_mode"] == "low"
