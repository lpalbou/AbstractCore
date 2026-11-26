"""MLX-Gen replaced under AbstractVision's real MLX-Gen backend: no weights, no MLX.

The stand-ins are the ones AbstractVision's own suite uses (tests/test_minimax_h3.py and
tests/test_mflux_backend.py). Everything above the engine is real: AbstractVision's request types,
memory modes, refusals, soundtrack handling, metadata and `has_loaded_model()`.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator
from unittest.mock import patch

import pytest

H3 = "AbstractFramework/minimax-h3-8bit"
FLUX = "AbstractFramework/flux.2-klein-4b-8bit"
SEEDVR2 = "AbstractFramework/seedvr2-3b-8bit"
MP4 = b"\x00\x00\x00\x18ftypmp42h3"
PREPARED_SHARD = b"mlx-gen prepared shard"


def _png(width: int, height: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 120, 40)).save(buf, format="PNG")
    return buf.getvalue()


def _wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(32000)
        handle.writeframes(b"\x00\x00" * 2 * 32)
    return buf.getvalue()


# A keyframe the backend can decode (it opens every keyframe before any weights load).
PNG = _png(16, 9)
WAV = _wav()


def _h3_row(task: str) -> SimpleNamespace:
    return SimpleNamespace(
        public_task=task,
        default_for_task=True,
        supports_negative_prompt=False,
        supports_guidance=False,
        supports_lora=True,
        lora_target_roles=("transformer",),
        supports_block_streaming=True,
        generates_audio=True,
        audio_channels=2,
        audio_sample_rate=32000,
        dimension_multiple=32,
        min_frames=124,
        max_frames=345,
        frame_multiple=17,
        frame_remainder=5,
        frame_rounding="up",
        output_fps=24.0,
        default_steps=8,
        default_width=960,
        default_height=544,
        default_frames=124,
        default_flow_shift=12.0,
        default_audio_shift=3.0,
        validated_quantization_bits=(4, 8),
        measured_runs=(SimpleNamespace(footprint_bytes=1, low_ram=True, stream_blocks=True),),
    )


def _capabilities(model=None, base_model=None, **_kwargs) -> SimpleNamespace:
    if "minimax" in str(model or "").lower() or "minimax" in str(base_model or "").lower():
        return SimpleNamespace(capabilities=(_h3_row("text-to-video"), _h3_row("image-to-video")))
    if "seedvr2" in str(model or "").lower() or "seedvr2" in str(base_model or "").lower():
        return SimpleNamespace(capabilities=(SimpleNamespace(public_task="image-upscale", supports_block_streaming=False),))
    return SimpleNamespace(
        capabilities=tuple(
            SimpleNamespace(public_task=task, supports_block_streaming=False) for task in ("text-to-image", "image-to-image")
        )
    )


class FakeH3Video:
    def __init__(self, kwargs: Dict[str, Any], *, mux: bool, model: "FakeMiniMaxH3") -> None:
        self.kwargs = kwargs
        self.mux = mux
        self.model = model

    def save(self, path=None, export_json_metadata=False, overwrite=True):
        path = Path(path)
        path.write_bytes(MP4)
        # As MLX-Gen writes it: with the model path it read and the keyframe path it was given.
        metadata: Dict[str, Any] = {
            "model": self.model.init_kwargs.get("model_path"),
            "image_path": self.kwargs.get("image_path"),
            "seed": self.kwargs["seed"],
            "frames": 124,
            "fps": 24,
        }
        if self.kwargs.get("generate_audio", True):
            metadata.update(audio_present=True, audio_source="generated", audio_channels=2, audio_sample_rate=32000, audio_duration_seconds=5.1667)
            if self.mux:
                metadata.update(audio_muxed=True, audio_codec="aac", audio_mux_mode="ffmpeg_copy_video_aac_audio")
            else:
                wav = path.with_suffix(".wav")
                wav.write_bytes(WAV)
                metadata.update(audio_muxed=False, audio_mux_reason="boom", audio_sidecar_path=str(wav))
        if export_json_metadata:
            path.with_suffix(".metadata.json").write_text(json.dumps(metadata))
        return path


class FakeMiniMaxH3:
    instances: list = []
    mux = True
    # Called with each denoise step index, then "decode": lets a test hold a run mid-way.
    during_step = None

    def __init__(self, **kwargs):
        self.init_kwargs = dict(kwargs)
        self.block_stream = object() if kwargs.get("stream_blocks") else None
        self.transformer = object()
        self.generate_calls: list = []
        FakeMiniMaxH3.instances.append(self)

    def generate_video(self, **kwargs):
        self.generate_calls.append(dict(kwargs))
        if self.transformer is None:
            self.transformer = object()
        progress = kwargs.get("progress_callback")

        def emit(phase, step):
            if progress is not None:
                progress(SimpleNamespace(phase=phase, step=step, total_steps=3, total_frames=124, frame=None, task=None, timestep=None))

        for step in range(3):
            if FakeMiniMaxH3.during_step is not None:
                FakeMiniMaxH3.during_step(step)
            emit("denoise", step + 1)
        emit("decode", 3)
        if kwargs.get("release_transformer") and self.block_stream is None:
            self.transformer = None
        emit("generated", 3)
        return FakeH3Video(kwargs, mux=FakeMiniMaxH3.mux, model=self)


class FakeH3Planning:
    min_aspect_ratio = 1 / 4
    max_aspect_ratio = 4.0
    quantization_bits = (8, 4)

    @staticmethod
    def compose_prompt(prompt, soundscape=None, music=None):
        return prompt

    @staticmethod
    def adapter_targets_modulation(path):
        return False


class FakeWeightLoader:
    @staticmethod
    def is_prepared(component):
        return any(shard.read_bytes() == PREPARED_SHARD for shard in Path(component).glob("*.safetensors"))


class _FakeImage:
    def save(self, fp, format=None):  # noqa: A002
        fp.write(PNG)


class _Callbacks:
    def subscribe_progress(self, callback, *, task=None):
        return lambda: None

    def register(self, callback) -> None:
        return None


class FakeFlux2:
    instances: list = []

    def __init__(self, **kwargs):
        self.init_kwargs = dict(kwargs)
        self.callbacks = _Callbacks()
        self.tiling_config = None
        FakeFlux2.instances.append(self)

    def generate_image(self, **kwargs):
        return SimpleNamespace(image=_FakeImage())


class FakeMemorySaver:
    instances: list = []

    def __init__(self, model, keep_transformer=True, cache_limit_bytes=1000**3, args=None):
        self.model = model
        self.keep_transformer = keep_transformer
        FakeMemorySaver.instances.append(self)


class _FakeModelConfig:
    @staticmethod
    def from_name(model_name, base_model=None):
        return ("config", str(model_name), base_model)


def write_snapshot(root: Path, repo_id: str) -> Path:
    """A Hugging Face cache entry for `repo_id` holding a prepared MLX-Gen package; returns the snapshot."""
    repo_dir = root / f"models--{repo_id.replace('/', '--')}"
    snap = repo_dir / "snapshots" / "abc123"
    (snap / "transformer").mkdir(parents=True, exist_ok=True)
    (snap / "transformer" / "0.safetensors").write_bytes(PREPARED_SHARD)
    (snap / "README.md").write_text("---\nlibrary_name: mlx-gen\n---\n")
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    return snap


def reset_server_state() -> None:
    from abstractcore.server import vision_endpoints

    with vision_endpoints._BACKEND_CACHE_LOCK:
        vision_endpoints._BACKEND_CACHE.clear()
        vision_endpoints._RESIDENCY_RECORDS.clear()
    with vision_endpoints._JOBS_LOCK:
        vision_endpoints._JOBS.clear()


@pytest.fixture()
def mlx_gen(tmp_path, monkeypatch) -> Iterator[Path]:
    """MLX-Gen stand-ins under the real backend, the two packages in a scratch cache; yields that cache."""
    for key in list(os.environ):
        if key.startswith(("ABSTRACTCORE_VISION_", "ABSTRACTVISION_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    class _UnconfiguredMgr:
        def get_capability_default(self, kind, modality=None, task=None):
            return {"source": "not_configured"}

    monkeypatch.setattr("abstractcore.config.manager.get_config_manager", lambda: _UnconfiguredMgr())
    cache = tmp_path / "hf-cache"
    write_snapshot(cache, H3)
    write_snapshot(cache, FLUX)
    write_snapshot(cache, SEEDVR2)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("ABSTRACTVISION_MEMORY_MODE", "")
    FakeMiniMaxH3.instances = []
    FakeMiniMaxH3.mux = True
    FakeMiniMaxH3.during_step = None
    FakeFlux2.instances = []
    FakeMemorySaver.instances = []
    target = "abstractvision.backends.mflux"
    with contextlib.ExitStack() as stack:
        for name, value in (
            ("_lazy_import_mflux", (_FakeModelConfig, FileNotFoundError, FakeFlux2, FakeFlux2, object, object)),
            ("_lazy_import_mflux_minimax_h3", FakeMiniMaxH3),
            ("_lazy_import_mlx_gen_h3_planning", FakeH3Planning),
            ("_lazy_import_mlx_gen_weight_loader", (FakeWeightLoader, ValueError)),
            ("_lazy_import_mlx_gen_task_inference", (_capabilities, None)),
            ("_lazy_import_mlx_gen_memory_saver", FakeMemorySaver),
            ("_set_mlx_cache_limit", 555),
        ):
            stack.enter_context(patch(f"{target}.{name}", return_value=value))
        # The engine is replaced, so its installed versions are not checked.
        stack.enter_context(patch(f"{target}._require_runtime_versions"))
        stack.enter_context(patch(f"{target}._apply_mlx_gen_default_cache_limit"))
        stack.enter_context(patch(f"{target}._clear_mlx_cache"))
        reset_server_state()
        yield cache
        FakeMiniMaxH3.during_step = None
        reset_server_state()
