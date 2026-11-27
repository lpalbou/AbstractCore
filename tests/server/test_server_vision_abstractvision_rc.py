"""AbstractVision on both sides of AbstractCore, with MLX-Gen replaced (`mlx_gen_stand_ins`).

`llm.vision` reaches AbstractVision's own AbstractCore plugin through the capability registry, and
AbstractVision's OpenAI-compatible client drives this server over loopback.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import patch

import pytest

from abstractcore.server.app import app
from tests.server.mlx_gen_stand_ins import FLUX, H3, MP4, WAV, FakeMiniMaxH3, mlx_gen  # noqa: F401
from tests.server.vision_jobs import HANG_GUARD_S, join_job

abstractvision_types = pytest.importorskip("abstractvision.types")

pytestmark = pytest.mark.usefixtures("mlx_gen")

_REAL_URLOPEN = __import__("urllib.request", fromlist=["urlopen"]).urlopen
# The id this server routes to MLX-Gen (a bare `org/name` id routes to Diffusers).
_ROUTED_H3 = f"mlx-gen/{H3}"


def _plugin_vision(model: str):
    """`llm.vision` backed by AbstractVision's AbstractCore plugin, configured for one MLX-Gen model."""
    from abstractcore.capabilities.registry import CapabilityRegistry
    from abstractvision.integrations.abstractcore_plugin import register

    owner = SimpleNamespace(config={"vision_backend": "mlx-gen", "vision_mflux_model": model})
    registry = CapabilityRegistry(owner)
    registry._plugins_loaded = True
    register(registry)
    return registry.vision


def test_llm_vision_hands_every_typed_option_to_the_plugin():
    vision = _plugin_vision(H3)
    out = vision.t2v(
        "fox",
        provider="mlx-gen",
        model=H3,
        memory_mode="low",
        soundscape="Paws in dry snow.",
        music="Sparse piano.",
        generate_audio=True,
        audio_shift=2.5,
        cancel_event=threading.Event(),
    )

    assert out == MP4
    call = FakeMiniMaxH3.instances[-1].generate_calls[-1]
    assert (call["soundscape"], call["music"], call["audio_shift"], call["generate_audio"]) == (
        "Paws in dry snow.",
        "Sparse piano.",
        2.5,
        True,
    )
    assert call["release_transformer"] is True  # memory_mode="low"
    entries = vision.list_provider_models(task="text_to_video")
    h3 = [e for e in entries if e.get("id") == H3]
    assert h3 and h3[0]["parameter_constraints"]["memory_mode"]["enum"] == ["default", "low", "lowest"]


def test_llm_vision_memory_peaks_returns_the_plugin_rows():
    rows = [{"task": "text_to_video", "memory_mode": "low", "width": 8, "height": 544, "frames": 124, "steps": 8, "peak_bytes": 1, "machine_total_ram_bytes": 2}]
    with patch("abstractvision.model_inventory.memory_peaks", return_value=rows) as peaks:
        assert _plugin_vision(H3).memory_peaks(H3) == rows
    peaks.assert_called_once_with(H3)


def test_llm_vision_control_image_is_refused_with_its_field():
    from abstractvision.errors import CapabilityNotSupportedError

    with pytest.raises(CapabilityNotSupportedError) as refused:
        _plugin_vision(FLUX).t2i("cube", provider="mlx-gen", model=FLUX, control_image=b"png")
    assert (refused.value.field, refused.value.supported) == ("control_image", ())


# ---------------------------------------------------------------------------
# AbstractVision's OpenAI-compatible client against this server, over loopback
# ---------------------------------------------------------------------------


@pytest.fixture()
def served(monkeypatch):
    """The AbstractCore app on a free loopback port, its MLX-Gen default MiniMax-H3."""
    import socket

    import uvicorn
    from abstractvision.backends import openai_compatible

    # The client module binds `urlopen` when first imported; a test that patched
    # urllib.request.urlopen at that moment would leave it offline.
    monkeypatch.setattr(openai_compatible, "urlopen", _REAL_URLOPEN)
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setenv("ABSTRACTCORE_VISION_BACKEND", "mlx-gen")
    monkeypatch.setenv("ABSTRACTCORE_VISION_MFLUX_MODEL", H3)
    # The server is handed the bound socket: no other process can take the port in between.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    started = threading.Event()
    startup = server.startup

    async def signalling_startup(sockets=None):
        try:
            await startup(sockets=sockets)
        finally:
            started.set()

    server.startup = signalling_startup  # type: ignore[method-assign]
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        assert started.wait(HANG_GUARD_S) and server.started
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        thread.join(HANG_GUARD_S)
        sock.close()


def _client(base_url: str, model_id=None):
    from abstractvision.backends.openai_compatible import OpenAICompatibleBackendConfig, OpenAICompatibleVisionBackend

    return OpenAICompatibleVisionBackend(
        config=OpenAICompatibleBackendConfig(base_url=base_url, model_id=model_id, text_to_video_path="/videos/generations", timeout_s=30.0)
    )


def test_the_abstractvision_client_drives_this_server_end_to_end(served):
    from abstractvision import VisionManager
    from abstractvision.errors import CapabilityNotSupportedError

    remote = _client(served, model_id=_ROUTED_H3)
    asset = VisionManager(backend=remote).generate_video(
        "A red fox trots through snow.",
        memory_mode="lowest",
        soundscape="Paws in dry snow.",
        music="Sparse piano.",
        generate_audio=True,
        seed=42,
    )

    model = FakeMiniMaxH3.instances[-1]
    call = model.generate_calls[-1]
    assert model.init_kwargs.get("stream_blocks") is True  # lowest reached the backend
    assert (call["soundscape"], call["music"], call["generate_audio"], call["seed"]) == ("Paws in dry snow.", "Sparse piano.", True, 42)
    assert asset.data == MP4
    assert (asset.audio.sample_rate, asset.audio.channels, asset.audio.muxed, asset.audio.codec) == (32000, 2, True, "aac")
    assert asset.metadata["memory_mode"] == "lowest"
    caps = remote.get_capabilities()
    assert tuple(caps.memory_modes) == ("default", "low", "lowest")
    assert {"soundscape", "music", "generate_audio", "audio_shift"} <= set(remote.supported_request_options("text_to_video"))

    # The server's 422 becomes CapabilityNotSupportedError on the client, before any model load.
    instances = len(FakeMiniMaxH3.instances)
    with pytest.raises(CapabilityNotSupportedError):
        remote.generate_video(abstractvision_types.VideoGenerationRequest(prompt="fox", memory_mode="lowest", quantize=4))
    assert len(FakeMiniMaxH3.instances) == instances

    # An unmuxed soundtrack reaches the client as WAV bytes with the codec the server reports.
    FakeMiniMaxH3.mux = False
    unmuxed = remote.generate_video(abstractvision_types.VideoGenerationRequest(prompt="fox", memory_mode="low"))
    assert (unmuxed.audio.muxed, unmuxed.audio.codec, unmuxed.audio.data) == (False, "pcm_s16le", WAV)
    FakeMiniMaxH3.mux = True

    # A mode the server does not list for the model is refused by the client from discovery.
    flux = _client(served, model_id=f"mlx-gen/{FLUX}")
    with pytest.raises(CapabilityNotSupportedError):
        flux.generate_image(abstractvision_types.ImageGenerationRequest(prompt="cube", memory_mode="lowest"))


def test_a_client_naming_no_model_uses_the_server_default_of_the_task(served):
    remote = _client(served)
    asset = remote.generate_video(abstractvision_types.VideoGenerationRequest(prompt="fox", memory_mode="lowest"))
    assert asset.metadata["memory_mode"] == "lowest"


def test_the_client_cancel_event_cancels_the_server_job(served):
    from abstractcore.server import vision_endpoints
    from abstractvision.errors import GenerationCancelledError

    reached = threading.Event()

    def the_job() -> Dict[str, Any]:
        with vision_endpoints._JOBS_LOCK:
            (job,) = vision_endpoints._JOBS.values()
            return job

    def during_step(step) -> None:
        if step == 1:
            reached.set()
            # Held here until the client's cancel reached the server; the next step then stops.
            assert the_job()[vision_endpoints._JOB_CANCEL_KEY].wait(HANG_GUARD_S), "the cancel never reached the job"

    FakeMiniMaxH3.during_step = during_step
    cancel = threading.Event()
    outcome: Dict[str, Any] = {}

    def run() -> None:
        try:
            _client(served, model_id=_ROUTED_H3).generate_video(
                abstractvision_types.VideoGenerationRequest(prompt="fox", memory_mode="low", cancel_event=cancel)
            )
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    worker = threading.Thread(target=run)
    worker.start()
    assert reached.wait(HANG_GUARD_S)
    cancel.set()
    worker.join(HANG_GUARD_S)
    assert not worker.is_alive()

    assert isinstance(outcome.get("error"), GenerationCancelledError), outcome
    join_job(the_job()["id"])
    assert the_job()["state"] == "cancelled"
    # The backend's run stopped at its next step: one call, never finished.
    assert len(FakeMiniMaxH3.instances[-1].generate_calls) == 1
