"""The vision HTTP contract for memory modes, load-time quantization and video soundtracks.

The backend is AbstractVision's real MLX-Gen backend with MLX-Gen replaced (`mlx_gen_stand_ins`):
the request types, the refusals with their `field` / `supported`, the memory modes, the soundtrack
summary, `metadata["memory_mode"]` and `has_loaded_model()` are AbstractVision's own. The server's
part is the wire: typed fields in, structured errors out, one generation at a time, no path.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List
from unittest.mock import patch

import pytest
from abstractvision.backends.base_backend import SOUNDTRACK_REQUEST_FIELDS
from abstractvision.backends.mflux import _require_runtime_versions as _REAL_VERSION_GATE
from fastapi import HTTPException
from fastapi.testclient import TestClient

from abstractcore.server import vision_endpoints
from abstractcore.server.app import app
from tests.server.mlx_gen_stand_ins import (  # noqa: F401 - `mlx_gen` is a fixture
    FLUX,
    H3,
    MP4,
    PNG,
    SEEDVR2,
    WAV,
    FakeFlux2,
    FakeMiniMaxH3,
    mlx_gen,
)
from tests.server.vision_jobs import HANG_GUARD_S, finished_job

pytestmark = pytest.mark.usefixtures("mlx_gen")


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def hf_cache() -> Path:
    """The scratch Hugging Face cache the stand-ins hold the packages in."""
    return Path(os.environ["HF_HUB_CACHE"])


@pytest.fixture()
def h3_route(monkeypatch) -> Dict[str, Any]:
    """A configured text-to-video route on MiniMax-H3; the test fills its `options`."""
    route: Dict[str, Any] = {"provider": "mlx-gen", "model": H3, "options": {}}

    class _Mgr:
        def get_capability_default(self, kind, modality=None, task=None):
            if kind == "output" and modality == "video":
                return dict(route)
            return {"source": "not_configured"}

    monkeypatch.setattr("abstractcore.config.manager.get_config_manager", lambda: _Mgr())
    return route


def _t2v(client: TestClient, **fields: Any):
    return client.post("/v1/videos/generations", json={"prompt": "A red fox trots through snow.", "provider": "mlx-gen", "model": H3, **fields})


def _t2i(client: TestClient, **fields: Any):
    return client.post("/v1/images/generations", json={"prompt": "cube", "provider": "mlx-gen", "model": FLUX, **fields})


def _wait_job(client: TestClient, job_id: str) -> Dict[str, Any]:
    job = finished_job(client, job_id)
    assert job["state"] in {"succeeded", "failed", "cancelled"}, job
    return job


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _paths_in(body: Any, roots: List[str]) -> List[str]:
    return [text for text in _strings(body) if any(root in text for root in roots)]


# --- Typed fields in, soundtrack and memory mode out ----------------------------------------------


def test_typed_sound_and_memory_fields_reach_the_engine_and_the_response_reports_them(client):
    resp = _t2v(client, soundscape="Paws in dry snow.", music="Sparse piano.", audio_shift=2.5, seed=42, memory_mode="low")

    assert resp.status_code == 200, resp.text
    call = FakeMiniMaxH3.instances[0].generate_calls[0]
    assert (call["soundscape"], call["music"], call["audio_shift"], call["seed"]) == ("Paws in dry snow.", "Sparse piano.", 2.5, 42)
    assert call["release_transformer"] is True  # low
    body = resp.json()
    assert base64.b64decode(body["data"][0]["b64_json"]) == MP4
    assert body["audio"] == body["data"][0]["audio"]
    assert {k: body["audio"][k] for k in ("present", "sample_rate", "channels", "codec", "muxed")} == {
        "present": True,
        "sample_rate": 32000,
        "channels": 2,
        "codec": "aac",
        "muxed": True,
    }
    assert body["memory_mode_used"] == "low"


def test_a_run_without_a_memory_mode_reports_default(client):
    body = _t2v(client).json()
    assert body["memory_mode_used"] == "default"
    assert FakeMiniMaxH3.instances[0].generate_calls[0].get("release_transformer") in (None, False)


def test_extra_carries_the_options_of_a_backend_that_lists_them(client):
    resp = _t2v(client, extra={"memory_mode": "low", "music": "piano"})

    assert resp.status_code == 200, resp.text
    assert FakeMiniMaxH3.instances[0].generate_calls[0]["music"] == "piano"
    assert resp.json()["memory_mode_used"] == "low"


def test_an_unmuxed_soundtrack_comes_inline_and_a_silent_clip_reports_no_audio(client):
    FakeMiniMaxH3.mux = False
    audio = _t2v(client).json()["audio"]
    assert (audio["present"], audio["muxed"], audio["codec"]) == (True, False, "pcm_s16le")
    assert base64.b64decode(audio["wav"]["b64_json"]) == WAV
    assert audio["wav"]["mime_type"] == "audio/wav"

    silent = _t2v(client, generate_audio=False).json()
    assert silent["audio"] == {"present": False}
    assert FakeMiniMaxH3.instances[0].generate_calls[-1]["generate_audio"] is False


def test_an_image_response_carries_no_audio(client):
    body = _t2i(client).json()
    assert "audio" not in body and "audio" not in body["data"][0]


def test_multipart_routes_take_the_typed_fields(client):
    resp = client.post(
        "/v1/videos/edits",
        data={"prompt": "fox", "provider": "mlx-gen", "model": H3, "music": "piano", "memory_mode": "low", "generate_audio": "false", "audio_shift": "2.0"},
        files={"image": ("k.png", PNG, "image/png")},
    )

    assert resp.status_code == 200, resp.text
    call = FakeMiniMaxH3.instances[0].generate_calls[0]
    assert (call["music"], call["generate_audio"], call["audio_shift"], call["release_transformer"]) == ("piano", False, 2.0, True)
    assert resp.json()["audio"] == {"present": False}


# --- Refusals: the backend decides, the server relays its field ----------------------------------


def test_a_memory_mode_the_model_cannot_honour_is_422_with_its_field_and_leaves_nothing_loaded(client):
    resp = _t2i(client, memory_mode="lowest")

    assert resp.status_code == 422, resp.text
    error = resp.json()["error"]
    assert (error["code"], error["field"], error["supported"]) == ("capability_not_supported", "memory_mode", ["default", "low"])
    assert FakeFlux2.instances == []
    assert vision_endpoints._BACKEND_CACHE == {}
    assert client.get("/acore/models/loaded").json()["data"] == []


_EVERY_ROUTE = [
    ("images/generations", {"json": {"prompt": "cube", "provider": "mlx-gen", "model": FLUX}}),
    ("images/edits", {"data": {"prompt": "x", "provider": "mlx-gen", "model": FLUX}, "files": {"image": ("i.png", PNG, "image/png")}}),
    ("images/upscale", {"data": {"provider": "mlx-gen", "model": SEEDVR2}, "files": {"image": ("i.png", PNG, "image/png")}}),
    ("videos/generations", {"json": {"prompt": "fox", "provider": "mlx-gen", "model": H3}}),
    ("videos/edits", {"data": {"prompt": "fox", "provider": "mlx-gen", "model": H3}, "files": {"image": ("k.png", PNG, "image/png")}}),
]


def _with_memory_mode(kwargs: Dict[str, Any], mode: str) -> Dict[str, Any]:
    body = "json" if "json" in kwargs else "data"
    return {**kwargs, body: {**kwargs[body], "memory_mode": mode}}


@pytest.mark.parametrize(("route", "kwargs"), _EVERY_ROUTE, ids=[r for r, _ in _EVERY_ROUTE])
def test_a_value_the_backend_rejects_as_invalid_is_400_on_every_route_and_job(client, route, kwargs):
    # AbstractVision: an unknown mode name is not a valid value (ValueError), unlike a valid mode
    # the model cannot honour (CapabilityNotSupportedError, 422).
    sync = client.post(f"/v1/{route}", **_with_memory_mode(kwargs, "turbo"))
    assert sync.status_code == 400, sync.text
    assert sync.json()["error"]["code"] == "invalid_request"
    assert "turbo" in sync.json()["error"]["message"]

    job_id = client.post(f"/v1/vision/jobs/{route}", **_with_memory_mode(kwargs, "turbo")).json()["job_id"]
    job = _wait_job(client, job_id)
    assert (job["state"], job["error_code"]) == ("failed", "invalid_request")
    assert "turbo" in job["error"]
    assert FakeFlux2.instances == [] and FakeMiniMaxH3.instances == []


def test_quantize_on_a_prepared_package_is_422_before_any_load(client):
    resp = _t2v(client, quantize=8)

    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "capability_not_supported"
    assert FakeMiniMaxH3.instances == []
    assert vision_endpoints._BACKEND_CACHE == {}


def test_the_prepared_package_refusal_names_its_field_and_no_path(client, hf_cache):
    error = _t2v(client, quantize=8).json()["error"]
    assert error["field"] == "quantize"
    assert str(hf_cache) not in error["message"]


def test_a_value_given_at_top_level_and_in_extra_must_agree(client):
    conflict = _t2v(client, memory_mode="low", extra={"memory_mode": "lowest"})
    assert conflict.status_code == 400
    assert conflict.json()["error"]["code"] == "invalid_request"
    assert FakeMiniMaxH3.instances == []

    same = _t2v(client, music="piano", extra={"music": "piano"})
    assert same.status_code == 200, same.text


@pytest.mark.parametrize(
    ("fields", "field"),
    [
        ({"quantize": True}, "quantize"),
        ({"quantize": "8"}, "quantize"),
        ({"memory_mode": 5}, "memory_mode"),
        ({"generate_audio": "yes"}, "generate_audio"),
        ({"music": 3}, "music"),
    ],
)
def test_malformed_typed_values_are_refused_with_their_field(client, fields, field):
    resp = _t2v(client, **fields)

    assert resp.status_code == 422, resp.text
    error = resp.json()["error"]
    assert (error["code"], error["field"]) == ("validation_error", field)
    assert FakeMiniMaxH3.instances == []


def test_a_non_finite_audio_shift_is_refused_and_the_error_is_valid_json(client):
    resp = client.post(
        "/v1/videos/generations",
        content=b'{"prompt": "fox", "provider": "mlx-gen", "model": "' + H3.encode() + b'", "audio_shift": NaN}',
        headers={"content-type": "application/json"},
    )

    assert resp.status_code == 422
    error = resp.json()["error"]
    assert (error["field"], error["details"][0]["input"]) == ("audio_shift", "nan")


@pytest.mark.parametrize(("name", "value"), [("generate_audio", "maybe"), ("audio_shift", "nan"), ("quantize", "abc")])
def test_malformed_form_values_are_400(client, name, value):
    resp = client.post(
        "/v1/videos/edits",
        data={"prompt": "fox", "provider": "mlx-gen", "model": H3, name: value},
        files={"image": ("k.png", PNG, "image/png")},
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["code"] == "invalid_request"


def test_a_boolean_quantize_is_a_type_error(client):
    assert _t2v(client, quantize=True).json()["error"]["field"] == "quantize"  # 422 validation_error
    in_extra = _t2v(client, extra={"quantize": True})
    assert (in_extra.status_code, in_extra.json()["error"]["code"]) == (400, "invalid_request")
    form = client.post(
        "/v1/videos/edits",
        data={"prompt": "fox", "provider": "mlx-gen", "model": H3, "quantize": "true"},
        files={"image": ("k.png", PNG, "image/png")},
    )
    assert (form.status_code, form.json()["error"]["code"]) == (400, "invalid_request")
    assert FakeMiniMaxH3.instances == []


@pytest.mark.skipif(sys.platform != "darwin", reason="AbstractVision checks the MLX version on macOS only")
def test_an_engine_outside_abstractvisions_version_bounds_is_501_with_its_message(client):
    from importlib import metadata

    from abstractvision.backends import mflux

    real_version = metadata.version

    def old_mlx(distribution):
        return {"mlx": "0.31.2", "mlx-gen": "0.40.2"}.get(distribution) or real_version(distribution)

    # The stand-ins skip the engine's version check; this test puts AbstractVision's own back.
    _REAL_VERSION_GATE.cache_clear()
    try:
        with patch("importlib.metadata.version", side_effect=old_mlx), patch.object(mflux, "_require_runtime_versions", _REAL_VERSION_GATE):
            resp = _t2i(client)
    finally:
        _REAL_VERSION_GATE.cache_clear()

    assert resp.status_code == 501, resp.text
    error = resp.json()["error"]
    assert error["code"] == "not_implemented"
    assert "mlx>=0.32.4" in error["message"] and "0.31.2" in error["message"]


def test_a_model_a_failed_default_run_left_loaded_is_listed_and_unloads(client):
    def boom(step):
        if step == 1:
            raise RuntimeError("engine stopped")

    FakeMiniMaxH3.during_step = boom
    assert _t2v(client).status_code == 500
    FakeMiniMaxH3.during_step = None

    (record,) = client.get("/acore/models/loaded").json()["data"]
    assert (record["model"], record["loaded"], record["pinned"]) == (H3, True, False)
    unload = client.post("/acore/models/unload", json={"task": "text_to_video", "provider": "mlx-gen", "model": H3})
    assert unload.status_code == 200, unload.text
    assert client.get("/acore/models/loaded").json()["data"] == []


# An empty form field is not sent (FastAPI reads it as absent), so "" cannot reach the backend by form.
@pytest.mark.parametrize("value", ["0", "none", "false", "off", "no", "8"])
@pytest.mark.parametrize("route", ["/v1/images/upscale", "/v1/vision/jobs/images/upscale"])
def test_upscale_quantize_reaches_the_backend_as_sent(client, route, value):
    from abstractvision.backends.mflux import MFluxVisionBackend
    from abstractvision.types import GeneratedAsset

    seen: List[Any] = []

    def upscale_image(self, request, progress_callback=None):
        # The backend's entry points (a job enters through the progress one); AbstractVision
        # normalizes the spelling after them.
        seen.append(request.quantize)
        return GeneratedAsset(media_type="image", data=PNG, mime_type="image/png", metadata={"memory_mode": "default"})

    with patch.object(MFluxVisionBackend, "upscale_image", upscale_image), patch.object(
        MFluxVisionBackend, "upscale_image_with_progress", upscale_image
    ):
        resp = client.post(route, data={"provider": "mlx-gen", "model": SEEDVR2, "quantize": value}, files={"image": ("i.png", PNG, "image/png")})
        if route.startswith("/v1/vision/jobs/"):
            assert _wait_job(client, resp.json()["job_id"])["state"] == "succeeded"
        else:
            assert resp.status_code == 200, resp.text

    assert seen == [int(value) if value.isdigit() else value]


# --- Jobs: the synchronous body, the refusal's fields, cancellation -------------------------------


def test_a_job_result_is_the_synchronous_body(client):
    job_id = client.post("/v1/vision/jobs/videos/generations", json={"prompt": "fox", "provider": "mlx-gen", "model": H3, "memory_mode": "low", "n": 2}).json()["job_id"]
    job = _wait_job(client, job_id)

    assert job["state"] == "succeeded", job
    result = job["result"]
    assert [item["audio"]["present"] for item in result["data"]] == [True, True]
    assert result["audio"] == result["data"][0]["audio"]
    assert result["memory_mode_used"] == "low"
    assert "_cancel_event" not in job


def test_a_refused_job_carries_the_refusal_fields(client):
    job_id = client.post("/v1/vision/jobs/images/generations", json={"prompt": "cube", "provider": "mlx-gen", "model": FLUX, "memory_mode": "lowest"}).json()["job_id"]
    job = _wait_job(client, job_id)

    assert job["state"] == "failed"
    assert (job["error_code"], job["error_field"], job["error_supported"]) == ("capability_not_supported", "memory_mode", ["default", "low"])


def test_cancelling_a_job_stops_the_run_at_its_next_step(client):
    reached, proceed = threading.Event(), threading.Event()

    def during_step(step):
        if step == 1:
            reached.set()
            assert proceed.wait(HANG_GUARD_S)

    FakeMiniMaxH3.during_step = during_step
    job_id = client.post("/v1/vision/jobs/videos/generations", json={"prompt": "fox", "provider": "mlx-gen", "model": H3}).json()["job_id"]
    assert reached.wait(HANG_GUARD_S)
    cancel = client.post(f"/v1/vision/jobs/{job_id}/cancel").json()
    proceed.set()

    assert cancel["cancel_requested"] is True
    assert _wait_job(client, job_id)["state"] == "cancelled"
    assert len(FakeMiniMaxH3.instances[0].generate_calls) == 1
    assert client.post("/v1/vision/jobs/nope/cancel").status_code == 404


# --- One generation at a time; a client that leaves cancels its run --------------------------------


class _ObservedGate:
    """The generation gate, signalling `waiting` once a run has found it held."""

    def __init__(self, gate: Any) -> None:
        self._gate = gate
        self.waiting = threading.Event()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        acquired = self._gate.acquire(blocking, timeout)
        if not acquired:
            self.waiting.set()
        return acquired

    def release(self) -> None:
        self._gate.release()

    def locked(self) -> bool:
        return self._gate.locked()

    def __enter__(self) -> "_ObservedGate":
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


def test_a_job_is_refused_while_a_synchronous_run_holds_the_machine_and_a_second_request_waits(client, monkeypatch):
    gate = _ObservedGate(vision_endpoints._GENERATION_GATE)
    monkeypatch.setattr(vision_endpoints, "_GENERATION_GATE", gate)
    reached, proceed = threading.Event(), threading.Event()

    def during_step(step):
        if step == 0 and not reached.is_set():
            reached.set()
            assert proceed.wait(HANG_GUARD_S)

    FakeMiniMaxH3.during_step = during_step
    results: Dict[str, Any] = {}
    first = threading.Thread(target=lambda: results.setdefault("video", _t2v(client)))
    first.start()
    assert reached.wait(HANG_GUARD_S)

    refused = client.post("/v1/vision/jobs/images/generations", json={"prompt": "cube", "provider": "mlx-gen", "model": FLUX})
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "conflict"

    second = threading.Thread(target=lambda: results.setdefault("image", _t2i(client)))
    second.start()
    assert gate.waiting.wait(HANG_GUARD_S)  # the image run found the gate held by the video run
    assert FakeFlux2.instances == []  # waiting for the gate, not running beside the video
    proceed.set()
    first.join(HANG_GUARD_S)
    second.join(HANG_GUARD_S)
    assert not first.is_alive() and not second.is_alive()
    assert results["video"].status_code == 200 and results["image"].status_code == 200
    assert not vision_endpoints._GENERATION_GATE.locked()


class _ClientThatLeaves:
    """A synchronous video request whose client reports a disconnect once `left` is set.

    The server polls the client while the run goes on; it then sets the run's cancel event.
    """

    def __init__(self, monkeypatch) -> None:
        self.left = threading.Event()
        self.cancel_events: List[threading.Event] = []
        execute = vision_endpoints._execute

        def recording_execute(run, cancel_event, item_hooks=None):
            self.cancel_events.append(cancel_event)
            return execute(run, cancel_event, item_hooks)

        monkeypatch.setattr(vision_endpoints, "_execute", recording_execute)

    def leave_and_wait_for_the_cancel(self) -> None:
        # Called on the run's thread, which recorded its cancel event before it ran.
        self.left.set()
        assert self.cancel_events[0].wait(HANG_GUARD_S), "the server never cancelled the run"

    def run(self) -> HTTPException:
        from starlette.requests import Request

        request = Request({"type": "http", "method": "POST", "path": "/v1/videos/generations", "headers": [], "query_string": b""})
        left = self.left

        async def disconnected() -> bool:
            return left.is_set()

        request.is_disconnected = disconnected  # type: ignore[method-assign]
        run = vision_endpoints._video_generation_run(request, {"prompt": "fox", "provider": "mlx-gen", "model": H3})
        with pytest.raises(HTTPException) as caught:
            asyncio.run(vision_endpoints._run_and_wait(request, run))
        return caught.value


def test_a_synchronous_run_is_cancelled_when_its_client_disconnects_mid_run(monkeypatch):
    client = _ClientThatLeaves(monkeypatch)

    def during_step(step):
        if step == 0:
            client.leave_and_wait_for_the_cancel()

    FakeMiniMaxH3.during_step = during_step

    error = client.run()

    assert error.status_code == 409
    assert client.cancel_events[0].is_set()
    assert len(FakeMiniMaxH3.instances[0].generate_calls) == 1  # stopped at the step after step 0
    assert not vision_endpoints._GENERATION_GATE.locked()


def test_a_client_that_disconnects_while_the_model_loads_gets_409_and_the_loaded_model_is_listed(monkeypatch):
    client = _ClientThatLeaves(monkeypatch)
    FakeMiniMaxH3.during_load = client.leave_and_wait_for_the_cancel

    error = client.run()

    assert error.status_code == 409
    assert FakeMiniMaxH3.instances[0].generate_calls == []  # cancelled before the engine ran
    assert not vision_endpoints._GENERATION_GATE.locked()
    # The model it loaded is held in memory: listed, so it can be unloaded.
    (record,) = TestClient(app).get("/acore/models/loaded").json()["data"]
    assert (record["model"], record["loaded"]) == (H3, True)


def test_a_client_that_disconnects_while_waiting_for_the_gate_gets_409_and_loads_nothing(monkeypatch):
    client = _ClientThatLeaves(monkeypatch)
    client.left.set()
    gate = vision_endpoints._GENERATION_GATE
    assert gate.acquire(blocking=False)  # another generation holds the machine
    try:
        error = client.run()
        assert gate.locked()  # the refused run never took nor released the other run's gate
    finally:
        gate.release()

    assert error.status_code == 409
    assert client.cancel_events[0].is_set()
    assert FakeMiniMaxH3.instances == []
    assert vision_endpoints._BACKEND_CACHE == {}


# --- Discovery ---------------------------------------------------------------------------------------


def test_model_entries_carry_their_memory_modes_and_soundtrack_fields(client):
    video = {e["id"]: e for e in client.get("/v1/vision/models", params={"task": "text_to_video"}).json()["models"]}
    image = {e["id"]: e for e in client.get("/v1/vision/models", params={"task": "text_to_image"}).json()["models"]}

    assert video[H3]["memory_modes"] == ["default", "low", "lowest"]
    assert (video[H3]["supports_audio"], video[H3]["audio_fields"]) == (True, list(SOUNDTRACK_REQUEST_FIELDS))
    assert image[FLUX]["memory_modes"] == ["default", "low"]
    assert (image[FLUX]["supports_audio"], image[FLUX]["audio_fields"]) == (False, [])
    assert "memory_peaks" not in video[H3]


def test_detail_relays_abstractvision_memory_peaks_as_they_are(client):
    rows = [{"task": "text_to_video", "memory_mode": "lowest", "width": 8, "height": 544, "frames": 124, "steps": 8, "peak_bytes": 11, "machine_total_ram_bytes": 64}]
    asked: List[Any] = []

    def memory_peaks(repo_id):
        asked.append(repo_id)
        return list(rows) if repo_id == H3 else []

    with patch("abstractvision.model_inventory.memory_peaks", side_effect=memory_peaks):
        entries = client.get("/v1/vision/models", params={"task": "text_to_video", "detail": "true"}).json()["models"]

    by_id = {e["id"]: e for e in entries}
    assert by_id[H3]["memory_peaks"] == rows
    assert all(e["memory_peaks"] == [] for e in entries if e["id"] != H3)
    assert H3 in asked


def test_a_local_default_is_listed_per_task_and_runs_by_its_routed_model(client, monkeypatch, hf_cache):
    snapshot = next((hf_cache / "models--AbstractFramework--minimax-h3-8bit" / "snapshots").iterdir())
    monkeypatch.setenv("ABSTRACTCORE_VISION_BACKEND", "mlx-gen")
    monkeypatch.setenv("ABSTRACTCORE_VISION_MFLUX_MODEL", str(snapshot))

    body = client.get("/v1/vision/models").json()

    default = body["defaults"]["text_to_video"]
    assert {k: default[k] for k in ("id", "model", "provider", "server_default", "memory_modes", "supports_audio")} == {
        "id": "default",
        "model": "default",
        "provider": "mlx-gen",
        "server_default": True,
        "memory_modes": ["default", "low", "lowest"],
        "supports_audio": True,
    }
    assert default["routed_model"] == "mlx-gen/default"
    assert "active" in body  # the 2.39.0 object, unchanged
    resp = client.post("/v1/videos/generations", json={"prompt": "fox", "model": default["routed_model"]})
    assert resp.status_code == 200, resp.text
    assert client.post("/v1/videos/generations", json={"prompt": "fox", "model": "default"}).status_code == 400
    (record,) = client.get("/acore/models/loaded").json()["data"]
    assert record["model"] == "default"


def test_no_vision_or_residency_response_carries_a_path(client, monkeypatch, hf_cache, tmp_path):
    snapshot = next((hf_cache / "models--AbstractFramework--minimax-h3-8bit" / "snapshots").iterdir())
    monkeypatch.setenv("ABSTRACTCORE_VISION_BACKEND", "mlx-gen")
    monkeypatch.setenv("ABSTRACTCORE_VISION_MFLUX_MODEL", str(snapshot))
    models_dir = tmp_path / "local-models" / "my-diffusers"
    models_dir.mkdir(parents=True)
    (models_dir / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionPipeline"}))
    monkeypatch.setenv("ABSTRACTCORE_VISION_MODELS_DIR", str(tmp_path / "local-models"))
    roots = [str(tmp_path), str(Path.home())]

    def boom(step):
        raise FileNotFoundError(str(tmp_path / "secret" / "weights.safetensors"))

    bodies: Dict[str, Any] = {}
    bodies["models"] = client.get("/v1/vision/models").json()
    bodies["models_detail"] = client.get("/v1/vision/models", params={"detail": "true"}).json()
    bodies["video"] = _t2v(client, model="mlx-gen/default", memory_mode="low").json()
    bodies["refusal"] = _t2i(client, memory_mode="lowest").json()
    job_id = client.post("/v1/vision/jobs/videos/generations", json={"prompt": "fox"}).json()["job_id"]
    bodies["job"] = _wait_job(client, job_id)
    bodies["load"] = client.post("/acore/models/load", json={"task": "image_generation", "provider": "mlx-gen", "model": FLUX}).json()
    bodies["loaded"] = client.get("/acore/models/loaded").json()
    FakeMiniMaxH3.during_step = boom
    bodies["engine_error"] = _t2v(client, model="mlx-gen/default").json()
    failed_id = client.post("/v1/vision/jobs/videos/generations", json={"prompt": "fox"}).json()["job_id"]
    bodies["failed_job"] = _wait_job(client, failed_id)
    FakeMiniMaxH3.during_step = None
    bodies["unload"] = client.post("/acore/models/unload", json={"task": "image_generation", "provider": "mlx-gen", "model": FLUX}).json()

    assert bodies["engine_error"]["error"]["code"] == "server_error"
    assert bodies["failed_job"]["state"] == "failed"
    assert {name: _paths_in(body, roots) for name, body in bodies.items() if _paths_in(body, roots)} == {}
    local = [m for m in bodies["models"]["local_models"] if m.get("id") == "my-diffusers"]
    assert local, bodies["models"]["local_models"]


def test_every_discovery_entry_routed_model_runs_it_on_the_backend_it_states(client, monkeypatch, hf_cache, tmp_path):
    from abstractcore.capabilities import vision_catalog

    snapshot = next((hf_cache / "models--AbstractFramework--minimax-h3-8bit" / "snapshots").iterdir())
    monkeypatch.setenv("ABSTRACTCORE_VISION_BACKEND", "mlx-gen")
    monkeypatch.setenv("ABSTRACTCORE_VISION_MFLUX_MODEL", str(snapshot))
    local = tmp_path / "local-models"
    for folder in ("my-diffusers", "org__named"):
        (local / folder).mkdir(parents=True)
        (local / folder / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionPipeline"}))
    monkeypatch.setenv("ABSTRACTCORE_VISION_MODELS_DIR", str(local))
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("ABSTRACTCORE_VISION_UPSTREAM_MODEL_ID", "remote-model")

    body = client.get("/v1/vision/models").json()
    # The same entries as only the server's cache and configuration list them (no plugin catalog).
    local_catalog = asyncio.run(vision_endpoints._local_vision_model_catalog())
    cache_only = vision_endpoints._merge_vision_provider_model_entries(
        vision_endpoints._cached_vision_provider_model_entries(local_catalog, None),
        vision_endpoints._configured_vision_provider_model_entries(None),
    )
    entries = body["models"] + cache_only + list(body["defaults"].values())
    stated = {vision_endpoints._listing_backend_kind(e) for e in entries}
    assert {"mlx-gen", "diffusers", "openai_compatible_proxy"} <= stated, stated

    for entry in entries:
        kind = vision_endpoints._listing_backend_kind(entry)
        for task in entry.get("capabilities") or []:
            if task not in vision_endpoints._VISION_TASK_MODALITY:
                continue
            modality = vision_endpoints._VISION_TASK_MODALITY[task]
            assert vision_endpoints._effective_backend_kind(entry["routed_model"], modality, task) == kind, (entry, task)

    # A local folder listed by its name: that name selects the folder again.
    named = {e["id"]: e["routed_model"] for e in entries if vision_endpoints._listing_backend_kind(e) == "diffusers"}
    assert named["my-diffusers"] == "diffusers/my-diffusers"
    assert vision_catalog.local_diffusers_model_path("my-diffusers") == str(local / "my-diffusers")
    assert named["org/named"] == "diffusers/org/named"
    cached_mlx = {e["id"]: e["routed_model"] for e in cache_only if e["provider"] == "mlx-gen"}
    assert cached_mlx[H3] == f"mlx-gen/{H3}"


# --- Route request defaults and residency ----------------------------------------------------------


def test_route_request_defaults_apply_to_requests_for_the_route_model_only(client, h3_route):
    h3_route["options"] = {"memory_mode": "low", "base_model": "minimax-h3", "music": "piano"}

    assert _t2v(client).json()["memory_mode_used"] == "low"
    call = FakeMiniMaxH3.instances[-1].generate_calls[-1]
    assert call["music"] == "piano" and call["release_transformer"] is True
    # A value the request sets wins; the backend setting never reaches the request.
    assert _t2v(client, memory_mode="default", music="harp").status_code == 200
    call = FakeMiniMaxH3.instances[-1].generate_calls[-1]
    assert call["music"] == "harp" and not call.get("release_transformer")


def test_a_route_default_the_model_cannot_honour_is_refused_at_load_with_its_field(client, h3_route):
    h3_route["options"] = {"quantize": 4}

    load = client.post("/acore/models/load", json={"task": "text_to_video", "provider": "mlx-gen", "model": H3})

    assert load.status_code == 422, load.text
    error = load.json()["error"]
    assert (error["code"], error["field"], error["supported"]) == ("capability_not_supported", "quantize", [])
    assert vision_endpoints._BACKEND_CACHE == {} and vision_endpoints._RESIDENCY_RECORDS == {}
    assert client.get("/acore/models/loaded").json()["data"] == []
    # A request that names nothing runs with the route's defaults, so it is refused the same way
    # until the route is fixed; the default is never dropped.
    later = client.post("/v1/videos/generations", json={"prompt": "fox"})
    assert (later.status_code, later.json()["error"]["field"]) == (422, "quantize")
    assert FakeMiniMaxH3.instances == []


def test_an_explicit_load_is_pinned_and_listed_as_loaded(client):
    resp = client.post("/acore/models/load", json={"task": "image_generation", "provider": "mlx-gen", "model": FLUX})

    assert resp.status_code == 200, resp.text
    runtime = resp.json()["runtime"]
    assert (runtime["loaded"], runtime["pinned"], runtime["model"]) == (True, True, FLUX)
    assert len(FakeFlux2.instances) == 1


def test_a_refused_load_leaves_the_pinned_model_loaded_and_listed(client, h3_route):
    first = client.post("/acore/models/load", json={"task": "text_to_video", "provider": "mlx-gen", "model": H3})
    assert first.status_code == 200, first.text
    h3_route["options"] = {"quantize": 4}

    refused = client.post("/acore/models/load", json={"task": "text_to_video", "provider": "mlx-gen", "model": H3})

    assert refused.status_code == 422 and refused.json()["error"]["field"] == "quantize"
    (record,) = client.get("/acore/models/loaded").json()["data"]
    assert (record["model"], record["loaded"], record["pinned"]) == (H3, True, True)
    assert len(FakeMiniMaxH3.instances) == 1


def test_a_route_lowest_default_preloads_streamed_and_the_next_request_reuses_that_instance(client, h3_route):
    h3_route["options"] = {"memory_mode": "lowest"}

    load = client.post("/acore/models/load", json={"task": "text_to_video", "provider": "mlx-gen", "model": H3})

    assert load.status_code == 200, load.text
    assert (load.json()["runtime"]["loaded"], load.json()["runtime"]["pinned"]) == (True, True)
    (model,) = FakeMiniMaxH3.instances
    assert model.init_kwargs.get("stream_blocks") is True

    resp = client.post("/v1/videos/generations", json={"prompt": "fox"})  # names nothing: the route's model

    assert resp.status_code == 200, resp.text
    assert resp.json()["memory_mode_used"] == "lowest"
    assert FakeMiniMaxH3.instances == [model] and len(model.generate_calls) == 1  # no second load
    # A lowest run drops the model; the pinned record says so.
    (record,) = client.get("/acore/models/loaded").json()["data"]
    assert (record["loaded"], record["state"], record["pinned"]) == (False, "not_loaded", True)


def test_a_pinned_model_dropped_by_a_low_request_stays_pinned_and_the_next_request_reloads_it(client):
    client.post("/acore/models/load", json={"task": "image_generation", "provider": "mlx-gen", "model": FLUX})
    assert len(FakeFlux2.instances) == 1

    assert _t2i(client, memory_mode="low").json()["memory_mode_used"] == "low"
    (record,) = client.get("/acore/models/loaded").json()["data"]
    assert (record["loaded"], record["state"], record["pinned"]) == (False, "not_loaded", True)

    assert _t2i(client).json()["memory_mode_used"] == "default"
    (record,) = client.get("/acore/models/loaded").json()["data"]
    assert (record["loaded"], record["state"], record["pinned"]) == (True, "loaded", True)
    assert len(vision_endpoints._BACKEND_CACHE) == 1  # the same backend reloaded its model
    assert len(FakeFlux2.instances) == 2  # the preload (which the low run used), then the reload


# --- Error bodies, catalog errors, the remote lane ------------------------------------------------


def test_every_error_body_has_a_code_and_a_structured_detail_passes_through(client):
    missing = client.post("/acore/models/unload", json={"task": "image_generation", "provider": "mlx-gen", "model": "nope/x"})
    assert missing.status_code == 404
    assert missing.json()["error"]["type"] == "not_found" and missing.json()["error"]["code"] == "not_found"
    assert client.get("/v1/vision/jobs/nope").json()["error"]["code"] == "not_found"
    assert client.post("/v1/images/generations", json={"prompt": ""}).json()["error"]["code"] == "invalid_request"


def test_catalog_errors_are_classified_by_type():
    from abstractvision.errors import BackendNotConfiguredError, CapabilityNotSupportedError, OptionalDependencyMissingError

    for exc in (CapabilityNotSupportedError("x"), BackendNotConfiguredError("x"), OptionalDependencyMissingError("x")):
        assert vision_endpoints._vision_catalog_error(exc).status_code == 501
    assert vision_endpoints._vision_catalog_error(RuntimeError("upstream down")).status_code == 502


@pytest.mark.parametrize(
    ("base_url", "is_openai"),
    [
        ("https://api.openai.com/v1", True),
        ("https://eu.api.openai.com/v1", True),
        ("https://api.openai.com.example.net/v1", False),
        ("https://notapi.openai.com/v1", False),
        ("http://127.0.0.1:9/api.openai.com", False),
    ],
)
def test_the_remote_lane_uses_abstractvisions_openai_host_rule(base_url, is_openai):
    from abstractvision.backends import openai_compatible

    assert vision_endpoints.looks_like_openai_api is openai_compatible.looks_like_openai_api
    assert vision_endpoints.looks_like_openai_api(base_url) is is_openai


class _Upstream:
    calls: List[Dict[str, Any]] = []

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None, data=None, files=None):
        _Upstream.calls.append({"url": url, "json": json, "data": data})
        png = base64.b64encode(PNG).decode()
        return type("R", (), {"status_code": 200, "text": "", "json": lambda self: {"data": [{"b64_json": png}]}})()


def test_the_remote_lane_refuses_typed_options_and_forwards_extra_as_before(client, monkeypatch):
    monkeypatch.setattr(vision_endpoints.httpx, "Client", _Upstream)
    _Upstream.calls = []
    remote = {"model": "openai-compatible/some-model", "base_url": "http://127.0.0.1:9/v1"}

    refused = client.post("/v1/images/generations", json={"prompt": "c", "memory_mode": "low", **remote})
    assert refused.status_code == 422
    error = refused.json()["error"]
    assert (error["field"], error["supported"]) == ("memory_mode", [])
    assert _Upstream.calls == []

    sent = client.post("/v1/images/generations", json={"prompt": "c", "extra": {"memory_mode": "low", "quantize": "q4"}, **remote})
    assert sent.status_code == 200, sent.text
    assert {k: _Upstream.calls[-1]["json"][k] for k in ("memory_mode", "quantize")} == {"memory_mode": "low", "quantize": "q4"}


def test_openapi_declares_the_typed_fields_and_the_cancel_route_404(client):
    schema = client.get("/openapi.json").json()
    video = schema["components"]["schemas"]["VideoGenerationBody"]["properties"]
    for name in ("memory_mode", "quantize", *SOUNDTRACK_REQUEST_FIELDS):
        assert name in video, name
    assert "404" in schema["paths"]["/v1/vision/jobs/{job_id}/cancel"]["post"]["responses"]
