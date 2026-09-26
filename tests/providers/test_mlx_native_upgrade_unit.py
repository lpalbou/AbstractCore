"""CPU-only checks of provider/native-runtime integration boundaries."""
from dataclasses import replace
import gc
import sys
import types
import weakref
from types import SimpleNamespace

import importlib.util

import pytest

from abstractcore.providers.mlx_provider import MLXProvider
from abstractcore.providers.mlx_runtime import NativeResult, NativeRuntimeError
from abstractcore.providers.speculation import SpeculationRequest


_requires_mlx_stack = pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("mlx", "mlx_lm", "mlx_vlm")),
    reason="requires the optional MLX stack (pip install \"abstractcore[mlx]\")",
)


def provider():
    value = MLXProvider.__new__(MLXProvider)
    value.model = "local/native"
    value._mtp_processor = object()
    value._mtp_drafter = object()
    value._mtp_kind = "mtp"
    value._mtp_drafter_id = "local/head"
    value._speculation_request = SpeculationRequest(mode="native_mtp", num_draft_tokens=2)
    value._mtp_block_size = 2
    value._mlx_cache_scope = "test-owner"
    value._native_owner_id = "lease"
    value._native_runtime = object()
    value._apply_per_call_speculation(None)
    return value


def test_lower_request_preserves_exact_controls_and_scope():
    value = provider()
    value._native_cache_key = "conversation"
    value._native_sampling_kwargs = {"min_p": .1}
    value._native_stop = ("END",)
    request = value._native_runtime_request("full prompt", {
        "max_tokens": 23, "temperature": .7, "top_p": .9, "top_k": 12, "seed": 42,
    })
    assert request.draft_tokens == 2
    assert (request.temperature, request.top_p, request.top_k, request.seed) == (.7, .9, 12, 42)
    assert (request.cache_key, request.cache_scope, request.owner_id) == ("conversation", "test-owner", "lease")
    assert request.sampling["min_p"] == .1
    assert request.stop == ("END",)


def test_lower_request_off_does_not_forward_drafter():
    value = provider()
    value._apply_per_call_speculation(False)
    assert value._native_runtime_request("prompt", {"max_tokens": 10}).draft_tokens == 0


@pytest.mark.parametrize("controls", [
    {"max_tokens": True}, {"max_tokens": 1.5},
    {"max_tokens": 10, "temperature": float("nan")},
    {"max_tokens": 10, "top_p": 0}, {"max_tokens": 10, "top_k": True},
])
def test_invalid_native_controls_are_typed_request_local_rejections(controls):
    with pytest.raises(NativeRuntimeError) as caught:
        provider()._native_runtime_request("prompt", controls)
    assert (caught.value.code, caught.value.http_status, caught.value.request_local) == (
        "invalid_request", 400, True)


def test_lower_request_copies_media_bytes_not_mutable_buffer():
    value = provider()
    raw = bytearray(b"image-data")
    value._native_media = [(3, SimpleNamespace(content=raw, file_path=None))]
    request = value._native_runtime_request("prompt", {"max_tokens": 10})
    raw[0] = 0
    assert request.media == ((3, b"image-data"),)


def test_runtime_result_preserves_actual_execution_not_requested_flag():
    value = provider()
    value._native_media_report = None
    result = NativeResult(text="answer", metadata={
        "execution": {"mode": "cohort", "peak_batch_size": 3},
        "speculation": {"used": True, "stats": {"rounds": 4, "accounting": "cohort"}},
    })
    value._observe_native_runtime_result(result)
    assert value._mtp_last_used is True
    assert value._mtp_call_stats == {"rounds": 4, "accounting": "cohort"}
    value._observe_native_runtime_result(replace(result, metadata={}))
    assert value._mtp_last_used is False
    assert value._mtp_call_stats == {}


def test_nonstream_consumes_deltas_once_and_closes_runtime_handle():
    value = provider()._native_request_view()
    class Handle:
        closed = False
        def __iter__(self):
            yield NativeResult(text="first ")
            yield NativeResult(text="second")
            yield NativeResult(finish_reason="length", prompt_tokens=7, generation_tokens=2)
        def close(self):
            self.closed = True
    handle = Handle()
    value._native_runtime = SimpleNamespace(stream=lambda request, **kwargs: handle)
    assert value._mtp_generate_fn(None, None, "prompt", max_tokens=2) == "first second"
    assert value._mtp_last_result.generation_tokens == 2
    assert value._mtp_last_result.finish_reason == "length"
    assert handle.closed
    assert value._native_runtime_stream is None


def test_exact_27b_usage_and_execution_metadata_replace_estimates():
    value = provider()
    value.logger = SimpleNamespace(debug=lambda *a, **k: None)
    value.llm = object()
    value.tokenizer = object()
    value.generate_fn = lambda *a, **k: "answer"
    value._postprocess_generated_text = lambda text, **_: (text, None)
    value._calculate_usage = lambda *a: {"input_tokens": 999, "output_tokens": 999}
    value._count_tokens = lambda text: 999
    value._mtp_last_result = NativeResult(prompt_tokens=17, generation_tokens=3,
        cached_tokens=8, prompt_tps=100, generation_tps=25, peak_memory=16,
        finish_reason="stop")
    value._native_runtime_metadata = {"execution": {"mode": "continuous"}}
    response = value._single_generate("prompt", 20, 0, 1, seed=123)
    assert response.usage["input_tokens"] == 17
    assert response.usage["output_tokens"] == 3
    assert response.usage["cached_input_tokens"] == 8
    assert response.usage["total_tokens"] == 20
    assert response.metadata["execution"]["mode"] == "continuous"
    assert response.metadata["performance"]["generation_tokens_per_second"] == 25


def test_partial_cancelled_nonstream_is_not_reported_as_success():
    value = provider()._native_request_view()
    class Handle:
        def __iter__(self):
            yield NativeResult(text="partial")
        def close(self):
            pass
    value._native_runtime = SimpleNamespace(stream=lambda request, **kwargs: Handle())
    with pytest.raises(Exception, match="without a terminal result"):
        value._mtp_generate_fn(None, None, "prompt", max_tokens=20)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
def test_native_incomplete_response_distinguishes_cancel_from_backend_failure(stream, cancelled):
    value = provider()._native_request_view()
    class Handle:
        def __iter__(self):
            yield NativeResult(text="partial")
            if cancelled:
                value._native_cancel_event.set()
        def close(self):
            # Cleanup itself must not reclassify an interrupted backend as a
            # caller cancellation. The classification happens before close.
            value._native_cancel_event.set()
    value._native_runtime = SimpleNamespace(stream=lambda request, **kwargs: Handle())
    with pytest.raises(NativeRuntimeError) as caught:
        if stream:
            list(value._mtp_stream_generate_fn(None, None, "prompt", max_tokens=20))
        else:
            value._mtp_generate_fn(None, None, "prompt", max_tokens=20)
    assert caught.value.code == ("cancelled" if cancelled else "backend_error")
    assert caught.value.request_local is cancelled


def test_repeated_runtime_media_snapshots_record_delivery_once():
    from abstractcore.media.delivery import MediaReport
    value = provider()
    value._native_media_report = MediaReport.for_request([object()], provider="mlx", model="m")
    result = NativeResult(media_records=[{"index": 0, "kind": "image", "content": b"image",
                                        "tokens": 4, "transport": "mlx_vlm_native"}])
    value._observe_native_runtime_result(result)
    value._observe_native_runtime_result(result)
    assert len(value._native_media_report.delivered) == 1


def test_automatic_native_caches_do_not_claim_manual_artifact_compatibility():
    value = provider()
    with pytest.raises(Exception, match="manual prompt_cache_save"):
        value.prompt_cache_save("key", "unused.safetensors")
    with pytest.raises(Exception, match="manual prompt_cache_load"):
        value.prompt_cache_load("unused.safetensors")


@pytest.fixture
def fake_session_binding(monkeypatch, tmp_path):
    from abstractcore.providers.mlx_native_session import NativeSession
    class Cache:
        def __init__(self, *args, **kwargs):
            self.identity = args[0]
            self.closed = 0
        def manager(self, request):
            return None
        def close(self):
            self.closed += 1
    class Runtime:
        def __init__(self, *args, on_close, **kwargs):
            self.owners = set()
            self.busy = set()
            self.on_close = on_close
            self.closed = 0
        def acquire(self):
            owner = str(len(self.owners))
            self.owners.add(owner)
            return owner
        def release(self, owner):
            if owner in self.busy:
                raise RuntimeError("active owner")
            self.owners.remove(owner)
            if not self.owners:
                self.closed += 1
                self.on_close()
        def retire(self, owner, *, keepalive=None):
            self.release(owner)
    monkeypatch.setattr("abstractcore.providers.mlx_native_cache.NativeCacheStore", Cache)
    monkeypatch.setattr("abstractcore.providers.mlx_runtime.NativeRuntime", Runtime)
    session = NativeSession()
    session.model, session.processor = object(), object()
    def create():
        value = provider()
        value._mlx_batching = True
        value._mlx_runtime_options = {}
        value._mlx_cache_options = {}
        value._resolved_model_id = str(tmp_path)
        value.prompt_cache_weights_fingerprint = lambda: "weights"
        value.prompt_cache_model_config_fingerprint = lambda: "config"
        value.prompt_cache_tokenizer_fingerprint = lambda: "tokens"
        value._unload_model_unlocked = lambda model: None
        value._bind_native_session(session)
        return value
    return session, create


@_requires_mlx_stack
def test_shared_native_lease_release_does_not_close_other_owner(fake_session_binding):
    session, create = fake_session_binding
    first, second = create(), create()
    runtime, cache = session.runtime, session.cache_store
    first.unload_model(first.model)
    assert runtime.closed == 0 and cache.closed == 0
    assert second._native_owner_id in runtime.owners
    second.unload_model(second.model)
    assert runtime.closed == 1 and cache.closed == 1
    assert session.runtime is None


@_requires_mlx_stack
def test_active_native_lease_rejection_preserves_provider(fake_session_binding):
    session, create = fake_session_binding
    value = create()
    runtime = session.runtime
    runtime.busy.add(value._native_owner_id)
    with pytest.raises(RuntimeError, match="active owner"):
        value.unload_model(value.model)
    assert value._native_runtime is runtime
    assert value in session.holders
    runtime.busy.clear()
    value.unload_model(value.model)


@_requires_mlx_stack
def test_dropping_last_provider_releases_worker_and_cache(fake_session_binding):
    session, create = fake_session_binding
    value = create()
    runtime, cache = session.runtime, session.cache_store
    ref = weakref.ref(value)
    del value
    gc.collect()
    assert ref() is None
    assert runtime.closed == 1 and cache.closed == 1
    # GC retirement is worker-owned, not synchronous provider detachment. This
    # test deliberately retains the session externally; its closed runtime is
    # harmless. Real deferred completion/keepalive release is tested separately.
    assert session.runtime is runtime


@_requires_mlx_stack
def test_request_facade_retains_originating_lease_until_released(fake_session_binding):
    session, create = fake_session_binding
    value = create()
    runtime = session.runtime
    view = value._native_request_view()
    ref = weakref.ref(value)
    del value
    gc.collect()
    assert ref() is not None and runtime.closed == 0
    del view
    gc.collect()
    assert ref() is None and runtime.closed == 1


def test_sidecar_alias_resolves_before_load_and_revision_changes_runtime_key(monkeypatch, tmp_path):
    from abstractcore.providers import mlx_native_session as native
    from abstractcore.providers.mlx_native_cache import NativeCacheStore
    first_dir = tmp_path / "snapshots" / ("a" * 40)
    second_dir = tmp_path / "snapshots" / ("b" * 40)
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    selected = [first_dir]
    monkeypatch.setattr("abstractcore.utils.model_cache.resolve_hf_snapshot_dir", lambda repo: selected[0])
    loaded_heads = []
    vlm = types.ModuleType("mlx_vlm")
    vlm.load = lambda path: (object(), object())
    drafters = types.ModuleType("mlx_vlm.speculative.drafters")
    def load_drafter(path):
        loaded_heads.append(path)
        return SimpleNamespace(), "mtp"
    drafters.load_drafter = load_drafter
    monkeypatch.setitem(sys.modules, "mlx_vlm", vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative.drafters", drafters)
    first = native.load_native_session(str(tmp_path / "target"), "publisher/head")
    selected[0] = second_dir
    second = native.load_native_session(str(tmp_path / "target"), "publisher/head")
    assert first is not second
    assert loaded_heads == [str(first_dir.resolve()), str(second_dir.resolve())]
    assert first.drafter_weights_fingerprint == "weights-revision:" + "a" * 40
    assert second.drafter_weights_fingerprint == "weights-revision:" + "b" * 40
    assert native.load_native_session(str(tmp_path / "target"), str(first_dir)) is first
    def namespace(session):
        return NativeCacheStore({"weights_fingerprint": "target-revision", "head": {
            "path": session.drafter_path, "weights_fingerprint": session.drafter_weights_fingerprint,
        }}).namespace
    assert namespace(first) != namespace(second)


@_requires_mlx_stack
def test_provider_disk_identity_uses_resolved_head_not_repo_alias(fake_session_binding, tmp_path):
    session, create = fake_session_binding
    session.drafter = object()
    session.drafter_path = str(tmp_path / "snapshots" / ("c" * 40))
    session.drafter_weights_fingerprint = "weights-revision:" + "c" * 40
    value = create()
    # A separate capture avoids relying on a mutable repo alias as identity.
    assert value._mtp_drafter_id == "local/head"
    assert session.cache_store.identity["head"] == {
        "path": session.drafter_path, "weights_fingerprint": session.drafter_weights_fingerprint,
    }
    value.unload_model(value.model)
