"""CPU-only Qwen4 head-publication races against the real provider binder."""

import importlib.metadata
import sys
import threading
import types
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from abstractcore.providers import mlx_qwen4 as loader
from abstractcore.providers.mlx_provider import MLXProvider


class ObservedRLock:
    """Report whether a selected thread actually contends on a real RLock."""

    def __init__(self):
        self.lock = threading.RLock()
        self.watch_thread = None
        self.observed = threading.Event()
        self.was_blocked = None

    def __enter__(self):
        acquired = self.lock.acquire(blocking=False)
        if threading.get_ident() == self.watch_thread:
            self.was_blocked = not acquired
            self.observed.set()
        if not acquired:
            self.lock.acquire()
        return self

    def __exit__(self, *exc):
        self.lock.release()


@pytest.fixture
def binding_case(monkeypatch, tmp_path):
    session = loader.Qwen4Session()
    session.model, session.processor = object(), SimpleNamespace(tokenizer=object())
    session.config_lock = ObservedRLock()
    registry = weakref.WeakValueDictionary()
    registry[(str(tmp_path.resolve()), False)] = session
    monkeypatch.setattr(loader, "_SESSIONS", registry)
    monkeypatch.setattr(loader, "checkpoint_config", lambda path: {"text_config": {}})
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "cpu-test")
    created, providers = [], []

    class Cache:
        def __init__(self, identity, **kwargs):
            self.identity = identity

        def manager(self, request):
            return None

        def close(self):
            pass

    class Runtime:
        def __init__(self, model, processor, drafter, kind, **kwargs):
            self.model, self.processor = model, processor
            self.drafter, self.draft_kind = drafter, kind
            created.append(self)

        def acquire(self):
            return "fake-owner"

        def release(self, owner):
            pass

    monkeypatch.setattr("abstractcore.providers.mlx_native_cache.NativeCacheStore", Cache)
    monkeypatch.setattr("abstractcore.providers.mlx_runtime.NativeRuntime", Runtime)

    def bind():
        value = MLXProvider.__new__(MLXProvider)
        providers.append(value)
        value._native_qwen4 = session
        value._mlx_batching = True
        value._mlx_runtime_options = {}
        value._mlx_cache_options = {}
        value._mtp_kind = None  # A genuinely target-only provider wins binding.
        value._resolved_model_id = str(tmp_path)
        value.prompt_cache_weights_fingerprint = lambda: "target-weights"
        value.prompt_cache_model_config_fingerprint = lambda: "target-config"
        value.prompt_cache_tokenizer_fingerprint = lambda: "target-tokenizer"
        session.config_lock.watch_thread = threading.get_ident()
        value._bind_native_session(session)
        return value

    yield SimpleNamespace(session=session, path=str(tmp_path), bind=bind, created=created)
    for provider in providers:
        finalizer = getattr(provider, "_native_finalizer", None)
        if finalizer is not None:
            finalizer.detach()
        session.holders.discard(provider)


def test_blocked_late_head_loader_serializes_actual_target_provider_binding(binding_case, monkeypatch):
    case = binding_case
    entered, release = threading.Event(), threading.Event()
    head = object()
    loads = []

    def load_head(path, config):
        loads.append((path, config))
        entered.set()
        assert release.wait(2), "Fake head load was not released"
        return head

    monkeypatch.setattr(loader, "_load_embedded_drafter", load_head)
    with ThreadPoolExecutor(max_workers=2) as pool:
        upgrading = pool.submit(loader.load_qwen4_session, case.path, mtp=True, ple_offload=False)
        try:
            assert entered.wait(1)
            binding = pool.submit(case.bind)
            assert case.session.config_lock.observed.wait(1), "Provider never attempted configuration ownership"
            assert case.session.config_lock.was_blocked is True, "Binding raced past the in-progress head publication"
            assert not binding.done() and not case.created
        finally:
            release.set()
        assert upgrading.result(timeout=1) is case.session
        provider = binding.result(timeout=1)
    assert len(loads) == len(case.created) == 1
    assert provider._native_runtime is case.session.runtime
    assert provider._native_runtime.drafter is case.session.drafter is head
    assert provider._native_runtime.draft_kind == case.session.draft_kind == "mtp"
    assert case.session.cache_store.identity["head"]["path"].endswith("#mtp")
    assert not case.session.lock.locked()


def test_scheduled_target_session_refuses_late_head_without_mutation(binding_case, monkeypatch):
    case = binding_case
    provider = case.bind()
    runtime = provider._native_runtime
    monkeypatch.setattr(loader, "_load_embedded_drafter", lambda *args: pytest.fail("Bound runtime must not load a head"))
    with pytest.raises(RuntimeError, match="target-only scheduled session"):
        loader.load_qwen4_session(case.path, mtp=True, ple_offload=False)
    assert case.session.runtime is runtime
    assert case.session.drafter is runtime.drafter is None
    assert case.session.draft_kind is runtime.draft_kind is None


def test_provider_binding_that_wins_the_lock_prevents_concurrent_late_head(binding_case, monkeypatch):
    from abstractcore.providers import mlx_runtime

    case = binding_case
    entered, release = threading.Event(), threading.Event()
    base_runtime = mlx_runtime.NativeRuntime

    class BlockingRuntime(base_runtime):
        def __init__(self, *args, **kwargs):
            entered.set()
            assert release.wait(2), "Fake runtime construction was not released"
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(mlx_runtime, "NativeRuntime", BlockingRuntime)
    monkeypatch.setattr(loader, "_load_embedded_drafter", lambda *args: pytest.fail("Binding must win before any head load"))

    def upgrade():
        case.session.config_lock.watch_thread = threading.get_ident()
        return loader.load_qwen4_session(case.path, mtp=True, ple_offload=False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        binding = pool.submit(case.bind)
        try:
            assert entered.wait(1)
            case.session.config_lock.observed.clear()
            upgrading = pool.submit(upgrade)
            assert case.session.config_lock.observed.wait(1), "Loader did not request configuration ownership"
            assert case.session.config_lock.was_blocked is True
            assert not upgrading.done()
        finally:
            release.set()
        provider = binding.result(timeout=1)
        with pytest.raises(RuntimeError, match="target-only scheduled session"):
            upgrading.result(timeout=1)
    assert provider._native_runtime.drafter is case.session.drafter is None
    assert len(case.created) == 1


def test_initial_embedded_head_publishes_its_kind_without_provider_defaults(binding_case, monkeypatch):
    case = binding_case
    head, model, processor = object(), object(), object()
    validated = []
    vlm = types.ModuleType("mlx_vlm")
    vlm.__path__ = []
    vlm.load = lambda *args: pytest.fail("Real target loader must not run")
    storage = types.ModuleType("mlx_vlm.models.qwen4_exp.ple_storage")
    storage.build_quantized_ple_manifest = lambda *args, **kwargs: pytest.fail("Offload disabled")
    drafters = types.ModuleType("mlx_vlm.speculative.drafters")
    drafters.validate_drafter_compatibility = lambda *args: validated.append(args)
    for name, module in (("mlx_vlm", vlm), (storage.__name__, storage), (drafters.__name__, drafters)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(loader, "_load_embedded_drafter", lambda *args: head)
    monkeypatch.setattr(loader, "_load_target", lambda *args: (model, processor))
    session = loader._create_qwen4_session(case.path, mtp=True, ple_offload=False)
    assert session.drafter is head and session.draft_kind == "mtp"
    assert validated == [(model, head, "mtp")]


def test_existing_mtp_runtime_is_returned_without_replacing_head(binding_case, monkeypatch):
    case = binding_case
    head = case.session.drafter = object()
    case.session.draft_kind = "mtp"
    provider = case.bind()
    monkeypatch.setattr(loader, "_load_embedded_drafter", lambda *args: pytest.fail("Existing head must be reused"))
    assert loader.load_qwen4_session(case.path, mtp=True, ple_offload=False) is case.session
    assert case.session.runtime is provider._native_runtime
    assert case.session.runtime.drafter is head
    assert case.session.runtime.draft_kind == "mtp"


def test_direct_generation_busy_refusal_releases_configuration_ownership(binding_case, monkeypatch):
    case = binding_case
    monkeypatch.setattr(loader, "_load_embedded_drafter", lambda *args: pytest.fail("Active direct generation forbids loading"))
    with case.session.lock:
        with pytest.raises(RuntimeError, match="generation is active"):
            loader.load_qwen4_session(case.path, mtp=True, ple_offload=False)
    with ThreadPoolExecutor(max_workers=1) as pool:
        provider = pool.submit(case.bind).result(timeout=1)
    assert case.session.config_lock.was_blocked is False
    assert provider._native_runtime.drafter is None


def test_failed_head_load_publishes_nothing_and_releases_both_locks(binding_case, monkeypatch):
    case = binding_case
    failure = ValueError("invalid fake embedded weights")

    def broken(*args):
        raise failure

    monkeypatch.setattr(loader, "_load_embedded_drafter", broken)
    with pytest.raises(ValueError) as caught:
        loader.load_qwen4_session(case.path, mtp=True, ple_offload=False)
    assert caught.value is failure
    assert case.session.drafter is case.session.draft_kind is case.session.runtime is None
    assert not case.session.lock.locked()
    head = object()
    monkeypatch.setattr(loader, "_load_embedded_drafter", lambda *args: head)
    with ThreadPoolExecutor(max_workers=1) as pool:
        retried = pool.submit(loader.load_qwen4_session, case.path, mtp=True, ple_offload=False).result(timeout=1)
    assert retried.drafter is head and retried.draft_kind == "mtp"
