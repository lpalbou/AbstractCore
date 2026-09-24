"""
PyTest configuration for AbstractCore tests
Provides fixtures and utilities for vision testing
"""

import pytest
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add abstractcore to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Server unit tests set auth explicitly with fixtures. Do not let a developer's
# persisted ~/.abstractcore server auth token leak into app imports during collection.
os.environ["ABSTRACTCORE_SERVER_DISABLE_CENTRALIZED_CONFIG"] = "1"

# ---------------------------------------------------------------------------
# Tests never touch your home or the network.
#
# Why (two incidents, 2026-09-24): an unisolated test downloaded a 1.3 GB
# vision model into ~/.abstractcore/models and rewrote the operator's
# abstractcore.json; a later run rewrote the operator's embedding caches under
# ~/.abstractcore/embeddings as EMPTY files. Isolating one config-file env var
# is not enough: many locations derive from the home directory itself
# (Path.home(), expanduser, huggingface_hub's IMPORT-TIME constants). So HOME
# itself moves -- once here, at conftest import, before any package or
# huggingface_hub is imported, and again for every test.
#
# The same block installs a socket guard: no test reaches a non-loopback host,
# nor the operator's live loopback services (gateway 8080, LM Studio 1234,
# Ollama 11434, the hermetic mission gateway 18850). Any other loopback port
# (TestClient, fake servers on scratch ports) stays allowed.
#
# Opt-outs, each with a MANDATORY reason (collection refuses a bare marker):
#   @pytest.mark.network("reason")    the test genuinely needs the network
#   @pytest.mark.real_home("reason")  the test reads the operator's real home
# ---------------------------------------------------------------------------
import ipaddress as _ipaddress
import os
import sys
import re as _re
import shutil as _shutil
import socket as _socket
import tempfile as _tempfile

_REAL_HOME = os.path.expanduser("~")
_SESSION_HOME = Path(_tempfile.mkdtemp(prefix="abstractcore-tests-home-"))

# Operator-exported path knobs that would steer a test at real data
# (ABSTRACTGATEWAY_FLOWS_DIR, HF_HUB_CACHE, ABSTRACTCORE_JOBS_DIR, ...).
_PATH_KNOB = _re.compile(
    r"^(ABSTRACT[A-Z0-9_]*|HF_[A-Z0-9_]*|HUGGINGFACE_[A-Z0-9_]*)_(DIR|DIRS|PATH|FILE|ROOT|ROOTS|HOME|CACHE|REGISTRY)$"
)
# XDG_* are REMOVED rather than set: every XDG default derives from HOME
# (already tmp), and tests that pass their own fake home expect the platform
# default layout under it, which an exported XDG_CONFIG_HOME would override.
_EXTRA_KNOBS = frozenset(
    {
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "SENTENCE_TRANSFORMERS_HOME",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "XDG_RUNTIME_DIR",
    }
)
_OWN_KNOBS = frozenset({"ABSTRACT_TEST_REAL_HOME"})


def _operator_path_knobs() -> list:
    return [k for k in os.environ if k not in _OWN_KNOBS and (_PATH_KNOB.match(k) or k in _EXTRA_KNOBS)]


def _hermetic_env(home: Path) -> dict:
    """Every location the packages derive from the home directory, under `home`."""
    env = {
        "HOME": str(home),
        "HF_HOME": str(home / ".cache" / "huggingface"),
        "HF_HUB_CACHE": str(home / ".cache" / "huggingface" / "hub"),
    }
    # ABSTRACTCORE_CONFIG_DIR is scrubbed (a path knob), not set: the config
    # dir then derives from the tmp HOME, which is also what tests that move
    # HOME themselves expect.
    if os.name == "nt":
        env["USERPROFILE"] = str(home)
        env["APPDATA"] = str(home / "AppData" / "Roaming")
        env["LOCALAPPDATA"] = str(home / "AppData" / "Local")
    return env


for _knob in _operator_path_knobs():
    os.environ.pop(_knob, None)
os.environ.update(_hermetic_env(_SESSION_HOME))
# Read-only pointer for a test that declares @pytest.mark.real_home (e.g. to
# READ installed tokenizers at collection); HOME itself stays tmp.
os.environ["ABSTRACT_TEST_REAL_HOME"] = _REAL_HOME


def _under(path: Path, root: str) -> bool:
    try:
        path.resolve().relative_to(Path(root).resolve())
        return True
    except (OSError, ValueError):
        return False


def _assert_hf_constants_isolated() -> None:
    """huggingface_hub freezes its cache path at import; fail loudly if it froze on the real home."""
    consts = sys.modules.get("huggingface_hub.constants")
    if consts is None:
        return
    cache = Path(str(getattr(consts, "HF_HUB_CACHE", "")))
    if _under(cache, _REAL_HOME) and not _under(cache, str(_SESSION_HOME)) and not _under(cache, _tempfile.gettempdir()):
        pytest.fail(
            f"huggingface_hub.constants.HF_HUB_CACHE={cache} points into the real home: huggingface_hub "
            "was imported before the test conftest isolated HOME (a plugin or sitecustomize imports it early).",
            pytrace=False,
        )


# -- network guard ----------------------------------------------------------
_LIVE_LOOPBACK_PORTS = frozenset({8080, 1234, 11434, 18850})


class NetworkGuardError(ConnectionRefusedError):
    """A test tried to open a connection the network guard refuses."""


class NetworkGuardResolveError(_socket.gaierror):
    """A test tried to resolve a non-loopback host name."""


_GUARD = {"test": None, "allowed": False, "hits": []}
_GUARD_ALL_HITS: list = []

_REAL_CONNECT = _socket.socket.connect
_REAL_CONNECT_EX = _socket.socket.connect_ex
_REAL_CREATE_CONNECTION = _socket.create_connection
_REAL_GETADDRINFO = _socket.getaddrinfo


def _is_loopback_host(host) -> bool:
    if host is None:
        return True
    if isinstance(host, (bytes, bytearray)):
        host = bytes(host).decode("ascii", "replace")
    text = str(host).strip().strip("[]").split("%", 1)[0].lower()
    if text in ("", "localhost", "localhost.localdomain") or text.endswith(".localhost"):
        return True
    try:
        ip = _ipaddress.ip_address(text)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return bool(ip.is_loopback or ip.is_unspecified)


def _is_this_machine_name(host) -> bool:
    """This machine's own host name (and its Bonjour `<name>.local`) resolves to itself."""
    if isinstance(host, (bytes, bytearray)):
        host = bytes(host).decode("ascii", "replace")
    text = str(host or "").strip().lower().rstrip(".")
    try:
        own = _socket.gethostname().strip().lower().rstrip(".")
    except OSError:
        return False
    short = own[: -len(".local")] if own.endswith(".local") else own.split(".", 1)[0]
    return bool(own) and text in {own, short, f"{short}.local"}


def _is_ip_literal(host) -> bool:
    if isinstance(host, (bytes, bytearray)):
        host = bytes(host).decode("ascii", "replace")
    try:
        _ipaddress.ip_address(str(host or "").strip().strip("[]").split("%", 1)[0])
        return True
    except ValueError:
        return False


def _port_int(port):
    try:
        return int(port)
    except (TypeError, ValueError):
        return None


def _guard_refuse(api: str, host, port, why: str, exc_type) -> None:
    where = f"{host}:{port}"
    test = _GUARD["test"] or "<collection/session>"
    hit = {"test": test, "api": api, "target": where, "why": why}
    _GUARD["hits"].append(hit)
    _GUARD_ALL_HITS.append(hit)
    raise exc_type(
        f"network guard: {test} tried to reach {where} via {api} ({why}). Point the test at a fake or a "
        "scratch loopback port, or mark it @pytest.mark.network(\"reason\") if it genuinely needs the network."
    )


def _guard_connect_target(api: str, host, port) -> None:
    if _GUARD["allowed"]:
        return
    if not _is_loopback_host(host):
        _guard_refuse(api, host, port, "non-loopback destination", NetworkGuardError)
    if _port_int(port) in _LIVE_LOOPBACK_PORTS:
        _guard_refuse(api, host, port, "the operator's live loopback service", NetworkGuardError)


def _inet_target(sock, address):
    # A datagram connect sends no packet (it only asks the kernel for a route;
    # the gateway's LAN-address probe relies on that), so only stream
    # sockets are guarded here.
    if sock.type != _socket.SOCK_STREAM:
        return None
    if sock.family in (_socket.AF_INET, _socket.AF_INET6) and isinstance(address, tuple) and len(address) >= 2:
        return address[0], address[1]
    return None


def _guarded_connect(self, address):
    target = _inet_target(self, address)
    if target is not None:
        _guard_connect_target("socket.connect", *target)
    return _REAL_CONNECT(self, address)


def _guarded_connect_ex(self, address):
    target = _inet_target(self, address)
    if target is not None:
        _guard_connect_target("socket.connect_ex", *target)
    return _REAL_CONNECT_EX(self, address)


def _guarded_create_connection(address, *args, **kwargs):
    if isinstance(address, tuple) and len(address) >= 2:
        _guard_connect_target("socket.create_connection", address[0], address[1])
    return _REAL_CREATE_CONNECTION(address, *args, **kwargs)


def _guarded_getaddrinfo(host, port, *args, **kwargs):
    # Resolving a loopback name, an IP literal or this machine's own name never
    # leaves the machine; the connect that follows is where a non-loopback
    # address or a live loopback port is refused.
    if (
        not _GUARD["allowed"]
        and not _is_loopback_host(host)
        and not _is_ip_literal(host)
        and not _is_this_machine_name(host)
    ):
        _guard_refuse("socket.getaddrinfo", host, port, "non-loopback name resolution", NetworkGuardResolveError)
    return _REAL_GETADDRINFO(host, port, *args, **kwargs)


_socket.socket.connect = _guarded_connect
_socket.socket.connect_ex = _guarded_connect_ex
_socket.create_connection = _guarded_create_connection
_socket.getaddrinfo = _guarded_getaddrinfo


def _marker_reason(item, name: str):
    """(present, reason) for an opt-out marker; reason must be a non-empty string."""
    marker = item.get_closest_marker(name)
    if marker is None:
        return False, None
    reason = marker.args[0] if marker.args else marker.kwargs.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise pytest.UsageError(
            f"{item.nodeid}: @pytest.mark.{name} needs a reason string, e.g. "
            f"@pytest.mark.{name}(\"why this test needs it\")"
        )
    return True, reason


def _register_hermetic_markers(config) -> None:
    # Collection runs before any test: module-level probes are refused too,
    # unless the whole run opted in with --allow-network.
    _GUARD["allowed"] = bool(config.getoption("--allow-network", default=False))
    config.addinivalue_line(
        "markers",
        "network(reason): the test genuinely needs the network (Hub lookup, real download, live-provider "
        "probe). The reason is mandatory. Without the marker the network guard refuses non-loopback "
        "destinations and the operator's live loopback ports 8080, 1234, 11434 and 18850; with it the test "
        "is skipped unless pytest runs with --allow-network.",
    )
    config.addinivalue_line(
        "markers",
        "real_home(reason): the test reads the operator's real home directory (read-only, through "
        "ABSTRACT_TEST_REAL_HOME; HOME itself stays a tmp directory). The reason is mandatory, and a test "
        "module that reads ABSTRACT_TEST_REAL_HOME without this marker is a collection error.",
    )


def _add_hermetic_options(parser) -> None:
    parser.addoption(
        "--allow-network",
        action="store_true",
        default=False,
        help="also run the tests marked @pytest.mark.network (they reach real hosts or the live local services); "
        "without it they are skipped with their reason",
    )


def _validate_hermetic_markers(config, items) -> None:
    allow_network = bool(config.getoption("--allow-network", default=False))
    for item in items:
        needs_network, why = _marker_reason(item, "network")
        if needs_network and not allow_network:
            item.add_marker(pytest.mark.skip(reason=f"needs the network ({why}); run with --allow-network"))
        present, _ = _marker_reason(item, "real_home")
        module = getattr(item, "module", None)
        if not present and module is not None and "ABSTRACT_TEST_REAL_HOME" in _module_source(module):
            raise pytest.UsageError(
                f"{item.nodeid}: reads ABSTRACT_TEST_REAL_HOME without @pytest.mark.real_home(\"reason\")"
            )


_SOURCE_CACHE: dict = {}


def _module_source(module) -> str:
    path = getattr(module, "__file__", None)
    if not path:
        return ""
    if path not in _SOURCE_CACHE:
        try:
            _SOURCE_CACHE[path] = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            _SOURCE_CACHE[path] = ""
    return _SOURCE_CACHE[path]


@pytest.fixture(autouse=True)
def isolate_home_and_network(request, monkeypatch, tmp_path_factory):
    """Per-test home under tmp, operator path knobs scrubbed, network guard armed."""
    for knob in _operator_path_knobs():
        monkeypatch.delenv(knob, raising=False)
    home = Path(str(tmp_path_factory.mktemp("home")))
    for key, value in _hermetic_env(home).items():
        monkeypatch.setenv(key, value)
    # HOME stays tmp even under @pytest.mark.real_home: such a test READS the
    # operator's home through ABSTRACT_TEST_REAL_HOME and so cannot write there
    # by accident (collection refuses the pointer in a module without the marker).
    monkeypatch.setenv("ABSTRACT_TEST_REAL_HOME", _REAL_HOME)
    _assert_hf_constants_isolated()

    allowed, _ = _marker_reason(request.node, "network")
    _GUARD.update(test=request.node.nodeid, allowed=allowed, hits=[])
    yield home
    hits = list(_GUARD["hits"])
    _GUARD.update(test=None, allowed=False, hits=[])
    if hits:
        lines = "\n".join(f"  {h['api']} -> {h['target']} ({h['why']})" for h in hits)
        pytest.fail(f"network guard refused {len(hits)} connection attempt(s):\n{lines}", pytrace=False)


@pytest.fixture
def fake_public_dns(monkeypatch):
    """Resolve every non-loopback name to one fixed public address, without DNS.

    For code that resolves a host before a faked fetch (the fetch_url SSRF
    check, the server's base-URL allowlist). The answer is 93.184.216.34, a
    public address, so the SSRF rules still see a public destination; a real
    connect to it is still refused by the guard.
    """

    def resolve(host, port, family=0, type=0, proto=0, flags=0):
        if _is_loopback_host(host) or _is_ip_literal(host):
            return _REAL_GETADDRINFO(host, port, family, type, proto, flags)
        return [(_socket.AF_INET, _socket.SOCK_STREAM, _socket.IPPROTO_TCP, "", ("93.184.216.34", _port_int(port) or 0))]

    monkeypatch.setattr(_socket, "getaddrinfo", resolve)
    return resolve


def _hermetic_terminal_summary(terminalreporter) -> None:
    if not _GUARD_ALL_HITS:
        return
    terminalreporter.section("network guard")
    for hit in _GUARD_ALL_HITS:
        terminalreporter.write_line(f"{hit['test']}: {hit['api']} -> {hit['target']} ({hit['why']})")


def _hermetic_sessionfinish(session) -> None:
    if any(h["test"] == "<collection/session>" for h in _GUARD_ALL_HITS) and session.exitstatus == 0:
        session.exitstatus = 1


def _hermetic_unconfigure() -> None:
    _shutil.rmtree(_SESSION_HOME, ignore_errors=True)
# ------------------------------------------------------------ end hermetic block


from abstractcore import create_llm
from abstractcore.media.capabilities import is_vision_model


@pytest.fixture(autouse=True)
def allow_unauthenticated_server_in_tests(monkeypatch):
    """Keep existing server unit tests explicit about using unauthenticated local mode."""
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")


@pytest.fixture(autouse=True)
def restore_api_key_environment():
    """Undo API keys a test exported into the process environment.

    `ConfigurationManager.set_api_key()` exports the configured key to its env
    var (for example OPENAI_API_KEY) by design. Without this guard a fake key
    set by a config test leaks into later tests, which then treat it as real
    credentials and call the live API.
    """
    saved = {k: v for k, v in os.environ.items() if k.endswith("_API_KEY")}
    yield
    for k in [k for k in os.environ if k.endswith("_API_KEY") and k not in saved]:
        os.environ.pop(k, None)
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def isolate_data_registry(tmp_path, monkeypatch):
    """Point register-at-first-write at a per-test registry file.

    Provider/embedding construction now registers data homes as a side effect;
    without isolation, unit tests would write the developer's real
    ~/.abstractframework/data_registry.json. Tests that need their own registry
    (tests/utils/test_data_registry.py) override this env with their fixture.
    """
    monkeypatch.setenv("ABSTRACTFRAMEWORK_DATA_REGISTRY", str(tmp_path / "test_data_registry.json"))


@pytest.fixture(autouse=True)
def isolate_context_calibration(tmp_path, monkeypatch):
    """Point the context-calibration store at a per-test directory.

    GGUF loads record where their n_ctx ladder settled as a side effect;
    without isolation, unit tests constructing (fake) GGUF providers would
    write the developer's real ~/.abstractcore/calibration store — and could
    read seeds from it, breaking hermeticity both ways.
    """
    monkeypatch.setenv("ABSTRACTCORE_CALIBRATION_DIR", str(tmp_path / "test_calibration"))


@pytest.fixture(scope="session")
def vision_examples_dir():
    """Path to vision examples directory."""
    return Path(__file__).parent / "vision_examples"


@pytest.fixture(scope="session")
def vision_test_images(vision_examples_dir):
    """List of all available vision test images."""
    if not vision_examples_dir.exists():
        pytest.skip("Vision examples directory not found")

    images = list(vision_examples_dir.glob("*.jpg"))
    if not images:
        pytest.skip("No test images found in vision_examples")

    return [str(img) for img in images]


@pytest.fixture(scope="session")
def vision_reference_files(vision_examples_dir):
    """Dictionary mapping image names to reference JSON files."""
    if not vision_examples_dir.exists():
        pytest.skip("Vision examples directory not found")

    references = {}
    for json_file in vision_examples_dir.glob("*.json"):
        image_name = json_file.name.replace('.json', '.jpg')
        references[image_name] = str(json_file)

    return references


def check_provider_availability(provider: str, model: str = None) -> tuple[bool, str]:
    """
    Check if a provider and optionally a specific model is available.
    Returns (is_available, skip_reason)
    """
    try:
        if provider == "ollama":
            if os.getenv("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS") != "1":
                return False, "Local provider tests disabled (set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1)"
            # Check if Ollama is running
            llm = create_llm(provider, model=model or "qwen2.5vl:7b", timeout=5.0)
            # Try to get model info to verify it's actually available
            if model and hasattr(llm, '_client'):
                # This will fail if model isn't installed
                pass
            return True, ""

        elif provider == "lmstudio":
            if os.getenv("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS") != "1":
                return False, "Local provider tests disabled (set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1)"
            # Check if LMStudio is running
            llm = create_llm(provider, model=model or "qwen/qwen2.5-vl-7b", timeout=5.0)
            return True, ""

        elif provider == "openai":
            if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
                return False, "Live API tests disabled (set ABSTRACTCORE_RUN_LIVE_API_TESTS=1)"
            # Check for API key
            if not os.getenv("OPENAI_API_KEY"):
                return False, "OPENAI_API_KEY not set"
            llm = create_llm(provider, model=model or "gpt-4o", timeout=5.0)
            return True, ""

        elif provider == "anthropic":
            if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
                return False, "Live API tests disabled (set ABSTRACTCORE_RUN_LIVE_API_TESTS=1)"
            # Check for API key
            if not os.getenv("ANTHROPIC_API_KEY"):
                return False, "ANTHROPIC_API_KEY not set"
            llm = create_llm(provider, model=model or "claude-haiku-4-5", timeout=5.0)
            return True, ""

        elif provider == "huggingface":
            # HuggingFace model loading can be heavy and may require large downloads / native backends.
            # Default to "unavailable" unless explicitly enabled.
            if os.getenv("ABSTRACTCORE_RUN_HUGGINGFACE_TESTS") != "1" and os.getenv("ABSTRACTCORE_RUN_GGUF_TESTS") != "1":
                return False, "HuggingFace tests disabled (set ABSTRACTCORE_RUN_HUGGINGFACE_TESTS=1)"

            # Best-effort lightweight check: if a specific model is requested, only claim availability
            # when it appears to be cached locally.
            if model:
                model_dir = (Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model.replace('/', '--')}")
                if not model_dir.exists():
                    return False, f"HuggingFace model not found in cache: {model}"
            return True, ""

        elif provider == "mlx":
            if os.getenv("ABSTRACTCORE_RUN_MLX_TESTS") != "1":
                return False, "MLX tests disabled (set ABSTRACTCORE_RUN_MLX_TESTS=1)"
            # Avoid loading a model here; tests that opt-in can create the provider directly.
            return True, ""

        else:
            return False, f"Unknown provider: {provider}"

    except ImportError as e:
        return False, f"Missing dependency for {provider}: {e}"
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["connection", "refused", "timeout", "operation not permitted", "not found", "not running"]):
            return False, f"{provider} not available: {e}"
        else:
            # Re-raise unexpected errors
            raise


def check_vision_capability(provider: str, model: str) -> tuple[bool, str]:
    """
    Check if a model actually supports vision.
    Returns (supports_vision, skip_reason)
    """
    try:
        if is_vision_model(model):
            return True, ""
        else:
            return False, f"Model {model} does not support vision"
    except Exception as e:
        return False, f"Could not determine vision capability: {e}"


@pytest.fixture(scope="session")
def available_vision_providers():
    """Dictionary of available vision providers and their models."""
    provider_models = {
        "ollama": [
            "qwen2.5vl:7b",
            "llama3.2-vision:11b",
            "gemma3:4b"
        ],
        "lmstudio": [
            "qwen/qwen2.5-vl-7b",
            "qwen/qwen3-vl-4b",
            "google/gemma-3n-e4b"
        ],
        "openai": [
            "gpt-4o",
            "gpt-4-turbo"
        ],
        "anthropic": [
            "claude-haiku-4-5",
        ],
        "huggingface": [
            "unsloth/Qwen2.5-VL-7B-Instruct-GGUF"
        ]
    }

    available = {}

    for provider, models in provider_models.items():
        available_models = []

        for model in models:
            # Check provider availability first
            provider_available, provider_reason = check_provider_availability(provider, model)
            if not provider_available:
                continue

            # Check vision capability
            vision_available, vision_reason = check_vision_capability(provider, model)
            if vision_available:
                available_models.append(model)

        if available_models:
            available[provider] = available_models

    return available


@pytest.fixture
def skip_if_provider_unavailable():
    """Decorator function to skip tests if provider is unavailable."""
    def skipper(provider: str, model: str = None):
        available, reason = check_provider_availability(provider, model)
        if not available:
            pytest.skip(f"Provider {provider} not available: {reason}")

        if model:
            vision_ok, vision_reason = check_vision_capability(provider, model)
            if not vision_ok:
                pytest.skip(f"Vision not supported: {vision_reason}")

    return skipper


@pytest.fixture
def create_vision_llm():
    """Factory function to create vision LLMs with proper error handling."""
    def factory(provider: str, model: str):
        try:
            llm = create_llm(provider, model=model)

            # Verify vision capability
            if not is_vision_model(model):
                pytest.skip(f"Model {model} does not support vision")

            return llm
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "refused", "timeout", "not found"]):
                pytest.skip(f"Provider {provider} not available: {e}")
            else:
                raise

    return factory


# Test markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    _register_hermetic_markers(config)
    config.addinivalue_line(
        "markers", "vision: mark test as requiring vision capabilities"
    )
    config.addinivalue_line(
        "markers", "provider_required: mark test as requiring specific provider"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "comprehensive: mark test as comprehensive vision test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers."""
    _validate_hermetic_markers(config, items)
    for item in items:
        # Add vision marker to tests in vision-related files
        if "vision" in str(item.fspath).lower():
            item.add_marker(pytest.mark.vision)

        # Add slow marker to comprehensive tests
        if "comprehensive" in item.name.lower():
            item.add_marker(pytest.mark.slow)


def pytest_addoption(parser):
    _add_hermetic_options(parser)

def pytest_terminal_summary(terminalreporter):
    _hermetic_terminal_summary(terminalreporter)


def pytest_sessionfinish(session, exitstatus):
    _hermetic_sessionfinish(session)


def pytest_unconfigure(config):
    _hermetic_unconfigure()
