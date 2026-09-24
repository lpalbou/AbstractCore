"""Offline-first is per LOAD CALL in the Hugging Face provider, never per process.

Mission U (2026-09-24). `huggingface_provider` used to write TRANSFORMERS_OFFLINE
/ HF_DATASETS_OFFLINE / HF_HUB_OFFLINE = 1 into `os.environ` at IMPORT. Every
child process inherited them (download jobs died with OfflineModeIsEnabled;
engine installs, app launches and tools ran "offline"), while in-process the
write only worked when this module happened to be imported before
huggingface_hub.

Separately, fully cached models failed to load offline with "We couldn't
connect to 'https://huggingface.co' ... couldn't find them in the cached
files". Root cause: AbstractCore's downloader pins the listed commit
(`snapshot_download(revision=<sha>)`), and huggingface_hub writes no
`refs/main` for a revision that already is a commit hash; transformers resolves
a repo id offline through `refs/main`, finds nothing, and reports the files
missing. Transformers also HEADs `adapter_config.json` on the Hub even with
`local_files_only=True` (the flag is not forwarded to its PEFT probe).

The fix hands every transformers call the cached snapshot DIRECTORY plus
`local_files_only=True`. These tests pin, with a socket guard:
  * import and load leave the three variables exactly as found (and a child
    process spawned after a load sees none);
  * a fully cached model loads with zero network in BOTH cache layouts;
  * an uncached / partial / README-only / adapter-without-base model fails fast
    with a plain message naming the model and the fix;
  * a PEFT adapter over a cached base loads offline;
  * an uncached MLX drafter is refused under offline-first, downloaded (and
    logged) when offline-first is off.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.models_engines_fakes import isolate_host

_NAMES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
_SHA = "0123456789abcdef0123456789abcdef01234567"
_REPO = "test-org/tiny-gpt2"


def _flags() -> dict:
    return {name: os.environ[name] for name in _NAMES if name in os.environ}


# ---------------------------------------------------------------------------
# Import: no environment write, whatever offline_first says.
# ---------------------------------------------------------------------------


def test_importing_the_provider_writes_no_offline_variable(tmp_path):
    """A fresh interpreter, default config (offline_first on), no flag set."""

    env = {k: v for k, v in os.environ.items() if k not in _NAMES}
    env["ABSTRACTCORE_CONFIG_DIR"] = str(tmp_path / "config")
    env.pop("ABSTRACTGATEWAY_AUTH_TOKEN", None)
    code = (
        "import json, os\n"
        "import abstractcore.providers.huggingface_provider as hfp\n"
        "assert hfp._config.is_offline_first() is True\n"
        f"print(json.dumps({{n: os.environ.get(n) for n in {list(_NAMES)!r}}}))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[2]))
    assert out.returncode == 0, out.stderr[-2000:]
    seen = json.loads(out.stdout.strip().splitlines()[-1])
    assert seen == {name: None for name in _NAMES}, f"import wrote process-wide offline flags: {seen}"


# ---------------------------------------------------------------------------
# Fixtures: an isolated host, a stub config, a socket guard, a tiny model.
# ---------------------------------------------------------------------------


@pytest.fixture
def host(tmp_path, monkeypatch):
    h = isolate_host(tmp_path, monkeypatch)
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)
    return h


def _stub_config(offline_first: bool, force_local_files_only: bool, cache_root: Path):
    return SimpleNamespace(
        is_offline_first=lambda: offline_first,
        should_force_local_files_only=lambda: force_local_files_only,
        config=SimpleNamespace(cache=SimpleNamespace(huggingface_cache_dir=str(cache_root))),
    )


@pytest.fixture
def offline_first(host, monkeypatch):
    import abstractcore.providers.huggingface_provider as hfp

    monkeypatch.setattr(hfp, "_config", _stub_config(True, True, host["home"] / "unused-hf"))
    return host


@pytest.fixture
def no_network(monkeypatch):
    """Refuse (and record) every DNS lookup and outbound connection."""

    attempts: list = []

    def _refuse(kind):
        def _fn(*args, **kwargs):
            attempts.append((kind, repr(args[:2])[:200]))
            raise OSError(f"network refused by test ({kind})")

        return _fn

    monkeypatch.setattr(socket.socket, "connect", _refuse("socket.connect"))
    monkeypatch.setattr(socket, "create_connection", _refuse("socket.create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", _refuse("socket.getaddrinfo"))
    return attempts


def _write_tiny_gpt2(dst: Path) -> Path:
    """A real (random, ~100 KB) GPT-2 checkpoint and fast tokenizer, built offline."""

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    dst.mkdir(parents=True, exist_ok=True)
    vocab = {"<|endoftext|>": 0}
    for word in "hello world the a of to and is in it".split():
        vocab[word] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<|endoftext|>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|endoftext|>", unk_token="<|endoftext|>"
    ).save_pretrained(str(dst))
    config = GPT2Config(vocab_size=len(vocab), n_positions=32, n_embd=8, n_layer=1, n_head=2)
    GPT2LMHeadModel(config).save_pretrained(str(dst))
    return dst


def _cache_snapshot(hub: Path, repo: str, sha: str = _SHA, *, refs_main: bool) -> Path:
    repo_dir = hub / ("models--" + repo.replace("/", "--"))
    snap = repo_dir / "snapshots" / sha
    snap.mkdir(parents=True, exist_ok=True)
    if refs_main:
        (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs" / "main").write_text(sha, encoding="utf-8")
    return snap


@pytest.fixture(scope="module")
def tiny_gpt2_files(tmp_path_factory):
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    return _write_tiny_gpt2(tmp_path_factory.mktemp("tiny-gpt2"))


def _install_tiny_gpt2(hub: Path, files: Path, repo: str = _REPO, *, refs_main: bool) -> Path:
    import shutil

    snap = _cache_snapshot(hub, repo, refs_main=refs_main)
    for f in files.iterdir():
        shutil.copy2(f, snap / f.name)
    return snap


def _load(model: str):
    # The provider cannot be constructed without transformers (the CI `[test]`
    # extra does not install it), so every load-path test skips there.
    pytest.importorskip("transformers", reason="the HuggingFace provider needs transformers to construct")
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    return HuggingFaceProvider(model=model, device="cpu")


# ---------------------------------------------------------------------------
# Cached loads: zero network, both cache layouts, nothing leaks to children.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("refs_main", [False, True], ids=["pinned-download-no-refs-main", "plain-snapshot-download"])
def test_a_fully_cached_model_loads_with_zero_network(offline_first, no_network, tiny_gpt2_files, refs_main):
    """`refs_main=False` is the layout AbstractCore's own downloader leaves:
    the load that used to fail with "couldn't find them in the cached files"."""

    snap = _install_tiny_gpt2(offline_first["hf"], tiny_gpt2_files, refs_main=refs_main)
    provider = _load(_REPO)

    assert no_network == [], f"a cached load reached for the network: {no_network}"
    assert provider.model == _REPO, "the handle the caller named is kept"
    assert provider.model_instance is not None and provider.tokenizer is not None
    assert provider.model_instance.config._commit_hash == _SHA, "the commit identity survives a directory load"
    assert provider._transformers_source.model == str(snap)
    assert _flags() == {}, f"a load wrote process-wide offline flags: {_flags()}"


def test_a_child_spawned_after_a_load_sees_no_offline_flag(offline_first, tiny_gpt2_files):
    _install_tiny_gpt2(offline_first["hf"], tiny_gpt2_files, refs_main=False)
    _load(_REPO)
    code = f"import json, os; print(json.dumps({{n: os.environ.get(n) for n in {list(_NAMES)!r}}}))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)  # env=None: inherits os.environ
    assert json.loads(out.stdout.strip()) == {name: None for name in _NAMES}


def test_a_load_keeps_exactly_what_the_operator_set(offline_first, monkeypatch, tiny_gpt2_files):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    _install_tiny_gpt2(offline_first["hf"], tiny_gpt2_files, refs_main=False)
    _load(_REPO)
    assert {name: os.environ.get(name) for name in _NAMES} == {
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": None,
        "HF_DATASETS_OFFLINE": None,
    }


def test_every_transformers_call_gets_the_directory_and_local_files_only(offline_first, monkeypatch, tiny_gpt2_files):
    """The per-call mechanism itself, entry point by entry point."""

    import transformers

    snap = _install_tiny_gpt2(offline_first["hf"], tiny_gpt2_files, refs_main=False)
    seen: list = []
    for cls_name in ("AutoConfig", "AutoTokenizer", "AutoModelForCausalLM"):
        cls = getattr(transformers, cls_name)
        original = cls.from_pretrained.__func__

        def _spy(klass, path, *a, __orig=original, __name=cls_name, **k):
            seen.append((__name, str(path), k.get("local_files_only")))
            return __orig(klass, path, *a, **k)

        monkeypatch.setattr(cls, "from_pretrained", classmethod(_spy))
    _load(_REPO)
    assert seen, "no transformers entry point was called"
    assert all(path == str(snap) and lfo is True for _, path, lfo in seen), seen


# ---------------------------------------------------------------------------
# Not loadable from disk: fail fast, plainly, without touching the network.
# ---------------------------------------------------------------------------


def _assert_plain_failure(excinfo, *needles):
    text = str(excinfo.value)
    for needle in needles:
        assert needle in text, f"{needle!r} missing from: {text}"
    assert "couldn't connect" not in text.lower()


def test_an_uncached_model_fails_fast_with_the_download_command(offline_first, no_network):
    from abstractcore.exceptions import ModelNotFoundError

    started = time.monotonic()
    with pytest.raises(ModelNotFoundError) as excinfo:
        _load("test-org/never-downloaded")
    assert time.monotonic() - started < 5.0
    assert no_network == []
    _assert_plain_failure(
        excinfo,
        "'test-org/never-downloaded'",
        "download it first",
        "abstractcore models download huggingface test-org/never-downloaded",
    )


def test_a_config_only_snapshot_is_reported_as_an_unfinished_download(offline_first, no_network):
    from abstractcore.exceptions import ModelNotFoundError

    snap = _cache_snapshot(offline_first["hf"], "test-org/half-done", refs_main=True)
    (snap / "config.json").write_text('{"model_type": "gpt2"}', encoding="utf-8")
    with pytest.raises(ModelNotFoundError) as excinfo:
        _load("test-org/half-done")
    assert no_network == []
    _assert_plain_failure(excinfo, "'test-org/half-done'", "no weight file", "download it first")


def test_a_readme_only_snapshot_is_not_a_transformers_model(offline_first, no_network):
    """The mission Q case: `ABDALLALSWAITI/linex-storybook-lora` is a 2.9 KB
    README in the cache -- nothing a text model can be built from."""

    from abstractcore.exceptions import ModelNotFoundError

    snap = _cache_snapshot(offline_first["hf"], "test-org/storybook-lora", refs_main=True)
    (snap / "README.md").write_text("# a diffusion LoRA\n", encoding="utf-8")
    with pytest.raises(ModelNotFoundError) as excinfo:
        _load("test-org/storybook-lora")
    assert no_network == []
    _assert_plain_failure(excinfo, "'test-org/storybook-lora'", "README.md", "no config.json")


# ---------------------------------------------------------------------------
# PEFT adapters (LoRA): base + adapter both from disk.
# ---------------------------------------------------------------------------


def _skip_unless_peft_works_with_this_transformers() -> None:
    """Skip, with the provider's own plain reason, when this environment holds a
    peft / transformers pair that cannot attach an adapter (for example
    transformers 5.17, whose MIN_PEFT_VERSION is 0.19.1, beside peft 0.18.1)."""
    from abstractcore.providers.huggingface_provider import _peft_adapter_support_problem

    problem = _peft_adapter_support_problem()
    if problem:
        pytest.skip(problem)


def _install_adapter(hub: Path, base_dir: Path, repo: str, base_repo: str) -> Path:
    """A real LoRA (trained weights: none, but a valid peft checkpoint) over `base_dir`."""
    _skip_unless_peft_works_with_this_transformers()
    import peft
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(str(base_dir), local_files_only=True)
    lora = peft.get_peft_model(base, peft.LoraConfig(r=2, target_modules=["c_attn"], task_type="CAUSAL_LM"))
    snap = _cache_snapshot(hub, repo, sha="f" * 40, refs_main=False)
    lora.save_pretrained(str(snap))
    cfg_path = snap / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["base_model_name_or_path"] = base_repo
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return snap


def _write_adapter_layout(hub: Path, repo: str, base_repo: str) -> Path:
    """The on-disk SHAPE of a PEFT adapter, without peft: enough for resolution."""
    snap = _cache_snapshot(hub, repo, sha="e" * 40, refs_main=False)
    (snap / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": base_repo, "r": 2,
                    "target_modules": ["c_attn"], "task_type": "CAUSAL_LM"}),
        encoding="utf-8",
    )
    (snap / "adapter_model.safetensors").write_bytes(b"")
    return snap


def test_a_cached_adapter_over_a_cached_base_loads_offline(offline_first, no_network, tiny_gpt2_files):
    base_snap = _install_tiny_gpt2(offline_first["hf"], tiny_gpt2_files, refs_main=False)
    adapter_snap = _install_adapter(offline_first["hf"], base_snap, "test-org/tiny-gpt2-lora", _REPO)

    provider = _load("test-org/tiny-gpt2-lora")
    assert no_network == [], f"an adapter load reached for the network: {no_network}"
    assert provider._transformers_source.adapter == str(adapter_snap)
    assert provider._transformers_source.model == str(base_snap)
    assert list(provider.model_instance.active_adapters()) == ["default"]
    assert _flags() == {}


def test_an_adapter_whose_base_is_not_cached_names_the_base(offline_first, no_network):
    from abstractcore.exceptions import ModelNotFoundError

    _write_adapter_layout(offline_first["hf"], "test-org/orphan-lora", "test-org/missing-base")
    with pytest.raises(ModelNotFoundError) as excinfo:
        _load("test-org/orphan-lora")
    assert no_network == []
    _assert_plain_failure(
        excinfo,
        "'test-org/orphan-lora'",
        "'test-org/missing-base'",
        "abstractcore models download huggingface test-org/missing-base",
    )


def _refuse_base_weight_load(monkeypatch):
    """The adapter check must fire BEFORE the base model's weights are read."""
    import transformers

    def _boom(*a, **k):
        raise AssertionError("the base model was loaded before the adapter runtime was checked")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", _boom)


def test_a_broken_peft_import_is_a_plain_provider_error(offline_first, no_network, monkeypatch, tiny_gpt2_files):
    """peft that cannot be imported (missing, or broken against this transformers)
    is named plainly -- never a raw ImportError wrapped in RuntimeError."""
    from abstractcore.exceptions import ModelNotFoundError, ProviderError

    _install_tiny_gpt2(offline_first["hf"], tiny_gpt2_files, refs_main=False)
    _write_adapter_layout(offline_first["hf"], "test-org/tiny-gpt2-lora", _REPO)
    monkeypatch.setitem(sys.modules, "peft", None)  # `import peft` -> ImportError
    _refuse_base_weight_load(monkeypatch)

    with pytest.raises(ProviderError) as excinfo:
        _load("test-org/tiny-gpt2-lora")
    assert not isinstance(excinfo.value, ModelNotFoundError), "the adapter IS on disk; its runtime is what is missing"
    text = str(excinfo.value)
    for needle in ("'test-org/tiny-gpt2-lora'", "adapter support needs peft", "compatible with transformers",
                   "installed: peft", "importing peft failed", "pip install -U"):
        assert needle in text, f"{needle!r} missing from: {text}"
    assert no_network == []


def test_a_peft_older_than_transformers_requires_is_a_plain_provider_error(offline_first, monkeypatch, tiny_gpt2_files):
    """The coordinator's venv: transformers 5.17 (MIN_PEFT_VERSION 0.19.1) + peft 0.18.1."""
    import importlib.metadata as md

    from abstractcore.exceptions import ProviderError

    pytest.importorskip("peft")
    from transformers.integrations import peft as tf_peft

    minimum = getattr(tf_peft, "MIN_PEFT_VERSION", None)
    if not minimum:
        pytest.skip("this transformers declares no MIN_PEFT_VERSION")
    _install_tiny_gpt2(offline_first["hf"], tiny_gpt2_files, refs_main=False)
    _write_adapter_layout(offline_first["hf"], "test-org/tiny-gpt2-lora", _REPO)
    real_version = md.version
    monkeypatch.setattr(md, "version", lambda dist: "0.0.1" if dist == "peft" else real_version(dist))
    _refuse_base_weight_load(monkeypatch)

    with pytest.raises(ProviderError) as excinfo:
        _load("test-org/tiny-gpt2-lora")
    text = str(excinfo.value)
    assert f"adapter support needs peft >= {minimum} compatible with transformers {real_version('transformers')}" in text
    assert "installed: peft 0.0.1" in text


# ---------------------------------------------------------------------------
# Offline-first OFF: the load may download, and says so.
# ---------------------------------------------------------------------------


def test_with_offline_first_off_an_uncached_model_is_handed_to_transformers_by_id(host, monkeypatch, caplog):
    import abstractcore.providers.huggingface_provider as hfp

    monkeypatch.setattr(hfp, "_config", _stub_config(False, False, host["home"] / "unused-hf"))
    with caplog.at_level(logging.INFO, logger="abstractcore.providers.huggingface"):
        source = hfp._resolve_transformers_load_source("test-org/not-here", local_only=hfp._hf_load_is_local_only())
    assert source.model == "test-org/not-here" and source.local_only is False
    assert any("may download" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# MLX drafter: explicit, never silent.
# ---------------------------------------------------------------------------


def _drafter_env(monkeypatch, *, offline_first: bool):
    import abstractcore.config.manager as manager

    monkeypatch.setattr(
        manager, "get_config_manager", lambda *a, **k: SimpleNamespace(is_offline_first=lambda: offline_first)
    )
    calls: list = []
    fake_utils = SimpleNamespace(get_model_path=lambda name: calls.append(name) or "/nonexistent/drafter")
    fake_mlx_vlm = SimpleNamespace(utils=fake_utils)
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", fake_utils)
    return calls


def test_an_uncached_drafter_is_refused_under_offline_first(host, monkeypatch):
    from abstractcore.exceptions import ModelNotFoundError
    from abstractcore.providers.mlx_native_session import resolve_native_drafter_path

    calls = _drafter_env(monkeypatch, offline_first=True)
    with pytest.raises(ModelNotFoundError) as excinfo:
        resolve_native_drafter_path("test-org/uncached-drafter")
    assert calls == [], "offline_first must never reach the downloading resolver"
    _assert_plain_failure(excinfo, "'test-org/uncached-drafter'", "abstractcore models download mlx test-org/uncached-drafter")


def test_an_uncached_drafter_is_downloaded_and_logged_when_offline_first_is_off(host, monkeypatch, caplog):
    from abstractcore.providers.mlx_native_session import resolve_native_drafter_path

    calls = _drafter_env(monkeypatch, offline_first=False)
    with caplog.at_level(logging.WARNING, logger="abstractcore.providers.mlx"):
        with pytest.raises(FileNotFoundError):  # the fake resolver returns a path that is not there
            resolve_native_drafter_path("test-org/uncached-drafter")
    assert calls == ["test-org/uncached-drafter"]
    assert any("being downloaded" in r.getMessage() for r in caplog.records)


def test_a_cached_drafter_never_asks_the_resolver(host, monkeypatch):
    from abstractcore.providers.mlx_native_session import resolve_native_drafter_path

    calls = _drafter_env(monkeypatch, offline_first=True)
    snap = _cache_snapshot(host["hf"], "test-org/cached-drafter", refs_main=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_native_drafter_path("test-org/cached-drafter") == str(snap.resolve())
    assert calls == []
