"""ADR 0009 — a named model handle is honoured, or the call fails loudly.

Regression suite for the silent artifact substitution in `HuggingFaceProvider`:
`create_llm(provider="huggingface", model="Qwen/Qwen3.6-27B")` resolved an LM Studio
Hub manifest, followed its `baseModel` dependency to the DIFFERENT repository
`lmstudio-community/Qwen3.6-27B-GGUF`, and loaded a Q4_K_M GGUF on llama.cpp in place
of the requested bf16 transformers weights — with no warning. Benchmark numbers taken
through that path were attributed to the model the caller named.

Every test here is cache-only and offline: HOME is redirected to `tmp_path`, and the
loaders/device probes are stubbed, so no model, torch or llama.cpp is ever touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from abstractcore.providers.huggingface_provider import HuggingFaceProvider


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

def _write_hub_manifest(home: Path, alias_org: str, alias_name: str, dep_user: str, dep_repo: str) -> Path:
    """Write an LM Studio Hub manifest aliasing `alias_org/alias_name` to a GGUF repo."""
    manifest_dir = home / ".lmstudio" / "hub" / "models" / alias_org / alias_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "type": "model",
                "owner": alias_org,
                "name": alias_name,
                "dependencies": [
                    {
                        "type": "model",
                        "purpose": "baseModel",
                        "modelKeys": [f"{dep_user}/{dep_repo}".lower()],
                        "sources": [{"type": "huggingface", "user": dep_user, "repo": dep_repo}],
                    }
                ],
                "revision": 1,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _write_lmstudio_gguf(home: Path, org: str, repo: str, *filenames: str) -> list[Path]:
    model_dir = home / ".lmstudio" / "models" / org / repo
    model_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for name in filenames:
        path = model_dir / name
        path.write_bytes(b"GGUF")
        out.append(path)
    return out


def _write_transformers_snapshot(home: Path, org: str, repo: str) -> Path:
    """Write a plausible transformers (safetensors) snapshot into the HF hub cache."""
    snapshot = (
        home / ".cache" / "huggingface" / "hub" / f"models--{org}--{repo}" / "snapshots" / "snapshot123"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"\x00")
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"\x00")
    return snapshot


def _substitution_topology(home: Path) -> dict:
    """The on-disk situation in which a substitution can actually occur.

    An LM Studio Hub manifest routes the requested name to a 4-bit GGUF in a
    DIFFERENT repository, and the requested repo's own transformers weights are NOT
    on this disk — so honouring the handle as named is impossible and returning the
    GGUF would be a substitution.
    """
    manifest = _write_hub_manifest(home, "qwen", "qwen3.6-27b", "lmstudio-community", "Qwen3.6-27B-GGUF")
    gguf = _write_lmstudio_gguf(home, "lmstudio-community", "Qwen3.6-27B-GGUF", "Qwen3.6-27B-Q4_K_M.gguf")[0]
    return {"manifest": manifest, "gguf": gguf, "snapshot": None}


def _coexistence_topology(home: Path) -> dict:
    """The same manifest and GGUF, but the requested weights ARE present.

    This is the real `Qwen/Qwen3.6-27B` cache on the benchmark host. Nothing is being
    substituted here: the handle names transformers weights, they exist, and they are
    what loads. The GGUF merely coexists.
    """
    topology = _substitution_topology(home)
    topology["snapshot"] = _write_transformers_snapshot(home, "Qwen", "Qwen3.6-27B")
    return topology


@pytest.fixture
def stub_loaders(monkeypatch) -> list[str]:
    """Record which artifact loader the resolution logic selected, loading nothing.

    Device setup is stubbed too: `_setup_device_gguf` probes llama.cpp, which
    initialises a Metal device we neither need nor want in a unit test.
    """
    calls: list[str] = []
    monkeypatch.setattr(HuggingFaceProvider, "_setup_device_gguf", lambda self: None)
    monkeypatch.setattr(HuggingFaceProvider, "_setup_device_transformers", lambda self: None)
    monkeypatch.setattr(HuggingFaceProvider, "_load_gguf_model", lambda self: calls.append("gguf"))
    monkeypatch.setattr(
        HuggingFaceProvider, "_load_transformers_model", lambda self: calls.append("transformers")
    )
    return calls


# ---------------------------------------------------------------------------
# the defect
# ---------------------------------------------------------------------------

@pytest.mark.basic
def test_bare_hf_handle_is_never_silently_promoted_to_a_cached_gguf(
    monkeypatch, tmp_path: Path, stub_loaders: list[str]
) -> None:
    """The exact Qwen/Qwen3.6-27B case: it must raise, not return a GGUF provider."""
    monkeypatch.setenv("HOME", str(tmp_path))
    topology = _substitution_topology(tmp_path)

    # `Exception` rather than the concrete class so that pre-fix code fails on the
    # behaviour ("DID NOT RAISE") instead of on an import of a symbol it lacks.
    with pytest.raises(Exception) as excinfo:
        HuggingFaceProvider(model="Qwen/Qwen3.6-27B")

    assert type(excinfo.value).__name__ == "ModelArtifactMismatchError"
    # No artifact was loaded at all — the gate runs before either loader.
    assert stub_loaders == []

    message = str(excinfo.value)
    # Actionable: what was requested, what was found, and how to ask for either one.
    assert "Qwen/Qwen3.6-27B" in message
    assert str(topology["gguf"]) in message
    assert "model_type='gguf'" in message
    assert "model_type='transformers'" in message


@pytest.mark.basic
def test_substitution_gate_names_the_manifest_that_caused_it(monkeypatch, tmp_path: Path) -> None:
    """The error must point at the manifest, or the substitution is unexplainable."""
    monkeypatch.setenv("HOME", str(tmp_path))
    topology = _substitution_topology(tmp_path)

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    with pytest.raises(Exception) as excinfo:
        provider._reject_silent_gguf_substitution("Qwen/Qwen3.6-27B")

    assert str(topology["manifest"]) in str(excinfo.value)


@pytest.mark.basic
def test_manifest_plus_gguf_does_not_block_a_handle_whose_weights_are_present(
    monkeypatch, tmp_path: Path, stub_loaders: list[str]
) -> None:
    """Coexistence is not substitution — the gate must not refuse a loadable handle.

    The first fix for the substitution defect keyed the refusal on "a Hub manifest
    exists AND some GGUF resolves", never asking whether the named artifact was
    itself loadable. That made `create_llm("huggingface", "Qwen/Qwen3.6-27B")` raise
    on a host whose complete bf16 snapshot was sitting in the cache: a model in the
    benchmark matrix became unloadable through its own public handle. ADR 0009
    forbids returning a DIFFERENT artifact, not loading the requested one.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    _coexistence_topology(tmp_path)

    provider = HuggingFaceProvider(model="Qwen/Qwen3.6-27B")

    assert provider.model_type == "transformers"
    # The decisive assertion: transformers weights were loaded and the GGUF was not.
    assert stub_loaders == ["transformers"]


# ---------------------------------------------------------------------------
# preserved legitimate behaviour
# ---------------------------------------------------------------------------

@pytest.mark.basic
def test_bare_hf_id_with_only_transformers_weights_resolves_normally(
    monkeypatch, tmp_path: Path, stub_loaders: list[str]
) -> None:
    """No manifest, no GGUF: an ordinary handle must still load transformers."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_transformers_snapshot(tmp_path, "Qwen", "Qwen3-4B-Instruct-2507")

    provider = HuggingFaceProvider(model="Qwen/Qwen3-4B-Instruct-2507")

    assert provider.model_type == "transformers"
    assert stub_loaders == ["transformers"]


@pytest.mark.basic
def test_genuine_gguf_repo_id_still_loads_as_gguf(
    monkeypatch, tmp_path: Path, stub_loaders: list[str]
) -> None:
    """A handle that names a GGUF repository is honoured, unchanged."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _substitution_topology(tmp_path)

    provider = HuggingFaceProvider(model="lmstudio-community/Qwen3.6-27B-GGUF")

    assert provider.model_type == "gguf"
    assert stub_loaders == ["gguf"]


@pytest.mark.basic
def test_gguf_file_path_still_loads_as_gguf(
    monkeypatch, tmp_path: Path, stub_loaders: list[str]
) -> None:
    """Naming the artifact directly is always allowed — it is unambiguous."""
    monkeypatch.setenv("HOME", str(tmp_path))
    gguf = _write_lmstudio_gguf(tmp_path, "org", "Model-GGUF", "model-Q4_K_M.gguf")[0]

    provider = HuggingFaceProvider(model=str(gguf))

    assert provider.model_type == "gguf"
    assert stub_loaders == ["gguf"]


@pytest.mark.basic
def test_explicit_model_type_gguf_opts_into_the_hub_alias(
    monkeypatch, tmp_path: Path, stub_loaders: list[str]
) -> None:
    """The escape hatch: a caller who genuinely wants the GGUF says so, and gets it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    topology = _substitution_topology(tmp_path)

    provider = HuggingFaceProvider(model="Qwen/Qwen3.6-27B", model_type="gguf")

    assert provider.model_type == "gguf"
    assert stub_loaders == ["gguf"]
    # And it resolves to the artifact the caller opted into.
    assert provider._find_gguf_in_cache("Qwen/Qwen3.6-27B") == str(topology["gguf"])


@pytest.mark.basic
def test_explicit_model_type_transformers_bypasses_the_gate(
    monkeypatch, tmp_path: Path, stub_loaders: list[str]
) -> None:
    """Declaring transformers is an answer to the ambiguity, so the gate stands down."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _substitution_topology(tmp_path)

    provider = HuggingFaceProvider(model="Qwen/Qwen3.6-27B", model_type="transformers")

    assert provider.model_type == "transformers"
    assert stub_loaders == ["transformers"]


@pytest.mark.basic
def test_lmstudio_hub_alias_probe_still_resolves(monkeypatch, tmp_path: Path) -> None:
    """`_find_gguf_in_cache` stays a pure locator; the policy gate lives at construction."""
    monkeypatch.setenv("HOME", str(tmp_path))
    topology = _substitution_topology(tmp_path)

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)

    assert provider._find_gguf_in_cache("qwen/qwen3.6-27b") == str(topology["gguf"])


# ---------------------------------------------------------------------------
# quantization: no silent substitution of an explicitly requested quant
# ---------------------------------------------------------------------------

@pytest.mark.basic
def test_unsatisfiable_quant_selector_raises_instead_of_downgrading(
    monkeypatch, tmp_path: Path
) -> None:
    """`:Q8_0` when only Q4_K_M exists must fail, not silently hand back Q4_K_M."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_lmstudio_gguf(tmp_path, "unsloth", "Model-GGUF", "Model-Q4_K_M.gguf", "Model-Q5_K_M.gguf")

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    with pytest.raises(Exception) as excinfo:
        provider._find_gguf_in_cache("unsloth/Model-GGUF:Q8_0")

    assert type(excinfo.value).__name__ == "ModelArtifactMismatchError"
    message = str(excinfo.value)
    assert "Q8_0" in message
    # Actionable: the caller is told what they could have asked for.
    assert "Model-Q4_K_M.gguf" in message
    assert "Model-Q5_K_M.gguf" in message


@pytest.mark.basic
def test_quant_selector_survives_hub_manifest_recursion(monkeypatch, tmp_path: Path) -> None:
    """An explicit quant must not be dropped when an alias redirects to another repo."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_hub_manifest(tmp_path, "qwen", "qwen3.6-27b", "lmstudio-community", "Qwen3.6-27B-GGUF")
    files = _write_lmstudio_gguf(
        tmp_path,
        "lmstudio-community",
        "Qwen3.6-27B-GGUF",
        "Qwen3.6-27B-Q4_K_M.gguf",
        "Qwen3.6-27B-Q8_0.gguf",
    )
    q8 = next(p for p in files if "Q8_0" in p.name)

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)

    # Without selector propagation this returns the Q4_K_M default pick.
    assert provider._find_gguf_in_cache("qwen/qwen3.6-27b:Q8_0") == str(q8)


@pytest.mark.basic
def test_exact_filename_selector_beats_a_loose_path_match(monkeypatch, tmp_path: Path) -> None:
    """Selector matching is ordered: exact filename wins over an incidental path hit."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # The directory name contains "Q4_K_M", so a path-substring match would hit
    # whichever file sorts first rather than the file the caller actually named.
    files = _write_lmstudio_gguf(
        tmp_path, "org", "Bundle-Q4_K_M-GGUF", "aaa-first.gguf", "Q8_0.gguf"
    )
    exact = next(p for p in files if p.name == "Q8_0.gguf")

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)

    assert provider._find_gguf_in_cache("org/Bundle-Q4_K_M-GGUF:Q8_0.gguf") == str(exact)


@pytest.mark.basic
def test_default_quant_pick_is_announced(monkeypatch, tmp_path: Path, caplog) -> None:
    """A repo id underdetermines the quant, so picking one is fine — but never silent."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_lmstudio_gguf(tmp_path, "unsloth", "Model-GGUF", "Model-Q4_K_M.gguf", "Model-Q8_0.gguf")

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    with caplog.at_level("WARNING", logger="abstractcore.providers.huggingface"):
        found = provider._find_gguf_in_cache("unsloth/Model-GGUF")

    assert found.endswith("Model-Q4_K_M.gguf")
    assert any("2 quantizations" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# selector validation
# ---------------------------------------------------------------------------

@pytest.mark.basic
def test_invalid_model_type_selector_is_rejected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(Exception) as excinfo:
        HuggingFaceProvider(model="Qwen/Qwen3-4B-Instruct-2507", model_type="safetensors")

    assert type(excinfo.value).__name__ == "InvalidRequestError"
    assert "safetensors" in str(excinfo.value)


@pytest.mark.basic
def test_artifact_mismatch_survives_the_constructor_as_its_own_type(
    monkeypatch, tmp_path: Path
) -> None:
    """ADR 0009 Enforcement: the refusal must reach the caller catchable as itself.

    `_load_gguf_model` wrapped every exception in `RuntimeError(f"Failed to load GGUF
    model ...")`, so an artifact-fidelity refusal raised during the constructor path
    arrived as a generic load failure — uncatchable by callers that handle the
    contract type, which is the same information loss the ADR exists to prevent.
    The existing suite missed it by calling `_find_gguf_in_cache` directly instead of
    going through the constructor.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_lmstudio_gguf(tmp_path, "unsloth", "Model-GGUF", "Model-Q4_K_M.gguf")

    with pytest.raises(Exception) as excinfo:
        HuggingFaceProvider(model="unsloth/Model-GGUF:Q8_0")

    assert type(excinfo.value).__name__ == "ModelArtifactMismatchError"


@pytest.mark.basic
def test_model_type_transformers_on_a_gguf_path_is_rejected(monkeypatch, tmp_path: Path) -> None:
    """A contradictory pair fails immediately rather than deep inside the loader."""
    monkeypatch.setenv("HOME", str(tmp_path))
    gguf = _write_lmstudio_gguf(tmp_path, "org", "Model-GGUF", "model-Q4_K_M.gguf")[0]

    with pytest.raises(Exception) as excinfo:
        HuggingFaceProvider(model=str(gguf), model_type="transformers")

    assert type(excinfo.value).__name__ == "ModelArtifactMismatchError"
