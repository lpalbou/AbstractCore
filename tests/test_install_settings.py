"""AbstractCore has exactly three install settings: light (no extra), `apple` and `gpu`.

Operator ruling (2026-09-29): "WE SHOULD ONLY HAVE 3 SETTINGS: NONE (for light install using
remote inferencers), APPLE (for apple machines!), GPU (for nvidia and amd). THAT IS IT!!!!"

These tests pin that contract statically (no network):

- `pyproject.toml` declares only `apple`, `gpu`, contributor tooling, and the deprecated aliases
  listed in the one table in docs/installation.md -- and each alias installs what the table says;
- the light install carries every remote provider's dependencies (the registry asks for no extra);
- no user-facing doc, and no install hint in the package, shows any setting other than
  `abstractcore`, `abstractcore[apple]` or `abstractcore[gpu]`.

The fresh-venv proofs (light installs and runs every remote provider; apple and gpu resolve for
their platforms) live in tests/install/test_install_settings_resolution.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.9/3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
EXTRAS: dict[str, list[str]] = PYPROJECT["project"]["optional-dependencies"]
BASE: list[str] = PYPROJECT["project"]["dependencies"]
INSTALL_DOC = ROOT / "docs" / "installation.md"

DOCUMENTED_SETTINGS = {"none", "apple", "gpu"}
SETTING_EXTRAS = {"apple", "gpu"}
CONTRIBUTOR_EXTRAS = {"dev", "test", "docs"}

REMOTE_PROVIDERS = ("openai", "anthropic", "openrouter", "portkey", "openai-compatible", "lmstudio", "ollama", "vllm")


def _name(requirement: str) -> str:
    return re.split(r"[\s\[<>=!~;]", requirement.strip(), maxsplit=1)[0].lower()


def _documented_aliases() -> dict[str, str]:
    """Parse the `## Deprecated aliases` table: alias -> 'light' | 'apple' | 'gpu' | 'unchanged'."""
    text = INSTALL_DOC.read_text(encoding="utf-8")
    section = text.split("## Deprecated aliases", 1)[1].split("\n## ", 1)[0]
    aliases: dict[str, str] = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 2 or not cells[0].startswith("`"):
            continue
        target = cells[1].split(":", 1)[0].strip()
        for alias in re.findall(r"`([^`]+)`", cells[0]):
            assert alias not in aliases, f"alias {alias!r} listed twice"
            aliases[alias] = target
    assert aliases, "docs/installation.md lost its deprecated-alias table"
    return aliases


def test_documented_settings_are_exactly_none_apple_gpu() -> None:
    text = INSTALL_DOC.read_text(encoding="utf-8")
    table = text.split("\n## ", 1)[0]
    commands = set(re.findall(r"`(pip install [^`]+)`", table))
    assert commands == {
        "pip install abstractcore",
        'pip install "abstractcore[apple]"',
        'pip install "abstractcore[gpu]"',
    }
    assert {"none" if "[" not in c else c.split("[")[1].split("]")[0] for c in commands} == DOCUMENTED_SETTINGS


def test_extras_are_exactly_the_settings_contributor_tooling_and_documented_aliases() -> None:
    aliases = _documented_aliases()
    assert not (set(aliases) & (SETTING_EXTRAS | CONTRIBUTOR_EXTRAS))
    assert set(EXTRAS) == SETTING_EXTRAS | CONTRIBUTOR_EXTRAS | set(aliases)


def test_each_alias_installs_what_the_table_says() -> None:
    for alias, target in _documented_aliases().items():
        deps = EXTRAS[alias]
        if target == "light":
            assert deps == [], f"{alias}: light alias must add nothing (the light install covers it): {deps}"
        elif target in SETTING_EXTRAS:
            assert deps == [f"abstractcore[{target}]"], f"{alias}: must be exactly abstractcore[{target}]: {deps}"
        elif target == "unchanged":
            assert deps, f"{alias}: an 'unchanged' alias keeps its packages"
        else:  # pragma: no cover - table typo
            pytest.fail(f"{alias}: unknown target {target!r} in docs/installation.md")


def test_apple_and_gpu_do_not_reference_another_extra() -> None:
    # Each setting is light + its own engine list; it never chains to an alias.
    for setting in SETTING_EXTRAS:
        for requirement in EXTRAS[setting]:
            assert _name(requirement) != "abstractcore", f"{setting} references {requirement}"


def test_light_install_carries_every_remote_provider_dependency() -> None:
    base = {_name(r) for r in BASE}
    # OpenAI and Anthropic talk through their SDKs; the other remote providers use httpx.
    assert {"openai", "anthropic", "httpx"} <= base
    # The light install is complete for normal use: tools, media inputs, server, plugins.
    for package in ("requests", "beautifulsoup4", "tiktoken", "pillow", "pypdf", "unstructured",
                    "fastapi", "uvicorn", "abstractvoice", "abstractvision", "abstractmusic", "abstract3d"):
        assert package in base, f"light install lacks {package}"
    # No local engine in light.
    # A browser is never a base-install cost (browser_tools design ruling); apple/gpu carry it.
    for engine in ("torch", "transformers", "mlx", "mlx-lm", "mlx-vlm", "vllm", "sentence-transformers", "omnivoice",
                   "playwright"):
        assert engine not in base, f"local engine {engine} leaked into the light install"


def test_apple_and_gpu_carry_the_local_engines_for_their_platform() -> None:
    apple = {_name(r) for r in EXTRAS["apple"]}
    gpu = {_name(r) for r in EXTRAS["gpu"]}
    assert {"mlx", "mlx-lm", "mlx-vlm", "torch", "transformers", "llama-cpp-python", "sentence-transformers",
            "abstractvoice", "abstractvision", "abstractmusic", "omnivoice", "playwright"} <= apple
    assert {"vllm", "torch", "transformers", "llama-cpp-python", "sentence-transformers",
            "abstractvoice", "abstractvision", "abstractmusic", "omnivoice", "playwright"} <= gpu
    assert "vllm" not in apple
    assert not {"mlx", "mlx-lm", "mlx-vlm"} & gpu
    assert any(r.startswith("abstractvoice[all-apple]") for r in EXTRAS["apple"])
    assert any(r.startswith("abstractvision[all-apple]") for r in EXTRAS["apple"])
    assert any(r.startswith("abstractmusic[all-apple]") for r in EXTRAS["apple"])
    assert any(r.startswith("abstractvoice[all-gpu]") for r in EXTRAS["gpu"])
    assert any(r.startswith("abstractvision[all-gpu]") for r in EXTRAS["gpu"])
    assert any(r.startswith("abstractmusic[all-gpu]") for r in EXTRAS["gpu"])


def test_registry_asks_remote_providers_for_no_extra_and_local_ones_for_a_setting() -> None:
    from abstractcore.providers.registry import get_provider_registry

    registry = get_provider_registry()
    for name in REMOTE_PROVIDERS:
        assert registry.get_provider_info(name).installation_extras is None, name
    assert registry.get_provider_info("mlx").installation_extras == "apple"
    assert registry.get_provider_info("huggingface").installation_extras in SETTING_EXTRAS


# --- Only the three settings are shown -------------------------------------------------------

_EXTRA_MENTION = re.compile(r"abstractcore\[([^\]\s\"'`]*)\]", re.IGNORECASE)


def _user_facing_files() -> list[Path]:
    files = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "llms.txt", ROOT / "llms-full.txt"]
    files += sorted((ROOT / "docs").glob("*.md"))
    files += sorted((ROOT / "docs" / "apps").glob("*.md"))
    files += sorted((ROOT / "abstractcore").rglob("*.md"))
    files += sorted((ROOT / "abstractcore").rglob("*.py"))
    files += sorted(p for p in (ROOT / "examples").rglob("*") if p.suffix in {".md", ".py"})
    files += sorted((ROOT / "docker").rglob("Dockerfile"))
    return [p for p in files if p.is_file()]


def test_no_user_facing_text_shows_a_setting_other_than_the_three() -> None:
    offenders: list[str] = []
    for path in _user_facing_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            for match in _EXTRA_MENTION.finditer(line):
                # `abstractcore[{setting}]` templates are checked through the registry test.
                if match.group(1) not in SETTING_EXTRAS and not match.group(1).startswith("{"):
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {match.group(0)}")
    assert not offenders, "only abstractcore / abstractcore[apple] / abstractcore[gpu] may be shown:\n" + "\n".join(
        offenders
    )


def test_deprecated_aliases_are_named_only_in_the_installation_doc() -> None:
    # The alias table is the one place the old names appear (llms-full.txt aggregates it).
    aliases = set(_documented_aliases()) - {"tools", "media", "server", "voice", "audio", "vision", "music",
                                             "browser", "3d", "all", "openai", "anthropic", "ollama",
                                             "lmstudio", "openrouter", "portkey", "huggingface", "embeddings",
                                             "tokens", "compression", "tool", "mlx", "vllm", "scene3d",
                                             "remote", "openai-compatible"}
    # Names that are also ordinary words or provider ids are checked through the
    # `abstractcore[...]` pattern above; the distinctive ones must not appear anywhere else.
    allowed = {INSTALL_DOC, ROOT / "llms-full.txt"}
    offenders = []
    for path in _user_facing_files():
        if path in allowed or path.suffix != ".md":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for alias in aliases:
            if re.search(rf"`{re.escape(alias)}`", text):
                offenders.append(f"{path.relative_to(ROOT)}: `{alias}`")
    assert not offenders, "\n".join(offenders)


def test_gpu_resolves_with_wheels_only_on_windows() -> None:
    """Backlog 0988: vLLM ships Linux wheels only and llama-cpp-python is a source build on PyPI,
    so `gpu` marks both out on Windows (the installer adds llama.cpp's prebuilt wheel). Every other
    engine keeps no platform marker; the aliases keep mapping to `gpu`."""
    gpu = {_name(r): r for r in EXTRAS["gpu"]}
    assert gpu["vllm"].endswith("; sys_platform == 'linux'"), gpu["vllm"]
    assert gpu["llama-cpp-python"].endswith("; sys_platform != 'win32'"), gpu["llama-cpp-python"]
    # 0.3.32 is the first abstractvision whose all-gpu marks stable-diffusion.cpp out on Windows.
    assert gpu["abstractvision"].startswith("abstractvision[all-gpu]>=0.3.32"), gpu["abstractvision"]
    for name, requirement in gpu.items():
        if name not in {"vllm", "llama-cpp-python", "numpy"}:
            assert "sys_platform" not in requirement, requirement
    # apple keeps llama-cpp-python unmarked (macOS wheels / Metal build).
    assert not [r for r in EXTRAS["apple"] if _name(r) == "llama-cpp-python" and ";" in r]
    assert EXTRAS["all-gpu"] == ["abstractcore[gpu]"] and EXTRAS["vllm"] == ["abstractcore[gpu]"]
