"""No install hint ever advises a bare package: only the three settings.

Operator ruling (2026-09-29, firm): users are only ever advised to install one of three
settings -- `abstractcore` (light), `abstractcore[apple]`, `abstractcore[gpu]`. No user-facing
message, CLI output, error hint, doc or generated doc may tell a user to `pip install <bare
package>` (mlx-lm, transformers torch, httpx, pydantic, pillow, PyYAML, llama-cpp-python,
diffusers, sentence-transformers, ...) or another extra as the way to get an AbstractCore
capability. A missing light dependency is a broken install (`pip install -U abstractcore`); a
local engine names this host's setting; a host with no setting is told plainly that the
capability is not available there (`abstractcore.utils.install_settings.not_available_here`).

This test scans the SOURCE STRINGS of the package, its user-facing docs and examples (tests may
scan text; product code never does) and parses every `pip install ...` command it finds. Each
install target must be one of the settings. Delete the fix of any hint (for example put back
`Install with: pip install httpx` in providers/openai_provider.py) and it goes RED.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

ALLOWED_TARGETS = {"abstractcore", "abstractcore[apple]", "abstractcore[gpu]"}
# Contributor editable installs (`pip install -e ".[apple,dev,test]"`) in contributor docs.
CONTRIBUTOR_EXTRAS = {"apple", "gpu", "dev", "test", "docs", "mlx-bench"}

# Deliberate, reviewed carve-outs: (relative path, target). Each is either not an AbstractCore
# capability or an explicit licence opt-in the operator decides on (listed in the hints-sweep
# report). Adding a row here is a decision, not a fix.
CARVE_OUTS = {
    # A shell-tool usage example for the agent's own project venv, not an AbstractCore capability.
    ("abstractcore/tools/shell_tools.py", "requests"),
    # PyMuPDF: AGPL / commercial licence opt-in, deliberately outside every setting.
    ("docs/installation.md", "pymupdf4llm"),
    ("docs/installation.md", "pymupdf-layout"),
    ("docs/media-handling-system.md", "pymupdf4llm"),
    ("docs/media-handling-system.md", "pymupdf-layout"),
    ("abstractcore/media/README.md", "pymupdf4llm"),
    ("abstractcore/media/README.md", "pymupdf-layout"),
    ("examples/media/README.md", "pymupdf4llm"),
    ("examples/media/README.md", "pymupdf-layout"),
    ("llms-full.txt", "pymupdf4llm"),
    ("llms-full.txt", "pymupdf-layout"),
    # Tooling, not a capability: upgrading pip itself, and a production process manager.
    ("docs/troubleshooting.md", "pip"),
    ("CONTRIBUTING.md", "pip"),
    ("docs/server.md", "gunicorn"),
    ("llms-full.txt", "pip"),
    ("llms-full.txt", "gunicorn"),
}

_PIP = re.compile(r"(?:uv\s+)?pip3?\s+install\b(?P<rest>[^\n`]*)")
# One requirement-like token: `name`, `name[extras]`, `name>=1.0`, `.[extras]`, a VCS/URL spec.
_REQUIREMENT = re.compile(r"^(?:\.|[A-Za-z0-9{][A-Za-z0-9._{}-]*)(?:\[[^\]\s]*\])?(?:[<>=!~]=?\S*)?$|^git\+\S+$")
# Prose after a command (`pip install -U abstractcore or ...`): the command has ended.
_STOPWORDS = {"or", "and", "then", "to", "in", "if", "when", "with", "for", "into", "from", "on", "is", "the",
              "a", "this", "once", "first", "again", "now"}
_OPTION_WITH_VALUE = {"--python", "-r", "--requirement", "-c", "--constraint", "--index-url", "--extra-index-url", "-i"}


def _files() -> list[Path]:
    files = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "llms.txt", ROOT / "llms-full.txt"]
    files += sorted((ROOT / "docs").glob("*.md"))
    files += sorted((ROOT / "docs" / "apps").glob("*.md"))
    files += sorted((ROOT / "abstractcore").rglob("*.py"))
    files += sorted((ROOT / "abstractcore").rglob("*.md"))
    files += sorted(p for p in (ROOT / "examples").rglob("*") if p.suffix in {".md", ".py"})
    return [p for p in files if p.is_file()]


def _targets(rest: str) -> list[str]:
    """The install targets of one `pip install` occurrence, up to where the command ends."""
    # `<this interpreter>`, `<setting>`: documentation placeholders, not packages.
    rest = re.sub(r"<[^>]*>", " <placeholder> ", rest)
    words = rest.replace('\\"', '"').replace("\\n", " \n ").split()
    out: list[str] = []
    skip = False
    for word in words:
        if skip:
            skip = False
            continue
        if word in {"&&", "||", ";", "|", "#", "\n"}:
            break
        token = word.strip("\"'`").rstrip(",.:;\"'`)")
        token = token.strip("\"'`")
        if not token:
            break
        if "<placeholder>" in token:
            continue
        if token in _OPTION_WITH_VALUE:
            skip = True
            continue
        if token.startswith("-"):
            continue
        if token.lower() in _STOPWORDS or not _REQUIREMENT.match(token):
            break
        out.append(token)
        if word.rstrip("\"'`").endswith((",", ".", ":", ";", ")")):
            break  # sentence punctuation ends the command
    return out


def _violations(rel: str, text: str) -> list[str]:
    found: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for match in _PIP.finditer(line):
            for target in _targets(match.group("rest")):
                lowered = target.lower()
                if lowered in ALLOWED_TARGETS:
                    continue
                if rel.endswith(".py") and ("{" in target or target in {"<pkg>", "<package>"}):
                    # f-string templates are built from the settings (checked by behaviour tests).
                    continue
                if lowered.startswith(".[") and set(lowered[2:].rstrip("]").split(",")) <= CONTRIBUTOR_EXTRAS:
                    continue
                if lowered == ".":
                    continue
                if (rel, lowered) in CARVE_OUTS:
                    continue
                found.append(f"{rel}:{lineno}: pip install ... {target}  <- {line.strip()[:160]}")
    return found


def test_the_parser_catches_a_bare_hint() -> None:
    # The check itself must go red on the patterns the sweep removed.
    for bad in (
        'raise ImportError("httpx package not installed. Install with: pip install httpx")',
        'print("💡 Install with: pip install transformers torch")',
        'hint = "pip install playwright && python -m playwright install --only-shell chromium"',
        "`pip install \"abstractvision[mlx-gen]\"`",
        "instruction='pip install \"abstractvoice[supertonic]\"',",
        'f"Install with: pip install \\"abstractcore[mlx]\\""',
    ):
        assert _violations("abstractcore/x.py", bad), bad
    for good in (
        "Repair with: pip install -U abstractcore",
        'Install with: pip install "abstractcore[apple]"',
        "`pip install -U \"abstractcore[gpu]\"`",
        'uv pip install --python /usr/bin/python3 "abstractcore[apple]"',
    ):
        assert not _violations("abstractcore/x.py", good), good


def test_no_install_hint_names_a_bare_package_or_another_extra() -> None:
    offenders: list[str] = []
    for path in _files():
        rel = path.relative_to(ROOT).as_posix()
        offenders += _violations(rel, path.read_text(encoding="utf-8", errors="replace"))
    assert not offenders, (
        "only `abstractcore`, `abstractcore[apple]` and `abstractcore[gpu]` may be advised "
        "(ruling 2026-09-29):\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    "os_id, arch, setting",
    [("darwin", "arm64", "apple"), ("linux", "x86_64", "gpu"), ("linux", "arm64", "gpu"),
     ("darwin", "x86_64", None), ("windows", "x86_64", None), ("windows", "arm64", None)],
)
def test_host_setting_and_hint(os_id, arch, setting) -> None:
    from abstractcore.utils import install_settings as s

    assert s.host_setting(os_id, arch) == setting
    hint = s.setting_hint("Local embeddings", setting)
    if setting:
        assert hint == f'Install with: pip install "abstractcore[{setting}]"'
        assert s.setting_hint("X", setting, upgrade=True) == f'Upgrade with: pip install -U "abstractcore[{setting}]"'
    else:
        assert "not available on this machine" in hint and "pip install" not in hint
