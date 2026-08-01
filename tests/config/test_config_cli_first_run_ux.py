"""THE FIRST THING A NEW USER DOES: `abstractcore config defaults`.

Setting a default is step one of every install, so the surface that reports the
problem must also state the fix. These pin the three ways that output used to
be a dead end:

  1. 24 rows of `not_configured` and no next step.
  2. Reasoning invisible, so the two entry points could disagree about what is
     configured without either one showing it.
  3. A CORRUPT store rendering identically to a fresh install -- the operator
     is told to configure what they already configured, and the next save
     overwrites the (recoverable) file.

Plus the accept-and-warn rule for provider names: unknown names stay writable
(media routes name plugin backends, endpoint profiles appear later), but a typo
on the TEXT route is reported at write time rather than at the first run.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from abstractcore.config.main import _handle_config_subcommand


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    rc = _handle_config_subcommand(list(argv))
    assert rc == 0, f"`abstractcore config {' '.join(argv)}` should succeed"
    return capsys.readouterr().out


def test_a_fresh_install_shows_the_seeded_recommended_defaults(capsys) -> None:
    # Operator ruling 2026-08-01: a truly fresh install (no config file has
    # ever existed) works out of the box on the recommended stack, and the
    # grid SHOWS those values rather than 24 rows of not_configured.
    d = Path(tempfile.mkdtemp())
    out = _run(capsys, "--config-file", str(d / "abstractcore.json"), "defaults")

    assert "lmstudio/qwen/qwen3.5-9b" in out
    assert "supertonic/supertonic-3" in out
    assert "mlx-gen/AbstractFramework/flux.2-klein-4b-8bit" in out
    assert "No text-generation default" not in out


def test_an_unconfigured_grid_names_the_command_that_fixes_it(capsys) -> None:
    # An EXISTING store with no routes (an operator cleared them, or a store
    # predating the recommendation seed) still gets the actionable guidance.
    d = Path(tempfile.mkdtemp())
    cfg = d / "abstractcore.json"
    cfg.write_text("{}")
    out = _run(capsys, "--config-file", str(cfg), "defaults")

    assert "output.text: -/- (not_configured)" in out
    assert "No text-generation default" in out, "silence about the missing default is the dead end"
    assert "abstractcore config set-default output.text --provider <provider> --model <model>" in out
    assert "abstractcore config clear-default output.text" in out


def test_the_grid_shows_reasoning_and_base_url_wherever_they_are_set(capsys) -> None:
    cfg = Path(tempfile.mkdtemp()) / "abstractcore.json"
    _run(
        capsys,
        "--config-file", str(cfg), "set-default", "output.text",
        "--provider", "lmstudio", "--model", "qwen3-0.6b",
        "--reasoning", "high", "--base-url", "http://localhost:1234/v1",
    )
    out = _run(capsys, "--config-file", str(cfg), "defaults")

    text_rows = [line for line in out.splitlines() if line.startswith("- output.text:")]
    assert text_rows, "the text route must be listed"
    assert "reasoning=high" in text_rows[0], "a configured reasoning effort must be visible"
    assert "base_url=http://localhost:1234/v1" in text_rows[0]
    assert "No text-generation default" not in out


def test_a_corrupt_store_is_reported_instead_of_looking_like_a_fresh_install(capsys) -> None:
    cfg = Path(tempfile.mkdtemp()) / "abstractcore.json"
    cfg.write_text('{"capability_defaults": {"routes": {"input.text": {"provider": "lms', encoding="utf-8")

    out = _run(capsys, "--config-file", str(cfg), "defaults")
    assert "could not be parsed" in out
    assert "DEFAULTS, not what you configured" in out
    assert "a save overwrites it" in out


def test_the_same_warning_fires_before_a_save_overwrites_a_corrupt_store(capsys) -> None:
    cfg = Path(tempfile.mkdtemp()) / "abstractcore.json"
    cfg.write_text("{ truncated", encoding="utf-8")

    out = _run(
        capsys, "--config-file", str(cfg), "set-default", "output.text",
        "--provider", "lmstudio", "--model", "qwen3-0.6b",
    )
    assert "could not be parsed" in out, "the operator must learn BEFORE the only copy is the backup"
    assert "✅ Set capability default for output.text" in out
    assert list(cfg.parent.glob("abstractcore.json.corrupt-*.bak")), "the unreadable file is preserved"


def test_an_unknown_text_provider_is_saved_and_warned_never_refused(capsys) -> None:
    cfg = Path(tempfile.mkdtemp()) / "abstractcore.json"
    out = _run(
        capsys, "--config-file", str(cfg), "set-default", "output.text",
        "--provider", "notaprovider", "--model", "nope-1",
    )

    assert "✅ Set capability default" in out, "unknown names stay writable"
    assert "is not a known AbstractCore text provider" in out
    stored = json.loads(cfg.read_text(encoding="utf-8"))
    assert stored["capability_defaults"]["routes"]["input.text"]["provider"] == "notaprovider"


@pytest.mark.parametrize(
    "route,provider,model",
    [
        ("output.text", "lmstudio", "qwen3-0.6b"),          # a real registry provider
        ("output.text", "endpoint:airelay", "gpt-5.4"),      # an endpoint profile reference
        ("output.voice", "supertonic", "supertonic-3"),      # a media plugin backend
        ("output.image.text_to_image", "mlx-gen", "flux"),   # a task route, plugin backend
    ],
)
def test_no_false_alarm_on_names_that_are_legitimate(capsys, route, provider, model) -> None:
    cfg = Path(tempfile.mkdtemp()) / "abstractcore.json"
    out = _run(
        capsys, "--config-file", str(cfg), "set-default", route,
        "--provider", provider, "--model", model,
    )
    assert "is not a known AbstractCore text provider" not in out


# --- a write failure must carry its reason ------------------------------------


def test_a_failed_write_names_the_route_and_the_cause(capsys) -> None:
    """Reported 2026-08-01: `set_capability_default` swallowed every exception
    into a bare `False`, so the CLI printed "❌ Failed to set capability default
    <route>" with nothing actionable and a one-off failure was undiagnosable."""
    cfg = Path(tempfile.mkdtemp()) / "abstractcore.json"

    rc = _handle_config_subcommand(
        ["--config-file", str(cfg), "set-default", "bogus.route", "--provider", "p", "--model", "m"]
    )
    out = capsys.readouterr().out

    assert rc == 1
    assert "Failed to set capability default 'bogus.route'" in out, "name the route"
    assert "Unknown capability route kind" in out, "and the reason it failed"


def test_the_typed_error_is_a_valueerror_so_every_entry_point_renders_it() -> None:
    """The AbstractCore server and the AbstractGateway seam both map ValueError
    to a 400; the type is chosen so neither has to learn a new one."""
    from abstractcore.config.manager import CapabilityDefaultWriteError, ConfigurationManager

    assert issubclass(CapabilityDefaultWriteError, ValueError)

    manager = ConfigurationManager(config_file=Path(tempfile.mkdtemp()) / "abstractcore.json", apply_env=False)
    with pytest.raises(CapabilityDefaultWriteError) as exc:
        manager.set_capability_default("bogus.route", provider="p", model="m")
    assert "bogus.route" in str(exc.value)
    assert isinstance(exc.value.__cause__, Exception), "the original failure stays attached"

    with pytest.raises(CapabilityDefaultWriteError):
        manager.clear_capability_default("bogus.route")

    # The happy path is unchanged: still a plain True.
    assert manager.update_capability_default("output.text", provider="lmstudio", model="m") is True


# --- options are a SET: the clear-all spelling is a contract -------------------


def test_option_semantics_keep_replace_and_clear(capsys) -> None:
    cfg = Path(tempfile.mkdtemp()) / "abstractcore.json"

    def stored_options():
        routes = json.loads(cfg.read_text(encoding="utf-8"))["capability_defaults"]["routes"]
        return routes["output.voice"].get("options")

    base = ["--config-file", str(cfg), "set-default", "output.voice"]
    _run(capsys, *base, "--provider", "supertonic", "--model", "supertonic-3", "--option", "voice=M1")
    assert stored_options() == {"voice": "M1"}

    # No --option at all: the set is kept.
    _run(capsys, *base, "--model", "supertonic-3")
    assert stored_options() == {"voice": "M1"}

    # --option given: the WHOLE set is replaced, never merged.
    _run(capsys, *base, "--option", "language=fr")
    assert stored_options() == {"language": "fr"}

    # --option "": the set is cleared. Both console-TUIs use this spelling.
    _run(capsys, *base, "--option", "")
    assert not stored_options()
    # ...and clearing options never touches the rest of the row.
    routes = json.loads(cfg.read_text(encoding="utf-8"))["capability_defaults"]["routes"]
    assert routes["output.voice"]["provider"] == "supertonic"
    assert routes["output.voice"]["model"] == "supertonic-3"


def test_the_help_states_the_option_set_semantics() -> None:
    """The contract is only a contract if the surface says it."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        _handle_config_subcommand(["set-default", "--help"])
    # argparse hard-wraps help text, so compare on collapsed whitespace.
    text = " ".join(buf.getvalue().split())

    assert "clears every option on the route" in text
    assert "whole-set replace" in text
