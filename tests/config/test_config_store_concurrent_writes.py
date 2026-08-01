"""ONE STORE, MANY WRITERS: a concurrent save must never publish a torn file.

`abstractcore config`, the AbstractCore console-TUI, the AbstractCore server's
PUT and the AbstractGateway seam all write the SAME `abstractcore.json`, and
nothing serializes them. Publishing was already atomic (`os.replace`), but the
temp file was a SHARED name (`abstractcore.json.tmp`), which made the atomicity
a lie: writer B's `open(..., "w")` truncated the shared temp while writer A was
mid-`json.dump` into the same inode; A then renamed that inode over the config
and B's still-open fd kept writing INTO THE LIVE CONFIG FILE.

Reproduced 2026-08-01 with 8 concurrent writers: 41 of 1200 concurrent reads
saw a truncated or interleaved store. A reader that hits one is not a cosmetic
failure -- the next process to start backs the file up and falls back to
DEFAULTS, so the operator's whole configuration appears to vanish.

Last-writer-wins is the contract. Torn state is not.
"""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path

from abstractcore.config.manager import ConfigurationManager


def test_the_temp_file_a_save_publishes_from_is_unique_per_writer() -> None:
    """The structural guarantee, checked without a race.

    A shared temp name is the whole bug; asserting only on a timing test would
    let the defect return whenever the race happened not to fire.
    """
    d = Path(tempfile.mkdtemp())
    cfg = d / "abstractcore.json"
    seen: list[str] = []

    manager = ConfigurationManager(config_file=cfg, apply_env=False)
    real_replace = Path.replace

    def spy(self: Path, target):  # noqa: ANN001 - Path.replace signature
        seen.append(self.name)
        return real_replace(self, target)

    Path.replace = spy  # type: ignore[method-assign]
    try:
        manager.update_capability_default("output.text", provider="lmstudio", model="a")
        manager.update_capability_default("output.text", provider="lmstudio", model="b")
    finally:
        Path.replace = real_replace  # type: ignore[method-assign]

    assert len(seen) == 2, "each save publishes via exactly one atomic replace"
    assert seen[0] != seen[1], "two saves must not share one temp file name"
    assert all(name.startswith("abstractcore.json.") and name.endswith(".tmp") for name in seen)
    assert not list(d.glob("*.tmp")), "a published save leaves no temp debris"


def test_concurrent_writers_never_publish_a_torn_store() -> None:
    d = Path(tempfile.mkdtemp())
    cfg = d / "abstractcore.json"

    seed = ConfigurationManager(config_file=cfg, apply_env=False)
    # A wide write window: the torn read is only observable while bytes are in
    # flight, so the store has to be big enough for a reader to land mid-write.
    for i in range(200):
        seed.set_provider_profile(f"prof{i}", base_url=f"http://h{i}.invalid/v1")
    seed.update_capability_default("output.text", provider="lmstudio", model="seed")

    torn: list[str] = []
    write_errors: list[str] = []
    stop = threading.Event()

    def writer(index: int) -> None:
        try:
            manager = ConfigurationManager(config_file=cfg, apply_env=False)
            for step in range(12):
                manager.update_capability_default(
                    "output.text", provider="lmstudio", model=f"w{index}-{step}"
                )
        except Exception as exc:  # noqa: BLE001 - the assertion is the report
            write_errors.append(f"{type(exc).__name__}: {exc}")

    def reader() -> None:
        while not stop.is_set():
            try:
                json.loads(cfg.read_text(encoding="utf-8"))
            except FileNotFoundError:
                torn.append("the live store disappeared mid-write")
            except json.JSONDecodeError as exc:
                torn.append(f"torn read: {exc}")
            except Exception:
                pass

    readers = [threading.Thread(target=reader, daemon=True) for _ in range(3)]
    writers = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in readers:
        t.start()
    for t in writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    for t in readers:
        t.join(timeout=5)

    assert not write_errors, f"concurrent saves must all succeed: {write_errors[:3]}"
    assert not torn, f"a reader must never observe a torn store: {torn[:3]}"

    final = json.loads(cfg.read_text(encoding="utf-8"))
    # Last writer wins, and it wins WHOLE: the unrelated sections a save did
    # not touch are still there.
    assert len(final["provider_profiles"]["profiles"]) == 200
    assert final["capability_defaults"]["routes"]["input.text"]["provider"] == "lmstudio"
    assert not list(d.glob("*.tmp")), "no temp debris survives the race"


# ---------------------------------------------------------------------------
# LAST-WRITER-WINS PER FIELD, NOT PER FILE
# ---------------------------------------------------------------------------
#
# The atomicity fix above made every published file parseable. It did NOT stop
# a save from being a silent revert: `_save_config` serialises the WHOLE
# in-memory config, so a manager that loaded the store at T0 and wrote one
# unrelated field at T2 republished its T0 snapshot over everything written in
# between. A long-lived process (a server, a console session, an entity loop)
# holds such a snapshot for hours.
#
# THE REPORTED LOSS (2026-08-01): the operator's
# `output.image: mflux/flux2-klein-9b` row was present in the Jul 30 backup and
# absent from the Aug 1 store, and the only other change in the whole file was
# a `timeouts` value -- exactly the shape a stale writer's timeout save leaves
# behind. Nothing warned. Nothing failed. The row simply was not there.


def _manager(cfg: Path) -> ConfigurationManager:
    return ConfigurationManager(config_file=cfg, apply_env=False)


def test_a_stale_writer_does_not_revert_another_writers_route() -> None:
    d = Path(tempfile.mkdtemp())
    cfg = d / "abstractcore.json"
    _manager(cfg).update_capability_default("output.text", provider="lmstudio", model="seed")

    stale = _manager(cfg)  # loads NOW; keeps its snapshot

    other = _manager(cfg)
    other.set_capability_default("output", "image", provider="mflux", model="flux2-klein-9b")
    other.set_capability_default(
        "output", "scene3d", task="image_to_scene3d", provider="exotic-engine", model="exotic/mesh-v9"
    )

    # One unrelated field, from the manager that never saw those rows.
    stale.set_default_timeout(7200.0)

    routes = json.loads(cfg.read_text(encoding="utf-8"))["capability_defaults"]["routes"]
    assert routes["output.image"] == {"provider": "mflux", "model": "flux2-klein-9b"}, (
        "a save of an UNRELATED field must not delete a route another writer added"
    )
    assert routes["output.scene3d.image_to_scene3d"]["provider"] == "exotic-engine", (
        "an exotic route no surface renders is preserved for exactly the same reason"
    )
    assert json.loads(cfg.read_text(encoding="utf-8"))["timeouts"]["default_timeout"] == 7200.0, (
        "the field the stale writer DID change still lands"
    )


def test_a_stale_writer_preserves_every_section_it_did_not_touch() -> None:
    """The rule is per FIELD, across the whole document -- not route-special."""
    d = Path(tempfile.mkdtemp())
    cfg = d / "abstractcore.json"
    _manager(cfg).set_api_key("openai", "sk-seed")

    stale = _manager(cfg)

    other = _manager(cfg)
    other.set_provider_profile("late-profile", base_url="http://late.invalid/v1")
    other.set_api_key("anthropic", "sk-late")
    other.set_console_log_level("DEBUG")

    stale.set_tool_timeout(1234.0)

    final = json.loads(cfg.read_text(encoding="utf-8"))
    assert "late-profile" in final["provider_profiles"]["profiles"]
    assert final["api_keys"]["anthropic"] == "sk-late"
    assert final["logging"]["console_level"] == "DEBUG"
    assert final["timeouts"]["tool_timeout"] == 1234.0


def test_an_explicit_delete_is_still_a_delete() -> None:
    """Preservation must not resurrect what an operator deliberately dropped.

    "Absent from mine but present in my baseline" is a DELETE; only "absent
    from both" is another writer's row. `clear-default` and `delete-provider`
    additionally adopt the on-disk value first, so they delete a row that
    appeared AFTER this manager loaded -- an explicit delete means the key
    goes, not "the key goes if I happened to know about it".
    """
    d = Path(tempfile.mkdtemp())
    cfg = d / "abstractcore.json"
    seed = _manager(cfg)
    seed.set_capability_default("output", "image", provider="mflux", model="flux2-klein-9b")
    seed.set_provider_profile("doomed", base_url="http://doomed.invalid/v1")

    seed.clear_capability_default("output", "image")
    seed.delete_provider_profile("doomed")
    final = json.loads(cfg.read_text(encoding="utf-8"))
    assert "output.image" not in final["capability_defaults"]["routes"]
    assert "doomed" not in final["provider_profiles"]["profiles"]

    # ... and a row that appeared after the manager loaded is still clearable.
    stale = _manager(cfg)
    _manager(cfg).set_capability_default("output", "image", provider="mflux", model="flux2-klein-9b")
    stale.clear_capability_default("output", "image")
    routes = json.loads(cfg.read_text(encoding="utf-8"))["capability_defaults"]["routes"]
    assert "output.image" not in routes, "an EXPLICIT clear beats preservation"


def test_the_happy_path_publishes_exactly_the_in_memory_document() -> None:
    """No concurrent writer means the merge is the identity function."""
    d = Path(tempfile.mkdtemp())
    cfg = d / "abstractcore.json"
    manager = _manager(cfg)
    manager.update_capability_default("output.text", provider="lmstudio", model="a")
    before = cfg.read_text(encoding="utf-8")
    manager._save_config()  # noqa: SLF001 - the unit under test
    assert cfg.read_text(encoding="utf-8") == before, (
        "a re-save with nothing changed and nobody else writing is byte-identical"
    )
