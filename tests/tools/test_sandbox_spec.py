"""Round 12 (R12.1): the sandbox spec, its backends' generated rules, the host policy and the
fail-closed paths. Platform-independent (no command runs here; see test_sandbox_macos.py and
test_sandbox_linux.py for the real OS tests)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from abstractcore.tools import sandbox as sb
from abstractcore.tools.sandbox import (
    KIND_NONE,
    KIND_UNSANDBOXED,
    REFUSED_NO_SPEC,
    SandboxError,
    SandboxRow,
    SandboxSpec,
    build_sandbox,
    configure_host,
    host_policy,
    sandbox_for_tool_call,
)


@pytest.fixture(autouse=True)
def _fresh_host():
    sb._reset_host_for_tests()
    yield
    sb._reset_host_for_tests()


@pytest.fixture
def tree(tmp_path, monkeypatch):
    home = tmp_path / "home"
    for d in ("home/.ssh", "priv", "refused/inner", "allowed/sub", "ro", "rw", "parent/child", "machine/LocalLow"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    real = Path(os.path.realpath(tmp_path))
    # What the Windows rules compute from the machine: here, from this tree.
    monkeypatch.setattr(sb, "_windows_user_data_roots", lambda: (str(real / "home"),))
    monkeypatch.setattr(sb, "_windows_low_places", lambda: (str(real / "machine/LocalLow"),))
    monkeypatch.setattr(sb, "_windows_state_folder", lambda: str(real / "machine/state"))
    monkeypatch.setattr(sb, "windows_unprotected_volumes", lambda: {})
    return real


def _spec(t: Path, **kw) -> SandboxSpec:
    base = dict(
        private_workspace=str(t / "priv"),
        posture="any_except_denied",
        default_mode="rw",
        allowed=(SandboxRow(str(t / "ro"), "ro"), SandboxRow(str(t / "rw"), "rw")),
        refused=(str(t / "refused"),),
        builtin_refused=(str(t / "home/.ssh"),),
    )
    base.update(kw)
    return SandboxSpec(**base)


# --- spec --------------------------------------------------------------------------------


def test_stamp_round_trip_carries_paths_only(tree):
    spec = _spec(tree, env={"SECRET_TOKEN": "x"}, unsandboxed_commands_allowed=True)
    stamp = spec.to_stamp()
    assert "env" not in stamp and "unsandboxed_commands_allowed" not in stamp
    assert "SECRET_TOKEN" not in json.dumps(stamp)
    back = SandboxSpec.from_stamp(json.dumps(stamp))
    assert back.allowed == spec.allowed and back.refused == spec.refused
    assert back.env is None and back.unsandboxed_commands_allowed is False


@pytest.mark.parametrize(
    "bad",
    [
        {"private_workspace": "relative/path", "posture": "allowed_only"},
        {"private_workspace": "/x", "posture": "whatever"},
        {"private_workspace": "/x", "posture": "allowed_only", "default_mode": "rwx"},
        {"private_workspace": "/x", "posture": "allowed_only", "allowed": [{"path": "/y", "mode": "deny"}]},
        {"private_workspace": "/x\n(allow default)", "posture": "allowed_only"},
        {"private_workspace": "", "posture": "allowed_only"},
        "not json",
        ["a list"],
    ],
)
def test_malformed_stamp_refuses_never_degrades(bad):
    box = sandbox_for_tool_call(bad)
    assert box.kind == KIND_NONE and box.refuses
    with pytest.raises(SandboxError):
        box.wrap("true")


def test_paths_are_realpathed(tree, tmp_path):
    link = tmp_path / "link_to_refused"
    try:
        link.symlink_to(tree / "refused")
    except OSError:
        pytest.skip("creating a symbolic link needs a privilege on this host")
    spec = sb.normalize_spec(_spec(tree, refused=(str(link),)))
    assert spec.refused == (str(tree / "refused"),)


# --- host policy -------------------------------------------------------------------------


def test_host_policy_set_once_and_reports_names_only():
    assert host_policy()["configured"] is False
    configure_host(env={"PATH": "/usr/bin", "LANG": "C"}, unsandboxed_commands_allowed=False)
    configure_host(env={"LANG": "C", "PATH": "/usr/bin"}, unsandboxed_commands_allowed=False)  # same = no-op
    with pytest.raises(RuntimeError):
        configure_host(env={"PATH": "/usr/bin"}, unsandboxed_commands_allowed=True)
    pol = host_policy()
    assert pol == {**pol, "configured": True, "unsandboxed_commands_allowed": False, "env_keys": ["LANG", "PATH"]}


def test_configured_host_refuses_a_call_without_a_stamp():
    assert build_sandbox(None).kind == KIND_UNSANDBOXED  # library use, no host
    configure_host(env={"PATH": "/usr/bin"})
    box = sandbox_for_tool_call(None)
    assert box.refuses and box.refusal() == REFUSED_NO_SPEC


def test_env_comes_from_the_host_never_os_environ(tree, monkeypatch):
    monkeypatch.setenv("AF_R12_LEAK_PROBE", "must-not-leak")
    configure_host(env={"PATH": "/usr/bin:/bin", "LANG": "C"})
    box = sandbox_for_tool_call(_spec(tree).to_stamp())
    env = box.env_for()
    assert "AF_R12_LEAK_PROBE" not in env
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["TMPDIR"] == str(tree / "priv" / ".tmp") and Path(env["TMPDIR"]).is_dir()


def test_stamp_cannot_switch_unsandboxed_on(tree, monkeypatch):
    monkeypatch.setattr(sb, "host_sandbox_kind", lambda posture="allowed_only": KIND_NONE)
    stamp = {**_spec(tree).to_stamp(), "unsandboxed_commands_allowed": True}
    assert sandbox_for_tool_call(stamp).kind == KIND_NONE


# --- fail closed -------------------------------------------------------------------------


def test_no_sandbox_binary_fails_closed(tree, monkeypatch):
    monkeypatch.setattr(sb, "host_sandbox_kind", lambda posture="allowed_only": KIND_NONE)
    box = build_sandbox(_spec(tree))
    assert box.refuses
    assert box.rendered_line() == "Sandbox: none — commands refused on this host"
    assert "serve --unsandboxed-commands" in box.refusal()
    res = sb.refusal_result(box, command="ls")
    assert res["success"] is False and res["sandbox"]["kind"] == "none"


def test_host_flag_reenables_unsandboxed(tree, monkeypatch):
    monkeypatch.setattr(sb, "host_sandbox_kind", lambda posture="allowed_only": KIND_NONE)
    configure_host(env={"PATH": "/usr/bin"}, unsandboxed_commands_allowed=True)
    box = sandbox_for_tool_call(_spec(tree).to_stamp())
    assert box.kind == KIND_UNSANDBOXED and not box.refuses
    argv, env = box.wrap("echo hi")
    assert argv == ["/bin/sh", "-c", "echo hi"] and env["PATH"] == "/usr/bin"


def test_macos_missing_sandbox_exec_is_none(monkeypatch):
    monkeypatch.setattr(sb.sys, "platform", "darwin")
    monkeypatch.setattr(sb, "SANDBOX_EXEC", "/nonexistent/sandbox-exec")
    assert sb.host_sandbox_kind() == KIND_NONE


def test_linux_without_bwrap_or_landlock_is_none(monkeypatch):
    monkeypatch.setattr(sb.sys, "platform", "linux")
    monkeypatch.setattr(sb, "_bwrap_path", lambda: None)
    monkeypatch.setattr(sb, "_landlock_abi", lambda: 0)
    assert sb.host_sandbox_kind("allowed_only") == KIND_NONE
    monkeypatch.setattr(sb, "_landlock_abi", lambda: 3)
    assert sb.host_sandbox_kind("allowed_only") == sb.KIND_LANDLOCK
    assert sb.host_sandbox_kind("any_except_denied") == KIND_NONE  # Landlock cannot deny a subpath


def test_windows_expresses_every_posture(monkeypatch):
    monkeypatch.setattr(sb.sys, "platform", "win32")
    assert sb.host_sandbox_kind("any_except_denied") == sb.KIND_WINDOWS
    assert sb.host_sandbox_kind("allowed_only") == sb.KIND_WINDOWS
    assert not hasattr(sb, "WINDOWS_ALLOW_LIST_UNEXPRESSIBLE") and not hasattr(sb, "WINDOWS_READ_ONLY_UNEXPRESSIBLE")


def _policy(argv):
    import base64
    import zlib

    return json.loads(zlib.decompress(base64.urlsafe_b64decode(argv[3])))


@pytest.mark.parametrize("posture,default_mode", [("any_except_denied", "ro"), ("allowed_only", "rw"), ("allowed_only", "ro")])
def test_windows_limits_writes_to_the_rows_with_labels(tree, monkeypatch, posture, default_mode):
    monkeypatch.setattr(sb.sys, "platform", "win32")
    spec = _spec(tree, posture=posture, default_mode=default_mode)
    box = build_sandbox(spec)
    assert box.kind == sb.KIND_WINDOWS and not box.refuses
    policy = _policy(box.wrap("echo hi")[0])
    assert policy["rules"] == sb.windows_rules(spec)
    # Every read & write rule is labelled (the run's own folder included), nothing else.
    assert policy["label"] == [r[1] for r in policy["rules"] if r[0] == "rw"]
    assert str(tree / "priv") in policy["label"] and str(tree / "rw") in policy["label"] and str(tree / "ro") not in policy["label"]
    assert policy["state"] == str(tree / "machine/state")


def test_windows_read_and_write_default_carries_no_label(tree, monkeypatch):
    monkeypatch.setattr(sb.sys, "platform", "win32")
    policy = _policy(build_sandbox(_spec(tree)).wrap("echo hi")[0])
    assert set(policy) == {"rules", "state"} and policy["state"] == str(tree / "machine/state")


def test_windows_wraps_commands_with_the_launcher(tree, monkeypatch):
    monkeypatch.setattr(sb.sys, "platform", "win32")
    box = build_sandbox(_spec(tree))
    assert box.kind == sb.KIND_WINDOWS and box.rendered_line() == "Sandbox: Windows restricted token"
    argv, env = box.wrap("echo hi")
    assert argv[1:3] == ["-I", sb.WINDOWS_LAUNCHER] and os.path.isfile(sb.WINDOWS_LAUNCHER)
    assert argv[3] == sb.windows_policy(sb.windows_rules(_spec(tree)))
    import base64
    import zlib

    assert json.loads(zlib.decompress(base64.urlsafe_b64decode(argv[3]))) == {"rules": sb.windows_rules(_spec(tree)), "state": str(tree / "machine/state")}
    assert argv[4:] == ["--shell", "echo hi"]  # one string: the launcher runs it as `shell=True` does
    assert box.wrap(["git", "status"])[0][4:] == ["--", "git", "status"]
    assert env["TEMP"] == env["TMP"] == str(tree / "priv" / ".tmp")
    with pytest.raises(SandboxError):
        box.wrap([])


# --- the Windows rules (nearest row wins; only rows that change their surroundings) -------


def _rules(spec: SandboxSpec):
    """The rules without the host's own Python (a system root, wherever this test runs) and
    the launcher's state folder (refused in every shape: see the test below)."""
    system = {os.path.normcase(p) for p in sb._system_readable_roots()}
    state = ["deny", sb._windows_state_folder()]
    return [r for r in sb.windows_rules(spec) if os.path.normcase(r[1]) not in system and os.path.normcase(r[-1]) not in system and r != state]


@pytest.mark.parametrize("posture,default_mode", [("any_except_denied", "rw"), ("any_except_denied", "ro"), ("allowed_only", "rw")])
def test_windows_rules_refuse_the_launchers_state_folder_in_every_shape(tree, posture, default_mode):
    # It holds the lock one launcher at a time takes and the list of labelled folders: no
    # command of any shape may read, change or hold them.
    rules = sb.windows_rules(_spec(tree, posture=posture, default_mode=default_mode, allowed=(SandboxRow(str(tree / "machine"), "rw"),)))
    assert ["deny", str(tree / "machine/state")] in rules  # even inside a read & write row


def test_windows_rules_keep_only_rows_that_change_their_surroundings(tree):
    t = tree
    spec = _spec(t, allowed=(SandboxRow(str(t / "ro"), "ro"), SandboxRow(str(t / "rw"), "rw")), builtin_refused=(str(t / "home/.ssh"),))
    # A read & write row in a read & write world and the run's own folder change nothing.
    assert _rules(spec) == [["ro", str(t / "ro")], ["deny", str(t / "refused")], ["deny", str(t / "home/.ssh")]]


def test_windows_rules_carve_a_refusal_around_the_rows_inside_it(tree):
    t = tree
    spec = _spec(
        t,
        private_workspace=str(t / "parent/a/b/priv"),
        allowed=(SandboxRow(str(t / "parent/child"), "rw"), SandboxRow(str(t / "refused/inner"), "ro")),
        refused=(str(t / "parent"), str(t / "parent/child/deny"), str(t / "refused")),
        builtin_refused=(),
    )
    assert _rules(spec) == [
        ["carve", str(t / "parent")],
        ["carve", str(t / "refused")],
        ["rw", str(t / "parent/child")],
        ["ro", str(t / "refused/inner")],
        ["read", str(t / "refused/inner")],  # read-only inside a refusal: its reading is re-opened too
        ["rw", str(t / "parent/a/b/priv")],
        ["deny", str(t / "parent/child/deny")],
        # Folders on the way to a row deeper than one level: traversal only.
        ["pass", str(t / "parent/a"), str(t / "parent")],
        ["pass", str(t / "parent/a/b"), str(t / "parent")],
    ]


def test_windows_rules_ties_and_redundant_rows(tree):
    t = tree
    p = str(t / "rw")
    # One path refused and allowed: the refusal. Read-only and read & write: read-only.
    assert ["deny", p] in _rules(_spec(t, allowed=(SandboxRow(p, "rw"),), refused=(p,)))
    assert ["ro", p] in _rules(_spec(t, allowed=(SandboxRow(p, "rw"), SandboxRow(p, "ro")), refused=()))
    # A refusal inside a refusal, a read-only row inside a read-only row: nothing to add.
    spec = _spec(t, allowed=(SandboxRow(str(t / "allowed"), "ro"), SandboxRow(str(t / "allowed/sub"), "ro")),
                 refused=(str(t / "refused"), str(t / "refused/inner")), builtin_refused=())  # fmt: skip
    assert _rules(spec) == [["ro", str(t / "allowed")], ["deny", str(t / "refused")]]


def test_windows_rules_builtin_refusals_are_absolute_except_the_host_exceptions(tree):
    t = tree
    data, run_dir = t / "home/.data", t / "home/.data/workspaces/session-1"
    run_dir.mkdir(parents=True)
    spec = _spec(t, private_workspace=str(run_dir), allowed=(SandboxRow(str(data / "other"), "rw"),), refused=(),
                 builtin_refused=(str(data),), builtin_allowed=(str(run_dir),))  # fmt: skip
    assert _rules(spec) == [
        ["carve", str(data)],
        ["rw", str(run_dir)],  # the host's exception; the row inside the built-in refusal is dropped
        ["pass", str(data / "workspaces"), str(data)],
    ]


def test_windows_refuses_more_rows_than_a_command_line_carries(tree, monkeypatch):
    monkeypatch.setattr(sb.sys, "platform", "win32")
    import secrets

    many = tuple(str(tree / "refused" / secrets.token_hex(40) / secrets.token_hex(40)) for _ in range(400))
    box = build_sandbox(_spec(tree, refused=many))
    assert box.refuses and box.reason == sb.WINDOWS_TOO_MANY_ROWS
    assert build_sandbox(_spec(tree, refused=many[:40])).kind == sb.KIND_WINDOWS


@pytest.mark.skipif(os.name != "nt", reason="Windows path spellings")
@pytest.mark.parametrize("bad", [r"C:\folder\file.txt:stream", r"C:\folder::$DATA", r"\\?\GLOBALROOT\Device\Nothing\x", r"\\.\pipe\x"])
def test_windows_paths_that_are_not_folders_or_files_are_refused(tree, bad):
    with pytest.raises(SandboxError):
        sb.normalize_spec(_spec(tree, refused=(bad,)))


def test_windows_rules_keep_the_hosts_python_readable_inside_a_refusal(tree, monkeypatch):
    python_home = tree / "allowed/sub"
    monkeypatch.setattr(sb, "_system_readable_roots", lambda: (str(python_home),))
    rules = sb.windows_rules(_spec(tree, allowed=(), refused=(str(tree / "allowed"),), builtin_refused=()))
    assert rules == [["carve", str(tree / "allowed")], ["read", str(python_home)], ["deny", str(tree / "machine/state")]]
    monkeypatch.setattr(sb, "_system_readable_roots", lambda: (str(tree / "rw"),))
    assert _rules(_spec(tree, allowed=(), refused=(), builtin_refused=())) == []  # readable anyway: no rule


def test_windows_rules_under_a_read_only_default(tree):
    t = tree
    spec = _spec(t, default_mode="ro", allowed=(SandboxRow(str(t / "ro"), "ro"), SandboxRow(str(t / "rw"), "rw"),
                                                 SandboxRow(str(t / "rw/frozen"), "ro"), SandboxRow(str(t / "rw/frozen/thaw"), "rw")))  # fmt: skip
    assert _rules(spec) == [
        ["rw", str(t / "rw")],  # every read & write row is a rule (it is labelled), but no read-only
        ["rw", str(t / "priv")],  # row: nothing is writable around it anyway; the run's own folder too
        ["deny", str(t / "refused")],
        ["ro", str(t / "rw/frozen")],
        ["deny", str(t / "home/.ssh")],
        ["rw", str(t / "rw/frozen/thaw")],
        ["guard", str(t / "machine/LocalLow")],  # what Windows labels Low: writes refused
    ]


def test_windows_rules_carve_a_refusal_under_a_read_only_default(tree):
    t = tree
    spec = _spec(t, default_mode="ro", private_workspace=str(t / "parent/a/b/priv"), allowed=(), refused=(str(t / "parent"),), builtin_refused=())
    assert _rules(spec) == [
        ["carve", str(t / "parent")],
        ["rw", str(t / "parent/a/b/priv")],
        ["guard", str(t / "machine/LocalLow")],
        ["pass", str(t / "parent/a"), str(t / "parent")],
        ["pass", str(t / "parent/a/b"), str(t / "parent")],
    ]


def test_windows_rules_allow_list_refuses_the_users_data(tree, monkeypatch):
    t = tree
    data = t / "home/.data"
    run_dir = data / "workspaces/session-1"
    run_dir.mkdir(parents=True)
    spec = _spec(t, posture="allowed_only", private_workspace=str(run_dir), allowed=(SandboxRow(str(t / "home/proj"), "rw"), SandboxRow(str(t / "ro"), "ro")),
                 refused=(), builtin_refused=(str(data),), builtin_allowed=(str(run_dir),))  # fmt: skip
    assert _rules(spec) == [
        ["carve", str(t / "home")],  # the user's data, with rows inside
        ["rw", str(t / "home/proj")],
        ["rw", str(run_dir)],
        ["pass", str(data), str(t / "home")],  # the built-in refusal lies in the refused profile: a folder on the way
        ["pass", str(data / "workspaces"), str(t / "home")],
    ]  # (the read-only row lies outside the user's data: readable and unwritable already; no Low guard: the profile is refused)
    # A row at or above a root of the user's data opens it.
    opened = _spec(t, posture="allowed_only", allowed=(SandboxRow(str(t / "home"), "ro"),), refused=(), builtin_refused=())
    assert _rules(opened) == [["rw", str(t / "priv")]]


def test_windows_rules_keep_the_hosts_python_readable_under_the_allow_list(tree, monkeypatch):
    python_home = tree / "home/AppData/uv/python"
    python_home.mkdir(parents=True)
    monkeypatch.setattr(sb, "_system_readable_roots", lambda: (str(python_home),))
    rules = sb.windows_rules(_spec(tree, posture="allowed_only", allowed=(), refused=(), builtin_refused=()))
    assert ["read", str(python_home)] in rules and ["carve", str(tree / "home")] in rules
    assert ["pass", str(tree / "home/AppData"), str(tree / "home")] in rules


def test_under_follows_the_platforms_path_rules():
    assert sb._under(os.path.join(os.sep, "a", "b"), os.path.join(os.sep, "a")) and sb._under(os.sep + "a", os.sep)
    assert not sb._under(os.path.join(os.sep, "ab"), os.path.join(os.sep, "a"))
    if os.name == "nt":
        assert sb._under("C:\\Users\\Me\\x.txt", "c:\\users\\me") and sb._under("C:\\x", "C:\\") and not sb._under("D:\\x", "C:\\")


# --- rule order (the R12 NESTING RULE in the profile) ------------------------------------


def _rule_lines(profile: str, needle: str):
    return [i for i, line in enumerate(profile.splitlines()) if needle in line]


posix_paths = pytest.mark.skipif(os.name == "nt", reason="compares the macOS profile with native paths: POSIX paths only")


@posix_paths
def test_profile_orders_rows_least_to_most_specific(tree):
    spec = _spec(
        tree,
        allowed=(SandboxRow(str(tree / "parent/child"), "rw"), SandboxRow(str(tree / "allowed"), "ro")),
        refused=(str(tree / "parent"), str(tree / "allowed/sub")),
    )
    prof = sb.macos_profile(spec)
    parent_deny = _rule_lines(prof, f'(deny file-read* file-write* (subpath "{tree / "parent"}"))')[0]
    child_allow = _rule_lines(prof, f'(allow file-read* file-write* (subpath "{tree / "parent/child"}"))')[0]
    allowed_ro = _rule_lines(prof, f'(allow file-read* (subpath "{tree / "allowed"}"))')[0]
    sub_deny = _rule_lines(prof, f'(deny file-read* file-write* (subpath "{tree / "allowed/sub"}"))')[0]
    assert parent_deny < child_allow  # an allowed child inside a refused parent stays reachable
    assert allowed_ro < sub_deny  # a refused row inside an allowed one refuses that subtree


@posix_paths
def test_profile_tie_refusal_wins(tree):
    p = str(tree / "rw")
    prof = sb.macos_profile(_spec(tree, allowed=(SandboxRow(p, "rw"),), refused=(p,)))
    assert _rule_lines(prof, f'(allow file-read* file-write* (subpath "{p}"))')[0] < _rule_lines(
        prof, f'(deny file-read* file-write* (subpath "{p}"))'
    )[0]


@posix_paths
def test_profile_builtin_refusals_come_last_except_host_allow(tree):
    data = tree / "home/.data"
    run_dir = data / "workspaces/session-1"
    run_dir.mkdir(parents=True)
    spec = _spec(
        tree,
        private_workspace=str(run_dir),
        allowed=(SandboxRow(str(data / "other"), "rw"),),  # a row cannot re-open a built-in
        builtin_refused=(str(data),),
        builtin_allowed=(str(run_dir),),
    )
    lines = sb.macos_profile(spec).splitlines()
    deny_data = max(i for i, l in enumerate(lines) if l == f'(deny file-read* file-write* (subpath "{data}"))')
    other = max(i for i, l in enumerate(lines) if f'(subpath "{data / "other"}")' in l)
    run = max(i for i, l in enumerate(lines) if l == f'(allow file-read* file-write* (subpath "{run_dir}"))')
    assert other < deny_data < run


def test_profile_posture_floor(tree):
    prof_allow = sb.macos_profile(_spec(tree, posture="allowed_only"))
    assert '(deny file-read* file-write* (subpath "/Users"))' in prof_allow
    assert '(deny file-write* (subpath "/"))' in prof_allow
    prof_any_rw = sb.macos_profile(_spec(tree))
    assert '(subpath "/Users"))' not in prof_any_rw.replace("(allow file-read-metadata", "")
    assert '(deny file-write* (subpath "/"))' not in prof_any_rw
    prof_any_ro = sb.macos_profile(_spec(tree, default_mode="ro"))
    assert '(deny file-write* (subpath "/"))' in prof_any_ro


def test_profile_never_parses_the_command(tree):
    box = sb.Sandbox(kind=sb.KIND_MACOS, spec=_spec(tree), _prefix=("/usr/bin/sandbox-exec", "-p", "PROFILE"))
    argv, _ = box.wrap("cat $(echo x) && cd .. ; ls")
    assert argv == ["/usr/bin/sandbox-exec", "-p", "PROFILE", "/bin/sh", "-c", "cat $(echo x) && cd .. ; ls"]


# --- bwrap / Landlock rule generation ----------------------------------------------------


def test_bwrap_argv_order_and_masks(tree):
    spec = _spec(
        tree,
        allowed=(SandboxRow(str(tree / "parent/child"), "rw"), SandboxRow(str(tree / "ro"), "ro")),
        refused=(str(tree / "parent"),),
    )
    argv = sb.bwrap_argv(spec)
    assert argv[:5] == ["bwrap", "--die-with-parent", "--bind", "/", "/"]
    i_mask = argv.index("--tmpfs", argv.index(str(tree / "parent")) - 1)
    assert argv[i_mask + 1] == str(tree / "parent")
    i_child = [i for i, a in enumerate(argv) if a == str(tree / "parent/child")][0]
    assert i_mask < i_child and argv[i_child - 1] == "--bind"
    # The mask is made read-only only AFTER the child is bound inside it (bwrap creates the
    # child's mount point in the tmpfs; a read-only tmpfs refused it on Linux).
    i_remount = [i for i, a in enumerate(argv) if a == "--remount-ro" and argv[i + 1] == str(tree / "parent")]
    assert len(i_remount) == 1 and i_remount[0] > i_child, argv
    assert argv[i_mask + 2] != "--remount-ro"
    assert argv[argv.index(str(tree / "ro")) - 1] == "--ro-bind"
    ssh = str(tree / "home/.ssh")
    assert argv[argv.index(ssh) - 1] == "--tmpfs"
    ro_argv = sb.bwrap_argv(_spec(tree, default_mode="ro"))
    assert ro_argv[2:5] == ["--ro-bind", "/", "/"]
    allow_argv = sb.bwrap_argv(_spec(tree, posture="allowed_only"))
    assert ["--bind", "/", "/"] != allow_argv[2:5] and "/home" not in allow_argv


def test_landlock_refuses_what_it_cannot_express(tree, monkeypatch):
    monkeypatch.setattr(sb, "host_sandbox_kind", lambda posture="allowed_only": sb.KIND_LANDLOCK)
    nested = _spec(tree, posture="allowed_only", allowed=(SandboxRow(str(tree / "allowed"), "rw"),), refused=(str(tree / "allowed/sub"),))
    box = build_sandbox(nested)
    assert box.refuses and "Landlock cannot refuse" in box.reason
    flat = _spec(tree, posture="allowed_only", refused=(), builtin_refused=())
    ok = build_sandbox(flat)
    assert ok.kind == sb.KIND_LANDLOCK
    argv, _ = ok.wrap("ls")
    assert argv[1:3] == ["-m", "abstractcore.tools.sandbox"] and argv[-3:] == ["/bin/sh", "-c", "ls"]
    rules = dict(json.loads(argv[3]))
    assert rules[str(tree / "rw")] == "rw" and rules[str(tree / "ro")] == "ro"
    assert str(tree / "refused") not in rules


def test_describe_counts_builtin_never_lists_them(tree):
    d = build_sandbox(_spec(tree)).describe()
    assert d["builtin_refused"] == 1
    assert str(tree / "home/.ssh") not in json.dumps(d)


def test_bwrap_masks_are_remounted_read_only_after_every_mount(tree):
    """R14.1 (backlog 1002 item 1): every mask is remounted read-only once, after all binds,
    unless a later rule of the very same path re-bound it."""
    (tree / "parent/child/sub/deeper").mkdir(parents=True, exist_ok=True)
    spec = _spec(
        tree,
        allowed=(SandboxRow(str(tree / "parent/child"), "rw"), SandboxRow(str(tree / "parent/child/sub/deeper"), "ro")),
        refused=(str(tree / "parent"), str(tree / "parent/child/sub")),
    )
    argv = sb.bwrap_argv(spec)
    last_bind = max(i for i, a in enumerate(argv) if a in ("--bind", "--ro-bind") and argv[i + 1] != "/")
    remounts = [argv[i + 1] for i, a in enumerate(argv) if a == "--remount-ro"]
    assert remounts and all(i > last_bind for i, a in enumerate(argv) if a == "--remount-ro"), argv
    for masked in (str(tree / "parent"), str(tree / "parent/child/sub"), str(tree / "home/.ssh")):
        assert remounts.count(masked) == 1, (masked, argv)
    # Order of masks and binds is still least specific first.
    order = [argv[i + 1] for i, a in enumerate(argv) if a in ("--tmpfs", "--bind", "--ro-bind") and argv[i + 1].startswith(str(tree / "parent"))]
    assert order == [str(tree / "parent"), str(tree / "parent/child"), str(tree / "parent/child/sub"), str(tree / "parent/child/sub/deeper")], order


def test_landlock_grants_an_allowed_child_of_a_refused_parent(tree, monkeypatch):
    """Landlock only allows: the child is granted, the refused parent never is (and the spec is expressible)."""
    monkeypatch.setattr(sb, "host_sandbox_kind", lambda posture="allowed_only": sb.KIND_LANDLOCK)
    spec = _spec(tree, posture="allowed_only", allowed=(SandboxRow(str(tree / "parent/child"), "rw"),), refused=(str(tree / "parent"),))
    box = build_sandbox(spec)
    assert box.kind == sb.KIND_LANDLOCK and not box.refuses, box.describe()
    rules = dict(sb.landlock_rules(spec))
    assert rules[str(tree / "parent/child")] == "rw"
    assert str(tree / "parent") not in rules
