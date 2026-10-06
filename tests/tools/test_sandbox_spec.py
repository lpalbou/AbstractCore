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
    for d in ("home/.ssh", "priv", "refused/inner", "allowed/sub", "ro", "rw", "parent/child"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    return Path(os.path.realpath(tmp_path))


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
    link.symlink_to(tree / "refused")
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


def test_windows_is_none(monkeypatch):
    monkeypatch.setattr(sb.sys, "platform", "win32")
    assert sb.host_sandbox_kind() == KIND_NONE
    assert "unsupported platform" in sb._no_sandbox_reason("allowed_only")


# --- rule order (the R12 NESTING RULE in the profile) ------------------------------------


def _rule_lines(profile: str, needle: str):
    return [i for i, line in enumerate(profile.splitlines()) if needle in line]


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


def test_profile_tie_refusal_wins(tree):
    p = str(tree / "rw")
    prof = sb.macos_profile(_spec(tree, allowed=(SandboxRow(p, "rw"),), refused=(p,)))
    assert _rule_lines(prof, f'(allow file-read* file-write* (subpath "{p}"))')[0] < _rule_lines(
        prof, f'(deny file-read* file-write* (subpath "{p}"))'
    )[0]


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
