### Added

- **Commands run sandboxed on Windows under every workspace posture** (`tools/sandbox.py`,
  `tools/sandbox_windows.py`). "Deny everything, allow listed workspaces" and "Allow everything,
  refuse listed workspaces" with a read-only default no longer refuse every command:
  `host_sandbox_kind` is `windows-restricted-token` for both postures. Under both, writes are
  limited to the read & write rows, the run's private workspace and its `.tmp`: the command's
  restricted token runs at an integrity level between Low and Medium (so every write elsewhere,
  the registry included, is refused) and those folders are labelled for it, once. A folder one
  run may write stays read-only to every other run's command that can reach it. "Deny
  everything" refuses the user's data computed from the machine: the profile, and on each fixed
  drive the drive itself or each folder at its root that the user may change (not `Windows`,
  `Program Files*`, `ProgramData`), plus the user's recycle bin folder; the host's own Python
  stays readable. Limits (Git Bash, folders and registry keys Windows labels Low, what the user
  cannot change) are listed in `abstractcore/tools/README.md` (*Command sandbox*, Windows).
- **Drives the Windows sandbox cannot protect are reported** (`sandbox.windows_unprotected_volumes`).
  A drive without permissions (FAT, exFAT) or a network share: where writes are limited to the
  rows, a command's result says so (`sandbox.unprotected`, `sandbox.unprotected_note`, the
  `Sandbox:` line); a refused or read-only row there stops the command with one line naming it.

### Fixed

- **git runs in rows inside a refused folder on Windows** (`tools/sandbox_windows.py`). A refused
  folder with a row inside it could not be listed, and git stopped with "unable to get current
  working directory" wherever a folder on the way had a name that could be a short (8.3) name (a
  `src` row in a refused project). Such folders now list the names directly inside them (never
  their content); folders carved by 2.38.0 are rewritten in place at the next command.
- **A folder can change its role between runs** (`tools/sandbox_windows.py`). A folder made inside
  a refused folder after a first command, then listed read & write or read-only (or on the way to
  a deeper row), answered "Access is denied"; it now takes its new role, and a row that is no
  longer listed is refused again.
- **What a command makes in a read & write row inside a refused folder is fully its own**
  (`tools/sandbox_windows.py`), its permissions included: `icacls` on it works, and
  `tempfile.mkdtemp` (hence `python -m venv`, `ensurepip`, `pip install`) no longer loops until
  the command times out. The row folder's own permissions stay closed to the command.
- **A sandboxed command can no longer stop every later command** (`tools/sandbox_windows.py`). The
  launchers' lock was a named object of the session, which any command could hold (every later
  launch waited) or create with a refusing access list (every later launch failed). It is now a
  file in the launcher's state folder, which every policy refuses; a launch waits for it at most
  10 minutes and then says which process holds it.
- **The launcher's state is refused to every command** (`tools/sandbox.py`), read & write by default
  included, and its list of labelled folders checks itself: a missing or damaged list stops the
  commands that rely on it instead of silently dropping another run's guard.
- **Reparse points that are not links are refused** (`tools/sandbox_windows.py`). Inside a
  refused folder that has a row in it, a cloud file placeholder (OneDrive) or an app alias was
  skipped like a junction and stayed readable. Only links (junctions, symbolic links), which a
  lookup follows to a target with its own rules, are left alone now. Paths longer than 260
  characters are handled whether or not Windows' long paths are enabled.
- **A drive without permissions no longer stops every "Deny everything" command on Windows**
  (`tools/sandbox.py`): an exFAT or FAT drive was taken for one whose permissions the user may
  change, and its refusal could not be written.

### Removed

- `WINDOWS_ALLOW_LIST_UNEXPRESSIBLE` and `WINDOWS_READ_ONLY_UNEXPRESSIBLE` (`tools/sandbox.py`):
  nothing is refused for its posture on Windows any more.
