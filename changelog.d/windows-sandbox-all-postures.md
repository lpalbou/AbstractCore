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

### Removed

- `WINDOWS_ALLOW_LIST_UNEXPRESSIBLE` and `WINDOWS_READ_ONLY_UNEXPRESSIBLE` (`tools/sandbox.py`):
  nothing is refused for its posture on Windows any more.
