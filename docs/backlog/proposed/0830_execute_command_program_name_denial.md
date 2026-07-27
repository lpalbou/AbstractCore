# Proposed: execute_command — workspace-scoped shell policy (program-name denial, honest authority)

## Metadata
- Created: 2026-07-25; REFINED 2026-07-25 with the operator's workspace-scoped allow/deny framing
  and a fable5 policy-design pass.
- Origin: tools quality audit; operator direction ("whitelist most execute commands except git
  mutable, rm -rf, and large-scale filesystem changes; but in the agent's OWN workspace I'd allow
  ANY tool usage — maybe that's why you see that style of checks?"). Aligns with the framework's
  recorded 2026-07-14 ruling (deny by program name, param-independent; wrapper peeling).

## Verdict on the operator's question: yes, the current screen IS that model — in SHAPE only
`_validate_command_security` (`common_tools.py:9345-9415`) is already default-allow + a blocking
denylist + a full bypass — i.e. "deny a catastrophic set, allow the rest." So the operator's
instinct matches the EXISTING design; this is a repair of the matcher, the dials, and WHERE the
relaxation authority lives — not a change of philosophy. Three precise deviations to fix:
- **Content** — the denied set is "text that RESEMBLES catastrophe", not the catastrophic set. It
  MISSES what the operator names (`rm -fr`/`rm -f` slip the flag-order regex `:9364`; git history
  rewrites appear NOWHERE in the screen) and it DENIES innocuous text (`\bshutdown\b`/`\breboot\b`/
  `\bhalt\b` `:9378-9380` block `grep reboot notes.txt` and `git commit -m "fix shutdown handler"`;
  substring keywords `destroy`/`wipe`/`kill -9` block `grep destroy docs.md`, `pkill -9 devserver`).
  Every false positive trains the model toward `allow_dangerous=True`, which disables ALL screening.
- **Authority (inverted)** — the operator's "in its own workspace, allow ANY" is an OPERATOR grant,
  but today the relaxation lever is `allow_dangerous`, a MODEL-FILLED tool argument (`:9064`, bypass
  `:9356`). And `require_confirmation` (`:9165-9169`) advertises an approval seam that does not
  exist — it logs "would normally ask" then executes anyway.

## Design: three layers, one classifier
| Layer | Owner | Role under this policy |
|---|---|---|
| In-tool shell screen | core (`execute_command`) | Catastrophic FLOOR — program-name denial of the irreversible set; last line even in bare hosts. run/refuse only. |
| Workspace wall | runtime (`resolve_user_path`) | Path containment for NAMED-target file tools (not the shell — cwd ≠ effect scope). |
| Approval lane | host (runtime/gateway/abstractcode) | The "allow most, ask on destructive" UX: `destructive_capable` → destroy band → permission-mode ceilings, lowered per-call by declared refiners. |

- **One classifier, hosted in core** (new `abstractcore/tools/command_screen.py`): program-basename
  extraction + the catastrophic set, consumed in-process by the tool and importable by enforcement
  lanes (import-never-copy; the abstractcode `bridge_policy.py` private copy retires onto it later —
  dependency arrow abstractcode→abstractcore is correct).

## Classification logic (layer 1)
- **Program-name denial** on each pipeline segment's resolved basename, ANY flag spelling (the
  2026-07-14 param-independence ruling); dotted families match by pre-dot stem (`mkfs.ext4`→`mkfs`).
- **Catastrophic default set (refuse):** deletion/erasure (`rm`,`rmdir`,`unlink`,`shred`,`srm`,
  `wipe`); device/filesystem (`dd`,`mkfs*`,`mke2fs`,`mkswap`,`fdisk`,`parted`,`sgdisk`); power AS
  PROGRAMS (`shutdown`,`reboot`,`halt`,`poweroff` — kills the keyword false positives); **git
  write/reset** not name-denied (git-read must stay usable) but gated by the two-stage read-only
  proof already declared core-side as refiner `git_read_only@v1` — unproven git = catastrophic tier.
- **Deliberately EXCLUDED from the tool default** (the tool can only refuse, so hard-refusing these
  contradicts "whitelist most"): `mv`,`chmod`,`chown`,`truncate`,`kill`/`pkill`. They stay ask-tier
  at the approval lane (execute_command already clamps to destroy band there).
- **Retained structural patterns** (argv/path facts, no English): device-write redirect
  `>/dev/(sd|nvme)`, destructive redirect onto critical system paths, fork bomb, kill-init exact.
- **Wrapper handling:** prefix wrappers (`env`,`sudo`,`doas`,`nohup`,`exec`,`setsid`,`stdbuf`) peel
  to the wrapped program; arg-wrappers (`timeout`,`nice`,`xargs`) trigger an all-token scan that
  fails SAFE.
- **Ambiguity → allow-with-segment-checks, not deny** (pipes are normal agent work; the tool's own
  example is `grep -R … | head`). Split on `&&`/`||`/`;`/`|`/newline and check each segment's leading
  program (`foo && rm -rf x` still surfaces `rm`). Optional `strict_ambiguous` config denies undecomposable commands for hardened hosts.
- **Honest limits (module docstring):** interpreter-mediated destruction (`python -c`, `find -delete`)
  and redirection truncation invoke non-denied programs and are NOT name-catchable — containment for
  that class is the approval lane + the workspace wall + OS sandboxing, never a claimed property of
  this screen. Delete the current "comprehensive security validation" claim.
- **Refusal shape:** structured, naming the matched program + the real rule + `abstractcore --config`
  pointer (replaces the shouting banner).

## Workspace scoping = a HOST-selected posture, never a cwd check in the tool
The tool must NOT relax on `working_directory` — cwd is not effect scope (a command run from a
scratch dir can still name any absolute path); doing so would be a containment lie. "Own workspace ⇒
allow ANY" is implemented where workspace identity actually lives — the host that created the run:
- The host (gateway `workspace_access_mode`, abstractcode permission mode) sets the screen mode for
  the process/session via an explicit host-only seam (`command_screen.set_mode(...)` or a policy
  object at tool-registration) — parameter-explicit, never ambient env, never a tool argument.
- Scratch-workspace run → host selects `off` (the operator's "allow ANY" posture).
- Real-filesystem run → the catastrophic floor applies + the approval lane's destroy-band ask above.
- Resolution, fail-closed: host posture > operator config > built-in `deny-catastrophic`; unknown → safest.

## require_confirmation / allow_dangerous — remove both (schema-only blast radius)
No live caller passes either (grep: only tests/docs/this backlog). `require_confirmation` cannot be
honest as a model-filled param (the approval seam is the HOST's `approval_required` wait); if schema
stability is preferred over removal, make it fail-closed (True + non-low → structured refusal naming
the host lane), never simulate-then-run. `allow_dangerous` relaxation moves to operator surfaces
(config + host posture). `_assess_command_risk` (substring-broken: `'rm '` matches `confirm `) loses
its only consumer — delete it. Replace the `rm temp_file.txt`+`require_confirmation` decorator example.

## Configuration surface (config-first, `abstractcore --config`)
New `execution` section in `AbstractCoreConfig`, read lazily like `analyze_media` reads the vision route:
```json
"execution": {
  "shell_screen_mode": "deny-catastrophic",   // "deny-catastrophic" | "off"
  "deny_programs_extra": [],                    // operator additions (by name)
  "allow_programs": [],                          // operator carve-outs from the built-in set
  "strict_ambiguous": false
}
```
Operator expresses "allowlist most, deny these" in the SAME name vocabulary the classifier enforces;
`mode:"off"` is the global sandboxed-host posture, logged once per process. Per-workspace
permissiveness rides the host posture seam (workspace identity is per-run, not global config).
- Approval-lane extension (opt-in follow-up): a declared refiner lowering unambiguous + zero-catastrophic
  + not-git-write shell calls to act band ("most shell auto-runs, catastrophic asks") — deny-safe,
  lower-only, default OFF (interpreters remain interpreters — a weaker proof than read-only-git).

## Note on the fetch_url tension
`fetch_url`'s screen was ruled "no config, no allowlist" for a single fixed exfil signature. Here the
operator's stated intent IS configurability, so the surface exists — kept minimal (one mode, two
lists, one flag) to stay in that spirit.

## Ownership
Core hosts the classifier + config schema + the in-tool floor. Runtime/gateway own the approval-lane
content (bands, refiners) and the host posture seam. Cross-package — coordinate before implementing.
