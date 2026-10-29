# 0869 — The console's theme copy re-syncs to ui-kit 0.9.6, with a CI-visible pin

**Status**: planned · **Priority**: P2 · **Created**: 2026-10-29
**Cross-package**: AbstractUIC 0.8.5 (ui-kit 0.9.6), on npm 2026-10-29.
> Root link: AbstractFramework release-closing wave 2026-10-29 (the lead's order, via the merge desk).

## Why
`abstractcore/console/themes.py` is generated from the kit's `theme.css` / `theme.ts` (`python -m
abstractcore.console.theme_sync`). Its drift pin, `tests/test_console_web.py::test_theme_module_matches_the_kit_source_exactly`,
fails against the AbstractUIC checkout now at 0.8.5 (reproduced 2026-10-29 with
`ABSTRACTUIC_SRC=<abstractuic>/ui-kit/src`), so core's copy is behind the kit. The pin is also invisible in CI: without
a sibling `abstractuic/ui-kit/src` (or `ABSTRACTUIC_SRC`) the test skips, so a stale copy stays green there.

## Scope
- Regenerate `themes.py` from ui-kit 0.9.6 (`ABSTRACTUIC_SRC=… python -m abstractcore.console.theme_sync`). Re-check the
  console's theme picker and styles (`test_every_kit_theme_is_offered_and_styled`).
- Record the kit version the copy was generated from inside `themes.py` (provenance: version and source hash). A CI test
  checks it against the kit floor core declares, so a stale copy fails CI without the sibling checkout. In a CI job that
  has the sibling, the byte-for-byte pin keeps running.

## Done when
`themes.py` is generated from ui-kit 0.9.6, the drift pin is green against the 0.8.5 checkout, the provenance test
fails on a copy from 0.9.5 (checked once), and the console on the operator's instance (8443 or 8080, at a stated sha)
shows the kit's themes. Until then: "in a PR, not done".
