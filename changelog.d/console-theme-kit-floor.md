### Fixed

- **The web console's theme copy is re-synced to ui-kit 0.9.14 and a stale copy fails CI everywhere**
  (`console/theme_sync.py`, `console/themes.py`, core backlog 0869).
  - `themes.py` records the ui-kit version it was generated from (`KIT_SOURCE_PROVENANCE["kit_version"]`), and a test
    compares it with `theme_sync.KIT_THEME_FLOOR` (0.9.14) without the abstractuic checkout. Before, the only check
    was the byte-for-byte drift test, which skips when the kit source is absent (as in CI).
  - The responsive token layer copies only bare `:root` (or `:root[attr]`) rules. A component rule written under
    `:root` (`:root .af-session-card.af-automation-card`, new in ui-kit 0.9.14) is no longer taken for a token.
