### Changed

- **Changelog entries are fragments** (`changelog.d/`, `scripts/changelog.py`). A pull request adds
  `changelog.d/<pr-or-slug>.md` instead of editing `CHANGELOG.md`'s `[Unreleased]` section, which conflicted on every
  merge of main; the release assembles the fragments (`python scripts/changelog.py assemble X.Y.Z`) and CI fails a pull
  request that changes `abstractcore/` without one. The `[Unreleased]` section as of 2026-10-29 is the first fragment.
