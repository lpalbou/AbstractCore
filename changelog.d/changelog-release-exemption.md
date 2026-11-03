### Changed

- **The changelog check recognises a release pull request** (`scripts/changelog.py check`). A PR that bumps
  `abstractcore/utils/version.py` to X.Y.Z, adds `CHANGELOG.md`'s dated `## [X.Y.Z] - YYYY-MM-DD` section and leaves
  no fragment in `changelog.d/` passes. Before, every release PR was red, because it edits `CHANGELOG.md` and source
  without adding a fragment. A release that leaves a fragment behind fails and names it. Any other PR that touches
  `abstractcore/` still needs a fragment.
