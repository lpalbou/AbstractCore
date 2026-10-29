# Changelog fragments

Each pull request that changes `abstractcore/` adds ONE file here instead of editing `CHANGELOG.md`:
`changelog.d/<pr-number-or-slug>.md` (for example `changelog.d/123.md` or `changelog.d/a-short-slug.md`). A new file never
conflicts with another branch; the `[Unreleased]` section did on every merge.

A fragment is Keep a Changelog markdown: one or more of `### Added`, `### Changed`, `### Deprecated`,
`### Removed`, `### Fixed`, `### Security`, each followed by its entries, written as before:

```markdown
### Fixed

- **A stop signal is what a reader needs** (`module.py`). What was wrong, what is true now,
  and how to use it ...
```

- CI (`changelog` job): a pull request that changes `abstractcore/` without adding a fragment fails, and so does a
  malformed fragment. `python scripts/changelog.py check` runs the same check locally.
- `python scripts/changelog.py preview` prints what the next release section will say.
- Release: `python scripts/changelog.py assemble X.Y.Z` writes `## [X.Y.Z] - <date>` from every fragment
  (sections in Keep a Changelog order) and deletes the fragments; commit that with the version bump.
