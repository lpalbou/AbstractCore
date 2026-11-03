### Fixed

- **The release-PR changelog check refuses an empty release section, and sees every fragment**
  (`scripts/changelog.py`). A release PR whose `## [X.Y.Z] - <date>` section had no entries passed when its
  fragments were deleted without being assembled; only the tag-time validation would have caught it. It now fails,
  naming the version. `fragment_files` reads `changelog.d/**/*.md` in any case (README excluded), so a fragment left
  in a subfolder or named `fix.MD` is listed by `check` and written by `assemble`, no longer invisible.
