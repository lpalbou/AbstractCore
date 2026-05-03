# Release Process

This repo publishes releases through the `Release` GitHub Actions workflow. The
preferred path is a manual workflow dispatch with the version number; the
workflow validates the release, creates the annotated tag, publishes to PyPI, and
creates the GitHub Release. Pushing an annotated tag like `v2.13.1` also starts
the same workflow.

The workflow:

- checks that the tag version matches `abstractcore/utils/version.py`
- checks that `CHANGELOG.md` has release notes for that version
- runs the test suite on Python 3.9, 3.10, 3.11, 3.12, and 3.13
- builds the documentation
- builds the source distribution and wheel
- runs `twine check`
- publishes to PyPI through Trusted Publishing
- creates a GitHub Release using the matching changelog section
- creates the annotated `vX.Y.Z` tag when launched manually

## One-time setup

Configure PyPI Trusted Publishing for the `abstractcore` project:

1. Open the PyPI project settings for `abstractcore`.
2. Add a Trusted Publisher for GitHub.
3. Use owner `lpalbou`, repository `abstractcore`, workflow
   `.github/workflows/release.yml`, and environment `pypi`.
4. In GitHub, create an environment named `pypi`. Add required reviewers there
   if you want a manual approval gate before PyPI publishing.

No PyPI API token is needed when Trusted Publishing is configured.

## Preferred automated release

1. Update `abstractcore/utils/version.py`.
2. Move the relevant `CHANGELOG.md` notes from `Unreleased` to the new version.
3. Run the local checks you want before pushing:

   ```bash
   ruff check --select W293 .
   python -m compileall -q abstractcore
   python -m pytest
   python -m build
   python -m twine check dist/*
   mkdocs build -q
   ```

4. Commit and push `main`:

   ```bash
   git push origin main
   ```

5. Open GitHub Actions, choose the `Release` workflow, click `Run workflow`,
   select `main`, and enter the version, for example:

   ```text
   2.13.1
   ```

6. The workflow will:

   ```text
   validate version/changelog
   run tests and docs build
   build and check dist/*
   create annotated tag v2.13.1
   publish to PyPI
   create GitHub Release "AbstractCore v2.13.1"
   ```

7. Verify the GitHub Release and the PyPI project page.

## Tag-push alternative

If you prefer to launch from the terminal, create and push the annotated tag
after `main` is pushed:

   ```bash
   git tag -a v2.13.1 -m "AbstractCore v2.13.1"
   git push origin main
   git push origin v2.13.1
   ```

Then watch the `Release` workflow in GitHub Actions and verify GitHub + PyPI.

## Manual fallback

If GitHub Actions or Trusted Publishing is unavailable, build and publish
manually after the same local checks:

```bash
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```
