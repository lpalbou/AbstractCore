# 0886 — Python 3.14 install blocked by the lxml ceiling

> Type: packaging
> Created: 2026-11-10
> Priority: normal
> Status: completed 2026-11-10 (2.35.1): `lxml>=4.9.0,<7.0.0`; BeautifulSoup's `lxml` and `xml` parser names are unchanged in lxml 6; Python 3.14 is not in the CI matrix yet (classifiers stop at 3.13).
> Labels: packaging, dependencies, ci
> Links: `pyproject.toml` (`lxml<6.0.0,>=4.9.0`), `abstractcore/tools/common_tools.py` (`BS4_PARSER`)

## Summary

`pip install abstractcore==2.35.0` on Python 3.14 (Ubuntu 26.04 host, 2026-11-10) fails before anything is
installed: the light install pins `lxml<6.0.0,>=4.9.0`, lxml 5.x publishes no cp314 wheels, so pip builds lxml from
source and stops on "Please make sure the libxml2 and libxslt development packages are installed". lxml 6.1.3 has
cp314 manylinux wheels. The package declares `requires-python >= 3.9` and classifiers up to 3.13; CI tests 3.9 to
3.13; the framework's installers provision Python 3.12, so the one-line installs are not affected. A plain pip user
on 3.14 is.

## Decision

Proposed: raise the ceiling to `lxml>=4.9.0,<7` after checking the two uses in `common_tools.py` (BeautifulSoup
parser name `lxml`, the `xml` parser choice for XML content) against lxml 6; add Python 3.14 to the CI matrix (or
state `requires-python <3.14` until then, which is the honest alternative). Same sweep for the other pinned
C-extension dependencies with no cp314 wheels, if any (`pip download --only-binary=:all: --python-version 3.14` of
the light set says which).

## Success criteria

- `pip install abstractcore` from wheels only succeeds on Python 3.14 Linux x86_64, or the package refuses 3.14 by
  its own metadata; CI matrix matches what is claimed.
