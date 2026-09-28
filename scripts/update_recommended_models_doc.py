"""Regenerate the recommended-models tables in docs/recommended-models.md.

The tables between the BEGIN/END markers are rendered from
`abstractcore.config.recommendations.recommendation_matrix()`, the same data
`abstractcore models recommendations --json` exports. Nobody edits them by
hand: run this script after changing a recommendation, and `--check` (run by
the test suite) fails when the page and the code disagree.

    python scripts/update_recommended_models_doc.py          # rewrite the block
    python scripts/update_recommended_models_doc.py --check  # exit 1 when stale
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "recommended-models.md"
BEGIN = "<!-- BEGIN GENERATED: recommended-models (scripts/update_recommended_models_doc.py) -->"
END = "<!-- END GENERATED: recommended-models -->"


def rendered_block() -> str:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from abstractcore.config.recommendations import recommendation_matrix, render_markdown

    return render_markdown(recommendation_matrix())


def updated_page(page: str, block: str) -> str:
    """`page` with the text between the markers replaced by `block`.

    Raises when the markers are missing or out of order: a page without them
    would silently stop being checked.
    """

    start = page.find(BEGIN)
    end = page.find(END)
    if start < 0 or end < 0 or end < start or page.count(BEGIN) != 1 or page.count(END) != 1:
        raise ValueError(f"{DOC.relative_to(ROOT)} must contain exactly one {BEGIN!r} ... {END!r} block")
    return page[: start + len(BEGIN)] + "\n\n" + block.rstrip("\n") + "\n\n" + page[end:]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Exit 1 when the page is not current (write nothing)")
    args = parser.parse_args(argv)
    page = DOC.read_text(encoding="utf-8")
    new = updated_page(page, rendered_block())
    if args.check:
        if new != page:
            print(f"{DOC.relative_to(ROOT)} is stale: run python scripts/update_recommended_models_doc.py")
            return 1
        print(f"{DOC.relative_to(ROOT)} is current")
        return 0
    if new != page:
        DOC.write_text(new, encoding="utf-8")
        print(f"updated {DOC.relative_to(ROOT)}")
    else:
        print(f"{DOC.relative_to(ROOT)} already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
