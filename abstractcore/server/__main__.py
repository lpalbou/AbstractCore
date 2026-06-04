"""Module entrypoint for `python -m abstractcore.server`."""

from __future__ import annotations

import sys
from typing import List, Optional

from .app import run_server_with_args


def main(argv: Optional[List[str]] = None) -> int:
    run_server_with_args(sys.argv[1:] if argv is None else argv, prog="python -m abstractcore.server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
