from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable


def _len_chars(value: Any) -> int:
    return len(str(value or ""))


def _json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False))
    except Exception:
        return _len_chars(value)


def _measure(label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    record: dict[str, Any] = {
        "tool": label,
        "elapsed_ms": round(elapsed_ms, 1),
        "type": type(out).__name__,
    }

    if isinstance(out, dict):
        record["json_chars"] = _json_size(out)
        record["rendered_chars"] = _len_chars(out.get("rendered"))
    else:
        record["text_chars"] = _len_chars(out)

    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lightweight footprint benchmarks for skim_* and fetch_* web tools.\n"
        'Requires: pip install "abstractcore[tools]"',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--url",
        action="append",
        dest="urls",
        default=[],
        help="URL to benchmark (repeatable). Defaults to a small stable set.",
    )
    parser.add_argument(
        "--query", default="abstractcore skim_url fetch_url", help="DuckDuckGo search query."
    )
    parser.add_argument(
        "--num-results",
        type=int,
        default=5,
        help="Results to return for web_search/skim_websearch.",
    )

    parser.add_argument("--max-bytes", type=int, default=200_000, help="skim_url max_bytes.")
    parser.add_argument(
        "--max-preview-chars", type=int, default=1200, help="skim_url max_preview_chars."
    )
    parser.add_argument("--max-headings", type=int, default=8, help="skim_url max_headings.")

    parser.add_argument(
        "--fetch-full",
        action="store_true",
        help="Use fetch_url(include_full_content=True). Default is False for smaller outputs.",
    )
    parser.add_argument(
        "--no-search", action="store_true", help="Skip web_search/skim_websearch benchmarks."
    )

    args = parser.parse_args()

    from abstractcore.tools.common_tools import fetch_url, skim_url, skim_websearch, web_search

    urls = list(args.urls or [])
    if not urls:
        urls = [
            "https://example.com",
            "https://www.rfc-editor.org/rfc/rfc9110",
        ]

    records: list[dict[str, Any]] = []

    if not args.no_search:
        records.append(
            _measure(
                "skim_websearch",
                skim_websearch,
                query=args.query,
                num_results=args.num_results,
            )
        )
        records.append(
            _measure(
                "web_search",
                web_search,
                query=args.query,
                num_results=args.num_results,
            )
        )

    for url in urls:
        records.append(
            _measure(
                "skim_url",
                skim_url,
                url,
                max_bytes=args.max_bytes,
                max_preview_chars=args.max_preview_chars,
                max_headings=args.max_headings,
            )
        )
        records.append(
            _measure(
                "fetch_url",
                fetch_url,
                url,
                include_full_content=bool(args.fetch_full),
            )
        )

    print(json.dumps({"benchmarks": records}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
