# fetch_url extraction fixtures + gold references

Committed raw-HTML fixtures and per-URL gold checklists for the `fetch_url`
extraction-quality regression (`../test_fetch_url_extraction_gold.py`).

## Why these exist

On 2026-07-11 a summoned entity's `fetch_url` returned "found no readable text"
on pages that are perfectly readable. Four adversarial subagents fetched each
page from raw bytes, curated the true article content, and attacked the
extraction pipeline. The root causes were:

1. **Contract gap (P0):** `fetch_url` returned `raw_text`/`normalized_text`/
   `rendered` but no obvious `content`/`title` key, so every clean HTML fetch
   looked empty to a consumer reaching for `content`. Fixed by exposing
   first-class `content` (structure-preserving markdown) + `title`.
2. **403 bot-challenge:** probabilistic Cloudflare challenge — fixed by a
   bounded same-profile retry with the *honest* identified UA (browser
   impersonation is worse: it manufactures an incoherent fingerprint and
   forfeits robots.txt compliance / whitelisting).
3. **Extraction junk/loss:** flat `get_text()` split sentences and dropped
   links; sidebar/author-box/consent overlays leaked in; body-level container
   selection dragged in nav/footer. Fixed by routing `content` through the
   structure-preserving markdown renderer, text-signature consent-banner
   removal, author-box/widget pruning, and a readability densest-container
   fallback.

## Files

- `*.html` — raw response bytes captured from the live pages (deterministic,
  offline; CI never hits the network).
- `gold_facts.json` — per-URL `facts` (distinct substrings that MUST appear in
  extracted content), `junk` (boilerplate that MUST NOT appear), and
  `title_contains`. Curated from the raw bytes, NEVER from the tool's output.
- `harness.py` — the mechanical scorer. `python harness.py` prints the
  fact-recall / junk table. Matching is whitespace/case/dash-normalized and
  markdown-link-aware (a fact wrapped in `[label](url)` still counts).

## The bar

Each URL's primary `content`: fact recall >= 0.90 AND junk ratio == 0.0. All
four fixtures currently pass at 100% recall / 0 junk.

## Refreshing a fixture

Re-capture with the tool's own honest UA so the fixture matches what the tool
sees:

```
curl -sS -A "AbstractCore-FetchTool/1.0 (+https://github.com/lpalbou/abstractcore)" \
  -H "Accept-Language: en-US,en;q=0.9" "<url>" -o <name>.html
```

If a site's real content genuinely changes, update `gold_facts.json` from the
new raw bytes — never from the tool's output.
