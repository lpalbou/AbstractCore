# fetch_url extraction fixtures + gold references

Committed raw-HTML/PDF fixtures and per-URL gold checklists for the `fetch_url`
extraction-quality regressions:

- `../test_fetch_url_extraction_gold.py` — the recall/junk bar
- `../test_fetch_url_adv_extraction.py` — the adversarial defect pins

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

On 2026-08-21 the corpus was expanded from 4 news articles to 17 fixtures
chosen for **structural diversity**, because the original four were all the
same shape (a single prose article on a CMS) and therefore could not see the
defects that live in tables, code blocks, discussion threads or documents.

## The corpus

| key | shape it stresses |
| --- | --- |
| `techxplore`, `budgyapp`, `newsletter`, `nextbigfuture` | prose article on a CMS (the 2026-07-11 production failures) |
| `techstartups` | WordPress article: ad rails, HTML build comments, Trending/Read-Next rail |
| `bbc_tech_hub` | news HUB/index: ~60 headline cards, duplicated timestamps, mega-footer |
| `pydocs_json` | Sphinx technical docs: syntax-highlighted code + conversion tables |
| `github_readme` | GitHub repo landing page: README under a mega-nav + file-list table |
| `wikipedia_bert` | Wikipedia: infobox table, `[edit]` links, refs, category footer |
| `ja_wikipedia` | non-English / heavy Unicode (Japanese, no word spaces) |
| `arxiv_abs` | academic abstract: metadata table, arXivLabs toggle rail |
| `hn_thread` | forum thread on a pure `<table>` layout: nested comments + vote chrome |
| `mdn_http_status` | MDN reference: definition lists, in-page TOC, feedback CTA |
| `do_pricing` | marketing pricing page: real price `<table>`s + styled-components chrome |
| `sqlite_datefunc` | small table-heavy reference (strftime substitution grid) |
| `spa_shell` | JS-only SPA shell — must FAIL loudly, never return empty success |
| `arxiv_paper_pdf` | a real 15-page PDF document |
| `discourse_thread` | Discourse forum thread: server-rendered OP + 21 replies, suggested-topics rail nested inside the first post |
| `github_issue` | GitHub issue page: a 556-char issue body under ~2,000 chars of mega-nav (replies are client-hydrated, so they are not in the bytes) |
| `supabase_pricing` | pricing page with real compute tables and a mega-footer, styled with semantically-empty Tailwind utility classes |

## Files

- `*.html` / `*.pdf` — raw response bytes captured from the live pages
  (deterministic, offline; CI never hits the network).
- `gold_facts.json` — per-URL `facts` (distinct substrings that MUST appear in
  extracted content), `structural_facts` (facts that must survive *with* their
  table/code/list/link structure), `junk` (boilerplate that MUST NOT appear),
  and `title_contains`. Curated from the raw bytes, NEVER from the tool's
  output; every string is machine-verified to occur in the fixture.
- `harness.py` — the mechanical scorer. `python harness.py` prints both tables.

## Web-search fixture

`websearch_fused_rows.json` supports `../test_websearch_adv_extraction.py` (the
W1-W4 pins). It holds four **verbatim production payloads** captured from the
2026-08-21 transcript plus one clearly-marked synthetic false-positive probe:
32 rows, 13 of them labelled `fused` (words glued together where the backend
stripped `<b>` highlight markup with no separator, e.g. `technewsstories
today,August21,2026`). `expect_fused_payload` encodes the payload-level gate
(>=2 fused rows AND >=30% of rows) that a detector must reproduce, so a single
long word or one CamelCase-heavy result can never trip it.

## The three metrics

`python tests/tools/fetch_url_fixtures/harness.py`

1. **`score(content, key)`** — `fact_recall` and `junk_ratio`. Matching is
   whitespace/case/dash-normalized and markdown-aware: `[label](url)`,
   `` `code` `` and `*emphasis*` reduce to their text, so the metric measures
   CONTENT and never punishes correct markup.
2. **`structure_score(content, key)`** — the fraction of `structural_facts`
   that survive *inside* their structure: a code fact still in a ``` fence, a
   table fact still on a `|` row, a list fact still on a `-`/`1.` line, a link
   fact still inside `[label](url)`. A fact that survives only as flat prose
   counts as LOST — that is exactly what the flat `get_text()` path destroys.
3. **`envelope_efficiency(result)`** — the token cost of the whole result dict:
   `content_chars`, `total_json_chars`, `amplification = total/content`, and
   `duplicate_payload_chars` (any other string field ≥200 chars whose 8-gram
   containment in `content` is ≥85%). `offline_envelope(key)` rebuilds the exact
   `fetch_url` envelope from a fixture so this is measurable with no network.

## The bar

Each URL's primary `content`: `fact_recall >= 0.90` AND `junk_ratio == 0.0` AND
`structure_score == 1.0`, with `amplification <= 3.0`.

Every `junk` string is machine-verified to occur in its fixture's raw bytes
(round 3 removed five in the original four fixtures that did not, and could
therefore never fail). A junk list that cannot fail is decoration.

## Refreshing a fixture

Re-capture with the tool's own honest UA so the fixture matches what the tool
sees:

```
curl -sS -A "AbstractCore-FetchTool/1.0 (+https://github.com/lpalbou/abstractcore)" \
  -H "Accept-Language: en-US,en;q=0.9" "<url>" -o <name>.html
```

If a site's real content genuinely changes, update `gold_facts.json` from the
new raw bytes — never from the tool's output.

---

## The JavaScript-render corpus (`js_*`, `bot_*`, `spa_shell.html`)

Added 2026-08-21 for the render-escalation work. These fixtures are NOT scored
against `gold_facts.json` — they exist to separate three failure modes that look
identical from the outside (a thin or empty `content` on a 200 response) and that
call for three completely different answers. Conflating them makes the render
feature look broken when it is not, and look fine when it is not.

**Class (a) — genuinely client-rendered, no embedded state.** A browser is the
only answer.

| fixture | source | static | after a headless render |
| --- | --- | ---: | --- |
| `spa_shell.html` | excalidraw.com | 0 | 189 chars — a browser-storage disclaimer |
| `js_hetzner_cloud.html` | hetzner.com/cloud | 7,357 | 7,371 (+14) |

**Class (b) — client-rendered in the browser, but the DATA is already in the
server bytes.** A browser is overkill; measured, it adds exactly zero characters.
The answer is to mine what is already there.

| fixture | source | static | embedded state |
| --- | --- | ---: | --- |
| `js_notion_blog.html` | notion.so/blog | 3,616 | `__NEXT_DATA__` |
| `js_nasa_news.html` | nasa.gov/news | 366 | JSON-LD + `<link rel=alternate>` to `/feed/` |
| `js_instagram_home.html` | instagram.com | 0 | 49 `application/json` blocks (logged out: no real content) |
| `js_vercel_blog.md` | vercel.com/blog | 51,481 | **not HTML** — see below |

**Class (c) — not a rendering problem.** Bot mitigation. A headless browser
carrying the same honest identity is blocked too, usually harder: rendering these
made the outcome WORSE in all three cases (reddit → an explicit network-block
notice, pubmed 203 → 403, imdb 202 → 403).

| fixture | source | status | tell |
| --- | --- | ---: | --- |
| `bot_reddit_programming.html` | reddit.com/r/programming | 200 | auto-submitted JS-computed `solution` form; **no vendor token** |
| `bot_pubmed_abstract.html` | pubmed.ncbi.nlm.nih.gov | 203 | `cookies-required` gate |
| `bot_imdb_title.html` | imdb.com/title/tt0111161 | 202 | `awswaf` / `gokuProps` / `challenge.js` |

### Two capture facts that will bite whoever refreshes these

1. **Capture with `Accept: */*`, not a browser-ish Accept.** That is what
   `fetch_url` sends, and sites content-negotiate on it. `vercel.com/blog`
   returns a 505 KB React app to `Accept: text/html,...` and 169 KB of
   **`text/markdown`** to `*/*` — which is why the fixture is `.md`, and why
   vercel is not a JS-rendering problem at all despite looking like the textbook
   case. A fixture captured with the wrong Accept describes a page the tool never
   sees.
2. **`imdb.com` returns a 0-byte body to `Accept: */*`** (HTTP 202) and the AWS
   WAF shell only to an HTML Accept. `bot_imdb_title.html` is the WAF shell,
   captured deliberately with an HTML Accept because the empty body is not an
   instructive artifact. Through `fetch_url` the URL takes the `empty_body` path
   and never reaches the render escalation.

### Why there are no rendered-DOM fixtures

A rendered DOM cannot be committed without going stale within days — it embeds
build hashes, session ids and A/B assignments. Tests that need one are gated
behind `ABSTRACT_E2E_FETCH_URL=1` (`test_fetch_url_adv_js_render.py`, group G)
rather than faked.
