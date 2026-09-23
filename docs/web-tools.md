# Web and Document Tools

AbstractCore ships four built-in tools for reading the open web, plus an optional headless-browser
path for pages that build themselves in JavaScript. They share one extraction pipeline, so a skim
and a fetch of the same URL agree about what the page says.

| Tool | Use it for |
| --- | --- |
| `skim_websearch` | A small, keyword-filtered list of candidate URLs |
| `web_search` | The full search result list |
| `skim_url` | A cheap look at one URL: title, description, heading outline, short preview |
| `fetch_url` | The full document as clean markdown, plus metadata and provenance |

The intended sequence is `skim_websearch` → `skim_url` → `fetch_url`: search for candidates, skim to
decide which are worth opening, then fetch the ones that are.

See also: [Tool Calling](tool-calling.md) for how these are registered and executed,
[API Reference](api-reference.md) for the full tool surface.

## Quick start

```python
from abstractcore.tools.common_tools import skim_websearch, skim_url, fetch_url

hits = skim_websearch(query="python http caching", required_terms=["cache"], num_results=5)
preview = skim_url(url="https://example.com/article")
page = fetch_url(url="https://example.com/article")

print(page["title"])
print(page["content"])          # structure-preserving markdown
```

## `fetch_url`

Fetches a URL and returns the document as clean markdown alongside first-class metadata.

```python
fetch_url(
    url,
    method="GET",
    headers=None,
    data=None,
    timeout=45,
    include_binary_preview=False,
    render_js="auto",
    keep_links=True,
    user_agent="AbstractCore-FetchTool/1.0 (+https://github.com/lpalbou/abstractcore)",
    include_full_content=True,
)
```

### What you get back

`content` is the field to read. It is structure-preserving markdown: headings, lists, links, fenced
code blocks with a language hint, and GFM pipe tables all survive. `title` and `description` are
first-class fields rather than something to parse out of the text.

| Key | Meaning |
| --- | --- |
| `content` | The document as markdown. This is the payload. |
| `content_chars` | Its length, so you can budget before reading it |
| `title`, `description` | Page metadata |
| `success`, `status_code`, `final_url` | Outcome and where the request landed |
| `detected_as` | `html`, `json`, `xml`, `text`, `pdf`, `image`, or `binary` |
| `link_dominant` | True when the page is a listing whose links are its content |
| `rendered_with_browser`, `render_note` | Whether a headless render produced this text, and why |
| `rendered` | A compact human-readable header, not a second copy of the document |

### One canonical copy

`content` is the only full copy of the document in the result. The raw source (`raw_text`), the
post-JavaScript DOM (`rendered_dom`) and the flattened text (`normalized_text`) are *evidence*: they
are withheld above `FETCH_URL_MAX_INLINE_EVIDENCE_CHARS` (2,000 characters) and replaced by a
descriptor recording what was dropped.

```python
page = fetch_url(url="https://example.com/long-article")
page["raw_text"]            # None once the source is over the cap
page["raw_text_withheld"]   # {'chars': …, 'bytes': …, 'sha256': …, 'content_type': …, 'reason': …}
```

The keys always exist, so a consumer that reads them keeps working. Raise
`FETCH_URL_MAX_INLINE_EVIDENCE_CHARS` if you want the evidence inline.

When extraction produces very little text, the evidence is kept as a bounded excerpt instead of
being withheld entirely, so you always have something to inspect.

### Links

With `keep_links=True` (the default) links stay inline in `content` as markdown, so an agent can
follow a citation without a second request. On a listing page — a news hub, a blog index — the links
*are* the content, so they are kept even at `keep_links=False`; `link_dominant` tells you when that
happened.

### The escalation ladder

`fetch_url` tries the cheapest thing that can work first and only pays for more when it has to. A
page that extracts fine statically costs one HTTP request, exactly as before.

| Rung | Runs when | Typical cost |
| --- | --- | --- |
| 1. Static fetch, honest `AbstractCore-FetchTool/1.0` User-Agent | always | 0.3–1.2 s |
| 2. Site adapter (the machine-readable view a site publishes for itself) | rung 1 produced no text and an adapter covers the URL | +0.3 s |
| 3. Real-browser render | rung 1 produced a JS shell/no text, **or** answered 403/429/451 with a challenge page (Cloudflare "Just a moment…", `Cf-Mitigated: challenge`, a script-only body) | +2.5–5 s |
| 4. The site's own RSS/Atom feed | the article was refused to every client, including the browser | +0.1–0.5 s |
| 5. Classed failure | nothing above produced readable text | — |

A 403 that explains itself (a geo or licensing refusal with real text) never launches a browser.
`render_js="never"` disables rung 3 only; adapters and feeds launch no browser.

**Site adapters** are deliberately few. Today there is one: a Reddit thread
(`/r/<sub>/comments/<id>/…`) is read from the Atom feed Reddit publishes at `<thread-url>/.rss`
(post, author, date and up to 60 comments), because Reddit serves every non-browser client an empty
app shell, `old.reddit.com` redirects to a login page and the `.json` view answers 403. The result
carries `adapter_used` and `adapter_source_url`.

**Feed recovery** (rung 4) reads the section feed (`/<section>/rss.xml`, then `/rss.xml`, `/feed`,
…) and returns the entry whose link *is* the requested URL. The result is `success=True` with
`degraded=True`, a `degraded_reason`, `recovered_from_feed`, and an `error_class` naming the refusal
— so an agent sees the headline and standfirst **and** that the body was not available.

### JavaScript rendering

Some pages ship a shell and build their content in the browser; others answer a non-browser with a
JavaScript challenge. `fetch_url` re-fetches such a page in a real browser and runs the *same*
extraction pipeline over the rendered DOM.

The escalation renders the page the way the operator's own browser would: the **full Chromium
build** (not `chrome-headless-shell`, which several sites recognise and serve a block page),
ordinary viewport, locale and User-Agent, a persistent cookie/consent profile, a short settle, one
scroll, and a bounded wait (≤10 s) for a self-clearing JavaScript interstitial to navigate to the
page. The layout text (`innerText`) is captured beside the DOM, so pages built from web components —
whose article lives in shadow roots that a serialised DOM does not contain — still yield text.
`render_note` names the browser that ran (`persistent:chromium`, `fresh:chromium`,
`fresh:headless-shell`).

It does **not** solve CAPTCHAs, rotate proxies, or spoof TLS fingerprints, and it never logs in. When
a site refuses, the result says so with a precise class (below).

```python
fetch_url(url=..., render_js="auto")    # render only when static extraction fails (default)
fetch_url(url=..., render_js="always")  # always render; the static result is kept if it is better
fetch_url(url=..., render_js="never")   # never launch a browser
```

Rendering requires the optional browser extra:

```bash
pip install "abstractcore[browser]"
python -m playwright install chromium            # full Chromium: what the fetch_url escalation prefers
python -m playwright install --only-shell chromium  # smaller; used as a fallback if the full build is absent
```

Without it, pages that need JavaScript return their normal actionable error with the install hint
attached. A page that extracts fine statically never launches a browser. If only the headless shell
is installed the escalation still runs on it, with a lower success rate on sites that detect it.

**The browser profile** lives at `~/.abstractcore/browser-profile` — the package's own state
directory, never a model-controlled path and never your real Chrome profile. It keeps cookies and
consent choices so a second visit to a site is not a first-ever visit; it never holds credentials,
because the tool never logs in. Deleting it is always safe. When two renders run at once, the second
cannot share the profile (Chromium locks it) and falls back to a private context automatically.

The render runs in a worker subprocess under one wall-clock budget taken from your `timeout`, with a
process-tree kill, so it cannot hang or leave a browser behind. Every navigation and subresource is
screened by the same SSRF guard the static path uses, so a page cannot redirect or script its way to
a destination the static fetch would refuse. `final_url` reports where the browser actually landed,
and `rendered_dom` carries the post-JavaScript DOM separately from `raw_text`, which always means
the bytes the server sent.

Rendering executes the page's JavaScript. Treat it as you would any untrusted code path.

### Errors are explicit

A fetch that cannot produce content fails with a class, a `retryable` flag and concrete
`suggestions` — never a silent empty success.

| `error_class` | Meaning |
| --- | --- |
| `js_required` | The page needs JavaScript and no browser was available |
| `empty_content` | A 2xx that yielded no extractable text |
| `empty_body` | A 2xx with a zero-byte body, usually bot mitigation |
| `bot_challenge` | An anti-bot challenge was detected; a retry or a render may clear it |
| `blocked_by_site` | A real browser was also refused (a challenge that did not clear, "you have been blocked"). Usually rate-based: waiting clears it. |
| `captcha_required` | The site demanded human verification. Out of scope by design; `retryable=False`. |
| `paywall` | The body is behind a subscription; the title/standfirst is what is legitimately readable |
| `login_required` | The page requires an account; the tool holds no credentials |
| `rate_limited` | Retry after the interval the server named |
| `auth_required`, `not_found`, `gone`, `client_error`, `server_error` | HTTP outcomes |
| `extraction_failed` | An internal extraction error, not a property of the page. Retryable. |
| `blocked_ssrf` | A non-public destination |
| `blocked_encoded_url` | The URL carried a base64-encoded payload |

Transient bot-challenge, rate-limit and 5xx statuses get a bounded retry with the same honest
User-Agent, honouring `Retry-After`. The static fetch keeps that honest identity: a browser
User-Agent on a non-browser HTTP stack is an incoherent fingerprint (measured: it turned
economist.com's plain 403 into a full Cloudflare challenge and helped none of the tested sites). A
browser identity is used only where a real browser is actually running — the render escalation.
AbstractCore does not solve CAPTCHAs.

`degraded=True` with an `error_class` on a `success=True` result means "partial content, and here
is why" — currently only the feed-recovery rung produces it.

Set `ABSTRACTCORE_DEBUG_EXTRACTION=1` to print the traceback behind an `extraction_failed`.

### Documents

PDFs are extracted with page anchors (`# Page 3`) so a model can cite a location, and typographic
ligatures are expanded so the text stays searchable. The default backend is `pypdf`
(BSD-3-Clause). Table recovery is available through `pymupdf4llm`, which is dual
AGPL-3.0/commercial and therefore never selected automatically — request it explicitly:

```python
from abstractcore.media.pdf_routing import route_pdf_bytes
route_pdf_bytes(pdf_bytes, preferred_backend="pymupdf4llm")
```

See [Media Handling](media-handling-system.md) for the wider document and image pipeline.

## `skim_url`

A cheap look at one URL before deciding whether to fetch it.

```python
skim_url(url, timeout=15, max_bytes=10485760, max_preview_chars=2400, max_headings=8)
```

`skim_url` uses the same extraction as `fetch_url` and then trims the *extracted* text to
`max_preview_chars`. The heading outline is harvested from the same markdown as the preview, so the
outline always describes content the preview came from.

`max_bytes` is a memory safety net, not the skim: it bounds the download at the same 10 MB limit
`fetch_url` uses. Pass a smaller value for a deliberate cheap peek — the report then states that the
extraction, not merely the transfer, was partial.

`skim_url` enforces the same SSRF guard as `fetch_url`, per redirect hop.

## `web_search` and `skim_websearch`

```python
web_search(query, num_results=10, safe_search="moderate", region="wt-wt", time_range=None)

skim_websearch(query, required_terms=None, num_results=5, safe_search="moderate",
               region="wt-wt", time_range=None, require_in="snippet", match="any")
```

`skim_websearch` returns a smaller list and can require keywords before you spend a fetch on a
result.

### Argument values

`time_range` accepts `h`, `d`, `w`, `m`, `y` and plain-English equivalents — `day`, `today`,
`past week`, `30 days`, `12 months` and similar all normalize. A value that cannot be mapped is
dropped and reported in `warnings`/`limitations`; it is never forwarded to the search backend, so an
unrecognised filter cannot cost you the primary backend.

`require_in` selects where `required_terms` must appear: `snippet`, `title`, `title_snippet`, or
`all`. Common synonyms (`body`, `text`, `title,body`, `both`, `any`) resolve to the right scope. A
value that cannot be resolved is substituted and reported, naming both what you sent and what was
used.

`match` is `any` (default) or `all`. Backend snippets are short, so `any` with
`require_in="title_snippet"` is usually the right combination.

### Result quality signals

Payloads carry `warnings` and `limitations` describing anything that degraded the search: a dropped
filter, a coerced argument, a fallback backend, or search-backend text with missing word separators
(`fused_result_text`). `degraded` is set whenever the result is not what you asked for, so a
consumer never has to guess.

`rank` is the result's position in the unfiltered backend list, so ranks are not contiguous after
filtering.

## Safety

All four tools screen destinations against the SSRF guard: loopback, private, link-local and
cloud-metadata addresses are refused, on the initial URL and on every redirect. Opt a specific
host and port in with `ABSTRACTCORE_FETCH_URL_ALLOW=127.0.0.1:8080`.

`fetch_url` keeps URL query parameters and passes your headers through — stripping them breaks real
fetches. The one screen is on base64-encoded payloads anywhere in the URL, which are refused with
`blocked_encoded_url`. Detection decodes the candidate and flags it only when it decodes to
meaningful data, so opaque identifiers such as Drive file ids, UUIDs and git SHAs fetch normally.

`fetch_url` is not read-only: `method` and `data` are caller-controlled, so it can issue writes. It
is tagged `write`/`remote_write` for host guards.
