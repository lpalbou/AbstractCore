"""AbstractCore web console: standalone page + embeddable Models/Engines fragments.

One vanilla-JS namespace (``window.AbstractCoreConsole``), no build step, no
other globals. ``abstractcore serve`` serves :func:`render_console_html` at
``/console``; abstractgateway splices :func:`fragment` into its own console's
"Models" (id ``catalog``) and "Engines" (id ``engines``) tabs and mounts them
with its CSRF-aware ``api()`` as the injected ``request``.

Data comes only from the shared contracts (host_profile_v1,
engines_status_v1, model_catalog_v1, models_installed_v1, host_job_v1; see
``docs/console.md``). The console never computes presence or fit itself: it
renders the server's verdicts and labels them with the shared vocabulary.

Embedding contract (the seam with abstractgateway)::

    from abstractcore.console.web import fragment
    frag = fragment("models")   # {"html": ..., "js": ..., "css": ...}
    # host page: <style>{css}</style> ... {html} ... <script>{js}</script>
    # then, in the host's JS:
    #   AbstractCoreConsole.mount("models", panelEl, {
    #       apiBase: "/api/gateway",
    #       request: (method, path, body) => api(path, {method, body: body == null ? undefined : JSON.stringify(body)}),
    #       isAdmin: () => Boolean(state.principal && state.principal.admin),
    #       onJob: (job, kind) => {...},
    #       hostName: "gateway-host",
    #       cliPrefix: "abstractgateway",
    #   });
"""

from __future__ import annotations

import json
from typing import Any, Dict

FRAGMENT_KINDS = ("models", "engines")

# Bump when the JS mount contract changes shape (hosts can read it from
# ``AbstractCoreConsole.version``).
CONSOLE_JS_VERSION = "1"


def _script_json(value: object) -> str:
    """JSON safe to inline inside a <script> element."""
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


# ---------------------------------------------------------------------------
# CSS. Fragment CSS is scoped under `.acc-root` and uses only the ui-kit token
# names (with fallbacks), so inside the gateway console it inherits whichever
# kit theme the gateway has active.
# ---------------------------------------------------------------------------

_FRAGMENT_CSS = """
.acc-root { color: var(--text-primary, #eee); font: var(--font-size-base, 14px)/1.5 var(--font-sans, system-ui, sans-serif); }
.acc-root *, .acc-root *::before, .acc-root *::after { box-sizing: border-box; }
.acc-root [hidden] { display: none !important; }
.acc-root h3 { margin: 0; font-size: var(--font-size-lg, 16px); }
.acc-root code, .acc-root pre { font-family: var(--font-mono, ui-monospace, monospace); font-size: var(--font-size-sm, 12px); }
.acc-root code { background: var(--ui-code-bg, rgba(0,0,0,.18)); border: 1px solid var(--ui-code-border, rgba(255,255,255,.08)); border-radius: var(--radius-sm, 4px); padding: 1px 5px; overflow-wrap: anywhere; }
.acc-root pre.acc-code { margin: 6px 0; padding: 10px 12px; background: var(--ui-code-block-bg, rgba(0,0,0,.25)); border: 1px solid var(--ui-code-block-border, rgba(255,255,255,.08)); border-radius: var(--radius-md, 8px); white-space: pre-wrap; word-break: break-all; }
.acc-root .acc-muted { color: var(--text-secondary, #aaa); }
.acc-root a { color: var(--info, #60a5fa); }
.acc-root .acc-hint { color: var(--text-muted, #666); font-size: var(--font-size-xs, 11px); margin: 10px 0 0; }
.acc-root .acc-toolbar { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; margin: 0 0 10px; }
.acc-root .acc-toolbar label { display: inline-flex; gap: 6px; align-items: center; color: var(--text-secondary, #aaa); font-size: var(--font-size-sm, 12px); }
.acc-root input[type=search], .acc-root input[type=password], .acc-root input[type=text], .acc-root select {
  background: var(--ui-surface-1, rgba(0,0,0,.16)); color: var(--text-primary, #eee); border: 1px solid var(--ui-border-2, rgba(255,255,255,.14));
  border-radius: var(--radius-sm, 4px); padding: 6px 8px; font: inherit; }
.acc-root input[type=search] { min-width: min(320px, 100%); flex: 1 1 220px; }
.acc-root .acc-toolbar > span { min-width: 0; overflow-wrap: anywhere; }
/* Long unbroken paths (config files under a deep HOME) in prose lines must
   wrap inside the pane, never widen the page (DESIGN §5.6). */
.acc-root .acc-hint, .acc-root .acc-muted, .acc-root .acc-host-line, .acc-root .acc-message, .acc-root .acc-section-head > * { min-width: 0; overflow-wrap: anywhere; }
.acc-root input:focus-visible, .acc-root select:focus-visible, .acc-root button:focus-visible, .acc-root tr:focus-visible { outline: 2px solid var(--info, #60a5fa); outline-offset: 1px; }
.acc-root .acc-btn { background: var(--bg-tertiary, #0f3460); color: var(--text-primary, #eee); border: 1px solid var(--ui-border-2, rgba(255,255,255,.14)); border-radius: var(--radius-sm, 4px); padding: 5px 10px; font: inherit; font-size: var(--font-size-sm, 12px); cursor: pointer; white-space: nowrap; }
.acc-root .acc-btn:hover:not(:disabled) { border-color: var(--info, #60a5fa); }
.acc-root .acc-btn:disabled { opacity: .5; cursor: not-allowed; }
.acc-root .acc-btn.acc-primary { background: var(--accent, #e94560); border-color: var(--accent, #e94560); color: #fff; }
.acc-root .acc-btn.acc-danger { background: var(--error, #e74c3c); border-color: var(--error, #e74c3c); color: #fff; }
.acc-root .acc-btn.acc-link { background: transparent; border-color: transparent; color: var(--info, #60a5fa); padding: 2px 4px; }
.acc-root .acc-actions { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.acc-root .acc-section { margin: 16px 0 0; }
/* List panels collapse (open by default, remembered per viewer; DESIGN §12). */
.acc-root .acc-collapse { display: inline-flex; align-items: center; gap: 6px; padding: 0; border: 0; background: transparent; color: inherit; font: inherit; cursor: pointer; }
.acc-root .acc-collapse::before { content: "\\25BE"; color: var(--text-secondary, #aaa); }
.acc-root .acc-collapse[aria-expanded="false"]::before { content: "\\25B8"; }
.acc-root .acc-section-head { display: flex; gap: 12px; align-items: baseline; justify-content: space-between; flex-wrap: wrap; margin: 0 0 6px; }
.acc-root .acc-table-scroll { overflow-x: auto; border: 1px solid var(--ui-border-1, rgba(255,255,255,.1)); border-radius: var(--radius-md, 8px); }
.acc-root table.acc-table { width: 100%; border-collapse: collapse; font-size: var(--font-size-sm, 12px); }
.acc-root .acc-table th, .acc-root .acc-table td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--ui-border-1, rgba(255,255,255,.1)); vertical-align: top; }
.acc-root .acc-table th { color: var(--text-secondary, #aaa); font-weight: 600; background: var(--ui-surface-1, rgba(0,0,0,.16)); position: sticky; top: 0; }
.acc-root .acc-table tbody tr:hover { background: var(--ui-overlay-bg-hover, rgba(255,255,255,.05)); }
.acc-root .acc-table tbody tr.acc-group-start td { border-top: 1px solid var(--ui-border-2, rgba(255,255,255,.14)); }
.acc-root .acc-table td.acc-num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.acc-root .acc-model-name { font-weight: 600; }
.acc-root .acc-sub { color: var(--text-secondary, #aaa); font-size: var(--font-size-xs, 11px); }
.acc-root .acc-badge { display: inline-block; border-radius: 999px; padding: 1px 8px; font-size: var(--font-size-xs, 11px); border: 1px solid var(--muted-border, rgba(148,163,184,.22)); background: var(--ui-pill-bg, rgba(0,0,0,.22)); white-space: nowrap; cursor: default; }
.acc-root .acc-tone-ok { color: var(--success, #27ae60); border-color: var(--success-border, rgba(39,174,96,.35)); background: var(--success-subtle, rgba(39,174,96,.12)); }
.acc-root .acc-tone-warn { color: var(--warning, #f39c12); border-color: var(--warning-border, rgba(243,156,18,.35)); background: var(--warning-subtle, rgba(243,156,18,.12)); }
.acc-root .acc-tone-err { color: var(--error, #e74c3c); border-color: var(--error-border, rgba(231,76,60,.35)); background: var(--error-subtle, rgba(231,76,60,.12)); }
.acc-root .acc-tone-info { color: var(--info, #60a5fa); border-color: var(--info-border, rgba(96,165,250,.35)); background: var(--info-subtle, rgba(96,165,250,.12)); }
.acc-root .acc-tone-off, .acc-root .acc-tone-muted { color: var(--text-secondary, #aaa); }
.acc-root .acc-chip { display: inline-block; font-size: var(--font-size-xxs, 10px); border: 1px solid var(--ui-chip-border, rgba(255,255,255,.12)); background: var(--ui-chip-bg, rgba(0,0,0,.22)); color: var(--ui-chip-text, rgba(255,255,255,.92)); border-radius: 999px; padding: 0 6px; margin: 2px 3px 0 0; }
.acc-root .acc-message { min-height: 0; margin: 6px 0; font-size: var(--font-size-sm, 12px); color: var(--text-secondary, #aaa); }
.acc-root .acc-message:empty { display: none; }
.acc-root .acc-message.acc-error, .acc-root .acc-error { color: var(--error, #e74c3c); }
.acc-root .acc-message.acc-ok { color: var(--success, #27ae60); }
.acc-root .acc-warn { color: var(--warning, #f39c12); }
.acc-root .acc-cli-line { font-size: var(--font-size-xs, 11px); color: var(--text-muted, #666); margin: 0 0 6px; }
.acc-root .acc-host-line { font-size: var(--font-size-sm, 12px); color: var(--text-secondary, #aaa); margin: 0 0 10px; }
.acc-root .acc-jobs { display: grid; gap: 8px; grid-template-columns: repeat(auto-fill, minmax(min(100%, 320px), 1fr)); }
.acc-root .acc-jobs:empty { display: none; }
.acc-root .acc-job { border: 1px solid var(--ui-border-2, rgba(255,255,255,.14)); border-radius: var(--radius-md, 8px); background: var(--bg-card, var(--bg-secondary, #16213e)); padding: 10px 12px; display: grid; gap: 6px; }
.acc-root .acc-job-head { display: flex; justify-content: space-between; gap: 8px; align-items: center; }
.acc-root .acc-job-title { font-weight: 600; overflow-wrap: anywhere; }
.acc-root .acc-progress { height: 6px; border-radius: 999px; background: var(--ui-surface-3, rgba(0,0,0,.25)); overflow: hidden; position: relative; }
.acc-root .acc-progress > span { display: block; height: 100%; background: var(--info, #60a5fa); transition: width .4s ease; }
.acc-root .acc-progress.acc-indeterminate > span { width: 30% !important; position: absolute; animation: acc-slide 1.4s ease-in-out infinite; }
@keyframes acc-slide { 0% { left: -30%; } 100% { left: 100%; } }
.acc-root .acc-progress-inline { display: inline-flex; gap: 6px; align-items: center; min-width: 140px; }
.acc-root .acc-progress-inline .acc-progress { flex: 1; }
.acc-root .acc-job pre { max-height: 160px; overflow: auto; margin: 4px 0 0; padding: 6px 8px; background: var(--ui-code-block-bg, rgba(0,0,0,.25)); border-radius: var(--radius-sm, 4px); white-space: pre-wrap; }
.acc-root details summary { cursor: pointer; color: var(--text-secondary, #aaa); font-size: var(--font-size-xs, 11px); }
.acc-root .acc-cards { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(min(100%, 300px), 1fr)); }
.acc-root .acc-card { border: 1px solid var(--ui-border-1, rgba(255,255,255,.1)); border-radius: var(--radius-lg, 10px); background: var(--bg-card, var(--bg-secondary, #16213e)); padding: 14px 16px; min-width: 0; }
.acc-root .acc-card h3 { margin-bottom: 8px; }
.acc-root dl.acc-kv { display: grid; grid-template-columns: minmax(0, max-content) minmax(0, 1fr); gap: 3px 12px; margin: 0; font-size: var(--font-size-sm, 12px); }
.acc-root dl.acc-kv dt { color: var(--text-secondary, #aaa); min-width: 0; overflow-wrap: anywhere; }
.acc-root dl.acc-kv dd { margin: 0; min-width: 0; overflow-wrap: anywhere; }
/* Status chips in a key/value card (Overview Engines) never widen the card:
   they wrap onto the next line and, alone, shrink with an ellipsis. */
.acc-root dl.acc-kv dd .acc-badge { max-width: 100%; overflow: hidden; text-overflow: ellipsis; vertical-align: middle; }
.acc-root .acc-modal-backdrop, .acc-root.acc-modal-backdrop { position: fixed; inset: 0; background: rgba(0,0,0,.55); display: grid; place-items: center; z-index: var(--z-connect-modal, 1000); padding: max(16px, var(--safe-top, 0px)) max(16px, var(--safe-right, 0px)) max(16px, var(--safe-bottom, 0px)) max(16px, var(--safe-left, 0px)); overscroll-behavior: contain; }
.acc-root .acc-modal { width: min(640px, 100%); max-height: calc(var(--vh-full, 100vh) - 32px); overflow: auto; overscroll-behavior: contain; background: var(--bg-secondary, #16213e); color: var(--text-primary, #eee); border: 1px solid var(--ui-border-2, rgba(255,255,255,.14)); border-radius: var(--radius-lg, 10px); box-shadow: var(--ui-shadow-1, 0 10px 30px rgba(0,0,0,.35)); padding: 18px 20px; display: grid; gap: 10px; }
.acc-root .acc-modal p { margin: 0; }
.acc-root .acc-modal ul { margin: 0; padding-left: 18px; }
.acc-root .acc-modal-actions { display: flex; gap: 8px; justify-content: flex-end; flex-wrap: wrap; margin-top: 4px; }
.acc-root .acc-note { font-size: var(--font-size-sm, 12px); color: var(--warning, #f39c12); }
.acc-root .acc-form { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 220px), 1fr)); gap: 8px 12px; }
.acc-root .acc-form label { display: grid; gap: 3px; font-size: var(--font-size-sm, 12px); color: var(--text-secondary, #aaa); }
.acc-root ul.acc-entries { list-style: none; margin: 8px 0; padding: 0; display: grid; gap: 4px; }
.acc-root ul.acc-entries li { display: flex; justify-content: space-between; gap: 8px; align-items: center; font-size: var(--font-size-sm, 12px); }
.acc-root ul.acc-entries li > :first-child { min-width: 0; overflow-wrap: anywhere; }
/* Responsive (DESIGN.md breakpoints; ui-kit >= 0.3.0 tokens, with fallbacks
   so the fragment also works on a host page without them). */
@media (max-width: 767.98px), (max-height: 500px) {
  .acc-root .acc-modal-backdrop, .acc-root.acc-modal-backdrop { place-items: end center; padding: var(--safe-top, 0px) 0 0; }
  .acc-root .acc-modal { width: 100%; max-height: calc(var(--vh-full, 100vh) - var(--safe-top, 0px)); border-radius: var(--radius-lg, 10px) var(--radius-lg, 10px) 0 0; border-bottom: 0; padding-bottom: max(18px, var(--safe-bottom, 0px)); }
}
@media (max-width: 479.98px) {
  .acc-root .acc-modal-actions { flex-direction: column; align-items: stretch; }
  .acc-root .acc-modal-actions .acc-btn { width: 100%; }
}
@media (pointer: coarse) {
  .acc-root .acc-btn, .acc-root .acc-toolbar label, .acc-root details summary { min-height: var(--tap-min, 44px); }
  .acc-root .acc-btn { display: inline-flex; align-items: center; justify-content: center; }
  .acc-root .acc-btn.acc-link { min-height: var(--tap-min, 44px); }
  .acc-root .acc-collapse { min-height: var(--tap-min, 44px); }
  .acc-root input[type=search], .acc-root input[type=password], .acc-root input[type=text], .acc-root select { font-size: var(--font-size-input, 16px); min-height: var(--tap-min, 44px); }
  .acc-root input[type=checkbox], .acc-root input[type=radio] { width: 20px; height: 20px; }
  .acc-root, .acc-root table.acc-table, .acc-root .acc-host-line, .acc-root .acc-message, .acc-root .acc-toolbar label, .acc-root dl.acc-kv, .acc-root ul.acc-entries li, .acc-root .acc-form label { font-size: var(--font-size-body, 14px); }
  .acc-root .acc-table th, .acc-root .acc-table td { padding: 8px 10px; }
  .acc-root .acc-hint { font-size: var(--font-size-body, 14px); }
  .acc-root .acc-sub, .acc-root .acc-cli-line { font-size: var(--font-size-md, 13px); }
  .acc-root .acc-chip, .acc-root .acc-badge { font-size: var(--font-size-sm, 12px); }
  /* Links that act as buttons (overview "Manage engines" / "Browse models"): 44 px rows. */
  .acc-root .acc-sub a, .acc-root .acc-card a { display: inline-flex; align-items: center; min-height: var(--tap-min, 44px); }
}
"""

_PAGE_CSS = """
* { box-sizing: border-box; }
html, body { margin: 0; min-height: 100%; }
html { -webkit-text-size-adjust: 100%; text-size-adjust: 100%; }
body { min-height: var(--vh-full, 100vh); background: var(--bg-primary, #1a1a2e); color: var(--text-primary, #eee); font: var(--font-size-base, 14px)/1.5 var(--font-sans, system-ui, sans-serif); -webkit-font-smoothing: antialiased; }
.acc-topbar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; padding: max(10px, var(--safe-top, 0px)) max(20px, var(--safe-right, 0px)) 10px max(20px, var(--safe-left, 0px)); background: var(--bg-secondary, #16213e); border-bottom: 1px solid var(--ui-border-1, rgba(255,255,255,.1)); }
.acc-brand { font-weight: 700; font-size: var(--font-size-lg, 16px); margin-right: auto; }
.acc-brand small { font-weight: 400; color: var(--text-secondary, #aaa); margin-left: 6px; }
.acc-health { display: inline-flex; gap: 6px; align-items: center; font-size: var(--font-size-sm, 12px); color: var(--text-secondary, #aaa); }
.acc-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-muted, #666); display: inline-block; }
.acc-dot.acc-ok { background: var(--success, #27ae60); }
.acc-dot.acc-err { background: var(--error, #e74c3c); }
.acc-topbar select, .acc-topbar button:not(.af-switch) { background: var(--ui-surface-1, rgba(0,0,0,.16)); color: var(--text-primary, #eee); border: 1px solid var(--ui-border-2, rgba(255,255,255,.14)); border-radius: var(--radius-sm, 4px); padding: 4px 8px; font: inherit; font-size: var(--font-size-sm, 12px); cursor: pointer; }
.acc-tabs { display: flex; gap: 2px; padding: 0 max(20px, var(--safe-right, 0px)) 0 max(20px, var(--safe-left, 0px)); background: var(--bg-secondary, #16213e); border-bottom: 1px solid var(--ui-border-1, rgba(255,255,255,.1)); overflow-x: auto; scrollbar-width: none; overscroll-behavior-x: contain; }
.acc-tabs button { background: transparent; color: var(--text-secondary, #aaa); border: 0; border-bottom: 2px solid transparent; padding: 10px 14px; font: inherit; cursor: pointer; white-space: nowrap; }
.acc-tabs button[aria-selected="true"] { color: var(--text-primary, #eee); border-bottom-color: var(--accent, #e94560); }
.acc-main { padding: 18px max(20px, var(--safe-right, 0px)) max(40px, var(--safe-bottom, 0px)) max(20px, var(--safe-left, 0px)); max-width: 1400px; margin: 0 auto; min-width: 0; }
.acc-panel[hidden] { display: none; }
/* Phones: the top bar keeps brand + health on one row, theme controls on the
   next; the tab strip scrolls sideways (never widens the page). */
@media (max-width: 767.98px) {
  .acc-topbar { padding-left: max(var(--gutter, 16px), var(--safe-left, 0px)); padding-right: max(var(--gutter, 16px), var(--safe-right, 0px)); gap: 8px 12px; }
  .acc-brand small { display: none; }
  .acc-tabs { padding-left: max(var(--gutter, 16px), var(--safe-left, 0px)); padding-right: max(var(--gutter, 16px), var(--safe-right, 0px)); }
  .acc-tabs button { padding: 10px 12px; }
  .acc-main { padding: var(--gutter, 16px) max(var(--gutter, 16px), var(--safe-right, 0px)) max(32px, var(--safe-bottom, 0px)) max(var(--gutter, 16px), var(--safe-left, 0px)); }
  /* Tables keep readable columns and scroll sideways inside their
     .acc-table-scroll wrapper (a 72-character artifact id wrapped over 16
     lines at 375 px). Long artifact ids stay on one line with an ellipsis;
     the full id is the cell's title. Standalone page only: the gateway
     stacks the embedded tables into cards itself. */
  .acc-main .acc-table-scroll > table.acc-table { min-width: 640px; }
  /* The meta line beside a toolbar's buttons gets its own full-width row. */
  .acc-main .acc-toolbar > span.acc-muted { flex: 1 1 100%; }
  .acc-main code.acc-artifact-id { display: block; max-width: 34ch; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; overflow-wrap: normal; }
}
@media (max-height: 500px) {
  .acc-topbar { padding-top: max(6px, var(--safe-top, 0px)); padding-bottom: 6px; }
  .acc-tabs button { padding-top: 6px; padding-bottom: 6px; }
}
@media (pointer: coarse) {
  .acc-topbar select, .acc-topbar button, .acc-tabs button { min-height: var(--tap-min, 44px); }
  .acc-topbar select { font-size: var(--font-size-input, 16px); }
  /* WebKit ignores min-height on native-appearance selects (23 px on iOS
     whatever the CSS says): drop the native look on touch and draw the
     chevron (the kit 0.3.2 rule; this standalone page does not load the kit's
     component CSS; embedded, the gateway page styles its selects itself). */
  /* .acc-root select: the fragment's later `background` shorthand would erase the chevron. */
  select:not([multiple]):where(:not([size]), [size="1"]), .acc-root select:not([multiple]):where(:not([size]), [size="1"]) { appearance: none; -webkit-appearance: none; min-height: var(--tap-min, 44px); font-size: var(--font-size-input, 16px); padding: 0 32px 0 10px; background-image: var(--af-select-chevron, url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1.5l5 5 5-5' fill='none' stroke='%23aaa' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E")); background-repeat: no-repeat; background-position: right 10px center; }
}
"""

_EMAIL_CSS = """
/* Email tab (standalone page only; DESIGN §3/§6/§12). Layout, cards and tabs
   are the kit's af-form / af-card / af-tabs rules (KIT_FORM_CSS, generated);
   this block only styles the console's own controls inside them. One type
   scale: card titles = the console's section heading, labels and switch
   labels = body size weight 500, helper text = sm muted. */
.acc-email.af-form { margin: 0 auto; }
.acc-email .acc-message { margin: 0; }
.acc-email input[type=text], .acc-email input[type=email], .acc-email input[type=password], .acc-email select {
  font-size: var(--font-size-base, 14px); font-weight: 400; padding: 8px 10px; min-height: 36px; font-family: inherit;
  background: var(--ui-surface-1, rgba(0,0,0,.16)); color: var(--text-primary, #eee);
  border: 1px solid var(--ui-border-2, rgba(255,255,255,.14)); border-radius: var(--radius-sm, 4px); }
.acc-email input[aria-invalid="true"] { border-color: var(--error, #e74c3c); }
.acc-email input:disabled { opacity: .6; cursor: not-allowed; }
.acc-email input[readonly] { opacity: .7; }
.acc-email .acc-btn { font-size: var(--font-size-base, 14px); padding: 7px 14px; min-height: 36px; }
.acc-email .acc-btn.acc-link { padding: 2px 4px; min-height: 0; }
.acc-email .acc-btn[aria-disabled="true"] { opacity: .55; cursor: not-allowed; }
.acc-email .acc-btn[aria-busy="true"] { cursor: progress; opacity: .8; }
.acc-email .af-form__actions { justify-content: flex-start; }
.acc-email .af-tabs { gap: var(--space-3, 12px); }
.acc-email [data-acc="email-connected"] { display: flex; flex-direction: column; gap: var(--space-3, 12px); }
.acc-email [data-acc="email-connected"][hidden] { display: none; }
.acc-email .af-tabs__panel { display: flex; flex-direction: column; gap: var(--space-4, 16px); }
.acc-email .af-tabs__panel[hidden] { display: none; }
.acc-email details.acc-disclosure > summary { color: var(--text-secondary, #aaa); font-size: var(--font-size-base, 14px); cursor: pointer; }
.acc-email details.acc-disclosure[open] > summary { margin-bottom: var(--space-4, 16px); }
.acc-email details.acc-disclosure > .af-form__field + .af-form__field,
.acc-email details.acc-disclosure > * + * { margin-top: var(--space-4, 16px); }
/* IMAP pane: the incoming and outgoing servers, each on one row from 768px up
   (Server | Port | Security), stacked below (Server, then Port + Security). */
.acc-email fieldset.acc-server-row { margin: 0; padding: 0; border: 0; min-width: 0; display: flex; flex-direction: column; gap: 6px; }
.acc-email fieldset.acc-server-row > legend { padding: 0; margin: 0 0 6px; }
.acc-email .acc-server-fields { display: grid; grid-template-columns: minmax(0, 1fr) 96px 132px; gap: var(--space-3, 12px); min-width: 0; }
.acc-email .acc-field-caption { margin: 0; color: var(--text-secondary, #aaa); font-size: max(var(--font-size-sm, 12px), 13px); font-weight: 400; line-height: 1.3; }
.acc-email .acc-source-line { overflow-wrap: anywhere; }
.acc-email .acc-fixed-address { margin: 0 0 12px; overflow-wrap: anywhere; }
.acc-email .acc-fixed-address .acc-btn.acc-link, .acc-email .acc-own-link { padding: 2px 0; }
.acc-email .acc-own-link { align-self: flex-start; font-size: max(var(--font-size-sm, 12px), 13px); }
.acc-email .acc-more-links { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: -6px 0 0; }
.acc-email .acc-more-links .acc-btn.acc-link { font-size: max(var(--font-size-sm, 12px), 13px); padding: 2px 0; }
.acc-email .acc-more-links .acc-btn.acc-link[aria-expanded="true"] { display: none; }
@media (max-width: 767.98px) {
  .acc-email .acc-server-fields { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
  .acc-email .acc-server-fields > .acc-server-host { grid-column: 1 / -1; }
}
.acc-email .acc-status-line { margin: 0; font-size: var(--font-size-base, 14px); line-height: 1.5; overflow-wrap: anywhere; }
.acc-email .acc-status-line .acc-status-error { color: var(--error, #e74c3c); }
.acc-email .acc-confirm { display: flex; flex-direction: column; gap: 10px; padding: 10px 12px; border: 1px solid var(--error-border, rgba(231,76,60,.35)); background: var(--error-subtle, rgba(231,76,60,.12)); border-radius: var(--radius-md, 8px); font-size: var(--font-size-base, 14px); }
.acc-email .acc-confirm[hidden] { display: none; }
.acc-email details.acc-advanced > summary { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px; list-style: none; cursor: pointer; }
.acc-email details.acc-advanced > summary::-webkit-details-marker { display: none; }
.acc-email details.acc-advanced > summary::before { content: "\\25B8"; color: var(--text-secondary, #aaa); font-size: var(--font-size-lg, 16px); }
.acc-email details.acc-advanced[open] > summary::before { content: "\\25BE"; }
.acc-email .acc-advanced-body { display: flex; flex-direction: column; gap: var(--space-4, 16px); margin-top: var(--space-3, 12px); }
/* Advanced: compact sentences with the controls inline (DESIGN v2 §3.5). */
.acc-email .acc-sentence-block { display: flex; flex-direction: column; gap: 8px; border-top: 1px solid var(--ui-border-1, rgba(255,255,255,.1)); padding-top: var(--space-4, 16px); min-width: 0; }
.acc-email .acc-sentence { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; margin: 0; font-size: var(--font-size-base, 14px); line-height: 1.5; }
.acc-email .acc-sentence label { font-weight: 500; }
.acc-email .acc-sentence select { width: auto; max-width: 100%; }
.acc-email .acc-sentence input.acc-num { width: 4.5em; text-align: right; }
.acc-email .acc-sentence input.acc-word { width: 12em; max-width: 100%; }
.acc-email .acc-saved { color: var(--success, #27ae60); font-size: max(var(--font-size-sm, 12px), 13px); font-weight: 400; }
.acc-email ul.acc-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 0; padding: 0; list-style: none; }
.acc-email ul.acc-chips li { display: inline-flex; align-items: center; gap: 2px; padding: 2px 4px 2px 10px; border: 1px solid var(--ui-border-2, rgba(255,255,255,.14)); border-radius: 999px; font-size: var(--font-size-base, 14px); max-width: 100%; overflow-wrap: anywhere; }
.acc-email ul.acc-chips li.acc-muted { border: 0; padding: 0; color: var(--text-muted, #999); }
.acc-email ul.acc-chips li .acc-btn.acc-link { padding: 0 6px; min-height: 0; line-height: 1.4; }
.acc-email .acc-chip-add { max-width: 420px; }
.acc-email .acc-list-title { margin: 4px 0 0; font-size: var(--font-size-base, 14px); font-weight: 600; }
.acc-email ul.acc-chips li.acc-chip-fixed { padding-right: 10px; }
.acc-email .af-switch--row { padding-top: 0; padding-bottom: 0; }
.acc-email .af-switch__desc { font-size: max(var(--font-size-sm, 12px), 13px); }
.acc-email .acc-cli-line { margin: 0; }
/* Phones (DESIGN §12): the full width; cards become flat sections with a
   hairline divider (no border, no card padding, no card in a card). */
@media (max-width: 767.98px) {
  .acc-email .af-card { border: 0; border-radius: 0; background: transparent; padding: var(--space-4, 16px) 0 0; border-top: 1px solid var(--ui-border-1, rgba(255,255,255,.1)); }
  .acc-email .acc-confirm { border-radius: var(--radius-sm, 4px); }
}
@media (pointer: coarse) {
  .acc-email input[type=text], .acc-email input[type=email], .acc-email input[type=password], .acc-email select { font-size: var(--font-size-input, 16px); min-height: var(--tap-min, 44px); }
  .acc-email .acc-btn, .acc-email .acc-btn.acc-link, .acc-email .acc-more-links .acc-btn.acc-link, .acc-email ul.acc-chips li .acc-btn.acc-link { min-height: var(--tap-min, 44px); }
  .acc-email details > summary { min-height: var(--tap-min, 44px); display: flex; align-items: center; }
  .acc-email .af-form__help, .acc-email .acc-field-caption, .acc-email .af-switch__desc, .acc-email .acc-saved, .acc-email .acc-more-links .acc-btn.acc-link, .acc-email .acc-own-link { font-size: max(var(--font-size-sm, 12px), 14px); }
  .acc-email .acc-sentence input.acc-num, .acc-email .acc-sentence input.acc-word, .acc-email .acc-sentence select { min-height: var(--tap-min, 44px); }
}
"""

# ---------------------------------------------------------------------------
# HTML templates. `data-acc` attributes are the roles the JS binds to (all
# lookups are scoped to the mounted root, so the ids below are only for the
# host page's convenience and for tests).
# ---------------------------------------------------------------------------

_MODELS_HTML = """<div class="acc-root" data-acc-kind="models" id="acc-models">
  <div class="acc-host-line" data-acc="host-line">Loading host profile...</div>
  <div class="acc-toolbar" role="search">
    <input type="search" data-acc="q" placeholder="Search models (press /)" aria-label="Search models">
    <label>Engine <select data-acc="engine" aria-label="Engine"><option value="">All engines</option></select></label>
    <label>Modality <select data-acc="modality" aria-label="Modality"><option value="">Any</option><option value="text">Text</option><option value="vision">Vision</option><option value="audio">Audio</option><option value="embedding">Embedding</option></select></label>
    <label title="Hide artifacts whose fit verdict is too large (f)"><input type="checkbox" data-acc="fits" checked> Fits this machine</label>
    <label><input type="checkbox" data-acc="installed-only"> Installed only</label>
    <label title="Also search the Hugging Face Hub (slower)"><input type="checkbox" data-acc="hub"> Search Hugging Face</label>
    <button type="button" class="acc-btn" data-acc-action="refresh" title="Refresh (r)">Refresh</button>
  </div>
  <div class="acc-cli-line">CLI equivalent: <code data-acc="view-cli">abstractcore models catalog --fits</code></div>
  <div class="acc-message" data-acc="message" role="status" aria-live="polite"></div>
  <section class="acc-jobs" data-acc="jobs" aria-label="Active jobs"></section>
  <section class="acc-section">
    <div class="acc-section-head"><h3><button type="button" class="acc-collapse" aria-expanded="true" data-acc-action="collapse" data-acc-section="catalog">Catalog</button></h3><span class="acc-muted" data-acc="catalog-count"></span></div>
    <div class="acc-table-scroll"><table class="acc-table" data-acc="catalog-table">
      <thead><tr><th>Model</th><th>Provider</th><th>Artifact</th><th>Quant</th><th>Download size</th><th>Weights</th><th>Fit</th><th>Actions</th></tr></thead>
      <tbody data-acc="catalog-body"><tr><td colspan="8" class="acc-muted">Loading catalog...</td></tr></tbody>
    </table></div>
  </section>
  <section class="acc-section">
    <div class="acc-section-head"><h3><button type="button" class="acc-collapse" aria-expanded="true" data-acc-action="collapse" data-acc-section="installed">Installed</button></h3><span class="acc-muted" data-acc="installed-summary"></span></div>
    <div class="acc-message acc-warn" data-acc="installed-errors"></div>
    <div class="acc-table-scroll"><table class="acc-table" data-acc="installed-table">
      <thead><tr><th>Provider</th><th>Artifact</th><th>Quant</th><th>Size</th><th>Location</th><th>Loaded</th><th>Actions</th></tr></thead>
      <tbody data-acc="installed-body"><tr><td colspan="7" class="acc-muted">Loading installed models...</td></tr></tbody>
    </table></div>
  </section>
  <p class="acc-hint">Keys: / search, f fits-only, r refresh; on a focused row: w download, d delete.</p>
</div>"""

_ENGINES_HTML = """<div class="acc-root" data-acc-kind="engines" id="acc-engines">
  <div class="acc-toolbar">
    <button type="button" class="acc-btn" data-acc-action="refresh" title="Refresh with a live probe (r)">Refresh</button>
    <span class="acc-muted" data-acc="engines-meta"></span>
  </div>
  <div class="acc-cli-line">CLI equivalent: <code data-acc="view-cli">abstractcore engines status --probe</code></div>
  <div class="acc-message" data-acc="message" role="status" aria-live="polite"></div>
  <section class="acc-jobs" data-acc="jobs" aria-label="Active jobs"></section>
  <div class="acc-table-scroll"><table class="acc-table" data-acc="engines-table">
    <thead><tr><th>Engine</th><th>Supported</th><th>Installed</th><th>Version</th><th>Running</th><th>Reachable</th><th>Base URL</th><th>Models</th><th>Actions</th></tr></thead>
    <tbody data-acc="engines-body"><tr><td colspan="9" class="acc-muted">Probing engines...</td></tr></tbody>
  </table></div>
  <p class="acc-hint">Keys: r refresh; on a focused row: i install, o open download page.</p>
</div>"""

_OVERVIEW_HTML = """<div class="acc-root" data-acc-kind="overview" id="acc-overview">
  <div class="acc-message" data-acc="message" role="status" aria-live="polite"></div>
  <div class="acc-cards">
    <section class="acc-card" data-acc="host-card"><h3>This machine</h3><div data-acc="host-body" class="acc-muted">Loading host profile...</div></section>
    <section class="acc-card" data-acc="engines-card"><h3>Engines</h3><div data-acc="engines-body" class="acc-muted">Probing engines...</div></section>
    <section class="acc-card" data-acc="server-card"><h3>Server</h3><div data-acc="server-body" class="acc-muted">Checking server health...</div></section>
  </div>
  <div class="acc-cli-line" style="margin-top:12px">CLI equivalent: <code>abstractcore host profile --json</code>, <code>abstractcore engines status --probe</code></div>
</div>"""

_PROVIDERS_HTML = """<div class="acc-root" data-acc-kind="providers" id="acc-providers">
  <div class="acc-toolbar">
    <button type="button" class="acc-btn" data-acc-action="refresh" title="Refresh (r)">Refresh</button>
    <span class="acc-muted">Read-only. API keys and defaults are managed on the host with <code>abstractcore --config</code>.</span>
  </div>
  <div class="acc-message" data-acc="message" role="status" aria-live="polite"></div>
  <section class="acc-section">
    <div class="acc-section-head"><h3><button type="button" class="acc-collapse" aria-expanded="true" data-acc-action="collapse" data-acc-section="providers">Providers</button></h3><span class="acc-muted" data-acc="providers-count"></span></div>
    <div class="acc-table-scroll"><table class="acc-table" data-acc="providers-table">
      <thead><tr><th>Provider</th><th>Type</th><th>Where</th><th>Status</th><th>API key</th><th>Features</th></tr></thead>
      <tbody data-acc="providers-body"><tr><td colspan="6" class="acc-muted">Loading providers...</td></tr></tbody>
    </table></div>
  </section>
  <section class="acc-section">
    <div class="acc-section-head"><h3><button type="button" class="acc-collapse" aria-expanded="true" data-acc-action="collapse" data-acc-section="defaults">Capability defaults</button></h3><span class="acc-muted" data-acc="defaults-source"></span></div>
    <div class="acc-message acc-warn" data-acc="defaults-message"></div>
    <div class="acc-table-scroll"><table class="acc-table" data-acc="defaults-table">
      <thead><tr><th>Capability</th><th>Provider</th><th>Model</th><th>Source</th></tr></thead>
      <tbody data-acc="defaults-body"><tr><td colspan="4" class="acc-muted">Loading capability defaults...</td></tr></tbody>
    </table></div>
  </section>
  <div class="acc-cli-line" style="margin-top:12px">CLI equivalent: <code>abstractcore config providers --probe</code>, <code>abstractcore config defaults</code></div>
</div>"""

_EMAIL_HTML = """<div class="acc-root acc-email af-form" data-acc-kind="email" id="acc-email">
  <div class="acc-message" data-acc="message" role="status" aria-live="polite"></div>
  <div class="acc-message acc-warn" data-acc="email-notices"></div>
  <section class="af-card" data-acc="email-identity-card" aria-labelledby="acc-email-identity-h">
    <div class="af-card__header"><h3 class="af-card__title" id="acc-email-identity-h">Email address</h3></div>
    <div class="af-form__field" data-acc="email-identity-field">
      <div class="af-form__inline">
        <input type="email" id="acc-email-registered" data-acc="email-registered" autocomplete="email" aria-labelledby="acc-email-identity-h" aria-describedby="acc-email-identity-help acc-email-identity-error">
        <button type="button" class="acc-btn" data-acc-action="identity-save" data-acc="email-identity-save">Save</button>
      </div>
      <p class="af-form__help" id="acc-email-identity-help">The first address your agents may write to. Empty = the mailbox's own address.</p>
      <p class="af-form__help" data-acc="email-identity-differs" hidden></p>
      <p class="af-form__error" id="acc-email-identity-error" data-acc="email-identity-error" role="alert" hidden></p>
    </div>
    <p class="af-form__help" data-acc="email-identity-pending" hidden>The first address your agents may write to. Connecting a mailbox below sets it to the mailbox's address.</p>
  </section>
  <section class="af-card" data-acc="email-mailbox-card" aria-labelledby="acc-email-mailbox-h">
    <div class="af-card__header"><h3 class="af-card__title" id="acc-email-mailbox-h">Mailbox</h3></div>
    <div class="af-tabs" data-acc="email-setup">
      <p class="af-card__desc">Connect a mailbox so your agents and automations can read and send mail as you.</p>
      <div class="af-tabs__list" role="tablist" aria-label="How to connect the mailbox">
        <button type="button" class="af-tabs__tab" role="tab" id="acc-email-tab-imap" data-acc-tab="imap" aria-controls="acc-email-panel-imap" aria-selected="true">IMAP</button>
        <button type="button" class="af-tabs__tab" role="tab" id="acc-email-tab-google" data-acc-tab="google" aria-controls="acc-email-panel-oauth" aria-selected="false" tabindex="-1">Google</button>
        <button type="button" class="af-tabs__tab" role="tab" id="acc-email-tab-microsoft" data-acc-tab="microsoft" aria-controls="acc-email-panel-oauth" aria-selected="false" tabindex="-1">Microsoft</button>
      </div>
      <div class="af-tabs__panel" role="tabpanel" id="acc-email-panel-imap" data-acc="email-panel-imap" aria-labelledby="acc-email-tab-imap">
        <p class="acc-fixed-address" data-acc="email-address-fixed" hidden>Mailbox account: <strong data-acc="email-address-fixed-value"></strong> — <button type="button" class="acc-btn acc-link" data-acc-action="mailbox-address-reveal" aria-controls="acc-email-address-field acc-email-oauth-address-field" aria-expanded="false">Use a different account</button></p>
        <div class="af-form__field" id="acc-email-address-field" data-acc="email-address-field">
          <label class="af-form__label" for="acc-email-address">Mailbox address</label>
          <input type="email" id="acc-email-address" data-acc="email-address" autocomplete="email" aria-describedby="acc-email-address-help">
          <p class="af-form__help" id="acc-email-address-help">The account your agents read and send from — usually your own address.</p>
          <button type="button" class="acc-btn acc-link acc-own-link" data-acc-action="mailbox-address-own" data-acc="email-address-own" hidden></button>
        </div>
        <div class="af-form__field">
          <label class="af-form__label" for="acc-email-password">Password</label>
          <input type="password" id="acc-email-password" data-acc="email-password" autocomplete="current-password" aria-describedby="acc-email-password-help">
          <p class="af-form__help" id="acc-email-password-help">Use an app password if your provider needs one. Stored encrypted, never shown again.</p>
        </div>
        <fieldset class="acc-server-row" data-acc="email-imap-row">
          <legend class="af-form__label">Incoming mail (IMAP)</legend>
          <div class="acc-server-fields">
            <div class="af-form__field acc-server-host"><label class="acc-field-caption" for="acc-email-imap-host">Server</label><input type="text" id="acc-email-imap-host" data-acc="email-imap-host" autocomplete="off" spellcheck="false"></div>
            <div class="af-form__field acc-server-port"><label class="acc-field-caption" for="acc-email-imap-port">Port</label><input type="text" id="acc-email-imap-port" data-acc="email-imap-port" inputmode="numeric"></div>
            <div class="af-form__field acc-server-sec"><label class="acc-field-caption" for="acc-email-imap-security">Security</label><select id="acc-email-imap-security" data-acc="email-imap-security"><option value="ssl">SSL</option><option value="starttls">STARTTLS</option></select></div>
          </div>
        </fieldset>
        <fieldset class="acc-server-row" data-acc="email-smtp-row">
          <legend class="af-form__label">Outgoing mail (SMTP)</legend>
          <div class="acc-server-fields">
            <div class="af-form__field acc-server-host"><label class="acc-field-caption" for="acc-email-smtp-host">Server</label><input type="text" id="acc-email-smtp-host" data-acc="email-smtp-host" autocomplete="off" spellcheck="false"></div>
            <div class="af-form__field acc-server-port"><label class="acc-field-caption" for="acc-email-smtp-port">Port</label><input type="text" id="acc-email-smtp-port" data-acc="email-smtp-port" inputmode="numeric"></div>
            <div class="af-form__field acc-server-sec"><label class="acc-field-caption" for="acc-email-smtp-security">Security</label><select id="acc-email-smtp-security" data-acc="email-smtp-security"><option value="ssl">SSL</option><option value="starttls">STARTTLS</option></select></div>
          </div>
        </fieldset>
        <p class="af-form__help acc-source-line" data-acc="email-servers-source" role="status" aria-live="polite">Enter the mailbox address: the servers fill in from it.</p>
        <div class="acc-more-links">
          <button type="button" class="acc-btn acc-link" data-acc-action="login-reveal" data-acc="email-login-link" aria-controls="acc-email-login-field" aria-expanded="false">My provider uses a different login name</button>
          <button type="button" class="acc-btn acc-link" data-acc-action="ca-reveal" data-acc="email-ca-link" aria-controls="acc-email-ca-field" aria-expanded="false">Custom certificate</button>
        </div>
        <div class="af-form__field" id="acc-email-login-field" data-acc="email-login-field" hidden>
          <label class="af-form__label" for="acc-email-username">Login</label>
          <input type="text" id="acc-email-username" data-acc="email-username" autocomplete="username" spellcheck="false" aria-describedby="acc-email-username-help">
          <p class="af-form__help" id="acc-email-username-help">The name you sign in to your mail with, when it is not the mailbox address.</p>
        </div>
        <div class="af-form__field" id="acc-email-ca-field" data-acc="email-ca-field" hidden>
          <label class="af-form__label" for="acc-email-ca-file">CA file</label>
          <input type="text" id="acc-email-ca-file" data-acc="email-ca-file" spellcheck="false" aria-describedby="acc-email-ca-help">
          <p class="af-form__help" id="acc-email-ca-help">A PEM file for a mail server signed by a private CA; empty = the system trust store.</p>
        </div>
        <div class="af-form__actions">
          <button type="button" class="acc-btn acc-primary" data-acc-action="email-connect" data-acc="email-connect">Connect</button>
        </div>
        <p class="af-form__error" data-acc="email-connect-error" role="alert" hidden></p>
        <p class="af-form__help">Connect tests reading and sending first and stores nothing if the test fails. The mailbox is only read: never marked read, moved or deleted.</p>
      </div>
      <div class="af-tabs__panel" role="tabpanel" id="acc-email-panel-oauth" data-acc="email-panel-oauth" aria-labelledby="acc-email-tab-google" hidden>
        <p class="acc-fixed-address" data-acc="email-oauth-address-fixed" hidden>Mailbox account: <strong data-acc="email-oauth-address-fixed-value"></strong> — <button type="button" class="acc-btn acc-link" data-acc-action="mailbox-address-reveal" aria-controls="acc-email-address-field acc-email-oauth-address-field" aria-expanded="false">Use a different account</button></p>
        <div class="af-form__field" id="acc-email-oauth-address-field" data-acc="email-oauth-address-field">
          <label class="af-form__label" for="acc-email-oauth-address">Mailbox address</label>
          <input type="email" id="acc-email-oauth-address" data-acc="email-oauth-address" autocomplete="email" aria-describedby="acc-email-oauth-help">
          <p class="af-form__help" id="acc-email-oauth-help" data-acc="email-oauth-help">The Google sign-in page opens with this address.</p>
          <button type="button" class="acc-btn acc-link acc-own-link" data-acc-action="mailbox-address-own" data-acc="email-oauth-address-own" hidden></button>
        </div>
        <div class="af-form__actions">
          <button type="button" class="acc-btn acc-primary" data-acc-action="oauth-start" data-acc="email-oauth-start">Sign in with Google</button>
          <button type="button" class="acc-btn" data-acc-action="oauth-cancel" data-acc="email-oauth-cancel" hidden>Cancel sign-in</button>
        </div>
        <p class="af-form__help" id="acc-email-oauth-reason" data-acc="email-oauth-reason" hidden></p>
        <p class="af-form__error" data-acc="email-oauth-error" role="alert" hidden></p>
        <div class="acc-message" data-acc="email-oauth-prompt" role="status" aria-live="polite"></div>
        <details class="acc-disclosure" data-acc="email-oauth-advanced">
          <summary>Advanced: your own sign-in client</summary>
          <div class="af-form__grid-2">
            <div class="af-form__field"><label class="af-form__label" for="acc-email-oauth-client-id">Client ID</label><input type="text" id="acc-email-oauth-client-id" data-acc="email-oauth-client-id" autocomplete="off"></div>
            <div class="af-form__field"><label class="af-form__label" for="acc-email-oauth-client-secret">Client secret</label><input type="password" id="acc-email-oauth-client-secret" data-acc="email-oauth-client-secret" autocomplete="new-password"></div>
          </div>
          <div class="af-form__grid-2" data-acc="email-oauth-microsoft-fields">
            <div class="af-form__field"><label class="af-form__label" for="acc-email-oauth-tenant">Tenant</label><input type="text" id="acc-email-oauth-tenant" data-acc="email-oauth-tenant" autocomplete="off" aria-describedby="acc-email-oauth-tenant-help"><p class="af-form__help" id="acc-email-oauth-tenant-help">Empty = common (any Microsoft account).</p></div>
            <div class="af-form__field"><label class="af-form__label" for="acc-email-oauth-flow">Sign-in flow</label><select id="acc-email-oauth-flow" data-acc="email-oauth-flow"><option value="">Provider default</option><option value="device">Device code (any browser)</option><option value="loopback">Browser on this machine</option></select></div>
          </div>
          <p class="af-form__help">A client you registered with the provider for the mail scopes. Without one, the built-in AbstractFramework client signs in when this version has one. Tokens are stored encrypted.</p>
        </details>
      </div>
    </div>
    <div data-acc="email-connected" hidden>
      <p class="acc-status-line" data-acc="email-status"></p>
      <span class="af-switch-wrap af-switch-wrap--row">
        <button type="button" role="switch" class="af-switch af-switch--row" aria-checked="false" data-acc-action="mailbox-switch" data-acc="email-enabled" data-reason-id="acc-email-enabled-reason">
          <span class="af-switch__track" aria-hidden="true"><span class="af-switch__thumb"></span></span>
          <span class="af-switch__text"><span class="af-switch__label">Active</span><span class="af-switch__desc">Off pauses watching and sending; your settings are kept.</span></span>
        </button>
        <span id="acc-email-enabled-reason" class="af-switch__reason" hidden></span>
      </span>
      <div class="af-form__actions">
        <button type="button" class="acc-btn" data-acc-action="email-test" data-acc="email-test">Test</button>
        <button type="button" class="acc-btn acc-danger" data-acc-action="email-disconnect">Disconnect</button>
      </div>
      <div class="acc-confirm" data-acc="email-disconnect-confirm" role="group" aria-label="Confirm disconnect" hidden>
        <span>Disconnect this mailbox? Your agents lose email until you connect again. Policy and limits are kept.</span>
        <div class="af-form__actions">
          <button type="button" class="acc-btn acc-danger" data-acc-action="email-disconnect-confirm">Disconnect</button>
          <button type="button" class="acc-btn" data-acc-action="email-disconnect-cancel">Cancel</button>
        </div>
      </div>
      <p class="af-form__help" data-acc="email-storage"></p>
    </div>
  </section>
  <section class="af-card" data-acc="email-agent-card" aria-label="Agent email tools">
    <span class="af-switch-wrap af-switch-wrap--row">
      <button type="button" role="switch" class="af-switch af-switch--row" aria-checked="false" data-acc-action="agent-tools-switch" data-acc="email-agent-tools" data-reason-id="acc-email-agent-tools-reason">
        <span class="af-switch__track" aria-hidden="true"><span class="af-switch__thumb"></span></span>
        <span class="af-switch__text"><span class="af-switch__label">Agent email tools</span><span class="af-switch__desc" data-acc="email-agent-tools-desc">Your agents and workflows may list, search, read, send and reply to your mail. Every send still follows your recipient rules, your limits and the approval gate.</span></span>
      </button>
      <span id="acc-email-agent-tools-reason" class="af-switch__reason" hidden></span>
    </span>
  </section>
  <details class="af-card acc-advanced" data-acc="email-advanced">
    <summary><h3 class="af-card__title">Advanced</h3><span class="af-card__desc">Who your agents may write to, send limits, the folder they watch</span></summary>
    <div class="acc-advanced-body">
      <div class="acc-sentence-block" data-acc="email-policy-card">
        <p class="acc-sentence"><label for="acc-email-policy-mode">Your agents may send to</label> <select id="acc-email-policy-mode" data-acc="email-policy-mode" aria-describedby="acc-email-policy-help"><option value="allowlist">Only the Allowed list</option><option value="denylist">Anyone not on the Denied list</option></select> <span class="acc-saved" data-acc="email-policy-saved" role="status" aria-live="polite"></span></p>
        <p class="acc-list-title" id="acc-email-allow-title">Always allowed</p>
        <ul class="acc-chips" data-acc="email-allow-entries" aria-labelledby="acc-email-allow-title"></ul>
        <div class="af-form__inline acc-chip-add"><input type="text" id="acc-email-allow-new" data-acc="email-allow-new" autocomplete="off" spellcheck="false" placeholder="e.g. abstractframework.ai" aria-label="Address or domain to always allow"><button type="button" class="acc-btn" data-acc-action="policy-add-allow">Add</button></div>
        <p class="acc-list-title" id="acc-email-deny-title">Always denied</p>
        <ul class="acc-chips" data-acc="email-deny-entries" aria-labelledby="acc-email-deny-title"></ul>
        <div class="af-form__inline acc-chip-add"><input type="text" id="acc-email-deny-new" data-acc="email-deny-new" autocomplete="off" spellcheck="false" placeholder="e.g. xxx.gov" aria-label="Address or domain to always deny"><button type="button" class="acc-btn" data-acc-action="policy-add-deny">Add</button></div>
        <p class="af-form__help" id="acc-email-policy-help">Denied always wins. Your own address is always allowed. A domain also covers its subdomains. To, Cc and Bcc are all checked: a message with any refused recipient is not sent.</p>
      </div>
      <div class="acc-sentence-block" data-acc="email-limits-card">
        <p class="acc-sentence">Send at most <input type="text" class="acc-num" id="acc-email-per-hour" data-acc="email-per-hour" inputmode="numeric" aria-label="Messages per hour"> per hour and <input type="text" class="acc-num" id="acc-email-per-day" data-acc="email-per-day" inputmode="numeric" aria-label="Messages per day"> per day. <span class="acc-saved" data-acc="email-limits-saved" role="status" aria-live="polite"></span></p>
        <p class="af-form__help" data-acc="email-usage"></p>
      </div>
      <div class="acc-sentence-block" data-acc="email-folder-card">
        <p class="acc-sentence"><label for="acc-email-folder">Watch folder</label> <input type="text" class="acc-word" id="acc-email-folder" data-acc="email-folder" autocomplete="off" spellcheck="false" aria-describedby="acc-email-folder-help"> <span class="acc-saved" data-acc="email-folder-saved" role="status" aria-live="polite"></span></p>
        <p class="af-form__help" id="acc-email-folder-help" data-acc="email-folder-help">The folder your agents read. Empty = INBOX.</p>
      </div>
    </div>
  </details>
  <p class="af-form__help" data-acc="email-scope">These are the email settings of this AbstractCore install. A gateway keeps its own per user: each user sets theirs in the gateway console (My account).</p>
  <div class="acc-cli-line">CLI equivalent: <code>abstractcore email status</code>, <code>abstractcore email registered-address &lt;email address&gt;</code>, <code>printf '%s\\n' "$PASSWORD" | abstractcore email connect --address &lt;email address&gt; --password-stdin</code>, <code>abstractcore email connect --address &lt;email address&gt; --oauth google|microsoft</code>, <code>abstractcore email agent-tools on|off</code>, <code>abstractcore email policy set --mode allowlist --always-allow &lt;address or domain&gt; --always-deny &lt;address or domain&gt;</code>, <code>abstractcore email limits set --per-hour 100 --per-day 1000</code>, <code>abstractcore email folder &lt;name&gt;</code>, <code>abstractcore email enable|disable</code> (Active)</div>
</div>"""

_TEMPLATES: Dict[str, str] = {
    "models": _MODELS_HTML,
    "engines": _ENGINES_HTML,
    "overview": _OVERVIEW_HTML,
    "providers": _PROVIDERS_HTML,
    "email": _EMAIL_HTML,
}

# ---------------------------------------------------------------------------
# JavaScript: one IIFE, one global (window.AbstractCoreConsole).
# ---------------------------------------------------------------------------

_JS_TEMPLATE = r"""
(function () {
  "use strict";
  if (window.AbstractCoreConsole && window.AbstractCoreConsole.version === "__ACC_VERSION__") return;

  const TEMPLATES = __ACC_TEMPLATES__;
  const POLL_MS = 1500;
  const REQUEST_TIMEOUT_MS = 60000;
  const TOKEN_KEY = "abstractcore_console_token";
  const JOBS_KEY = "abstractcore_console_jobs_v1";
  const THEME_KEY = "abstractcore_console_theme_v1";
  const TERMINAL = new Set(["completed", "failed", "cancelled"]);

  // Shared vocabulary (contract G). Keep in sync with the TUI and gateway.
  const WEIGHT_LABELS = {
    installed: { label: "installed", tone: "ok" },
    absent: { label: "not downloaded", tone: "off" },
    unknown: { label: "unknown", tone: "muted" },
    not_applicable: { label: "remote", tone: "muted" },
  };
  const FIT_LABELS = {
    fits: { label: "fits", tone: "ok" },
    tight: { label: "tight", tone: "warn" },
    too_large: { label: "too large", tone: "err" },
    partial_offload: { label: "partial offload", tone: "warn" },
    // Fits once the Mac's GPU memory limit is raised; the fit notes (the
    // badge title) carry the exact sysctl command.
    needs_gpu_limit: { label: "needs GPU limit", tone: "warn" },
    unknown: { label: "unknown", tone: "muted" },
  };
  const JOB_LABELS = {
    queued: { label: "queued", tone: "info" },
    running: { label: "running", tone: "info" },
    completed: { label: "completed", tone: "ok" },
    failed: { label: "failed", tone: "err" },
    cancelled: { label: "cancelled", tone: "muted" },
  };
  const BLOCKER_TEXT = {
    loaded: "The model is loaded in memory right now.",
    remote_engine: "The engine runs on another machine; its files are not on this host.",
  };

  // ---------------------------------------------------------------- helpers
  function esc(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function isNum(n) { return typeof n === "number" && isFinite(n); }
  function fmtBytes(n) {
    if (!isNum(n)) return "?";
    if (n < 1024) return `${n} B`;
    const units = ["KiB", "MiB", "GiB", "TiB"];
    let v = n / 1024; let i = 0;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
    return `${v >= 100 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
  }
  function fmtParams(n) {
    if (!isNum(n)) return "";
    if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
    return String(n);
  }
  function shq(s) {
    const v = String(s === null || s === undefined ? "" : s);
    if (/^[A-Za-z0-9_\-.\/:@=+,%]+$/.test(v)) return v;
    return "'" + v.replace(/'/g, "'\\''") + "'";
  }
  function argvText(argv) { return Array.isArray(argv) ? argv.map(shq).join(" ") : ""; }
  function badge(view, title) {
    return `<span class="acc-badge acc-tone-${esc(view.tone)}"${title ? ` title="${esc(title)}"` : ""}>${esc(view.label)}</span>`;
  }
  function triBadge(value, yes, no) {
    if (value === true) return badge({ label: yes, tone: "ok" });
    if (value === false) return badge({ label: no, tone: "off" });
    return badge({ label: "unknown", tone: "muted" });
  }
  function weightBadge(presence) {
    const p = presence || {};
    const view = WEIGHT_LABELS[p.status] || WEIGHT_LABELS.unknown;
    return badge(view, [p.location, p.evidence].filter(Boolean).join(" | "));
  }
  function fitTitle(fit) {
    const f = fit || {};
    const lines = [];
    // `usable_bytes` (ceiling minus the system reserve) is what the verdict
    // compared `need_bytes` with; the raw ceiling is only the context.
    if (isNum(f.need_bytes) && isNum(f.usable_bytes)) lines.push(`Needs ${fmtBytes(f.need_bytes)} of ${fmtBytes(f.usable_bytes)} usable memory${isNum(f.ceiling_bytes) ? ` (ceiling ${fmtBytes(f.ceiling_bytes)} minus the system reserve)` : ""}`);
    else if (isNum(f.need_bytes) || isNum(f.ceiling_bytes)) lines.push(`Needs ${fmtBytes(f.need_bytes)}; model memory ceiling ${fmtBytes(f.ceiling_bytes)}`);
    if (isNum(f.free_now_bytes)) lines.push(`Free now ${fmtBytes(f.free_now_bytes)}${f.fits_now === true ? " (fits now)" : f.fits_now === false ? " (does not fit now)" : ""}`);
    if (f.disk_ok === false) lines.push("Not enough free disk for the download");
    if (isNum(f.max_context)) lines.push(`Max context ${f.max_context.toLocaleString()} tokens`);
    if (f.confidence) lines.push(`Confidence: ${f.confidence}`);
    for (const note of Array.isArray(f.notes) ? f.notes : []) lines.push(String(note));
    return lines.join("\n");
  }
  function fitBadge(fit) {
    const f = fit || {};
    return badge(FIT_LABELS[f.verdict] || FIT_LABELS.unknown, fitTitle(f));
  }
  function role(root, name) { return root.querySelector(`[data-acc="${name}"]`); }
  function setMessage(ctx, text, tone) {
    const el = role(ctx.host, "message");
    if (!el) return;
    el.textContent = text || "";
    el.className = "acc-message" + (tone ? ` acc-${tone}` : "");
  }
  function errorText(err) { return (err && err.message) || String(err); }
  // Copy to the clipboard. navigator.clipboard exists only on https or
  // localhost; a console opened over plain http from another machine falls
  // back to a hidden textarea + execCommand("copy"). True when it copied.
  async function copyText(text) {
    try {
      if (navigator.clipboard && window.isSecureContext) { await navigator.clipboard.writeText(text); return true; }
    } catch (err) { /* fall through to the textarea copy */ }
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed"; ta.style.top = "-1000px"; ta.style.opacity = "0";
    document.body.appendChild(ta);
    try { ta.select(); return document.execCommand("copy"); } catch (err) { return false; } finally { ta.remove(); }
  }
  // Collapsible list panels (DESIGN §12): open by default, the state remembered per viewer.
  const COLLAPSE_KEY = "abstractcore_console_collapsed_v1";
  function collapsedSections() { try { return JSON.parse(storeGet("local", COLLAPSE_KEY) || "{}") || {}; } catch (e) { return {}; } }
  function setCollapsed(btn, collapsed, remember) {
    const section = btn.closest(".acc-section");
    btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
    if (section) for (const child of Array.from(section.children)) if (!child.classList.contains("acc-section-head")) child.hidden = collapsed;
    if (remember) { const m = collapsedSections(); if (collapsed) m[btn.dataset.accSection] = true; else delete m[btn.dataset.accSection]; storeSet("local", COLLAPSE_KEY, Object.keys(m).length ? JSON.stringify(m) : ""); }
  }
  function storage(kind) { try { return kind === "local" ? window.localStorage : window.sessionStorage; } catch (e) { return null; } }
  function storeGet(kind, key) { const s = storage(kind); try { return s ? s.getItem(key) : null; } catch (e) { return null; } }
  function storeSet(kind, key, value) {
    const s = storage(kind); if (!s) return;
    try { if (value === null || value === undefined || value === "") s.removeItem(key); else s.setItem(key, value); } catch (e) { /* storage full or blocked */ }
  }
  function query(params) {
    const parts = [];
    for (const [k, v] of Object.entries(params)) {
      if (v === undefined || v === null || v === "" || v === false) continue;
      parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(v === true ? "1" : String(v))}`);
    }
    return parts.length ? `?${parts.join("&")}` : "";
  }
  // A job answer is the host_job_v1 dict itself; a `{job: {...}}` envelope
  // (the gateway's historical download route) is unwrapped. Anything else is
  // a contract break and fails loudly rather than rendering a blank card.
  function asJob(payload) {
    if (payload && payload.schema === "host_job_v1" && payload.job_id) return payload;
    if (payload && payload.job && typeof payload.job === "object" && payload.job.job_id) return payload.job;
    const err = new Error("Unexpected job payload (expected schema host_job_v1 with job_id).");
    err.data = payload;
    throw err;
  }

  // ------------------------------------------------------- default request
  function getToken() { return storeGet("session", TOKEN_KEY) || ""; }
  function setToken(token) { storeSet("session", TOKEN_KEY, token || ""); }
  let tokenPrompt = null;
  function promptToken(reason) {
    if (tokenPrompt) return tokenPrompt;
    tokenPrompt = new Promise((resolve) => {
      const wrap = document.createElement("div");
      wrap.className = "acc-root acc-modal-backdrop";
      wrap.id = "acc-auth-modal";
      wrap.innerHTML = `<div class="acc-modal" role="dialog" aria-modal="true" aria-labelledby="acc-auth-title">
        <h3 id="acc-auth-title">Server token required</h3>
        ${reason ? `<p class="acc-error">${esc(reason)}</p>` : ""}
        <p>This AbstractCore server requires its bearer token. On the server's machine, <code>abstractcore serve --print-token</code> prints it (or <code>abstractcore serve --claim-url</code> prints a one-time console link). It is kept in this browser tab's session storage only.</p>
        <form data-acc="auth-form">
          <input type="password" data-acc="auth-token" autocomplete="off" placeholder="Bearer token" aria-label="Bearer token" style="width:100%">
          <div class="acc-modal-actions">
            <button type="button" class="acc-btn" data-acc="auth-cancel">Cancel</button>
            <button type="submit" class="acc-btn acc-primary">Use token</button>
          </div>
        </form>
      </div>`;
      document.body.appendChild(wrap);
      const input = wrap.querySelector('[data-acc="auth-token"]');
      const done = (token) => { wrap.remove(); tokenPrompt = null; resolve(token); };
      wrap.querySelector('[data-acc="auth-form"]').addEventListener("submit", (e) => {
        e.preventDefault();
        const token = input.value.trim();
        if (!token) return;
        setToken(token);
        done(token);
      });
      wrap.querySelector('[data-acc="auth-cancel"]').addEventListener("click", () => done(""));
      setTimeout(() => input.focus(), 0);
    });
    return tokenPrompt;
  }
  function httpError(status, data) {
    const detail = data && data.detail;
    let msg;
    if (detail && typeof detail === "object") msg = detail.message || detail.reason_code || (detail.error && detail.error.message) || JSON.stringify(detail);
    else if (typeof detail === "string" && detail) msg = detail;
    else if (data && data.error && typeof data.error === "object") msg = data.error.message || data.error.type;
    else if (data && typeof data.error === "string") msg = data.error;
    else msg = `HTTP ${status}`;
    const err = new Error(String(msg));
    err.status = status;
    err.data = data;
    if (detail && typeof detail === "object") err.detail = detail;
    return err;
  }
  async function defaultRequest(method, path, body) {
    for (let attempt = 0; attempt < 2; attempt++) {
      const headers = { Accept: "application/json" };
      const token = getToken();
      if (token) headers.Authorization = `Bearer ${token}`;
      const init = { method: String(method || "GET").toUpperCase(), headers, credentials: "same-origin" };
      if (body !== undefined && body !== null) { headers["Content-Type"] = "application/json"; init.body = JSON.stringify(body); }
      const controller = typeof AbortController === "function" ? new AbortController() : null;
      if (controller) init.signal = controller.signal;
      const timer = controller ? setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS) : null;
      let res; let text;
      try {
        res = await fetch(path, init);
        text = await res.text();
      } catch (e) {
        if (e && e.name === "AbortError") throw new Error(`Request timed out after ${REQUEST_TIMEOUT_MS / 1000}s (${path}).`);
        throw new Error(`AbstractCore server unreachable: ${(e && e.message) || e}`);
      } finally {
        if (timer !== null) clearTimeout(timer);
      }
      let data = {};
      try { data = text ? JSON.parse(text) : {}; } catch (e) { data = { detail: text }; }
      if (res.status === 401 && attempt === 0) {
        if (token) setToken("");
        const fresh = await promptToken(token ? "The stored token was rejected." : "");
        if (fresh) continue;
      }
      if (!res.ok) throw httpError(res.status, data);
      return data;
    }
    throw new Error("Unauthorized: no server token provided.");
  }

  // ------------------------------------------------------------ CLI mirror
  function cliFor(ctx, action, a) {
    const p = ctx.cliPrefix;
    switch (action) {
      case "download": return `${p} models download ${shq(a.provider)} ${shq(a.artifact)}`;
      case "delete": return `${p} models delete ${shq(a.provider)} ${shq(a.artifact)}${a.force ? " --force" : ""} --yes`;
      case "engine_install": return `${p} engines install ${shq(a.engine)} --yes`;
      case "open": return `${p} engines open ${shq(a.engine)}`;
      default: return "";
    }
  }
  function jobCli(ctx, job) {
    if (job.cli_equivalent) return String(job.cli_equivalent);
    const action = job.kind === "engine_install" ? "engine_install" : job.kind;
    return cliFor(ctx, action, { provider: job.provider, artifact: job.artifact, engine: job.engine });
  }
  function jobTitle(job) {
    if (job.kind === "engine_install") return `Install ${job.engine || "engine"}`;
    const verb = job.kind === "delete" ? "Delete" : "Download";
    return `${verb} ${job.provider || ""} ${job.artifact || ""}`.trim();
  }

  // ------------------------------------------------------------ job tracker
  function makeJobTracker(ctx, hooks) {
    const jobs = new Map();
    const polling = new Set();
    const storeKey = `${JOBS_KEY}:${ctx.apiBase}:${ctx.kind}`;
    function persist() {
      const live = [];
      for (const job of jobs.values()) if (!TERMINAL.has(job.status) && !job._lost) live.push(job.job_id);
      storeSet("session", storeKey, live.length ? JSON.stringify(live) : "");
    }
    function notify(job) {
      if (ctx.onJob) { try { ctx.onJob(job, ctx.kind); } catch (e) { /* host callback errors must not stop polling */ } }
    }
    function schedule(id) {
      if (polling.has(id) || !ctx.alive) return;
      polling.add(id);
      const t = setTimeout(() => { ctx.timers.delete(t); polling.delete(id); poll(id); }, POLL_MS);
      ctx.timers.add(t);
    }
    async function poll(id) {
      if (!ctx.alive) return;
      let job;
      try {
        job = asJob(await ctx.request("GET", `${ctx.apiBase}/jobs/${encodeURIComponent(id)}`));
      } catch (err) {
        const prev = jobs.get(id) || { job_id: id, kind: "download", status: "running" };
        if (err && err.status === 404) {
          // In-process registry: a server restart forgets jobs. The files may
          // well have landed, so re-probe instead of claiming a failure.
          const lost = Object.assign({}, prev, { _lost: true, message: "The server no longer knows this job (restarted?). Re-checking what is on disk." });
          jobs.set(id, lost); persist(); notify(lost); hooks.changed(); hooks.finished(lost);
          return;
        }
        jobs.set(id, Object.assign({}, prev, { _pollError: errorText(err) }));
        hooks.changed();
        schedule(id);
        return;
      }
      jobs.set(id, job); notify(job); hooks.changed();
      if (TERMINAL.has(job.status)) { persist(); hooks.finished(job); } else schedule(id);
    }
    return {
      track(job) {
        jobs.set(job.job_id, job); notify(job); persist(); hooks.changed();
        if (TERMINAL.has(job.status)) hooks.finished(job); else schedule(job.job_id);
      },
      restore() {
        let ids = [];
        try { ids = JSON.parse(storeGet("session", storeKey) || "[]"); } catch (e) { ids = []; }
        for (const id of Array.isArray(ids) ? ids : []) {
          if (typeof id !== "string" || jobs.has(id)) continue;
          jobs.set(id, { job_id: id, kind: "download", status: "queued", message: "Re-attaching..." });
          poll(id);
        }
      },
      async cancel(id) {
        const job = asJob(await ctx.request("POST", `${ctx.apiBase}/jobs/${encodeURIComponent(id)}/cancel`));
        jobs.set(job.job_id, job); notify(job); persist(); hooks.changed();
        if (!TERMINAL.has(job.status)) schedule(job.job_id); else hooks.finished(job);
      },
      dismiss(id) { jobs.delete(id); persist(); hooks.changed(); },
      activeFor(kind, a, b) {
        for (const job of jobs.values()) {
          if (job.kind !== kind || TERMINAL.has(job.status) || job._lost) continue;
          if (kind === "engine_install" ? job.engine === a : job.provider === a && job.artifact === b) return job;
        }
        return null;
      },
      list() { return Array.from(jobs.values()); },
    };
  }
  function progressHtml(job, inline) {
    const running = job.status === "running" || job.status === "queued";
    const pct = isNum(job.percent) ? Math.max(0, Math.min(100, job.percent)) : null;
    const bar = `<div class="acc-progress${pct === null && running ? " acc-indeterminate" : ""}" role="progressbar" aria-valuemin="0" aria-valuemax="100"${pct === null ? "" : ` aria-valuenow="${pct.toFixed(0)}"`}><span style="width:${pct === null ? (running ? 30 : 0) : pct}%"></span></div>`;
    if (!inline) return bar;
    return `<span class="acc-progress-inline" title="${esc(job.message || "")}">${bar}<span>${pct === null ? esc(job.status) : `${pct.toFixed(0)}%`}</span></span>`;
  }
  function jobCardHtml(ctx, job) {
    const view = job._lost ? { label: "unknown", tone: "muted" } : (JOB_LABELS[job.status] || { label: job.status || "unknown", tone: "muted" });
    const bytes = isNum(job.total_bytes) && job.total_bytes > 0
      ? `${fmtBytes(job.downloaded_bytes || 0)} of ${fmtBytes(job.total_bytes)}`
      : (isNum(job.downloaded_bytes) && job.downloaded_bytes > 0 ? fmtBytes(job.downloaded_bytes) : "");
    const tail = Array.isArray(job.log_tail) ? job.log_tail : [];
    const cli = jobCli(ctx, job);
    const active = !TERMINAL.has(job.status) && !job._lost;
    return `<article class="acc-job" data-job-id="${esc(job.job_id)}">
      <div class="acc-job-head"><span class="acc-job-title">${esc(jobTitle(job))}${job.dry_run ? " (dry run)" : ""}</span>${badge(view)}</div>
      ${progressHtml(job, false)}
      <div class="acc-sub">${esc([job.message, bytes].filter(Boolean).join(" | "))}</div>
      ${job.error ? `<div class="acc-error">${esc(job.error)}</div>` : ""}
      ${job._pollError ? `<div class="acc-warn">Polling: ${esc(job._pollError)}</div>` : ""}
      ${Array.isArray(job.command) && job.command.length ? `<div class="acc-sub">Command: <code>${esc(argvText(job.command))}</code></div>` : ""}
      ${cli ? `<div class="acc-sub">CLI equivalent: <code>${esc(cli)}</code> <button type="button" class="acc-btn acc-link" data-acc-action="copy" data-text="${esc(cli)}">Copy</button></div>` : ""}
      ${tail.length ? `<details><summary>Log (${tail.length} lines)</summary><pre>${esc(tail.join("\n"))}</pre></details>` : ""}
      <div class="acc-actions">${active
        ? `<button type="button" class="acc-btn" data-acc-action="cancel-job" data-job-id="${esc(job.job_id)}" title="Cancel (c)">Cancel</button>`
        : `<button type="button" class="acc-btn" data-acc-action="dismiss-job" data-job-id="${esc(job.job_id)}">Dismiss</button>`}</div>
    </article>`;
  }
  function renderJobs(ctx) {
    const el = role(ctx.host, "jobs");
    if (!el) return;
    el.innerHTML = ctx.jobs.list().map((job) => jobCardHtml(ctx, job)).join("");
  }

  // ------------------------------------------------------------------ modal
  function openModal(ctx, spec) {
    closeModal(ctx);
    const wrap = document.createElement("div");
    wrap.className = "acc-modal-backdrop";
    wrap.setAttribute("data-acc", "modal");
    wrap.innerHTML = `<div class="acc-modal" role="dialog" aria-modal="true" aria-labelledby="acc-modal-title-${esc(ctx.kind)}">
      <h3 id="acc-modal-title-${esc(ctx.kind)}">${esc(spec.title)}</h3>
      ${spec.bodyHtml}
      <div class="acc-message" data-acc="modal-message" role="status"></div>
      <div class="acc-modal-actions">
        ${spec.previewLabel ? `<button type="button" class="acc-btn" data-modal="preview">${esc(spec.previewLabel)}</button>` : ""}
        <button type="button" class="acc-btn" data-modal="cancel">Cancel</button>
        <button type="button" class="acc-btn ${spec.danger ? "acc-danger" : "acc-primary"}" data-modal="confirm">${esc(spec.confirmLabel)}</button>
      </div>
    </div>`;
    ctx.host.appendChild(wrap);
    ctx.modal = wrap;
    const msg = wrap.querySelector('[data-acc="modal-message"]');
    const buttons = () => Array.from(wrap.querySelectorAll("button[data-modal]"));
    const run = async (fn, closeAfter) => {
      buttons().forEach((b) => { b.disabled = true; });
      msg.className = "acc-message"; msg.textContent = "Working...";
      try {
        const out = await fn(wrap);
        if (closeAfter) { closeModal(ctx); return; }
        msg.textContent = typeof out === "string" ? out : "";
      } catch (err) {
        msg.className = "acc-message acc-error";
        msg.textContent = errorText(err);
      }
      buttons().forEach((b) => { b.disabled = false; });
    };
    wrap.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-modal]");
      if (e.target === wrap || (b && b.dataset.modal === "cancel")) { closeModal(ctx); return; }
      if (!b) return;
      if (b.dataset.modal === "confirm") run(spec.onConfirm, true);
      else if (b.dataset.modal === "preview" && spec.onPreview) run(spec.onPreview, false);
    });
    setTimeout(() => { const c = wrap.querySelector('[data-modal="cancel"]'); if (c) c.focus(); }, 0);
    return wrap;
  }
  function closeModal(ctx) {
    if (ctx.modal) { ctx.modal.remove(); ctx.modal = null; }
  }

  // ---------------------------------------------------------- Models (catalog)
  function modelsController(ctx) {
    const st = { catalog: null, installed: null, seenEngines: new Set(), seq: 0, debounce: null };
    const ui = {
      q: role(ctx.host, "q"), engine: role(ctx.host, "engine"), modality: role(ctx.host, "modality"),
      fits: role(ctx.host, "fits"), installedOnly: role(ctx.host, "installed-only"), hub: role(ctx.host, "hub"),
    };
    function filters() {
      return {
        q: ui.q.value.trim(), engine: ui.engine.value, modality: ui.modality.value,
        fits: ui.fits.checked, installedOnly: ui.installedOnly.checked, hub: ui.hub.checked,
      };
    }
    function viewCli() {
      const f = filters();
      let cmd = f.q ? `${ctx.cliPrefix} models search ${shq(f.q)}` : `${ctx.cliPrefix} models catalog`;
      if (f.engine) cmd += ` --engine ${shq(f.engine)}`;
      if (f.fits) cmd += " --fits";
      if (f.hub) cmd += " --hub";
      const el = role(ctx.host, "view-cli");
      if (el) el.textContent = `${cmd} --json`;
    }
    function hostLine(profile) {
      const el = role(ctx.host, "host-line");
      if (!el) return;
      if (!profile) { el.textContent = "Host profile unavailable."; return; }
      const mem = profile.unified_memory ? `${fmtBytes(profile.ram_bytes)} unified memory` : `${fmtBytes(profile.ram_bytes)} RAM${isNum(profile.vram_bytes) ? `, ${fmtBytes(profile.vram_bytes)} VRAM` : ""}`;
      const parts = [
        `${profile.os || "?"}/${profile.arch || "?"}`,
        profile.gpu_name || profile.accelerator || "no accelerator",
        mem,
        `model ceiling ${fmtBytes(profile.ceiling_bytes)}${profile.ceiling_source ? ` (${profile.ceiling_source})` : ""}`,
        `free now ${fmtBytes(profile.free_now_bytes)}`,
      ];
      el.innerHTML = `This machine (<strong>${esc(ctx.hostName)}</strong>): ${esc(parts.join(" | "))}`;
    }
    function engineOptions() {
      for (const row of (st.catalog && st.catalog.rows) || []) for (const a of row.artifacts || []) if (a.provider) st.seenEngines.add(a.provider);
      for (const row of (st.installed && st.installed.rows) || []) if (row.provider) st.seenEngines.add(row.provider);
      const current = ui.engine.value;
      const opts = Array.from(st.seenEngines).sort();
      ui.engine.innerHTML = `<option value="">All engines</option>` + opts.map((id) => `<option value="${esc(id)}">${esc(id)}</option>`).join("");
      ui.engine.value = opts.includes(current) ? current : "";
    }
    function installedRow(provider, artifact) {
      for (const row of (st.installed && st.installed.rows) || []) if (row.provider === provider && row.artifact === artifact) return row;
      return null;
    }
    function modalityOk(caps, modality) {
      if (!modality) return true;
      const c = caps || {};
      return c[modality] === true;
    }
    function actionCell(a) {
      const admin = ctx.isAdmin();
      const dis = admin ? "" : ` disabled title="Admin only"`;
      const attrs = `data-provider="${esc(a.provider)}" data-artifact="${esc(a.artifact)}"`;
      const job = ctx.jobs.activeFor("download", a.provider, a.artifact) || ctx.jobs.activeFor("delete", a.provider, a.artifact);
      if (job) {
        return `<div class="acc-actions">${progressHtml(job, true)}<button type="button" class="acc-btn" data-acc-action="cancel-job" data-job-id="${esc(job.job_id)}">Cancel</button></div>`;
      }
      const status = (a.presence && a.presence.status) || "unknown";
      if (status === "installed") return `<div class="acc-actions"><button type="button" class="acc-btn" data-acc-action="delete" ${attrs}${dis || ' title="Delete (d)"'}>Delete</button></div>`;
      if (status === "not_applicable") return `<span class="acc-muted">-</span>`;
      if (a.downloadable) return `<div class="acc-actions"><button type="button" class="acc-btn acc-primary" data-acc-action="download" ${attrs}${dis || ' title="Download (w)"'}>Download</button></div>`;
      return `<span class="acc-muted" title="This artifact cannot be downloaded from here">-</span>`;
    }
    function renderCatalog() {
      const body = role(ctx.host, "catalog-body");
      const count = role(ctx.host, "catalog-count");
      if (st.catalogError) {
        body.innerHTML = `<tr><td colspan="8" class="acc-error">${esc(st.catalogError)}</td></tr>`;
        count.textContent = "";
        return;
      }
      if (!st.catalog) return;
      const f = filters();
      const out = [];
      let models = 0; let artifacts = 0; let hidden = 0;
      for (const row of st.catalog.rows || []) {
        if (!modalityOk(row.capabilities, f.modality)) continue;
        const arts = (row.artifacts || []).filter((a) => {
          if (f.engine && a.provider !== f.engine) return false;
          if (f.installedOnly && !(a.presence && a.presence.status === "installed")) return false;
          if (f.fits && a.fit && a.fit.verdict === "too_large") { hidden++; return false; }
          return true;
        });
        if (!arts.length) continue;
        models++;
        const caps = row.capabilities || {};
        const chips = [];
        for (const k of ["text", "vision", "audio", "embedding", "thinking"]) if (caps[k] === true) chips.push(k);
        if (caps.tools && caps.tools !== "none") chips.push(`tools:${caps.tools}`);
        const params = fmtParams(row.params_total) + (isNum(row.params_active) ? ` (${fmtParams(row.params_active)} active)` : "");
        arts.forEach((a, i) => {
          artifacts++;
          const first = i === 0;
          const modelCell = first
            ? `<div class="acc-model-name">${esc(row.display_name || row.id)}${row.source === "hf_search" ? ' <span class="acc-chip">hub</span>' : ""}</div>
               <div class="acc-sub">${esc([row.vendor, params, row.license].filter(Boolean).join(" | "))}</div>
               <div>${chips.map((c) => `<span class="acc-chip">${esc(c)}</span>`).join("")}</div>`
            : "";
          const size = isNum(a.download_bytes) ? fmtBytes(a.download_bytes) : "?";
          out.push(`<tr tabindex="0" class="${first ? "acc-group-start" : ""}" data-acc-row="artifact" data-provider="${esc(a.provider)}" data-artifact="${esc(a.artifact)}">
            <td>${modelCell}</td>
            <td>${esc(a.provider)}</td>
            <td><code>${esc(a.artifact)}</code>${a.recommended ? ' <span class="acc-badge acc-tone-info">recommended</span>' : ""}</td>
            <td>${esc(a.quant || "")}${isNum(a.bits) ? `<div class="acc-sub">${esc(a.bits)} bits</div>` : ""}</td>
            <td class="acc-num" title="Size source: ${esc(a.size_source || "unknown")}">${esc(size)}</td>
            <td>${weightBadge(a.presence)}</td>
            <td>${fitBadge(a.fit)}</td>
            <td>${actionCell(a)}</td>
          </tr>`);
        });
      }
      body.innerHTML = out.length ? out.join("") : `<tr><td colspan="8" class="acc-muted">No model matches these filters.</td></tr>`;
      count.textContent = `${models} models, ${artifacts} artifacts${hidden ? `, ${hidden} hidden as too large (untick "Fits this machine" to show)` : ""}`;
    }
    function renderInstalled() {
      const body = role(ctx.host, "installed-body");
      const summary = role(ctx.host, "installed-summary");
      const errs = role(ctx.host, "installed-errors");
      if (st.installedError) {
        body.innerHTML = `<tr><td colspan="7" class="acc-error">${esc(st.installedError)}</td></tr>`;
        summary.textContent = ""; errs.textContent = "";
        return;
      }
      if (!st.installed) return;
      const f = filters();
      const needle = f.q.toLowerCase();
      const admin = ctx.isAdmin();
      const rows = (st.installed.rows || []).filter((r) => (!f.engine || r.provider === f.engine) && (!needle || String(r.artifact || "").toLowerCase().includes(needle) || String(r.catalog_id || "").toLowerCase().includes(needle)));
      let total = 0; let unknownSizes = 0;
      for (const r of rows) { if (isNum(r.size_bytes)) total += r.size_bytes; else unknownSizes++; }
      body.innerHTML = rows.length ? rows.map((r) => {
        const job = ctx.jobs.activeFor("delete", r.provider, r.artifact);
        const blockers = Array.isArray(r.delete_blockers) ? r.delete_blockers : [];
        const action = job
          ? `<div class="acc-actions">${progressHtml(job, true)}</div>`
          : `<button type="button" class="acc-btn" data-acc-action="delete" data-provider="${esc(r.provider)}" data-artifact="${esc(r.artifact)}"${admin ? ' title="Delete (d)"' : ' disabled title="Admin only"'}>Delete</button>`;
        return `<tr tabindex="0" data-acc-row="installed" data-provider="${esc(r.provider)}" data-artifact="${esc(r.artifact)}">
          <td>${esc(r.provider)}</td>
          <td><code class="acc-artifact-id" title="${esc(r.artifact)}">${esc(r.artifact)}</code>${r.catalog_id ? `<div class="acc-sub">${esc(r.catalog_id)}</div>` : ""}${blockers.length ? `<div class="acc-sub acc-warn">${esc(blockers.join(", "))}</div>` : ""}</td>
          <td>${esc(r.quant || "")}</td>
          <td class="acc-num">${esc(fmtBytes(r.size_bytes))}</td>
          <td><span class="acc-sub">${esc(r.location || "")}</span></td>
          <td>${r.loaded === true ? badge({ label: "loaded", tone: "info" }) : r.loaded === false ? '<span class="acc-muted">no</span>' : '<span class="acc-muted" title="unknown">?</span>'}</td>
          <td>${action}</td>
        </tr>`;
      }).join("") : `<tr><td colspan="7" class="acc-muted">No installed model matches.</td></tr>`;
      const probed = Array.isArray(st.installed.engines_probed) ? st.installed.engines_probed.join(", ") : "";
      summary.textContent = `${rows.length} models, ${fmtBytes(total)} on disk${unknownSizes ? ` (+${unknownSizes} of unknown size)` : ""}${probed ? ` | probed: ${probed}` : ""}`;
      const errors = st.installed.errors && typeof st.installed.errors === "object" ? Object.entries(st.installed.errors) : [];
      errs.textContent = errors.length ? `Could not list: ${errors.map(([k, v]) => `${k} (${v})`).join(", ")}` : "";
    }
    function render() { renderJobs(ctx); renderCatalog(); renderInstalled(); viewCli(); }
    async function loadCatalog() {
      const seq = ++st.seq;
      const f = filters();
      try {
        const data = await ctx.request("GET", `${ctx.apiBase}/models/catalog${query({ q: f.q, engine: f.engine, fits: f.fits, hub: f.hub })}`);
        if (seq !== st.seq || !ctx.alive) return;
        st.catalog = data; st.catalogError = null;
        hostLine(data && data.host_profile);
      } catch (err) {
        if (seq !== st.seq || !ctx.alive) return;
        st.catalogError = `Catalog: ${errorText(err)}`;
      }
      engineOptions(); render();
    }
    async function loadInstalled() {
      try {
        st.installed = await ctx.request("GET", `${ctx.apiBase}/models/installed`);
        st.installedError = null;
      } catch (err) {
        st.installedError = `Installed models: ${errorText(err)}`;
      }
      if (!ctx.alive) return;
      engineOptions(); render();
    }
    function refresh() { return Promise.all([loadCatalog(), loadInstalled()]); }
    async function download(provider, artifact) {
      if (!ctx.isAdmin()) { setMessage(ctx, "Downloads are admin-only.", "error"); return; }
      setMessage(ctx, `Starting download of ${provider} ${artifact}...`);
      try {
        const job = asJob(await ctx.request("POST", `${ctx.apiBase}/models/download`, { provider, artifact, dry_run: false }));
        ctx.jobs.track(job);
        setMessage(ctx, `Download started. CLI equivalent: ${jobCli(ctx, job)}`, "ok");
      } catch (err) {
        setMessage(ctx, `Download ${provider} ${artifact}: ${errorText(err)}`, "error");
      }
    }
    function confirmDelete(provider, artifact) {
      if (!ctx.isAdmin()) { setMessage(ctx, "Deletes are admin-only.", "error"); return; }
      const row = installedRow(provider, artifact) || {};
      const blockers = Array.isArray(row.delete_blockers) ? row.delete_blockers : [];
      const blockerItems = blockers.map((b) => {
        if (BLOCKER_TEXT[b]) return BLOCKER_TEXT[b];
        if (String(b).startsWith("shared_cache:")) return `The files are shared by ${String(b).slice(13).split(",").join(", ")}; deleting removes them for all of them.`;
        return String(b);
      });
      const needsForce = blockers.length > 0 || row.deletable === false;
      const cli = (force) => cliFor(ctx, "delete", { provider, artifact, force });
      openModal(ctx, {
        title: `Delete ${artifact}?`,
        danger: true,
        confirmLabel: "Delete",
        previewLabel: "Preview (dry run)",
        bodyHtml: `<p>Removes the <strong>${esc(provider)}</strong> weights <code>${esc(artifact)}</code> from host <strong>${esc(ctx.hostName)}</strong>${isNum(row.size_bytes) ? `, freeing about ${esc(fmtBytes(row.size_bytes))}` : ""}.</p>
          ${row.location ? `<p class="acc-sub">Location: <code>${esc(row.location)}</code></p>` : ""}
          ${blockerItems.length ? `<ul class="acc-note">${blockerItems.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>` : ""}
          ${needsForce ? `<label><input type="checkbox" data-acc="force"> Force (delete despite the warnings above)</label>` : ""}
          <p class="acc-sub">CLI equivalent: <code data-acc="modal-cli">${esc(cli(false))}</code></p>
          <div data-acc="preview-out"></div>`,
        onPreview: async (wrap) => {
          const force = !!(wrap.querySelector('[data-acc="force"]') || {}).checked;
          const job = asJob(await ctx.request("POST", `${ctx.apiBase}/models/delete`, { provider, artifact, dry_run: true, force }));
          const out = wrap.querySelector('[data-acc="preview-out"]');
          out.innerHTML = `<p class="acc-sub">Dry run: ${esc(job.message || job.status)}</p>${Array.isArray(job.command) && job.command.length ? `<pre class="acc-code">${esc(argvText(job.command))}</pre>` : ""}`;
          return "";
        },
        onConfirm: async (wrap) => {
          const force = !!(wrap.querySelector('[data-acc="force"]') || {}).checked;
          const job = asJob(await ctx.request("POST", `${ctx.apiBase}/models/delete`, { provider, artifact, dry_run: false, force }));
          ctx.jobs.track(job);
          setMessage(ctx, `Delete started. CLI equivalent: ${jobCli(ctx, job)}`, "ok");
        },
      });
      const forceBox = role(ctx.modal, "force");
      if (forceBox) forceBox.addEventListener("change", () => { role(ctx.modal, "modal-cli").textContent = cli(forceBox.checked); });
    }
    function onInput() {
      viewCli();
      clearTimeout(st.debounce);
      st.debounce = setTimeout(loadCatalog, 300);
    }
    ui.q.addEventListener("input", onInput);
    ui.engine.addEventListener("change", () => { loadCatalog(); });
    ui.fits.addEventListener("change", () => { loadCatalog(); });
    ui.hub.addEventListener("change", () => { loadCatalog(); });
    ui.modality.addEventListener("change", render);
    ui.installedOnly.addEventListener("change", render);
    return {
      refresh,
      render,
      jobFinished() { loadInstalled(); loadCatalog(); },
      action(name, el) {
        if (name === "refresh") { setMessage(ctx, ""); refresh(); }
        else if (name === "download") download(el.dataset.provider, el.dataset.artifact);
        else if (name === "delete") confirmDelete(el.dataset.provider, el.dataset.artifact);
      },
      key(key, row) {
        if (key === "/") { ui.q.focus(); ui.q.select(); return true; }
        if (key === "f") { ui.fits.checked = !ui.fits.checked; loadCatalog(); return true; }
        if (key === "r") { refresh(); return true; }
        if (key === "w" || key === "d") {
          if (!row) { setMessage(ctx, "Focus a row first (click it or Tab to it)."); return true; }
          const p = row.dataset.provider; const a = row.dataset.artifact;
          if (key === "w") download(p, a); else confirmDelete(p, a);
          return true;
        }
        return false;
      },
      destroy() { clearTimeout(st.debounce); },
    };
  }

  // -------------------------------------------------------------- Engines
  function hostNotes(install) {
    const out = [];
    const argv = Array.isArray(install.argv) ? install.argv : [];
    if (argv[0] === "sudo") out.push("Runs with sudo on the host.");
    switch (install.method) {
      case "brew": out.push("Uses Homebrew on the host (brew must already be on PATH); no sudo needed."); break;
      case "script": out.push("The vendor script may ask for sudo. A background job cannot type a password: if it needs one, run the CLI equivalent in a terminal on the host."); break;
      case "winget": out.push("Windows may raise a UAC prompt on the host's desktop; someone at the host must approve it."); break;
      case "pip": out.push("Installs into the Python environment that runs this server; no sudo needed."); break;
      case "download_page": out.push("The app itself is installed by hand from the vendor's download page."); break;
      default: break;
    }
    return out;
  }
  function enginesController(ctx) {
    const st = { data: null, error: null };
    function byId(id) { return ((st.data && st.data.engines) || []).find((e) => e.id === id) || null; }
    function actionsHtml(e) {
      const admin = ctx.isAdmin();
      const install = e.install || {};
      const parts = [];
      const job = ctx.jobs.activeFor("engine_install", e.id);
      if (job) {
        parts.push(progressHtml(job, true));
        parts.push(`<button type="button" class="acc-btn" data-acc-action="cancel-job" data-job-id="${esc(job.job_id)}">Cancel</button>`);
      } else if (e.supported_on_host !== false && install.available && Array.isArray(install.argv) && install.argv.length && e.installed !== true) {
        parts.push(`<button type="button" class="acc-btn acc-primary" data-acc-action="install" data-engine="${esc(e.id)}"${admin ? ' title="Install (i)"' : ' disabled title="Admin only"'}>Install</button>`);
      }
      if (install.url && (install.method === "download_page" || e.installed !== true)) {
        parts.push(`<button type="button" class="acc-btn" data-acc-action="open" data-engine="${esc(e.id)}" title="Open download page (o)">Open download page</button>`);
      }
      if (e.docs_url) parts.push(`<a class="acc-btn acc-link" href="${esc(e.docs_url)}" target="_blank" rel="noopener noreferrer">Docs</a>`);
      return parts.length ? `<div class="acc-actions">${parts.join("")}</div>` : '<span class="acc-muted">-</span>';
    }
    function render() {
      renderJobs(ctx);
      const body = role(ctx.host, "engines-body");
      const meta = role(ctx.host, "engines-meta");
      if (st.error) { body.innerHTML = `<tr><td colspan="9" class="acc-error">${esc(st.error)}</td></tr>`; return; }
      if (!st.data) return;
      const engines = st.data.engines || [];
      body.innerHTML = engines.length ? engines.map((e) => `<tr tabindex="0" data-acc-row="engine" data-engine="${esc(e.id)}">
          <td><div class="acc-model-name">${esc(e.name || e.id)}</div><div class="acc-sub">${esc(e.id)} | ${esc(e.kind || "")}</div></td>
          <td>${e.supported_on_host === false ? badge({ label: "not supported", tone: "off" }, e.unsupported_reason || "") + (e.unsupported_reason ? `<div class="acc-sub">${esc(e.unsupported_reason)}</div>` : "") : e.supported_on_host === true ? badge({ label: "supported", tone: "ok" }) : badge({ label: "unknown", tone: "muted" })}</td>
          <td>${triBadge(e.installed, "installed", "not installed")}${e.install_location ? `<div class="acc-sub">${esc(e.install_location)}</div>` : ""}</td>
          <td>${esc(e.version || "")}</td>
          <td>${e.kind === "local_engine" && e.running === null ? '<span class="acc-muted" title="In-process engine: nothing to run">n/a</span>' : triBadge(e.running, "running", "stopped")}</td>
          <td>${e.base_url ? triBadge(e.reachable, "reachable", "unreachable") : '<span class="acc-muted">-</span>'}</td>
          <td>${e.base_url ? `<code>${esc(e.base_url)}</code>` : '<span class="acc-muted">-</span>'}</td>
          <td class="acc-num">${isNum(e.models_count) ? esc(e.models_count) : '<span class="acc-muted">?</span>'}</td>
          <td>${actionsHtml(e)}</td>
        </tr>`).join("") : `<tr><td colspan="9" class="acc-muted">No engines reported.</td></tr>`;
      const installed = engines.filter((e) => e.installed === true).length;
      const running = engines.filter((e) => e.running === true).length;
      meta.textContent = `${installed} of ${engines.length} installed, ${running} running${st.data.generated_at ? ` | probed ${new Date(st.data.generated_at).toLocaleTimeString()}` : ""}`;
    }
    async function refresh() {
      try {
        st.data = await ctx.request("GET", `${ctx.apiBase}/engines${query({ probe: true })}`);
        st.error = null;
      } catch (err) {
        st.error = `Engines: ${errorText(err)}`;
      }
      if (ctx.alive) render();
    }
    function openPage(id) {
      const e = byId(id);
      const url = e && e.install && e.install.url;
      if (!url) { setMessage(ctx, `No download page known for ${id}.`, "error"); return; }
      window.open(url, "_blank", "noopener,noreferrer");
      setMessage(ctx, `Opened ${url}. CLI equivalent: ${cliFor(ctx, "open", { engine: id })}`);
    }
    function confirmInstall(id) {
      const e = byId(id);
      if (!e) return;
      if (!ctx.isAdmin()) { setMessage(ctx, "Engine installs are admin-only.", "error"); return; }
      const install = e.install || {};
      const notes = hostNotes(install);
      openModal(ctx, {
        title: `Install ${e.name || e.id}`,
        confirmLabel: "Install",
        previewLabel: "Preview (dry run)",
        bodyHtml: `<p>Runs on host <strong>${esc(ctx.hostName)}</strong>: the machine that runs this server, which may not be the computer in front of you.</p>
          <div class="acc-sub">Exact command (fixed argument list, no shell):</div>
          <pre class="acc-code" data-acc="install-argv">${esc(argvText(install.argv))}</pre>
          <dl class="acc-kv">
            <dt>Method</dt><dd>${esc(install.method || "?")}</dd>
            ${isNum(install.estimated_bytes) ? `<dt>Download</dt><dd>about ${esc(fmtBytes(install.estimated_bytes))}</dd>` : ""}
            ${install.notes ? `<dt>What it changes</dt><dd>${esc(install.notes)}</dd>` : ""}
            ${install.url ? `<dt>Vendor page</dt><dd><a href="${esc(install.url)}" target="_blank" rel="noopener noreferrer">${esc(install.url)}</a></dd>` : ""}
          </dl>
          ${notes.length ? `<ul class="acc-note">${notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}
          <p class="acc-sub">CLI equivalent: <code>${esc(cliFor(ctx, "engine_install", { engine: e.id }))}</code></p>
          <div data-acc="preview-out"></div>`,
        onPreview: async (wrap) => {
          const job = asJob(await ctx.request("POST", `${ctx.apiBase}/engines/${encodeURIComponent(e.id)}/install`, { dry_run: true }));
          wrap.querySelector('[data-acc="preview-out"]').innerHTML = `<p class="acc-sub">Dry run: ${esc(job.message || job.status)}</p>${Array.isArray(job.command) && job.command.length ? `<pre class="acc-code">${esc(argvText(job.command))}</pre>` : ""}`;
          return "";
        },
        onConfirm: async () => {
          const job = asJob(await ctx.request("POST", `${ctx.apiBase}/engines/${encodeURIComponent(e.id)}/install`, { dry_run: false }));
          ctx.jobs.track(job);
          setMessage(ctx, `Install started on ${ctx.hostName}. CLI equivalent: ${jobCli(ctx, job)}`, "ok");
        },
      });
    }
    return {
      refresh,
      render,
      jobFinished() { refresh(); },
      action(name, el) {
        if (name === "refresh") { setMessage(ctx, ""); refresh(); }
        else if (name === "install") confirmInstall(el.dataset.engine);
        else if (name === "open") openPage(el.dataset.engine);
      },
      key(key, row) {
        if (key === "r") { refresh(); return true; }
        if (key === "i" || key === "o") {
          if (!row) { setMessage(ctx, "Focus an engine row first (click it or Tab to it)."); return true; }
          if (key === "i") confirmInstall(row.dataset.engine); else openPage(row.dataset.engine);
          return true;
        }
        return false;
      },
      destroy() {},
    };
  }

  // ------------------------------------------------------------- Overview
  function overviewController(ctx) {
    function kv(pairs) {
      return `<dl class="acc-kv">${pairs.filter((p) => p[1] !== null && p[1] !== undefined && p[1] !== "").map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;
    }
    async function loadHost() {
      const el = role(ctx.host, "host-body");
      try {
        const p = await ctx.request("GET", `${ctx.apiBase}/host/profile`);
        const disks = p.disk && typeof p.disk === "object" ? Object.entries(p.disk) : [];
        el.className = "";
        el.innerHTML = kv([
          ["Host", esc(ctx.hostName)],
          ["OS", esc(`${p.os || "?"} / ${p.arch || "?"}`)],
          ["Accelerator", esc([p.accelerator, p.gpu_name].filter(Boolean).join(" | ") || "none")],
          [p.unified_memory ? "Unified memory" : "RAM", esc(fmtBytes(p.ram_bytes))],
          ["VRAM", isNum(p.vram_bytes) ? esc(fmtBytes(p.vram_bytes)) : null],
          ["Model ceiling", esc(`${fmtBytes(p.ceiling_bytes)}${p.ceiling_source ? ` (${p.ceiling_source})` : ""}`)],
          ["Free now", esc(fmtBytes(p.free_now_bytes))],
          ...disks.map(([name, d]) => [`Disk: ${name}`, `${esc(fmtBytes(d && d.free_bytes))} free <span class="acc-sub">${esc((d && d.path) || "")}</span>`]),
          ["Python", esc(p.python || "")],
        ]);
      } catch (err) {
        el.className = "acc-error"; el.textContent = `Host profile: ${errorText(err)}`;
      }
    }
    async function loadEngines() {
      const el = role(ctx.host, "engines-body");
      try {
        const data = await ctx.request("GET", `${ctx.apiBase}/engines${query({ probe: true })}`);
        const engines = data.engines || [];
        el.className = "";
        el.innerHTML = kv(engines.map((e) => [e.name || e.id,
          e.supported_on_host === false ? badge({ label: "not supported", tone: "off" }, e.unsupported_reason || "")
            : `${triBadge(e.installed, "installed", "not installed")} ${e.base_url ? triBadge(e.running, "running", "stopped") : ""}${isNum(e.models_count) ? ` <span class="acc-sub">${esc(e.models_count)} models</span>` : ""}`]))
          + `<p class="acc-sub"><a href="#engines">Manage engines</a> | <a href="#catalog">Browse models</a></p>`;
      } catch (err) {
        el.className = "acc-error"; el.textContent = `Engines: ${errorText(err)}`;
      }
    }
    async function loadServer() {
      const el = role(ctx.host, "server-body");
      const rows = [];
      try {
        const h = await ctx.request("GET", `${ctx.serverRoot}/health`);
        rows.push(["Status", badge({ label: h.status || "ok", tone: h.status === "healthy" ? "ok" : "warn" })]);
        rows.push(["Version", esc(h.version || "?")]);
      } catch (err) {
        rows.push(["Status", badge({ label: "unreachable", tone: "err" }, errorText(err))]);
      }
      try {
        const a = await ctx.request("GET", `${ctx.apiBase}/auth/validate`);
        rows.push(["Server auth", a.server_auth_enabled ? "bearer token required" : "disabled (local/dev mode)"]);
      } catch (err) { /* auth/validate is informational only */ }
      rows.push(["API", `<code>${esc(ctx.apiBase)}</code>`]);
      rows.push(["Docs", `<a href="${esc(ctx.serverRoot)}/docs">Swagger UI</a>`]);
      el.className = "";
      el.innerHTML = kv(rows);
    }
    function refresh() { return Promise.all([loadHost(), loadEngines(), loadServer()]); }
    return { refresh, render() {}, jobFinished() {}, action(name) { if (name === "refresh") refresh(); }, key(k) { if (k === "r") { refresh(); return true; } return false; }, destroy() {} };
  }

  // ------------------------------------------------------------ Providers
  function providersController(ctx) {
    async function loadProviders() {
      const body = role(ctx.host, "providers-body");
      try {
        const data = await ctx.request("GET", `${ctx.serverRoot}/providers`);
        const rows = data.providers || [];
        role(ctx.host, "providers-count").textContent = `${rows.length} available${data.error ? ` | ${data.error}` : ""}`;
        body.innerHTML = rows.length ? rows.map((p) => `<tr>
            <td><div class="acc-model-name">${esc(p.display_name || p.name)}</div><div class="acc-sub">${esc(p.name)}</div></td>
            <td>${esc(p.type || "")}</td>
            <td>${p.local_provider ? "local" : "cloud"}</td>
            <td>${badge({ label: p.status || "unknown", tone: p.status === "available" ? "ok" : "warn" }, p.error || "")}</td>
            <td>${p.authentication_required ? "required" : '<span class="acc-muted">not needed</span>'}</td>
            <td>${(Array.isArray(p.supported_features) ? p.supported_features : []).map((f) => `<span class="acc-chip">${esc(f)}</span>`).join("")}</td>
          </tr>`).join("") : `<tr><td colspan="6" class="acc-muted">No providers reported.</td></tr>`;
      } catch (err) {
        body.innerHTML = `<tr><td colspan="6" class="acc-error">Providers: ${esc(errorText(err))}</td></tr>`;
      }
    }
    async function loadDefaults() {
      const body = role(ctx.host, "defaults-body");
      const msg = role(ctx.host, "defaults-message");
      try {
        const data = await ctx.request("GET", `${ctx.serverRoot}/v1/config/capability-defaults`);
        const routes = Array.isArray(data.routes) ? data.routes : [];
        role(ctx.host, "defaults-source").textContent = data.config_file ? `from ${data.config_file}` : "";
        msg.textContent = Array.isArray(data.errors) && data.errors.length ? data.errors.map(String).join("; ") : "";
        body.innerHTML = routes.length ? routes.map((r) => `<tr>
            <td><div>${esc(r.label || r.key)}</div><div class="acc-sub">${esc(r.key || "")}</div></td>
            <td>${esc(r.provider || "")}</td>
            <td>${r.model ? `<code>${esc(r.model)}</code>` : '<span class="acc-muted">not configured</span>'}${r.covered_by ? `<div class="acc-sub">covered by ${esc(r.covered_by)}</div>` : ""}</td>
            <td><span class="acc-sub">${esc(r.source || "")}</span></td>
          </tr>`).join("") : `<tr><td colspan="4" class="acc-muted">No capability defaults configured.</td></tr>`;
      } catch (err) {
        body.innerHTML = `<tr><td colspan="4" class="acc-muted">Capability defaults are not exposed by this server (${esc(errorText(err))}).</td></tr>`;
      }
    }
    function refresh() { return Promise.all([loadProviders(), loadDefaults()]); }
    return { refresh, render() {}, jobFinished() {}, action(name) { if (name === "refresh") refresh(); }, key(k) { if (k === "r") { refresh(); return true; } return false; }, destroy() {} };
  }


  // ------------------------------------------------------------------ Email
  // DESIGN v2 §3 (round 2), in this order: Email address (one field + its own
  // inline Save) -> Mailbox (tabs IMAP | Google | Microsoft, IMAP first and
  // the default; or the connected status with the Active switch) -> Agent
  // email tools (switch) -> Advanced (compact sentences: who agents may write
  // to, send limits, the watched folder). The IMAP pane shows every server
  // field, pre-filled with the standard imap./smtp.<domain> values as soon as
  // the address has a domain, then with discovery's `defaults`
  // (POST /acore/email/discover, `discovery.server_defaults`), never over a
  // field the user edited. No user name field (a small link reveals Login)
  // and no display name field (the server keeps the stored one, else the
  // address's local part). Core is single-user: no admin switches, and no
  // notification preferences (core has none). Data: email_settings_v1 from
  // /acore/email. On/off settings are af-switch buttons (ui-kit
  // docs/state-toggles.md): they apply at once, stay busy while saving, revert
  // on failure, and the confirmation names the new state.
  const EMAIL_TAB_KEY = "abstractcore_console_email_tab_v1";
  const OAUTH_LABELS = { google: "Google", microsoft: "Microsoft" };
  const EMAIL_TABS = ["imap", "google", "microsoft"];
  // The standard ports per security (RFC 8314 implicit TLS / STARTTLS submission).
  const STANDARD_PORTS = { imap: { ssl: 993, starttls: 143 }, smtp: { ssl: 465, starttls: 587 } };
  const SERVER_FIELDS = ["email-imap-host", "email-imap-port", "email-imap-security", "email-smtp-host", "email-smtp-port", "email-smtp-security", "email-username"];
  function setSwitch(btn, s) {
    if (!btn) return;
    const reasonId = btn.getAttribute("data-reason-id");
    const reasonEl = reasonId ? btn.ownerDocument.getElementById(reasonId) : null;
    btn.setAttribute("aria-checked", s.checked ? "true" : "false");
    if (s.reason) {
      btn.setAttribute("aria-disabled", "true");
      if (reasonId) btn.setAttribute("aria-describedby", reasonId);
      btn.title = s.reason;
      if (reasonEl) { reasonEl.textContent = s.reason; reasonEl.hidden = false; }
    } else {
      btn.removeAttribute("aria-disabled");
      btn.removeAttribute("aria-describedby");
      btn.title = s.hint || "";
      if (reasonEl) { reasonEl.textContent = ""; reasonEl.hidden = true; }
    }
    if (s.busy) btn.setAttribute("aria-busy", "true"); else btn.removeAttribute("aria-busy");
  }
  function switchBlocked(btn) { return btn.getAttribute("aria-disabled") === "true" || btn.getAttribute("aria-busy") === "true"; }
  function ago(iso) {
    const t = new Date(iso).getTime();
    if (!iso || isNaN(t)) return "";
    const s = Math.max(0, Math.round((Date.now() - t) / 1000));
    if (s < 60) return "just now";
    const m = Math.round(s / 60);
    if (m < 60) return `${m} min ago`;
    const h = Math.round(m / 60);
    if (h < 48) return `${h} h ago`;
    return `${Math.round(h / 24)} days ago`;
  }
  function emailController(ctx) {
    const base = `${ctx.apiBase}/email`;
    let doc = null;
    // Recipient rules: the two lists as stored, and the own addresses (always allowed, shown fixed).
    let lists = { always_allow: [], always_deny: [] };
    let selves = [];
    let confirming = false;
    let oauthFlow = null; // {id, cancelled}: the sign-in being awaited
    let tab = null; // imap | google | microsoft
    let discovery = null; // {address, domain, pending, defaults, error}
    let discoverTimer = null;
    let otherAccount = false; // "Use a different account" was chosen: the mailbox forms edit their own address
    const edited = new Set(); // server fields the user typed in: never overwritten by a pre-fill
    const busy = new Set();
    const timers = {};
    const el = (name) => role(ctx.host, name);
    const val = (name) => { const e = el(name); return e ? e.value.trim() : ""; };
    function errMsg(err) {
      const e = err && err.data && err.data.error;
      if (e && typeof e === "object" && e.cause) return e.fix ? `${e.cause} Fix: ${e.fix}` : e.cause;
      return errorText(err);
    }
    function errBody(err) { const e = err && err.data && err.data.error; return e && typeof e === "object" ? e : {}; }
    // The inline Connect error names the step that failed (DESIGN §6.2).
    function connectErrorText(err) {
      const e = errBody(err);
      const d = e.details || {};
      const where = d.host ? `${d.host}${d.port ? `:${d.port}` : ""}` : "";
      if (e.code === "email_auth_failed") return `Sign-in refused by ${d.host || "the mail server"} — check the password${e.fix ? ` (${e.fix})` : ""}.`;
      if (e.code === "email_unreachable") return `Couldn't reach ${where || "the mail server"}. ${e.cause || ""}`.trim();
      if (e.code === "email_tls_failed") return `The secure connection to ${where || "the mail server"} failed. ${e.fix || e.cause || ""}`.trim();
      if (e.code === "email_discovery_failed") return e.message || errMsg(err);
      return errMsg(err);
    }
    function showError(name, text) { const e = el(name); if (!e) return; e.textContent = text || ""; e.hidden = !text; }
    function setVal(name, v) { const e = el(name); if (e) e.value = v === null || v === undefined ? "" : String(v); }
    function setIfIdle(name, v) { const e = el(name); if (e && e.ownerDocument.activeElement !== e) setVal(name, v); }
    function flash(name, text, restore) {
      const e = el(name);
      if (!e) return;
      e.textContent = text;
      clearTimeout(timers[name]);
      timers[name] = setTimeout(() => { e.textContent = restore || ""; }, 2000);
    }
    function adopt(out) {
      if (out && out.schema === "email_settings_v1") {
        if (out.configured) otherAccount = false; // a later disconnect starts again from the Email address
        doc = { ...out, notices: (doc && doc.notices) || [], oauth_providers: out.oauth_providers || (doc && doc.oauth_providers) };
        fill(doc);
      }
      render();
    }
    function ownAddress(d) { return (d && (d.registered_address_stored || "")) || ""; }
    // ONE address question (DESIGN v2 §11 G5). Not connected and no Email
    // address: the mailbox form's address is the only address field (Connect
    // sets the Email address from it, EmailAccountStore.connect). An Email
    // address set: the mailbox forms show it as a read-only line, "Use a
    // different account" reveals their field (prefilled) and the Email address
    // turns read-only meanwhile. Connected: only the Email address field.
    // Never two editable address fields on screen.
    const ADDRESS_PANES = ["email-address", "email-oauth-address"];
    function renderAddressMode() {
      const d = doc || {};
      const connected = Boolean(d.configured);
      const own = ownAddress(d);
      const fixed = !connected && Boolean(own) && !otherAccount;
      const locked = !connected && Boolean(own) && otherAccount;
      el("email-identity-field").hidden = !connected && !own;
      el("email-identity-pending").hidden = connected || Boolean(own);
      const reg = el("email-registered");
      reg.readOnly = locked;
      if (locked) reg.setAttribute("aria-readonly", "true"); else reg.removeAttribute("aria-readonly");
      el("email-identity-save").hidden = locked;
      for (const name of ADDRESS_PANES) {
        el(`${name}-fixed`).hidden = !fixed;
        el(`${name}-fixed-value`).textContent = own;
        el(`${name}-field`).hidden = fixed;
        const back = el(`${name}-own`);
        back.hidden = !locked;
        back.textContent = locked ? `Use ${own} instead` : "";
        if (fixed && val(name) !== own) { setVal(name, own); if (name === "email-address") prefill(); }
      }
      for (const b of ctx.host.querySelectorAll('[data-acc-action="mailbox-address-reveal"]')) b.setAttribute("aria-expanded", otherAccount ? "true" : "false");
    }
    function fill(d) {
      setIfIdle("email-registered", ownAddress(d));
      const pol = d.policy || { mode: "allowlist", always_allow: [], always_deny: [], self_addresses: [] };
      if (!Array.isArray(pol.always_allow) || !Array.isArray(pol.always_deny) || !Array.isArray(pol.self_addresses)) {
        throw new Error("The email policy has no always_allow / always_deny / self_addresses (server older than the console?)");
      }
      setVal("email-policy-mode", pol.mode);
      lists = { always_allow: pol.always_allow.slice(), always_deny: pol.always_deny.slice() };
      selves = pol.self_addresses.slice();
      const lim = d.limits || {};
      setIfIdle("email-per-hour", lim.per_hour); setIfIdle("email-per-day", lim.per_day);
      setIfIdle("email-folder", d.imap ? d.imap.folder || "INBOX" : "");
      // The mailbox forms start from the Email address (IMAP: the mailbox to connect; Google/Microsoft: the sign-in hint).
      const own = ownAddress(d) || d.registered_address || "";
      if (!val("email-address") && own) { setVal("email-address", own); prefill(); }
      if (!val("email-oauth-address")) setVal("email-oauth-address", own);
    }
    function method(d) {
      if (d.auth_kind === "oauth2" && d.oauth) return OAUTH_LABELS[d.oauth.provider] || d.oauth.provider;
      return d.auth_kind === "oauth2" ? "OAuth2 sign-in" : "IMAP";
    }
    function oauthProvider(id) {
      const list = doc && doc.oauth_providers;
      if (!Array.isArray(list)) throw new Error("email_settings_v1 has no oauth_providers list (server older than the console?)");
      return list.find((p) => p && p.id === id) || { id, available: false, reason: `No ${OAUTH_LABELS[id] || id} sign-in is known to this server.` };
    }
    function chooseTab(next, opts) {
      const o = opts || {};
      tab = next;
      if (o.user) storeSet("local", EMAIL_TAB_KEY, next);
      for (const b of ctx.host.querySelectorAll("[data-acc-tab]")) {
        const on = b.dataset.accTab === next;
        b.setAttribute("aria-selected", on ? "true" : "false");
        b.tabIndex = on ? 0 : -1;
        if (on && o.focus) b.focus();
      }
      const oauth = el("email-panel-oauth");
      oauth.hidden = next === "imap";
      oauth.setAttribute("aria-labelledby", `acc-email-tab-${next === "imap" ? "google" : next}`);
      el("email-panel-imap").hidden = next !== "imap";
      const from = next === "imap" ? "email-oauth-address" : "email-address";
      const to = next === "imap" ? "email-address" : "email-oauth-address";
      if (!val(to) && val(from)) { setVal(to, val(from)); if (next === "imap") prefill(); }
      renderOauth();
    }
    function renderOauth() {
      if (!tab || tab === "imap") return;
      const label = OAUTH_LABELS[tab];
      const btn = el("email-oauth-start");
      btn.textContent = `Sign in with ${label}`;
      el("email-oauth-help").textContent = `The ${label} sign-in page opens with this address.`;
      el("email-oauth-microsoft-fields").hidden = tab !== "microsoft";
      let reason = "";
      if (!val("email-oauth-client-id")) {
        try {
          const p = oauthProvider(tab);
          if (!p.available) reason = p.reason || `No built-in ${label} sign-in client in this version: add your own client ID under Advanced.`;
        } catch (err) { reason = errorText(err); }
      }
      const reasonEl = el("email-oauth-reason");
      if (reason) {
        btn.setAttribute("aria-disabled", "true");
        btn.setAttribute("aria-describedby", "acc-email-oauth-reason");
        btn.title = reason;
        reasonEl.textContent = reason; reasonEl.hidden = false;
      } else {
        btn.removeAttribute("aria-disabled"); btn.removeAttribute("aria-describedby"); btn.title = "";
        reasonEl.textContent = ""; reasonEl.hidden = true;
      }
    }
    function serverText(s) { return s && s.host ? `${s.host} · ${s.port} · ${String(s.security || "").toUpperCase()}` : ""; }
    function domainOf(address) {
      const at = address.lastIndexOf("@");
      const domain = at > 0 ? address.slice(at + 1).trim().toLowerCase() : "";
      return domain && domain.includes(".") && !domain.endsWith(".") ? domain : "";
    }
    // Pre-fill every server field the user has not edited. `d` = the
    // `defaults` shape of `discovery.server_defaults` ({imap, smtp, login}).
    function applyDefaults(d) {
      const put = (name, v) => { if (!edited.has(name)) setVal(name, v); };
      put("email-imap-host", d.imap.host); put("email-imap-port", d.imap.port); put("email-imap-security", d.imap.security);
      put("email-smtp-host", d.smtp.host); put("email-smtp-port", d.smtp.port); put("email-smtp-security", d.smtp.security);
      put("email-username", d.login);
    }
    // The standard servers of a domain, the moment the address has one (the
    // same fallback `server_defaults` answers when discovery finds nothing).
    function standardDefaults(address, domain) {
      return {
        imap: { host: `imap.${domain}`, port: STANDARD_PORTS.imap.ssl, security: "ssl" },
        smtp: { host: `smtp.${domain}`, port: STANDARD_PORTS.smtp.ssl, security: "ssl" },
        login: address, source: "standard", provider: null,
        message: `Standard settings for ${domain} — change them if your provider uses others.`,
      };
    }
    function renderSource() {
      const line = el("email-servers-source");
      if (!discovery) { line.textContent = "Enter the mailbox address: the servers fill in from it."; return; }
      if (discovery.error) { line.textContent = discovery.error; return; }
      const d = discovery.defaults;
      let text = d.message;
      if (d.source === "discovered" && (d.provider === "google" || d.provider === "microsoft")) {
        text += ` You can also sign in on the ${OAUTH_LABELS[d.provider]} tab.`;
      }
      if (discovery.pending) text += " Checking…";
      line.textContent = text;
    }
    function prefill() {
      const address = val("email-address");
      const domain = domainOf(address);
      clearTimeout(discoverTimer);
      if (!domain) { discovery = null; renderSource(); return; }
      if (discovery && discovery.address === address && !discovery.error) return;
      const mine = { address, domain, pending: true, defaults: standardDefaults(address, domain), error: "" };
      discovery = mine;
      applyDefaults(mine.defaults);
      renderSource();
      discoverTimer = setTimeout(() => discover(mine), 400);
    }
    async function discover(mine) {
      try {
        const r = await ctx.request("POST", `${base}/discover`, { address: mine.address });
        if (discovery !== mine) return;
        if (!r || !r.defaults) throw new Error("the server's discovery answer has no defaults (server older than the console?)");
        mine.defaults = r.defaults;
        applyDefaults(r.defaults);
      } catch (err) {
        if (discovery !== mine) return;
        mine.error = `The server lookup failed (${errMsg(err)}): the standard settings for ${mine.domain} are filled in; change them if your provider uses others.`;
      }
      mine.pending = false;
      renderSource();
    }
    // Default tab: IMAP, unless this viewer picked another tab before.
    function pickDefaultTab() {
      const last = storeGet("local", EMAIL_TAB_KEY);
      chooseTab(EMAIL_TABS.includes(last) ? last : "imap", {});
    }
    function renderSwitches() {
      const d = doc || {};
      const at = d.agent_tools || { enabled: false, active: false, reason: "" };
      const connected = Boolean(d.configured);
      setSwitch(el("email-agent-tools"), {
        checked: Boolean(at.enabled),
        busy: busy.has("agent"),
        reason: connected ? "" : "Connect a mailbox first.",
      });
      el("email-agent-tools-desc").textContent = connected && at.enabled && d.enabled === false
        ? "On, but not working: the mailbox is paused (Mailbox → Active)."
        : "Your agents and workflows may list, search, read, send and reply to your mail. Every send still follows your recipient rules, your limits and the approval gate.";
      setSwitch(el("email-enabled"), {
        checked: connected && d.enabled !== false,
        busy: busy.has("mailbox"),
        reason: connected ? "" : "Connect a mailbox first.",
      });
    }
    function render() {
      const d = doc || {};
      const connected = Boolean(d.configured);
      const notices = (Array.isArray(d.notices) ? d.notices : []).concat(d.secret_warning ? [d.secret_warning] : []);
      el("email-notices").textContent = notices.join(" ");
      el("email-setup").hidden = connected;
      el("email-connected").hidden = !connected;
      if (connected) {
        const st = d.status || {};
        const parts = [`Connected as <strong>${esc(d.address)}</strong>`, esc(method(d))];
        if (d.enabled === false) parts.push("paused");
        let line = parts.join(" · ");
        if (st.last_error) line += ` · <span class="acc-status-error">last check failed: ${esc(st.last_error.cause || st.last_error.code)}${st.last_error.fix ? ` ${esc(st.last_error.fix)}` : ""}</span>`;
        else if (st.last_ok) line += ` · checked <span title="${esc(st.last_ok)}">${esc(ago(st.last_ok))}</span>`;
        else line += " · not tested yet";
        el("email-status").innerHTML = line;
      } else if (!tab) {
        pickDefaultTab();
      } else renderOauth();
      el("email-disconnect-confirm").hidden = !(connected && confirming);
      renderAddressMode();
      const differs = el("email-identity-differs");
      const own = ownAddress(d).toLowerCase();
      if (connected && own && d.address && own.trim() !== String(d.address).trim().toLowerCase()) {
        differs.textContent = `Your mailbox is a different account: ${d.address}.`;
        differs.hidden = false;
      } else { differs.textContent = ""; differs.hidden = true; }
      renderSwitches();
      renderEntries();
      const lim = d.limits || {};
      el("email-usage").textContent = lim.per_hour !== undefined && lim.per_hour !== null
        ? `${lim.used_last_hour} sent this hour, ${lim.used_last_day} today.` : "";
      const folder = el("email-folder");
      folder.disabled = !(connected && d.imap);
      el("email-folder-help").textContent = connected && d.imap
        ? `The folder your agents read on ${d.imap.host}. Empty = INBOX.`
        : "Connect a mailbox first: the folder is read on its IMAP server.";
      el("email-storage").textContent = d.secret_storage === "os-keychain" ? "The password or tokens are encrypted; the key is in the OS keychain."
        : d.secret_storage === "key-file" ? "The password or tokens are encrypted; the key is in a 0600 file." : "";
      if (d.config_file) el("email-scope").textContent = `These are the email settings of this AbstractCore install (${d.config_file}). A gateway keeps its own per user: each user sets theirs in the gateway console (My account).`;
    }
    function renderEntries() {
      for (const [which, name] of [["always_allow", "email-allow-entries"], ["always_deny", "email-deny-entries"]]) {
        const items = lists[which];
        const chip = (e) => selves.includes(e)
          // The own address is always allowed: a fixed chip, never removable.
          ? `<li class="acc-chip-fixed"><span>${esc(e)} (your address)</span></li>`
          : `<li><span>${esc(e)}</span><button type="button" class="acc-btn acc-link" data-acc-action="policy-remove" data-list="${which}" data-entry="${esc(e)}" aria-label="Remove ${esc(e)}" title="Remove ${esc(e)}">×</button></li>`;
        el(name).innerHTML = items.length
          ? items.map(chip).join("")
          : `<li class="acc-muted">${which === "always_allow" && val("email-policy-mode") === "allowlist" ? "Nobody yet: your agents may send only to your own address." : "Nobody yet."}</li>`;
      }
    }
    async function load() {
      try { doc = await ctx.request("GET", base); fill(doc); render(); }
      catch (err) { setMessage(ctx, `Email: ${errMsg(err)}`, "error"); }
    }
    function num(name) { const v = val(name); return v === "" ? null : Number(v); }
    async function run(label, fn) {
      try { const out = await fn(); adopt(out); return out; }
      catch (err) { setMessage(ctx, `${label}: ${errMsg(err)}`, "error"); return null; }
    }
    async function flip(key, btn, path, label, stateText) {
      if (switchBlocked(btn)) return;
      const next = btn.getAttribute("aria-checked") !== "true";
      busy.add(key);
      setSwitch(btn, { checked: next, busy: true });
      try {
        const out = await ctx.request("PUT", path, { enabled: next });
        busy.delete(key);
        adopt(out);
        setMessage(ctx, stateText(next), "ok");
      } catch (err) {
        busy.delete(key);
        render(); // reverts to the stored state
        setMessage(ctx, `Could not change "${label}" (it stays as it was): ${errMsg(err)}`, "error");
      }
    }
    async function savePolicy() {
      const out = await run("Recipient rules", () => ctx.request("PUT", `${base}/policy`, { mode: val("email-policy-mode"), always_allow: lists.always_allow, always_deny: lists.always_deny }));
      if (out) flash("email-policy-saved", "Saved");
      else if (doc) { fill(doc); renderEntries(); } // not saved: show what is stored
    }
    async function saveLimits() {
      const lim = (doc && doc.limits) || {};
      const hour = num("email-per-hour"); const day = num("email-per-day");
      if (hour === lim.per_hour && day === lim.per_day) return;
      const out = await run("Send limits", () => ctx.request("PUT", `${base}/limits`, { per_hour: hour, per_day: day }));
      if (out) flash("email-limits-saved", "Saved");
      else { setVal("email-per-hour", lim.per_hour); setVal("email-per-day", lim.per_day); }
    }
    async function saveFolder() {
      const d = doc || {};
      if (!d.imap) return;
      const v = val("email-folder") || "INBOX";
      if (v === (d.imap.folder || "INBOX")) { setVal("email-folder", v); return; }
      const out = await run("Folder", () => ctx.request("PUT", `${base}/folder`, { folder: v }));
      if (out) flash("email-folder-saved", "Saved");
      else setVal("email-folder", d.imap.folder || "INBOX");
    }
    async function saveIdentity() {
      const btn = el("email-identity-save");
      if (btn.getAttribute("aria-busy") === "true") return;
      const v = val("email-registered");
      showError("email-identity-error", "");
      el("email-registered").removeAttribute("aria-invalid");
      btn.setAttribute("aria-busy", "true");
      try {
        const out = await ctx.request("PUT", `${base}/registered-address`, { address: v });
        adopt(out);
        setVal("email-registered", ownAddress(doc));
        flash("email-identity-save", "Saved", "Save");
        setMessage(ctx, v ? "Email address saved." : "Email address cleared: the mailbox's own address is used.", "ok");
      } catch (err) {
        el("email-registered").setAttribute("aria-invalid", "true");
        showError("email-identity-error", errMsg(err));
      } finally { btn.removeAttribute("aria-busy"); }
    }
    async function connectOther() {
      const btn = el("email-connect");
      if (btn.getAttribute("aria-busy") === "true") return;
      showError("email-connect-error", "");
      const addr = val("email-address");
      const need = (name, text) => { if (val(name)) return false; showError("email-connect-error", text); el(name).focus(); return true; };
      if (need("email-address", "Enter the mailbox address.")) return;
      if (!el("email-password").value) { showError("email-connect-error", "Enter the password (or app password) of the mailbox."); el("email-password").focus(); return; }
      if (need("email-imap-host", "Enter the incoming mail (IMAP) server.")) return;
      if (need("email-smtp-host", "Enter the outgoing mail (SMTP) server.")) return;
      const ca = val("email-ca-file");
      // Every server field is on screen: they are sent as shown. The login is
      // the discovered form (or what the user typed under "different login
      // name"); no display name: the server keeps the stored one, else the
      // address's local part.
      const body = {
        address: addr, username: val("email-username"), password: el("email-password").value, test: true,
        imap: { host: val("email-imap-host"), port: num("email-imap-port"), security: val("email-imap-security"), folder: "INBOX", ca_file: ca },
        smtp: { host: val("email-smtp-host"), port: num("email-smtp-port"), security: val("email-smtp-security"), ca_file: ca },
      };
      btn.setAttribute("aria-busy", "true");
      btn.textContent = "Connecting…";
      try {
        const out = await ctx.request("PUT", base, body);
        setVal("email-password", "");
        adopt(out);
        setMessage(ctx, `Mailbox connected as ${out.address}.`, "ok");
      } catch (err) {
        showError("email-connect-error", connectErrorText(err));
      } finally { btn.removeAttribute("aria-busy"); btn.textContent = "Connect"; }
    }
    async function startOauth() {
      const btn = el("email-oauth-start");
      if (oauthFlow || btn.getAttribute("aria-disabled") === "true") return;
      showError("email-oauth-error", "");
      const provider = tab;
      const body = {
        address: val("email-oauth-address"), provider,
        client_id: val("email-oauth-client-id"), client_secret: el("email-oauth-client-secret").value,
        tenant: provider === "microsoft" ? val("email-oauth-tenant") : "", flow: provider === "microsoft" ? val("email-oauth-flow") : "",
      };
      if (!body.address) { showError("email-oauth-error", "Enter the email address of the mailbox."); el("email-oauth-address").focus(); return; }
      const prompt = el("email-oauth-prompt");
      prompt.textContent = "";
      btn.setAttribute("aria-busy", "true");
      let start;
      try { start = await ctx.request("POST", `${base}/oauth/start`, body); }
      catch (err) { btn.removeAttribute("aria-busy"); showError("email-oauth-error", errMsg(err)); return; }
      const flow = { id: start.flow_id, cancelled: false };
      oauthFlow = flow;
      el("email-oauth-cancel").hidden = false;
      // Only https links are rendered as links (the endpoints are validated https server-side too).
      const link = (url, label) => String(url || "").toLowerCase().startsWith("https://")
        ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(label)}</a>` : `<code>${esc(url)}</code>`;
      if (start.flow === "device") {
        prompt.innerHTML = `Open ${link(start.verification_uri, start.verification_uri)} and enter the code <code>${esc(start.user_code)}</code>. Waiting for the approval…`;
      } else {
        prompt.innerHTML = `Open ${link(start.authorization_url, "the provider's sign-in page")} in a browser on the machine running AbstractCore. Waiting for the approval…`;
      }
      try {
        while (!flow.cancelled) {
          const out = await ctx.request("POST", `${base}/oauth/finish`, { flow_id: flow.id, wait_s: 20 });
          if (flow.cancelled) break;
          if (out && out.pending) continue;
          adopt(out);
          prompt.textContent = "";
          setVal("email-oauth-client-secret", "");
          setMessage(ctx, `Mailbox connected as ${out.address} (${OAUTH_LABELS[provider] || provider}).`, "ok");
          break;
        }
      } catch (err) {
        if (!flow.cancelled) { prompt.textContent = ""; showError("email-oauth-error", errMsg(err)); }
      } finally {
        if (oauthFlow === flow) oauthFlow = null;
        btn.removeAttribute("aria-busy");
        el("email-oauth-cancel").hidden = true;
      }
    }
    async function action(name, target) {
      if (name === "refresh") { await load(); return; }
      if (name === "identity-save") { await saveIdentity(); return; }
      if (name === "email-connect") { await connectOther(); return; }
      if (name === "mailbox-address-reveal" || name === "mailbox-address-own") {
        otherAccount = name === "mailbox-address-reveal";
        renderAddressMode();
        const pane = tab === "imap" ? "email-address" : "email-oauth-address";
        if (otherAccount) { el(pane).focus(); el(pane).select(); }
        else el(`${pane}-fixed`).querySelector("button").focus();
        return;
      }
      if (name === "login-reveal" || name === "ca-reveal") {
        const [field, link, input] = name === "login-reveal"
          ? ["email-login-field", "email-login-link", "email-username"] : ["email-ca-field", "email-ca-link", "email-ca-file"];
        el(field).hidden = false;
        el(link).setAttribute("aria-expanded", "true");
        el(input).focus();
        return;
      }
      if (name === "email-test") {
        const btn = el("email-test");
        if (btn.getAttribute("aria-busy") === "true") return;
        btn.setAttribute("aria-busy", "true"); btn.textContent = "Testing…";
        try {
          const out = await ctx.request("POST", `${base}/test`);
          const failed = ["imap", "smtp"].map((k) => out[k]).filter((l) => l && l.ok === false);
          setMessage(ctx, failed.length ? `Test failed: ${failed[0].cause} ${failed[0].fix || ""}`.trim() : "Test passed: reading and sending work.", failed.length ? "error" : "ok");
        } catch (err) { setMessage(ctx, `Test: ${errMsg(err)}`, "error"); }
        finally { btn.removeAttribute("aria-busy"); btn.textContent = "Test"; }
        await load();
        return;
      }
      if (name === "agent-tools-switch") {
        await flip("agent", el("email-agent-tools"), `${base}/agent-tools`, "Agent email tools", (on) => (on ? "Agent email tools are on." : "Agent email tools are off."));
        return;
      }
      if (name === "mailbox-switch") {
        await flip("mailbox", el("email-enabled"), `${base}/enabled`, "Active", (on) => (on ? "The mailbox is active." : "The mailbox is paused: no watching, no sending; your settings are kept."));
        return;
      }
      if (name === "oauth-start") { await startOauth(); return; }
      if (name === "oauth-cancel") {
        const flow = oauthFlow;
        if (!flow) return;
        flow.cancelled = true;
        oauthFlow = null;
        el("email-oauth-cancel").hidden = true;
        el("email-oauth-start").removeAttribute("aria-busy");
        el("email-oauth-prompt").textContent = "";
        try { await ctx.request("POST", `${base}/oauth/cancel`, { flow_id: flow.id }); } catch (_err) { /* the flow expires on its own */ }
        setMessage(ctx, "Sign-in cancelled.", "ok");
        return;
      }
      if (name === "email-disconnect") { confirming = true; render(); return; }
      if (name === "email-disconnect-cancel") { confirming = false; render(); return; }
      if (name === "email-disconnect-confirm") {
        confirming = false;
        const addr = (doc && doc.address) || "";
        const out = await run("Disconnect", () => ctx.request("DELETE", base));
        if (out) setMessage(ctx, `Mailbox ${addr} disconnected. Policy and limits are kept.`, "ok");
        return;
      }
      if (name === "policy-add-allow" || name === "policy-add-deny") {
        const which = name === "policy-add-allow" ? "always_allow" : "always_deny";
        const field = which === "always_allow" ? "email-allow-new" : "email-deny-new";
        const v = val(field);
        if (!v || lists[which].includes(v.toLowerCase())) { setVal(field, ""); return; }
        lists[which] = lists[which].concat([v]);
        setVal(field, "");
        renderEntries();
        await savePolicy();
        return;
      }
      if (name === "policy-remove") {
        const which = target.dataset.list;
        lists[which] = lists[which].filter((e) => e !== target.dataset.entry);
        renderEntries();
        await savePolicy();
        return;
      }
    }
    const listeners = [];
    function on(name, type, fn) { const e = el(name); if (e) { e.addEventListener(type, fn); listeners.push([e, type, fn]); } }
    const onEnter = (fn) => (e) => { if (e.key === "Enter") { e.preventDefault(); fn(); } };
    on("email-policy-mode", "change", () => { renderEntries(); savePolicy(); });
    on("email-per-hour", "change", saveLimits);
    on("email-per-day", "change", saveLimits);
    on("email-folder", "change", saveFolder);
    on("email-registered", "keydown", onEnter(saveIdentity));
    on("email-address", "input", prefill);
    on("email-password", "keydown", onEnter(connectOther));
    on("email-oauth-client-id", "input", renderOauth);
    on("email-allow-new", "keydown", onEnter(() => action("policy-add-allow")));
    on("email-deny-new", "keydown", onEnter(() => action("policy-add-deny")));
    for (const f of SERVER_FIELDS) on(f, "input", () => edited.add(f));
    // A security change moves an unedited port to that security's standard port.
    for (const leg of ["imap", "smtp"]) {
      on(`email-${leg}-security`, "change", () => {
        edited.add(`email-${leg}-security`);
        if (!edited.has(`email-${leg}-port`)) setVal(`email-${leg}-port`, STANDARD_PORTS[leg][val(`email-${leg}-security`)]);
      });
    }
    const tablist = ctx.host.querySelector('[role="tablist"]');
    const onTabClick = (e) => { const b = e.target.closest("[data-acc-tab]"); if (b) chooseTab(b.dataset.accTab, { user: true }); };
    const onTabKey = (e) => {
      const i = EMAIL_TABS.indexOf(tab);
      let next = null;
      if (e.key === "ArrowRight") next = EMAIL_TABS[(i + 1) % EMAIL_TABS.length];
      else if (e.key === "ArrowLeft") next = EMAIL_TABS[(i + EMAIL_TABS.length - 1) % EMAIL_TABS.length];
      else if (e.key === "Home") next = EMAIL_TABS[0];
      else if (e.key === "End") next = EMAIL_TABS[EMAIL_TABS.length - 1];
      if (next) { e.preventDefault(); chooseTab(next, { user: true, focus: true }); }
    };
    if (tablist) { tablist.addEventListener("click", onTabClick); tablist.addEventListener("keydown", onTabKey); }
    return {
      refresh: load, render, jobFinished() {},
      action(name, target) { action(name, target); },
      key(k) { if (k === "r") { load(); return true; } return false; },
      destroy() {
        if (oauthFlow) oauthFlow.cancelled = true;
        clearTimeout(discoverTimer);
        for (const t of Object.values(timers)) clearTimeout(t);
        for (const [e, type, fn] of listeners) e.removeEventListener(type, fn);
        if (tablist) { tablist.removeEventListener("click", onTabClick); tablist.removeEventListener("keydown", onTabKey); }
      },
    };
  }

  // ----------------------------------------------------------------- mount
  const CONTROLLERS = { models: modelsController, engines: enginesController, overview: overviewController, providers: providersController, email: emailController };
  const mounts = new Map();
  function normalizeKind(kind) {
    const k = String(kind || "").toLowerCase();
    if (k === "catalog") return "models";
    if (!CONTROLLERS[k]) throw new Error(`AbstractCoreConsole.mount: unknown kind "${kind}" (expected models or engines)`);
    return k;
  }
  function deriveServerRoot(apiBase) { return apiBase.endsWith("/acore") ? apiBase.slice(0, -"/acore".length) : ""; }
  function isVisible(el) { return !!(el && el.isConnected && el.getClientRects().length); }

  function mount(kind, rootEl, options) {
    const k = normalizeKind(kind);
    if (!rootEl || typeof rootEl.querySelector !== "function") throw new Error("AbstractCoreConsole.mount: rootEl must be a DOM element");
    if (mounts.has(rootEl)) mounts.get(rootEl).unmount();
    const opts = options || {};
    const selector = `.acc-root[data-acc-kind="${k}"]`;
    let host = rootEl.matches && rootEl.matches(selector) ? rootEl : rootEl.querySelector(selector);
    if (!host) { rootEl.innerHTML = TEMPLATES[k]; host = rootEl.querySelector(selector); }
    const apiBase = String(opts.apiBase === undefined || opts.apiBase === null ? "/acore" : opts.apiBase).replace(/\/+$/, "");
    const adminOpt = opts.isAdmin;
    const ctx = {
      kind: k,
      root: rootEl,
      host,
      apiBase,
      serverRoot: typeof opts.serverRoot === "string" ? opts.serverRoot.replace(/\/+$/, "") : deriveServerRoot(apiBase),
      request: typeof opts.request === "function" ? opts.request : defaultRequest,
      isAdmin: () => (typeof adminOpt === "function" ? !!adminOpt() : adminOpt === undefined ? true : !!adminOpt),
      onJob: typeof opts.onJob === "function" ? opts.onJob : null,
      hostName: String(opts.hostName || (window.location && window.location.hostname) || "this host"),
      cliPrefix: String(opts.cliPrefix || "abstractcore"),
      alive: true,
      timers: new Set(),
      modal: null,
    };
    let controller = null;
    ctx.jobs = makeJobTracker(ctx, {
      changed() { if (controller && ctx.alive) controller.render(); },
      finished() { if (controller && ctx.alive) controller.jobFinished(); },
    });
    controller = CONTROLLERS[k](ctx);
    for (const b of host.querySelectorAll('[data-acc-action="collapse"]')) setCollapsed(b, Boolean(collapsedSections()[b.dataset.accSection]), false);

    const onClick = async (e) => {
      const el = e.target.closest("[data-acc-action]");
      if (!el || !host.contains(el)) return;
      const name = el.dataset.accAction;
      if (name === "copy") {
        el.textContent = (await copyText(el.dataset.text || "")) ? "Copied" : "Copy failed — select and copy";
        setTimeout(() => { el.textContent = "Copy"; }, 1500);
        return;
      }
      if (name === "cancel-job") {
        try { await ctx.jobs.cancel(el.dataset.jobId); } catch (err) { setMessage(ctx, `Cancel: ${errorText(err)}`, "error"); }
        return;
      }
      if (name === "dismiss-job") { ctx.jobs.dismiss(el.dataset.jobId); return; }
      if (name === "collapse") { setCollapsed(el, el.getAttribute("aria-expanded") === "true", true); return; }
      controller.action(name, el);
    };
    const onKey = (e) => {
      if (!ctx.alive || !isVisible(host) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (ctx.modal) { if (e.key === "Escape") closeModal(ctx); return; }
      if (document.getElementById("acc-auth-modal")) return;
      const t = e.target;
      const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
      if (typing) { if (e.key === "Escape" && host.contains(t)) t.blur(); return; }
      if (t && t !== document.body && !host.contains(t) && t.closest && t.closest("[contenteditable], dialog")) return;
      const row = document.activeElement && host.contains(document.activeElement) ? document.activeElement.closest("tr[data-acc-row]") : null;
      if (e.key === "c" && row) {
        const job = ctx.jobs.list().find((j) => !TERMINAL.has(j.status) && !j._lost && ((row.dataset.engine && j.engine === row.dataset.engine) || (j.provider === row.dataset.provider && j.artifact === row.dataset.artifact)));
        if (job) { e.preventDefault(); ctx.jobs.cancel(job.job_id).catch((err) => setMessage(ctx, `Cancel: ${errorText(err)}`, "error")); }
        return;
      }
      if (controller.key(e.key, row)) e.preventDefault();
    };
    host.addEventListener("click", onClick);
    document.addEventListener("keydown", onKey);

    const handle = {
      kind: k,
      refresh: () => controller.refresh(),
      unmount() {
        ctx.alive = false;
        for (const t of ctx.timers) clearTimeout(t);
        ctx.timers.clear();
        closeModal(ctx);
        controller.destroy();
        host.removeEventListener("click", onClick);
        document.removeEventListener("keydown", onKey);
        mounts.delete(rootEl);
      },
    };
    mounts.set(rootEl, handle);
    ctx.jobs.restore();
    controller.refresh();
    return handle;
  }

  // ------------------------------------------------- standalone page (boot)
  function boot(config) {
    const cfg = config || {};
    const apiBase = cfg.apiBase || "/acore";
    const themes = Array.isArray(cfg.themes) ? cfg.themes : [];
    const lightIds = new Set((cfg.lightThemeIds || []).concat(["light"]));
    const mq = window.matchMedia ? window.matchMedia("(prefers-color-scheme: light)") : null;
    const themeSelect = document.getElementById("acc-theme");
    const themeToggle = document.getElementById("acc-theme-toggle");
    function effectiveTheme(choice) { return !choice || choice === "auto" ? (mq && mq.matches ? "light" : "dark") : choice; }
    function applyTheme(choice) {
      const id = effectiveTheme(choice);
      const rootEl = document.documentElement;
      for (const c of Array.from(rootEl.classList)) if (c.startsWith("theme-")) rootEl.classList.remove(c);
      if (id !== "dark") rootEl.classList.add(`theme-${id}`);
      rootEl.setAttribute("data-acc-theme", id);
      // A switch showing the state (on = a dark theme is showing), never the action.
      if (themeToggle) themeToggle.setAttribute("aria-checked", lightIds.has(id) ? "false" : "true");
    }
    let choice = storeGet("local", THEME_KEY) || "auto";
    if (themeSelect) {
      themeSelect.innerHTML = `<option value="auto">Auto (system)</option>` + themes.map((t) => `<option value="${esc(t.id)}">${esc(t.label)}</option>`).join("");
      themeSelect.value = themes.some((t) => t.id === choice) ? choice : "auto";
      themeSelect.addEventListener("change", () => { choice = themeSelect.value; storeSet("local", THEME_KEY, choice === "auto" ? "" : choice); applyTheme(choice); });
    }
    if (themeToggle) themeToggle.addEventListener("click", () => {
      choice = lightIds.has(effectiveTheme(choice)) ? "dark" : "light";
      storeSet("local", THEME_KEY, choice);
      if (themeSelect) themeSelect.value = choice;
      applyTheme(choice);
    });
    if (mq && mq.addEventListener) mq.addEventListener("change", () => { if (!choice || choice === "auto") applyTheme(choice); });
    applyTheme(choice);

    const forget = document.getElementById("acc-forget-token");
    const syncForget = () => { if (forget) forget.hidden = !getToken(); };
    if (forget) forget.addEventListener("click", () => { setToken(""); syncForget(); });

    // About (kit 0.7.0 compact card, rendered by the server): a native modal
    // dialog; a click on the backdrop (the dialog element itself) closes it.
    const aboutDialog = document.getElementById("acc-about");
    const aboutOpen = document.getElementById("acc-about-open");
    if (aboutDialog && aboutOpen && typeof aboutDialog.showModal === "function") {
      aboutOpen.addEventListener("click", () => {
        aboutDialog.showModal();
        const close = aboutDialog.querySelector('[data-action="close-about"]');
        if (close) close.focus();
      });
      aboutDialog.addEventListener("click", (ev) => {
        const t = ev.target;
        if (t === aboutDialog || (t && t.closest && t.closest('[data-action="close-about"]'))) aboutDialog.close();
      });
      aboutDialog.addEventListener("close", () => aboutOpen.focus());
    } else if (aboutOpen) {
      aboutOpen.hidden = true;
    }

    const healthDot = document.getElementById("acc-health-dot");
    const healthText = document.getElementById("acc-health-text");
    async function health() {
      try {
        const h = await defaultRequest("GET", `${deriveServerRoot(apiBase)}/health`);
        healthDot.className = "acc-dot acc-ok";
        healthText.textContent = `${h.status || "ok"} | v${h.version || "?"}`;
      } catch (err) {
        healthDot.className = "acc-dot acc-err";
        healthText.textContent = errorText(err);
      }
      syncForget();
    }

    const TABS = ["overview", "catalog", "engines", "providers", "email"];
    const mounted = {};
    function show(tab) {
      if (!TABS.includes(tab)) tab = "overview";
      for (const id of TABS) {
        const btn = document.getElementById(`acc-tab-button-${id}`);
        const panel = document.getElementById(`acc-tab-${id}`);
        const on = id === tab;
        if (btn) { btn.setAttribute("aria-selected", on ? "true" : "false"); btn.tabIndex = on ? 0 : -1; }
        if (panel) panel.hidden = !on;
      }
      const panel = document.getElementById(`acc-tab-${tab}`);
      if (!mounted[tab]) {
        mounted[tab] = mount(tab === "catalog" ? "models" : tab, panel, { apiBase, isAdmin: true, cliPrefix: "abstractcore" });
      } else {
        mounted[tab].refresh();
      }
      health();
    }
    for (const id of TABS) {
      const btn = document.getElementById(`acc-tab-button-${id}`);
      if (btn) btn.addEventListener("click", () => { if (window.location.hash !== `#${id}`) window.location.hash = id; else show(id); });
    }
    // One-time console link (`/console#claim=<code>`, printed by
    // `abstractcore serve`): strip it from the address bar at once, trade it
    // for the bearer token, keep the token in this tab only.
    const claim = /^#claim=([A-Za-z0-9_-]{16,128})$/.exec(window.location.hash || "");
    if (claim) {
      try { window.history.replaceState(null, "", window.location.pathname + window.location.search); } catch (e) { window.location.hash = ""; }
    }
    const start = () => {
      window.addEventListener("hashchange", () => show(window.location.hash.slice(1)));
      show(window.location.hash.slice(1) || "overview");
    };
    if (!claim) { start(); return; }
    redeemClaim(apiBase, claim[1]).then((problem) => {
      if (problem) {
        const main = document.querySelector(".acc-main");
        if (main) main.insertAdjacentHTML("afterbegin", `<p class="acc-error" id="acc-claim-error" role="alert">${esc(problem)}</p>`);
      }
      syncForget();
      start();
    });
  }

  async function redeemClaim(apiBase, code) {
    try {
      const res = await fetch(`${apiBase}/session/claim`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ code }),
      });
      let data = {};
      try { data = await res.json(); } catch (e) { data = {}; }
      if (res.ok && data && data.token) { setToken(data.token); return ""; }
      if (getToken()) return "";  // this tab is already signed in; the used link is harmless
      return (data && data.message) || `The console link was refused (HTTP ${res.status}). Run \`abstractcore serve --claim-url\` for a fresh one.`;
    } catch (e) {
      return `Could not redeem the console link: ${(e && e.message) || e}`;
    }
  }

  window.AbstractCoreConsole = Object.freeze({
    version: "__ACC_VERSION__",
    kinds: Object.freeze(["models", "engines"]),
    labels: Object.freeze({ weights: WEIGHT_LABELS, fit: FIT_LABELS, jobs: JOB_LABELS }),
    mount,
    unmount(rootEl) { const h = mounts.get(rootEl); if (h) h.unmount(); },
    defaultRequest,
    setToken,
    boot,
  });
})();
"""


def _console_js() -> str:
    return (
        _JS_TEMPLATE.replace("__ACC_TEMPLATES__", _script_json(_TEMPLATES))
        .replace("__ACC_VERSION__", CONSOLE_JS_VERSION)
        .strip()
        + "\n"
    )


def fragment(kind: str) -> Dict[str, str]:
    """Self-contained tab body for embedding (contract G).

    Returns ``{"html", "js", "css"}``. ``js`` and ``css`` are identical for
    every kind (one shared namespace/stylesheet, safe to include once or
    twice); ``html`` is the tab skeleton, which ``AbstractCoreConsole.mount``
    also injects itself if the host did not splice it.
    """
    k = str(kind or "").strip().lower()
    if k not in FRAGMENT_KINDS:
        raise ValueError(f"unknown console fragment kind {kind!r}; expected one of {FRAGMENT_KINDS}")
    return {"html": _TEMPLATES[k], "js": _console_js(), "css": _FRAGMENT_CSS.strip() + "\n"}


_ABOUT_DIALOG_CSS = """
.acc-icon-btn { display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 34px; padding: 0; }
dialog.acc-about { max-width: min(460px, calc(100vw - 32px)); margin: auto; color: var(--text-primary, #eee); }
dialog.acc-about::backdrop { background: rgba(0, 0, 0, 0.45); -webkit-backdrop-filter: blur(6px); backdrop-filter: blur(6px); }
dialog.acc-about[open] { display: block; padding: 16px 18px; border-radius: var(--radius-lg, 12px); border: 1px solid var(--ui-border-1, rgba(255,255,255,.12)); background: var(--bg-card, var(--bg-secondary, #16213e)); box-shadow: var(--ui-shadow-1, 0 8px 30px rgba(0,0,0,.35)); }
@media (pointer: coarse) { .acc-icon-btn { width: var(--tap-min, 44px); height: var(--tap-min, 44px); } }
"""


def console_about_versions() -> Dict[str, Any]:
    """Versions the AbstractCore console's About states: this AbstractCore,
    and the AbstractFramework and AbstractGateway installed on this host
    (the console is served by ``abstractcore serve``, not through a gateway).
    A missing distribution is said ("not installed on this host"), never
    invented. No package list (operator, round 5)."""
    from importlib import metadata

    from ..utils.identity import installed_version

    def _version(dist: str) -> str:
        try:
            return installed_version(dist)
        except metadata.PackageNotFoundError:
            return ""

    framework = _version("abstractframework")
    gateway = _version("abstractgateway")
    return {
        "core": _version("abstractcore") or "version not reported",
        "framework": framework or None,
        "framework_note": "" if framework else "not installed on this host",
        "gateway": gateway or None,
        "gateway_note": "" if gateway else "not installed on this host",
    }


def console_about_card_html() -> str:
    """The About card of the standalone console: the kit's compact card markup
    (`abstractcore.utils.identity.about_card_html`, the `AfAbout` twin) with a
    Close button in its heading row."""
    from ..utils.identity import about_card_html, app_identity

    v = console_about_versions()
    close = (
        '<button type="button" class="af-about-card__close" data-action="close-about" aria-label="Close" title="Close">'
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6 6 18"></path><path d="m6 6 12 12"></path></svg>'
        "</button>"
    )
    return about_card_html(
        app_identity("abstractcore", v["core"]),
        v["framework"],
        v["gateway"],
        framework_note=v["framework_note"],
        gateway_note=v["gateway_note"],
        title_id="acc-about-title",
        action_html=close,
    )


def render_console_html(api_base: str = "/acore", title: str = "AbstractCore Console") -> str:
    """The standalone console page (Overview, Models, Engines, Providers, Email)."""
    from html import escape

    from .themes import KIT_ABOUT_CSS, KIT_FORM_CSS, KIT_LIGHT_THEME_IDS, KIT_RESPONSIVE_CSS, KIT_ROOT_CSS, KIT_SWITCH_CSS, KIT_THEME_CSS, KIT_THEME_SPECS

    about_card = console_about_card_html()
    base = str(api_base or "").rstrip("/")
    config = {
        "apiBase": base,
        "title": title,
        "themes": [{"id": s["id"], "label": s["label"], "group": s["group"]} for s in KIT_THEME_SPECS],
        "lightThemeIds": list(KIT_LIGHT_THEME_IDS),
    }
    t = escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, interactive-widget=resizes-content">
<meta name="theme-color" content="#1a1a2e">
<title>{t}</title>
<style>
/* ui-kit base tokens + per-theme blocks: GENERATED copy (abstractcore/console/themes.py). */
{KIT_ROOT_CSS}

{KIT_RESPONSIVE_CSS}

{KIT_THEME_CSS}

/* ui-kit af-switch (the on/off control) and af-form (form layout, cards,
   tabs): GENERATED verbatim copies (themes.py KIT_SWITCH_CSS, KIT_FORM_CSS). */
{KIT_SWITCH_CSS}
{KIT_FORM_CSS}
/* About: the kit's compact card (themes.py KIT_ABOUT_CSS, GENERATED) in a
   native modal dialog element (Escape + focus containment by the browser). */
{KIT_ABOUT_CSS}
{_ABOUT_DIALOG_CSS}
{_PAGE_CSS}
{_FRAGMENT_CSS}
{_EMAIL_CSS}
</style>
</head>
<body>
<header class="acc-topbar">
  <div class="acc-brand">{t}<small>{escape(base)}</small></div>
  <span class="acc-health" id="acc-health"><span class="acc-dot" id="acc-health-dot"></span><span id="acc-health-text">checking...</span></span>
  <label class="acc-health">Theme <select id="acc-theme" aria-label="Theme"></select></label>
  <button type="button" role="switch" class="af-switch af-switch--sm" id="acc-theme-toggle" aria-checked="true" title="Dark or light theme (the Theme list has every theme)">
    <span class="af-switch__track" aria-hidden="true"><span class="af-switch__thumb"></span></span>
    <span class="af-switch__text"><span class="af-switch__label">Dark theme</span></span>
  </button>
  <button type="button" id="acc-forget-token" hidden title="Forget the server token stored in this tab">Forget token</button>
  <button type="button" id="acc-about-open" class="acc-icon-btn" aria-haspopup="dialog" aria-controls="acc-about" aria-label="About AbstractCore" title="About"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"></circle><path d="M12 16v-4"></path><path d="M12 8h.01"></path></svg></button>
</header>
<dialog id="acc-about" class="af-appearance af-about acc-about" aria-labelledby="acc-about-title">{about_card}</dialog>
<nav class="acc-tabs" role="tablist" aria-label="Console sections">
  <button type="button" role="tab" id="acc-tab-button-overview" aria-controls="acc-tab-overview" aria-selected="true">Overview</button>
  <button type="button" role="tab" id="acc-tab-button-catalog" aria-controls="acc-tab-catalog" aria-selected="false">Models</button>
  <button type="button" role="tab" id="acc-tab-button-engines" aria-controls="acc-tab-engines" aria-selected="false">Engines</button>
  <button type="button" role="tab" id="acc-tab-button-providers" aria-controls="acc-tab-providers" aria-selected="false">Providers</button>
  <button type="button" role="tab" id="acc-tab-button-email" aria-controls="acc-tab-email" aria-selected="false">Email</button>
</nav>
<main class="acc-main">
  <section class="acc-panel" role="tabpanel" id="acc-tab-overview" aria-labelledby="acc-tab-button-overview">{_OVERVIEW_HTML}</section>
  <section class="acc-panel" role="tabpanel" id="acc-tab-catalog" aria-labelledby="acc-tab-button-catalog" hidden>{_MODELS_HTML}</section>
  <section class="acc-panel" role="tabpanel" id="acc-tab-engines" aria-labelledby="acc-tab-button-engines" hidden>{_ENGINES_HTML}</section>
  <section class="acc-panel" role="tabpanel" id="acc-tab-providers" aria-labelledby="acc-tab-button-providers" hidden>{_PROVIDERS_HTML}</section>
  <section class="acc-panel" role="tabpanel" id="acc-tab-email" aria-labelledby="acc-tab-button-email" hidden>{_EMAIL_HTML}</section>
</main>
<noscript><p style="padding:20px">The AbstractCore console needs JavaScript.</p></noscript>
<script>
{_console_js()}</script>
<script>
window.AbstractCoreConsole.boot({_script_json(config)});
</script>
</body>
</html>
"""
