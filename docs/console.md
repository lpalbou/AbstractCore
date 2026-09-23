# Web Console

`abstractcore serve` ships a browser console at `http://localhost:8000/console`. Use it to see
what this machine can run, browse and download models, delete weights you no longer need, and
install local inference engines such as Ollama, without writing a command. Every action the
console takes shows the equivalent `abstractcore` CLI command, so you can repeat it in a terminal
or a script.

The console is a view over the HTTP API described in [Server](server.md). It adds no logic of its
own: presence, fit verdicts, download progress and install plans all come from the server. The
same Models and Engines screens are embedded in the AbstractGateway console, so both entry points
look and behave the same.

## Open the console

```bash
pip install "abstractcore[server]"
abstractcore serve
# then open http://localhost:8000/console
```

The server landing page (`/`) links to it as well.

## Tabs

| Tab | What you see | What you can do | CLI equivalent |
|---|---|---|---|
| **Overview** | This machine: OS, architecture, accelerator, RAM or unified memory, the usable model-memory ceiling and its source, free memory now, free disk for each model cache, Python version. An engines summary and server health (version, whether a bearer token is required). | Jump to Models or Engines. | `abstractcore host profile --json`, `abstractcore engines status --probe` |
| **Models** | The model catalog, one row per downloadable artifact (provider, artifact id, quantization, download size, weights status, fit badge), the models already installed with their sizes, and any running jobs with progress bars. | Search, filter, Download, Delete, Cancel a job. | `abstractcore models search <q> --engine <id> --fits`, `abstractcore models download`, `abstractcore models delete`, `abstractcore models list` |
| **Engines** | Each engine (Ollama, LM Studio, MLX, llama.cpp, vLLM, Hugging Face): supported on this host, installed and version, running and reachable, base URL, model count. | Install (with confirmation), Open download page, Refresh (live probe). | `abstractcore engines status --probe`, `abstractcore engines install <id>`, `abstractcore engines open <id>` |
| **Providers** | Read-only list of providers (`GET /providers`) and the configured capability defaults (`GET /v1/config/capability-defaults`). | Nothing to edit here. | `abstractcore config providers --probe`, `abstractcore config defaults` |

API keys and default models are configured on the host with `abstractcore --config`; the console
does not store or edit keys. See [Centralized Config](centralized-config.md).

### Models tab

- **Search box**: queries the catalog by name (`/` focuses it).
- **Engine** and **Modality** filters: narrow to one provider or to text, vision, audio or
  embedding models.
- **Fits this machine** (on by default): hides artifacts whose fit verdict is `too large`. The
  count line tells you how many were hidden.
- **Installed only**: shows only artifacts whose weights are already on disk.
- **Search Hugging Face**: adds live Hub search results to the curated catalog (slower).

Weights status uses the same four labels everywhere in AbstractFramework:

| Label | Meaning |
|---|---|
| `installed` | The weights are on this host. |
| `not downloaded` | The weights can be downloaded; they are not on disk. |
| `unknown` | The provider's tool could not be consulted (for example, the engine is not running). |
| `remote` | Nothing to download: the model runs on a remote API. |

Fit badges: `fits`, `tight`, `too large`, `partial offload`, `unknown`. Hover a badge to see the
evidence: memory needed versus the usable ceiling, free memory now, whether the download fits on
disk, the maximum context, the confidence of the estimate and any notes.

**Download** starts immediately (one click). **Delete** asks for confirmation, shows the files'
location and size, and warns when the model is loaded or when the files are shared with another
provider (for example MLX and Hugging Face share the Hugging Face cache). In those cases you must
tick **Force** to proceed. Both dialogs offer **Preview (dry run)**, which asks the server for the
exact command without running it.

### Engines tab

**Install** opens a confirmation dialog that shows:

- the exact command as a fixed argument list (for example `brew install ollama`); the server
  never runs a shell string built from user input;
- the host it runs on: the machine running the server, which may not be the computer in front of
  you;
- the install method and what it changes, plus host notes (Homebrew needs no sudo, a vendor
  script may ask for sudo, which a background job cannot answer, `winget` may raise a UAC prompt
  on the host's desktop, `pip` installs into the server's Python environment).

Engines that are installed by hand, such as the LM Studio desktop app, offer **Open download
page** instead. The server may refuse an install (for example when another install is running, or
when engine installs are disabled for a remote server); the console shows the server's reason.

### Jobs

Downloads, deletes and engine installs run as server-side jobs. The console polls each job every
1.5 seconds and shows its status (`queued`, `running`, `completed`, `failed`, `cancelled`),
progress, bytes, the last log lines, the command it runs and its CLI equivalent. Active job ids
are remembered for the browser tab, so a page reload re-attaches to them. Jobs live in the server
process: after a server restart the console reports that the job is no longer known and
re-checks what is on disk.

### Keyboard

| Key | Where | Action |
|---|---|---|
| `/` | Models | Focus the search box |
| `f` | Models | Toggle **Fits this machine** |
| `r` | any tab | Refresh |
| `w` | Models, focused row | Download |
| `d` | Models, focused row | Delete (confirm) |
| `i` | Engines, focused row | Install (confirm) |
| `o` | Engines, focused row | Open download page |
| `c` | focused row with a running job | Cancel the job |
| `Esc` | dialog | Close |

Click a row or move to it with `Tab` to focus it. These are the same verbs as the terminal
consoles.

## Authentication

The console page and its fragments are static code and never require a token. The data behind
them does: when `ABSTRACTCORE_AUTH_TOKEN` is set, the first API call answers `401` and the console
asks for the token. The token is kept in the browser tab's `sessionStorage` only (cleared when
the tab closes) and sent as `Authorization: Bearer <token>`. Use **Forget token** in the top bar
to clear it.

When no token is configured and `ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED` is not set, the API
answers `503 server_auth_not_configured`; the console shows that message. See
[Server](server.md) for the auth settings.

## Themes

The console uses the AbstractFramework UI kit themes. By default it follows your system's light
or dark preference; the top bar has a theme picker (every kit theme) and a light/dark toggle. The
choice is stored in `localStorage`.

## Embedding the Models and Engines screens

Hosts such as AbstractGateway embed the Models and Engines screens in their own console. Two ways
to obtain them:

- In Python: `abstractcore.console.web.fragment(kind)` with `kind` `"models"` or `"engines"`
  returns `{"html": str, "js": str, "css": str}`.
- Over HTTP: `GET /console/fragment/{kind}` returns the same object plus `"kind"`.

`render_console_html(api_base="/acore", title="AbstractCore Console")` returns the standalone page
served at `/console`.

Place `css` in a `<style>` element, `html` where the tab body goes, and `js` in a `<script>`
element (the script is idempotent; both kinds return the same `js` and `css`, so include each
once). The script defines one global, `window.AbstractCoreConsole`; then mount each screen:

```js
AbstractCoreConsole.mount("models", document.getElementById("tab-catalog"), {
  apiBase: "/api/gateway",                 // default "/acore"
  request: (method, path, body) =>          // default: fetch + optional bearer token
    api(path, { method, body: body == null ? undefined : JSON.stringify(body) }),
  isAdmin: () => Boolean(currentUser.admin),  // boolean or function; default true
  onJob: (job, kind) => {},                 // called on every job update
  hostName: "gateway-host",                 // shown in install/delete dialogs; default location.hostname
  cliPrefix: "abstractgateway",             // CLI shown for composed commands; default "abstractcore"
});
```

- `request(method, path, body)` receives the full path (`apiBase` + route) and a JSON-serializable
  body or `undefined`; it must resolve to the parsed JSON and reject with an `Error` carrying
  `status` on HTTP errors. The host keeps full control of auth, CSRF headers and timeouts.
- The screens call these routes under `apiBase`: `GET /host/profile`, `GET /engines?probe=1`,
  `GET /models/catalog?q=&engine=&fits=1&hub=1`, `GET /models/installed`,
  `POST /models/download`, `POST /models/delete`, `POST /engines/{id}/install`,
  `GET /jobs/{id}`, `POST /jobs/{id}/cancel`. Verbs answer a `host_job_v1` job (a `{"job": ...}`
  envelope is also accepted).
- When `isAdmin` is false, Download, Delete and Install are disabled.
- `mount()` returns `{kind, refresh(), unmount()}`; `AbstractCoreConsole.unmount(rootEl)` does the
  same. Mounting injects the tab body itself when the host did not place `html`.
- Styles are scoped under `.acc-root` and read the UI kit's CSS variables (`--bg-secondary`,
  `--text-primary`, `--accent`, `--info`, `--success`, `--warning`, `--error`, ...), so the
  screens follow the host's active theme.

## Related

- [Server](server.md): endpoints, auth settings, runtime control plane.
- [Centralized Config](centralized-config.md): API keys and default models (`abstractcore --config`).
- [Memory and Model Residency](memory-management.md): loading, unloading and locking models.
