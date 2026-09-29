# Terminal console (`abstractcore-console`)

`abstractcore-console` is AbstractCore's keyboard-first terminal console.
It configures `abstractcore.json` (default models, providers and API
keys, capability routes, media, embeddings, server) and manages local
models and engines on this machine.

```bash
pip install abstractcore               # the CLI the console drives
cargo install abstractcore-console     # the console (Rust 1.87+)
abstractcore-console                   # --wizard for the guided setup
```

The console reads the config file directly and runs the `abstractcore`
CLI for everything derived or written. It finds the CLI through
`$ABSTRACTCORE_CLI`, then `abstractcore` on `PATH`, then
`~/.local/bin/abstractcore` (the `uv tool` shim), then
`./.venv/bin/abstractcore` in the current directory.

## Screens

| Key | Screen | What it is for |
|---|---|---|
| 1 | Overview | every config section's state; Enter jumps to its owner |
| 2 | Model | default models and app defaults |
| 3 | Providers | providers, endpoint profiles, API keys, model discovery |
| 4 | Routes | capability routes (vision, audio, image, video, embeddings); `w` downloads a route's weights, `a` applies the recommended routes |
| 5 | Media | vision / audio / video settings |
| 6 | Embeddings | the embeddings route |
| 7 | Server | server, logging, timeouts, cache |
| 8 | Review | the file identity and test evidence |
| 9 | **Models** | the model catalog fitted to this host (see the keys below) |
| 0 | **Engines** | Ollama, LM Studio, MLX, llama.cpp, vLLM, Hugging Face (see the keys below) |
| @ | **Email** | the email account, its last test, recipient policy and send limits: `c` connect (Save and test), `t` test, `o` turn off/on, `x` disconnect, `p` recipient policy, `l` send limits — the web console's Email tab, same fields and words (see [Email](email.md)) |

To move between screens, press the screen's key, `Ctrl+N` / `Ctrl+P`, or
`←` / `→` for the next and previous screen (wrapping from Engines to
Overview and back). The arrows keep their own meaning while a text field
has the caret (they move it), while the screen bar has the focus (it moves
itself), and in any open dialog. In the wizard the arrows do not jump
screens; `Ctrl+N` walks the wizard.

### Models keys

| Key | Action |
|---|---|
| `w` | download the selected artifact; several downloads run at once |
| `d` | delete an installed artifact (the confirm names what blocks it: a loaded model, a shared cache) |
| `/` | filter by name, family, tag or artifact |
| `t` | cycle the model type: text, thinking, tools, vision, audio, embedding, voice, image, video |
| `f` | show only what fits this host |
| `e` | cycle the engine |
| `h` | search Hugging Face (an empty query returns to the catalog) |
| `u` | make the selected installed text model the default text model (`output.text`); the route becomes exactly that model |
| `v` | cycle catalog → installed → downloads (every download the backend knows, live) |
| `c` | cancel the selected download (asks first) |
| `r` | re-read the host, the catalog and what is installed |

### Engines keys

| Key | Action |
|---|---|
| `i` | install the selected engine (the confirm shows the command, the host it runs on, and any administrator step; a dry run is one of the answers) |
| `o` | open the vendor's download page |
| `r` | probe the local servers |
| `c` | cancel the engine's install |
| `s` | start or stop the engine's server (Ollama, LM Studio) |
| `a` | continue an install that waits for an administrator or for developer tools |
| `y` | copy that install's command |

`s`, `a` and `y`, the install location (just you or all users) and
Hugging Face search are optional backend verbs: over a backend without
them the key answers "not available here" and the footer says "not
here". The `abstractcore` CLI backend offers `h`, `u` and the downloads
view; the gateway console offers them all.

Models and Engines speak the same vocabulary as the web console and the
gateway console: weights `installed / not downloaded / unknown /
remote`, fit `fits / tight / too large / partial offload / unknown`.
Every action there is a CLI command you can run yourself:

| Console action | CLI equivalent |
|---|---|
| Models list | `abstractcore models catalog --json` / `models search <q> [--engine X] [--fits] --json` |
| Installed view | `abstractcore models list --json` |
| `w` download | `abstractcore models download <provider> <artifact> --json` |
| `d` delete | `abstractcore models delete <provider> <artifact> --yes [--force] --json` |
| Engines list | `abstractcore engines status [--probe] --json` |
| `i` install | `abstractcore engines install <id> --yes [--dry-run] --json` |

Downloads and installs run on the machine that runs the console.
Downloads run in parallel. An install that waits for a person (an
administrator password, developer tools) blocks nothing and shows the
exact command to run. `q` will not quit while a download or install is
working; a waiting install does not hold it.

## Who may change the host

`w`, `d`, `u` and `c` on Models and `i`, `s`, `a` and `c` on Engines
change the host, so they are administrator verbs, as in the web console.
In `abstractcore-console` you are the local operator and every verb is
yours. In the gateway console they follow your sign-in: a non-admin
browses freely, each of these keys answers "only an admin can … —
<reason>", and the footer marks them "admin only".

## Same screens in the gateway console

The Models and Engines screens are a library inside the crate
(`abstractcore_console::screens`). `abstractgateway-console` mounts them
over an HTTP transport to the gateway's `/api/gateway/models/*` and
`/api/gateway/engines/*` routes, so a gateway operator sees the same
screens, acting on the gateway host, with the administrator rules of the
gateway's sign-in. See the crate README
(`console-tui/README.md`) for the library API.
