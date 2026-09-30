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
| @ | **Email** | the email address and the mailbox of this install, card by card (see [Email keys](#email-keys)) — the web console's Email tab, same cards and words; the password and client secret go to `abstractcore email connect` on stdin, never on its command line (see [Email](email.md)) |

To move between screens, press the screen's key, `Ctrl+N` / `Ctrl+P`, or
`←` / `→` for the next and previous screen (wrapping from Engines to
Overview and back). The arrows keep their own meaning while a text field
has the caret (they move it), while the screen bar has the focus (it moves
itself), and in any open dialog. In the wizard the arrows do not jump
screens; `Ctrl+N` walks the wizard.

### Switches

A persistent on/off setting is a switch, labelled by what it controls:
`[x] Agent email tools` is on (accent, bold), `[ ] Agent email tools` is off,
`[-] Agent email tools — Connect a mailbox first.` is unavailable (dimmed, with
what unlocks it). `Space` or `Enter` switches the selected one and the status line
names the new state ("Agent email tools are on."). On the Model, Media, Embeddings
and Server screens every on/off field reads `[x] on` / `[ ] off` and switches at
once; only switching on a flag AbstractCore marks UNSAFE
(`server.allow_unauthenticated`, `server.allow_local_files`) asks first.

### Email keys

`Tab` / `Shift+Tab` move between the controls, `Enter` presses a button or saves a
field, `Space` switches a switch, `←` / `→` change the Mailbox tab while the tab
bar has the focus. The screen has four cards:

| Card | What it holds |
|---|---|
| Email address | one field with its own **Save** ("✓ Saved" for two seconds): where notifications go, and the first address your agents may write to |
| Mailbox | not connected: tabs **Google** / **Microsoft** / **Other**. Other asks for the email address and the password only; the mail servers are found from the address (`abstractcore email discover`) and shown as one line, and **Server settings** stay folded unless nothing is found (they then open with the reason). One **Connect** stores and tests; its error says what failed. Google and Microsoft sign in through the provider (the device code or sign-in address shows while it waits; **Cancel sign-in** stops it); your own client goes under their Advanced. Connected: "Connected as … · method · checked … ago", **Test** and **Disconnect** (asks inline first) |
| Agent email tools | the switch (off by default); unavailable until a mailbox is connected and in use |
| Advanced (folded) | recipient rules (mode, entries added and removed at once), send limits per hour and per day (saved on `Enter` or when the field loses focus), the folder the mailbox is read from (saved the same way), and the **Use this mailbox** switch (off keeps the settings but stops watching, sending and notifications) |

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
