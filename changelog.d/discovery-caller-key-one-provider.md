### Security

- **A caller's provider key on a discovery route reaches only the provider the request names** (`server/app.py`,
  root backlog 0955).
  - Before, `X-AbstractCore-Provider-API-Key` on `/v1/capabilities` and the capability and music discovery routes
    was handed to every key slot the capability has (OpenAI and the OpenAI-compatible server for voice, AceMusic and
    ElevenLabs for music, all of them on `/v1/capabilities`), so a key meant for one vendor could reach another.
  - Now the key goes to one provider: the one named with `?provider=<id>` (voice: `openai` or an OpenAI-compatible
    server; music: a backend such as `acemusic`, and the request's registry uses only that backend), else the
    configured music backend, or the single vision endpoint.
  - A request that sends the key without naming one provider (`GET /v1/capabilities`,
    `GET /v1/capabilities/voice/providers`, an unknown music provider) is refused with 400 and the key reaches
    nobody. Authenticate to the server and leave the key out to list every provider.
