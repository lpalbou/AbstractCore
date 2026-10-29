### Fixed

- **`/v1/embeddings` reuses the loaded embedding model** (operator rule 2026-10-29: a request on a loaded model reuses
  it, never reloads it). The route built a new `EmbeddingManager` for every request, and the manager loads the weights
  in its constructor, so each request was a full load of the embedding model and another copy beside the previous
  request's (measured on the residency probe: 4 managers alive after 2 requests). The server now keeps one manager per
  (provider, model, dimensions, provider settings) for the life of the process; an ejected model reloads on its next
  call as before, and the residency listing and eject see the same manager.
