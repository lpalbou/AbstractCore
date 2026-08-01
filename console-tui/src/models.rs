//! Model-id classification for the pickers. Discovery gives NAMES
//! ONLY (`config models`/`test-provider` carry no type metadata —
//! probed 2026-07-25), so class filtering is a name heuristic:
//! embedding models are strongly conventional (they exist to be found
//! by name — *embed*, minilm, bge, e5, gte, …) while generative model
//! names are unbounded. The pickers therefore filter INTO the
//! embedding class or OUT of it, never pretend finer classes
//! (vision-capable vs text cannot be told apart by name), and every
//! consumer keeps a free-typing lane so a heuristic miss can never
//! block a valid configuration.

/// What a picker is choosing FOR.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelClass {
    /// Embedding models only (the embeddings pair, embedding.* routes).
    Embedding,
    /// Everything that is not embedding-shaped (text/vision/audio
    /// generation — indistinguishable from each other by name).
    Generative,
}

/// The conventional embedding-family markers, checked against the
/// lowercased id (org prefixes and quant suffixes ride along
/// harmlessly — substring match).
const EMBEDDING_MARKERS: &[&str] = &[
    "embed",       // text-embedding-*, *-embedding-*, nomic-embed, arctic-embed…
    "minilm",      // all-minilm-l6-v2 (sentence-transformers)
    "bge-",        // bge-small-en-v1.5, bge-m3
    "gte-",        // gte-small/base/large
    "e5-",         // e5-small-v2 (also matches multilingual-e5-*)
    "sentence-t",  // sentence-transformers/…
    "paraphrase-", // paraphrase-multilingual-…
];

pub fn is_embedding_shaped(id: &str) -> bool {
    let lower = id.to_ascii_lowercase();
    EMBEDDING_MARKERS.iter().any(|m| lower.contains(m))
}

pub fn matches_class(id: &str, class: ModelClass) -> bool {
    match class {
        ModelClass::Embedding => is_embedding_shaped(id),
        ModelClass::Generative => !is_embedding_shaped(id),
    }
}

/// Split a discovery list for a picker: (matching, hidden_count).
/// When the heuristic leaves NOTHING, the full list comes back with
/// hidden 0 — a picker over live data must never render empty because
/// a name convention failed (the caller labels the fallback).
pub fn filter_for(class: ModelClass, models: &[String]) -> (Vec<String>, usize) {
    let matching: Vec<String> = models
        .iter()
        .filter(|m| matches_class(m, class))
        .cloned()
        .collect();
    if matching.is_empty() {
        (models.to_vec(), 0)
    } else {
        let hidden = models.len() - matching.len();
        (matching, hidden)
    }
}

/// The class a capability-route picker filters for.
///
/// The embedding routes are `embedding.text` / `embedding.image`: the
/// word "embedding" is the row's KIND, and its MODALITY is the thing
/// being embedded ("text", "image"). Reading the modality here — as
/// this did until 2026-08-01 — classified `embedding.text` as
/// Generative, so the one picker whose filter earns its keep offered
/// chat models for an embeddings route.
pub fn class_for_route(kind: &str, _modality: &str) -> ModelClass {
    if kind.eq_ignore_ascii_case("embedding") {
        ModelClass::Embedding
    } else {
        ModelClass::Generative
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real ids from the live LM Studio listing (2026-07-25) — the
    /// heuristic must split THIS list correctly, not invented names.
    #[test]
    fn classifies_the_live_listing_shapes() {
        for embedding in [
            "text-embedding-qwen3-embedding-0.6b",
            "all-minilm-l6-v2@4bit",
            "all-minilm-l6-v2@8bit",
            "bge-small-en-v1.5",
            "nomic-embed-text-v1.5",
            "text-embedding-qwen3-embedding-4b",
            "sentence-transformers/all-mpnet-base-v2",
            "intfloat/multilingual-e5-large",
        ] {
            assert!(is_embedding_shaped(embedding), "{embedding}");
        }
        for generative in [
            "gemma-3-1b-it",
            "granite-4.1-3b",
            "qwen/qwen3.6-35b-a3b",
            "google/gemma-4-26b-a4b",
            "huihui-ai_-_llama-3.2-3b-instruct-abliterated",
            "unsloth/Qwen3-4B-Instruct-2507-GGUF",
            "llava-v1.6-mistral-7b", // vision — still generative-class
            "whisper-large-v3",     // audio — still generative-class
        ] {
            assert!(!is_embedding_shaped(generative), "{generative}");
        }
    }

    #[test]
    fn filter_splits_and_never_returns_empty_over_data() {
        let models: Vec<String> = [
            "gemma-3-1b-it",
            "bge-small-en-v1.5",
            "all-minilm-l6-v2@4bit",
            "granite-4.1-3b",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let (m, hidden) = filter_for(ModelClass::Embedding, &models);
        assert_eq!(m, vec!["bge-small-en-v1.5", "all-minilm-l6-v2@4bit"]);
        assert_eq!(hidden, 2);

        let (m, hidden) = filter_for(ModelClass::Generative, &models);
        assert_eq!(m, vec!["gemma-3-1b-it", "granite-4.1-3b"]);
        assert_eq!(hidden, 2);

        // A heuristic whiff must not blank the picker: full list back.
        let only_gen: Vec<String> = vec!["gemma-3-1b-it".into()];
        let (m, hidden) = filter_for(ModelClass::Embedding, &only_gen);
        assert_eq!(m, only_gen);
        assert_eq!(hidden, 0);
    }

    /// Pinned against the LIVE row shapes: `embedding.text` arrives as
    /// kind "embedding" / modality "text", so a modality-only read
    /// mis-classifies the exact route the filter exists for.
    #[test]
    fn route_class_reads_the_kind_not_the_modality() {
        assert_eq!(class_for_route("embedding", "text"), ModelClass::Embedding);
        assert_eq!(class_for_route("embedding", "image"), ModelClass::Embedding);
        assert_eq!(class_for_route("input", "text"), ModelClass::Generative);
        assert_eq!(class_for_route("output", "voice"), ModelClass::Generative);
        assert_eq!(class_for_route("rerank", "text"), ModelClass::Generative);
    }
}
