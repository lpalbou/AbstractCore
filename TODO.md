# Random notes

Do not pay attention to this.

- async ? (huge dev I fear)

- handle images
    - with multimodal models

- better media : 
    - parse txt, csv, tsv, py etc
    - parse xlsx, docx, pdf

SHOULD IT BE IN ABSTRACT MEMORY OR ABSTRACT CORE ?
- handle file attachment
    - session : every file ever referenced
    - file currently attached to the context


- timeout vs RetryConfig ?

need a token estimator to share

summmarizer : cli ?
- gemma3:1b-it-qat
- gemma3:270m-it-qat
- gemma3n:e2b

function similar(text1, text2) with embeddings


update the @CHANGELOG.md to a new minor version. explain we fixed the init of both providers and processing. 


Embeddings
ok, so for hf, the embedding models are : google/embeddinggemma-300m, Qwen/Qwen3-Embedding-0.6B, sentence-transformers/all-MiniLM-L6-v2, 
ibm-granite/granite-embedding-107m-multilingual, ibm-granite/granite-embedding-278m-multilingual, ibm-granite/granite-embedding-30m-english (only english, careful), 
nomic-ai/nomic-embed-text-v1.5 and nomic-ai/nomic-embed-text-v2-moe. Those are the models that we favor and should recognize. any HF models should be stored in the HF cache 
~/.cache/huggingface/ and reused whenever possible - do not download them over and over again. by default, i suggest to stick to sentence-transformers/all-MiniLM-L6-v2


we created some benchmarks : @examples/embeddings_benchmark.py , please adjust the code if needed (at minima to suit the only models we are considering : """"google/embeddinggemma-300m, Qwen/Qwen3-Embedding-0.6B, sentence-transformers/all-MiniLM-L6-v2, ibm-granite/granite-embedding-107m-multilingual, ibm-granite/granite-embedding-278m-multilingual, ibm-granite/granite-embedding-30m-english, nomic-ai/nomic-embed-text-v1.5 and nomic-ai/nomic-embed-text-v2-moe""". Then run the benchmarks and tell me your conclusions. Note that all the models are already in the cache 



    Encountered exception while importing einops: No module named 'einops'
    Failed to save persistent cache: 'EmbeddingManager' object has no attribute 'cache_file'
    Failed to save normalized cache: 'EmbeddingManager' object has no attribute 'normalized_cache_file'
    Encountered exception while importing einops: No module named 'einops'
    Failed to save persistent cache: 'EmbeddingManager' object has no attribute 'cache_file'
    Failed to save normalized cache: 'EmbeddingManager' object has no attribute 'normalized_cache_file'