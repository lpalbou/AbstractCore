# Random notes

Do not pay attention to this.

1) rename to AbstractCore
2) LLM-as-a-judge
3) Media handling


uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000 --reload


time python -m abstractllm.apps.extractor /Users/albou/projects/promptons/examples/docs/christmas_carol_ebook_chapter_1.txt --provider lmstudio --model bytedance/seed-oss-36b --timeout 3600

time python -m abstractllm.apps.extractor /Users/albou/projects/promptons/examples/docs/christmas_carol_ebook_chapter_1.txt --provider lmstudio --model qwen/qwen3-4b-2507 --iterate=2

time python -m abstractllm.apps.extractor /Users/albou/projects/promptons/examples/docs/christmas_carol_ebook_chapter_1.txt --provider lmstudio --model qwen/qwen3-4b-2507 --iterate=3

time python -m abstractllm.apps.extractor /Users/albou/projects/promptons/examples/docs/christmas_carol_ebook_chapter_1.txt --provider lmstudio --model qwen/qwen3-next-80b

time python -m abstractllm.apps.extractor /Users/albou/projects/promptons/examples/docs/christmas_carol_ebook_chapter_1.txt --provider lmstudio --model qwen/qwen3-coder-30b


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