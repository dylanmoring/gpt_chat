try:
    from .cache_embeddings import CacheEmbeddingsOpenAIClient as OpenAIClient
except ImportError:
    try:
        from .numpy_embedding import NumpyEmbeddingOpenAIClient as OpenAIClient
    except ImportError:
        from .open_ai_client import OpenAIClient
