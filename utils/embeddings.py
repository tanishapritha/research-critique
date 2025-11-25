from __future__ import annotations

from typing import Iterable, List, Optional

# Primary: OpenRouter via OpenAI protocol (works with langchain-openai embeddings)
try:
    from langchain_openai import OpenAIEmbeddings
    _HAS_LC_OPENAI = True
except Exception:
    _HAS_LC_OPENAI = False

# Fallback: local sentence-transformers for offline / Ollama setups
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False


class EmbeddingProvider:
    """
    Unified embedding provider.
    - Default: OpenRouter (text-embedding-3-small)
    - Fallback: sentence-transformers (all-MiniLM-L6-v2)
    """

    def __init__(
        self,
        use_openrouter: bool = True,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
    ) -> None:
        self._mode = None

        if use_openrouter and _HAS_LC_OPENAI:
            import os
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise RuntimeError(f"{api_key_env} not set in environment")

            self._emb = OpenAIEmbeddings(
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
            self._mode = "openrouter"
        elif _HAS_ST:
            self._st = SentenceTransformer("all-MiniLM-L6-v2")
            self._mode = "sentencetransformers"
        else:
            raise RuntimeError(
                "No embedding backend available. "
                "Install `langchain-openai` (and set OPENROUTER_API_KEY) or `sentence-transformers`."
            )

    # API
    def embed_documents(self, docs: Iterable[str]) -> List[List[float]]:
        if self._mode == "openrouter":
            return self._emb.embed_documents(list(docs))
        return self._st.encode(list(docs), normalize_embeddings=True).tolist()

    def embed_query(self, query: str) -> List[float]:
        if self._mode == "openrouter":
            return self._emb.embed_query(query)
        return self._st.encode([query], normalize_embeddings=True)[0].tolist()
