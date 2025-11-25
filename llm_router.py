from __future__ import annotations

import os
from typing import Literal, Dict

from langchain_openai import ChatOpenAI
from utils.embeddings import EmbeddingProvider

Task = Literal["search", "summarize", "synthesize", "critique", "gaps"]

# ✅ token caps per step
TOKEN_LIMITS: Dict[Task, int] = {
    "search": 128,
    "summarize": 256,
    "synthesize": 512,
    "critique": 256,
    "gaps": 256,
}


# --------- LLM FACTORY (OpenRouter via OpenAI protocol) ----------
def _make_openrouter_llm(model: str, temperature: float = 0.2, max_tokens: int = 256) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set. Put it in .env")

    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,   # ✅ MAIN CHANGE
    )


def _maybe_make_ollama(model: str, temperature: float = 0.2):
    try:
        from langchain_community.llms import Ollama
        return Ollama(model=model, temperature=temperature)
    except Exception:
        return None


def get_llm_for_task(task: Task):
    """
    Cost-aware routing:
      - summarize -> cheap (fast)
      - synthesize -> more capable
      - critique/gaps -> mid
    """
    routing: Dict[Task, str] = {
        "search": "mistralai/mistral-7b-instruct",
        "summarize": "mistralai/mistral-7b-instruct",
        "synthesize": "anthropic/claude-3.5-haiku",
        "critique": "anthropic/claude-3.5-haiku",
        "gaps": "meta-llama/llama-3.1-8b-instruct",
    }

    model = routing.get(task, "mistralai/mistral-7b-instruct")
    max_tok = TOKEN_LIMITS.get(task, 256)

    return _make_openrouter_llm(
        model=model,
        temperature=0.2,
        max_tokens=max_tok,
    )


def get_embeddings():
    try:
        return EmbeddingProvider(
            use_openrouter=True,
            model="text-embedding-3-small",
        )
    except RuntimeError:
        # No OPENROUTER_API_KEY → fall back to local ST
        return EmbeddingProvider(use_openrouter=False)
