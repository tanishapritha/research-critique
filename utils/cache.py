# utils/cache.py
from __future__ import annotations
import chromadb

COLLECTION_NAME = "papers"

# ✅ new recommended API (persistent local DB)
chroma_client = chromadb.PersistentClient(path="./.chroma")

def get_collection():
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

def cache_key(query: str, model: str, version: str) -> str:
    return f"{query}:{model}:{version}".lower()


# ----- Cache add -----
def cache_add(query: str, model: str, version: str, result: dict):
    key = cache_key(query, model, version)
    col = get_collection()
    col.upsert(
        ids=[key],
        documents=[str(result)],
        metadatas=[{"model": model, "version": version}],
    )


# ----- Cache lookup -----
def cache_get(query: str, model: str, version: str):
    key = cache_key(query, model, version)
    col = get_collection()
    try:
        res = col.get(ids=[key])
        if res and res.get("documents"):
            return eval(res["documents"][0])   # stored as string
    except Exception:
        return None
    return None
