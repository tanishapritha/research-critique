from __future__ import annotations

import arxiv
from typing import List, Dict


def search_arxiv(query: str, max_results: int = 5) -> List[Dict]:

    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=0.3,  # gentle rate-limiting
        num_retries=2,
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    out: List[Dict] = []
    for r in client.results(search):
        out.append(
            {
                "title": r.title.strip(),
                "abstract": (r.summary or "").strip(),
                "url": r.entry_id,
                "pdf_url": getattr(r, "pdf_url", None),
            }
        )
    return out


def node(state: dict) -> dict:

    state["papers"] = search_arxiv(state["query"], max_results=5)
    return state
