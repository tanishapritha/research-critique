# graph.py
from __future__ import annotations

from typing import AsyncIterator, Dict, Any
from functools import partial
from langgraph.graph import StateGraph, END

from utils.cache import cache_get, cache_add


from nodes import (
    search as search_node,
    summarize as summarize_node,
    synthesize as synthesize_node,
    critique as critique_node,
    gaps as gaps_node,
)

from llm_router import get_llm_for_task, get_embeddings


def build_workflow():
    summarize_llm = get_llm_for_task("summarize")
    synthesize_llm = get_llm_for_task("synthesize")
    critique_llm = get_llm_for_task("critique")
    gaps_llm = get_llm_for_task("gaps")

    try:
        from nodes import retrieve as retrieve_node
        emb = get_embeddings()
        retrieve_fn = partial(retrieve_node, emb=emb)
        has_retrieve = True
    except Exception:
        retrieve_fn = None
        has_retrieve = False

    summarize_fn = partial(summarize_node, llm=summarize_llm)
    synthesize_fn = partial(synthesize_node, llm=synthesize_llm)
    critique_fn = partial(critique_node, llm=critique_llm)
    gaps_fn = partial(gaps_node, llm=gaps_llm)

    graph = StateGraph(dict)
    graph.set_entry_point("search")

    graph.add_node("search", search_node)
    if has_retrieve:
        graph.add_node("retrieve", retrieve_fn)
    graph.add_node("summarize", summarize_fn)
    graph.add_node("synthesize", synthesize_fn)
    graph.add_node("critique", critique_fn)
    graph.add_node("gaps", gaps_fn)

    if has_retrieve:
        graph.add_edge("search", "retrieve")
        graph.add_edge("retrieve", "summarize")
    else:
        graph.add_edge("search", "summarize")

    graph.add_edge("summarize", "synthesize")
    graph.add_edge("synthesize", "critique")
    graph.add_edge("critique", "gaps")
    graph.add_edge("gaps", END)

    return graph.compile()



async def ainvoke(workflow, query: str) -> Dict[str, Any]:
    MODEL_VERSION = "v1"
    MODEL_NAME = "workflow"

    cached = cache_get(query, MODEL_NAME, MODEL_VERSION)
    if cached:
        print("Cache hit")
        return cached

    print("Running workflow")
    initial_state = {
        "query": query,
        "papers": [],
        "summaries": [],
        "synthesis": None,
        "critique": None,
        "gaps": None,
    }

    final_state = await workflow.ainvoke(initial_state)

    cache_add(query, MODEL_NAME, MODEL_VERSION, final_state)

    return final_state



async def astream_states(workflow, query: str) -> AsyncIterator[Dict[str, Any]]:
    initial_state = {
        "query": query,
        "papers": [],
        "summaries": [],
        "synthesis": None,
        "critique": None,
        "gaps": None,
    }
    async for chunk in workflow.astream(initial_state):
        yield chunk
