from __future__ import annotations

from typing import Dict, List, Any
from prompts import SUMMARY_PROMPT


def _call_llm(llm: Any, prompt: str) -> str:
    """
    Works with:
    - langchain_openai.ChatOpenAI (returns object with .content)
    - custom wrappers returning plain str
    """
    res = llm.invoke(prompt)
    return getattr(res, "content", res)


def _safe_text(text: str, cap: int = 5000) -> str:
    return text[:cap]


def node(state: Dict, llm) -> Dict:

    summaries: List[Dict] = []
    for p in state.get("papers", []):
        prompt = SUMMARY_PROMPT.format(
            title=p.get("title", ""),
            abstract=_safe_text(p.get("abstract", "")),
        )
        summary = _call_llm(llm, prompt).strip()
        summaries.append(
            {
                "title": p.get("title", ""),
                "summary": summary,
                "url": p.get("url", ""),
                "pdf_url": p.get("pdf_url"),
            }
        )

    state["summaries"] = summaries
    return state
