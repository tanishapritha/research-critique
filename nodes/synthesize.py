from __future__ import annotations

from typing import Dict, Any, List
from prompts import SYNTH_PROMPT


def _call_llm(llm: Any, prompt: str) -> str:
    res = llm.invoke(prompt)
    return getattr(res, "content", res)


def node(state: Dict, llm) -> Dict:

    parts: List[str] = [s.get("summary", "") for s in state.get("summaries", []) if s.get("summary")]
    merged = "\n\n".join(parts).strip()

    if not merged:
        state["synthesis"] = ""
        return state

    prompt = SYNTH_PROMPT.format(summaries=merged)
    state["synthesis"] = _call_llm(llm, prompt).strip()
    return state
