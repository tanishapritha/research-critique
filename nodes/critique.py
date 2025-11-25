from __future__ import annotations

from typing import Dict, Any
from prompts import CRIT_PROMPT


def _call_llm(llm: Any, prompt: str) -> str:
    res = llm.invoke(prompt)
    return getattr(res, "content", res)


def node(state: Dict, llm) -> Dict:
    """
    Critique the synthesis for bias, gaps, and clarity.
    """
    synthesis = (state.get("synthesis") or "").strip()
    if not synthesis:
        state["critique"] = ""
        return state

    prompt = CRIT_PROMPT.format(s=synthesis)
    state["critique"] = _call_llm(llm, prompt).strip()
    return state
