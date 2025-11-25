from __future__ import annotations

from typing import Dict, Any

GAP_PROMPT = """
You are a senior research scientist. From the synthesis below, extract:
1) Missing or underrepresented perspectives
2) Open research questions
3) Data/benchmark gaps
4) Methodological risks and failure modes
5) Concrete next-step experiments (bullet points)

SYNTHESIS:
{summary}
"""


def _call_llm(llm: Any, prompt: str) -> str:
    res = llm.invoke(prompt)
    return getattr(res, "content", res)


def node(state: Dict, llm) -> Dict:
    """
    Produces a concise, actionable research-gap section.
    """
    synthesis = (state.get("synthesis") or "").strip()
    if not synthesis:
        state["gaps"] = ""
        return state

    state["gaps"] = _call_llm(llm, GAP_PROMPT.format(summary=synthesis)).strip()
    return state
