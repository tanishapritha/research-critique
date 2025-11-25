# app.py
from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from models import ResearchResponse, Paper, Summary
from graph import build_workflow, ainvoke, astream_states

load_dotenv()

app = FastAPI(
    title="PaperMind AI — Research Agent",
    version="1.0.0",
    description="Search → Summarize → Synthesize → Critique → Gaps (LangGraph + OpenRouter)",
)

# Build workflow once at startup
workflow = build_workflow()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get(
    "/research",
    response_model=ResearchResponse,
    summary="Run the full pipeline and return structured JSON",
)
async def research(q: str = Query(..., min_length=3, description="Research topic or question")):
    try:
        final_state = await ainvoke(workflow, q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Coerce into typed response
    papers = [Paper(**p) for p in final_state.get("papers", [])]
    summaries = [Summary(**s) for s in final_state.get("summaries", [])]
    return ResearchResponse(
        query=q,
        papers=papers,
        summaries=summaries,
        synthesis=final_state.get("synthesis") or "",
        critique=final_state.get("critique") or "",
        gaps=final_state.get("gaps") or "",
    )


@app.get(
    "/research/stream",
    summary="Stream partial results (NDJSON over text/event-stream)",
)
async def research_stream(q: str = Query(..., min_length=3)):
    """
    Streams state deltas as NDJSON (one JSON object per line) over SSE-compatible content-type.
    Frontends can read line-by-line for live updates.
    """

    async def gen() -> AsyncIterator[bytes]:
        try:
            async for delta in astream_states(workflow, q):
                # Normalize to a lightweight structure to avoid huge payloads
                payload = {
                    "delta": delta,
                }
                # For SSE, each event is `data: <json>\n\n`
                line = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                yield line.encode("utf-8")
            # Signal end of stream
            yield b"data: [DONE]\n\n"
        except Exception as e:
            # Send error as an SSE event then finish
            err = {"error": str(e)}
            yield f"data: {json.dumps(err)}\n\n".encode("utf-8")

    return StreamingResponse(gen(), media_type="text/event-stream")
