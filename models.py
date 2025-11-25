from typing import TypedDict, List, Optional

class RState(TypedDict):
    query: str
    papers: List[dict]
    summaries: List[dict]
    synthesis: Optional[str]
    critique: Optional[str]



from pydantic import BaseModel
from typing import List, Optional


class Paper(BaseModel):
    title: str
    abstract: str
    url: str


class Summary(BaseModel):
    title: str
    summary: str
    url: str


class ResearchResponse(BaseModel):
    query: str
    papers: List[Paper]
    summaries: List[Summary]
    synthesis: Optional[str]
    critique: Optional[str]
    gaps: Optional[str]
