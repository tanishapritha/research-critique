from langchain.tools import tool
from nodes.search import search_arxiv
from llm_router import get_llm_for_task
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

@tool
def search_tool(query: str):
    """Search for research papers on Arxiv. Returns titles and abstracts. 
    Use this to find relevant papers to answer the user's research question.
    """
    # We limit results to 3 to save tokens and keep it concise
    results = search_arxiv(query, max_results=3)
    # Format nicely for the LLM
    formatted = []
    for r in results:
        formatted.append(f"Title: {r['title']}\nAbstract: {r['abstract']}\nURL: {r['url']}")
    return "\n\n".join(formatted)

def build_react_agent():
    # Use a capable model for the agent loop
    llm = get_llm_for_task("synthesize") 
    memory = MemorySaver()
    return create_react_agent(llm, tools=[search_tool], checkpointer=memory)
