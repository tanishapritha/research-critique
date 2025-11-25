import streamlit as st
import asyncio
from dotenv import load_dotenv
import json

from graph import build_workflow, ainvoke, astream_states
from models import Paper, Summary
from reAct import build_react_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

st.set_page_config(page_title="PaperMind AI — Research Agent", layout="wide")

st.title("PaperMind AI — Research Agent")
st.markdown("Search → Summarize → Synthesize → Critique → Gaps")



with st.sidebar:
    st.header("Configuration")
    workflow_type = st.radio("Workflow Type", ["Standard Graph", "ReAct Agent"])


if "workflow" not in st.session_state:
    with st.spinner("Initializing workflow..."):
        st.session_state.workflow = build_workflow()

if "react_agent" not in st.session_state:
    with st.spinner("Initializing ReAct Agent..."):
        st.session_state.react_agent = build_react_agent()

query = st.text_input("Research Topic", placeholder="Enter a research topic or question...")

if st.button("Start Research"):
    if not query or len(query) < 3:
        st.warning("Please enter a valid research topic (min 3 chars).")
    else:
        async def run_research():
            status_container = st.empty()
            
            if workflow_type == "Standard Graph":
                try:
                    async for chunk in astream_states(st.session_state.workflow, query):
                        for node_name, node_state in chunk.items():
                            status_container.info(f"Processing: {node_name}...")
                    
                    status_container.info("Finalizing results...")
                    final_state = await ainvoke(st.session_state.workflow, query)
                    status_container.success("Research Complete!")
                    return final_state, "standard"
                    
                except Exception as e:
                    status_container.error(f"An error occurred: {str(e)}")
                    return None, None

            else:
                st.subheader("Agent Thinking Process")
                thinking_container = st.container()
                
                try:
                    messages = [HumanMessage(content=query)]
                    final_response = ""
                    

                    async for chunk in st.session_state.react_agent.astream({"messages": messages}):
                        
                        if "agent" in chunk:

                            msg = chunk["agent"]["messages"][0]
                            if isinstance(msg, AIMessage):
                                if msg.tool_calls:
                                    for tool_call in msg.tool_calls:
                                        with thinking_container.expander(f"Thinking: Decided to use {tool_call['name']}", expanded=True):
                                            st.write(f"Arguments: {tool_call['args']}")
                                else:
                                    if msg.content:
                                        # If it's the final chunk, it might be the answer
                                        pass
                        
                        if "tools" in chunk:
                            # Tool has responded
                            msg = chunk["tools"]["messages"][0]
                            if isinstance(msg, ToolMessage):
                                with thinking_container.expander(f"Tool Output: {msg.name}", expanded=False):
                                    st.text(msg.content[:500] + "..." if len(msg.content) > 500 else msg.content)

                    # Final answer
                    result = await st.session_state.react_agent.ainvoke({"messages": messages})
                    final_response = result["messages"][-1].content
                    status_container.success("Research Complete!")
                    return final_response, "react"

                except Exception as e:
                    status_container.error(f"An error occurred: {str(e)}")
                    return None, None

        result, mode = asyncio.run(run_research())

        if result:
            if mode == "standard":
                final_state = result

                st.header("Synthesis")
                st.markdown(final_state.get("synthesis") or "*No synthesis generated.*")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Critique")
                    st.markdown(final_state.get("critique") or "*No critique generated.*")
                    
                with col2:
                    st.subheader("Identified Gaps")
                    st.markdown(final_state.get("gaps") or "*No gaps identified.*")
                
                # Papers
                st.subheader("Papers Found")
                papers_data = final_state.get("papers", [])
                if papers_data:
                    for p_data in papers_data:
                        try:
                            paper = Paper(**p_data)
                            with st.expander(f"{paper.title}"):
                                st.markdown(f"**Abstract:** {paper.abstract}")
                                st.markdown(f"[Link]({paper.url})")
                        except Exception as e:
                            st.error(f"Error parsing paper data: {e}")
                else:
                    st.info("No papers found.")

                # Summaries (Detailed)
                st.subheader("Detailed Summaries")
                summaries_data = final_state.get("summaries", [])
                if summaries_data:
                    for s_data in summaries_data:
                        try:
                            summary = Summary(**s_data)
                            with st.expander(f"Summary for: {summary.title}"):
                                st.markdown(summary.summary)
                        except Exception as e:
                            st.error(f"Error parsing summary data: {e}")
                else:
                    st.info("No detailed summaries generated.")
            
            else: # ReAct Mode
                st.header("Final Answer")
                st.markdown(result)