# Prompt for AI Architecture Diagram Generator

**Role:** Expert Software Architect and Technical Illustrator.

**Task:** Create a comprehensive architecture diagram for an **Educational Agentic RAG (Retrieval-Augmented Generation) System** that uses a **ReAct (Reasoning + Acting)** workflow.

**System Name:** PaperMind AI — Research Agent

**Components to Visualize:**

1.  **User Interface (Frontend)**
    *   **Streamlit App**: The entry point where users input research topics and view results.
    *   **Features**: Workflow Selector (Standard vs. ReAct), Real-time "Thinking" Stream, Final Report Display.

2.  **Orchestration Layer (The Brain)**
    *   **LangGraph Controller**: Manages the state and flow of the application.
    *   **Dual Workflows**:
        *   *Path A (Standard)*: Linear Pipeline (Search -> Summarize -> Synthesize -> Critique -> Gaps).
        *   *Path B (ReAct)*: Cyclic Agent Loop (Thought -> Action -> Observation -> Repeat).

3.  **Intelligence (LLMs)**
    *   **LLM Router**: dynamically selects models based on task complexity (e.g., Haiku for summaries, Sonnet/GPT-4 for synthesis).
    *   **Providers**: OpenRouter / OpenAI / Ollama.

4.  **Tools & Knowledge (The Arms & Legs)**
    *   **Arxiv Search Tool**: Fetches real-time academic papers.
    *   **Retriever**: Queries the Vector Database for stored knowledge.
    *   **Vector Database**: ChromaDB (stores embeddings of papers).

5.  **Data Flow (The Arrows)**
    *   User Query -> Streamlit -> LangGraph.
    *   LangGraph -> LLM (Reasoning).
    *   LLM -> Tools (Search/Retrieve).
    *   Tools -> Data (Arxiv/Chroma) -> LLM (Observation).
    *   LLM -> Final Synthesis -> Streamlit -> User.

**Visual Style:**
*   Clean, modern, and educational.
*   Use distinct colors for: UI (Blue), Logic/Agents (Purple), Data/Tools (Green), External APIs (Orange).
*   Show the cyclic nature of the ReAct loop clearly.

**Output Format:**
Please generate a **Mermaid.js** code block representing this architecture.
