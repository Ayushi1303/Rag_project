# =============================================================
#  graph.py — LangGraph Workflow Builder
#
#  Builds and compiles the RAG state graph:
#
#    START
#      └─► intake_node
#            └─► rag_node
#                  └─► router_node
#                        ├── "answer"   → END
#                        └── "escalate" → hitl_node → END
#
#  The compiled app is a runnable object:
#    result = app.invoke({"query": "..."}, config=config)
# =============================================================

from langchain_core.retrievers import BaseRetriever
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import GraphState
from nodes import (
    make_intake_node,
    make_rag_node,
    router_node,
    route_decision,
    hitl_node,
)
from config import LANGSMITH_PROJECT


def build_graph(
    retriever: BaseRetriever,
    use_reformulation: bool = False,
) -> any:
    """
    Construct and compile the LangGraph workflow.

    Args:
        retriever:         The configured LangChain retriever from retriever.py.
        use_reformulation: If True, the intake node will call GPT to fix
                           typos and expand abbreviations before retrieval.
                           Costs 1 extra LLM call. Default: False.

    Returns:
        A compiled LangGraph app (CompiledGraph). Call with:
            app.invoke({"query": "..."}, config={"configurable": {"thread_id": "..."}})

    Graph structure:
        START → intake → rag → router → [END | hitl → END]

    LangSmith tracing:
        When LANGSMITH_TRACING=true, every node execution, every LLM call,
        and every retriever call is automatically captured as a nested span.
        View traces at https://smith.langchain.com under the project name
        defined by LANGSMITH_PROJECT in config.py.

    State persistence:
        MemorySaver stores graph state in memory keyed by thread_id, enabling
        HITL interrupt/resume and multi-turn conversation history.
    """

    # ── 1. Instantiate node functions ─────────────────────────
    intake = make_intake_node(use_reformulation=use_reformulation)
    rag    = make_rag_node(retriever=retriever)

    # ── 2. Build the StateGraph ───────────────────────────────
    workflow = StateGraph(GraphState)

    workflow.add_node("intake",  intake)
    workflow.add_node("rag",     rag)
    workflow.add_node("router",  router_node)
    workflow.add_node("hitl",    hitl_node)

    # ── 3. Wire unconditional edges ───────────────────────────
    workflow.add_edge(START,    "intake")
    workflow.add_edge("intake", "rag")
    workflow.add_edge("rag",    "router")
    workflow.add_edge("hitl",   END)

    # ── 4. Wire conditional edge from router ──────────────────
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "answer":   END,
            "escalate": "hitl",
        },
    )

    # ── 5. Compile with checkpointer ─────────────────────────
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    return app


def get_graph_config(
    thread_id: str = "default",
    metadata: dict = None,
    tags: list = None,
    run_name: str = None,
) -> dict:
    """
    Build the config dict for app.invoke() with full LangSmith metadata.

    Args:
        thread_id: Unique session identifier.
        metadata:  Extra key-value pairs shown in the LangSmith trace UI.
        tags:      Filterable string tags in LangSmith.
        run_name:  Human-readable name for this run in the trace tree.

    Returns:
        Config dict with LangSmith tracing fields populated.
    """
    config: dict = {"configurable": {"thread_id": thread_id}}

    # LangSmith run metadata — visible in the trace detail panel
    if metadata:
        config["metadata"] = metadata
    if tags:
        config["tags"] = tags
    if run_name:
        config["run_name"] = run_name

    return config
