# =============================================================
#  state.py — LangGraph State Definition
#
#  The GraphState TypedDict is the single shared data structure
#  that flows through every node in the LangGraph pipeline.
#
#  Every node receives the full state dict and returns a dict
#  containing ONLY the keys it wants to update. LangGraph merges
#  the returned dict into the existing state automatically.
# =============================================================

from typing import Optional
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Central state object for the RAG LangGraph workflow.

    Fields are populated progressively as the graph executes:

    ─── Set by intake_node ──────────────────────────────────────
    query        : str  — The raw user input, exactly as typed.
    clean_query  : str  — Validated + optionally reformulated query.

    ─── Set by rag_node ─────────────────────────────────────────
    context      : str  — Formatted context string (joined chunks).
    context_chunks: list[str]  — Raw chunk texts (for inspection).
    sources      : list[dict]  — [{"source": "file.pdf", "page": 3}]
    answer       : str  — The LLM-generated answer.
    confidence   : str  — "high" or "low" (set by rag_node pre-router).

    ─── Set by router_node ──────────────────────────────────────
    route        : str  — "answer" or "escalate".

    ─── Set by hitl_node ────────────────────────────────────────
    human_answer : Optional[str]  — The human agent's typed response.
                                    None if HITL was not triggered.

    ─── Set on errors ───────────────────────────────────────────
    error        : Optional[str]  — Error message if something failed.
    """

    # Input
    query: str
    clean_query: str

    # Retrieval
    context: str
    context_chunks: list
    sources: list

    # Generation
    answer: str
    confidence: str

    # Routing
    route: str

    # HITL
    human_answer: Optional[str]

    # Error handling
    error: Optional[str]
