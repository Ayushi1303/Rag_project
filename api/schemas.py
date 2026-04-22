# =============================================================
#  api/schemas.py — Pydantic Request & Response Schemas
#
#  All FastAPI endpoint input/output types are defined here.
#  Pydantic validates incoming JSON automatically and produces
#  clean OpenAPI docs at /docs.
# =============================================================

from typing import Optional
from pydantic import BaseModel, Field


# ── Chat endpoint ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    """POST /chat — send a user question to the RAG pipeline."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The customer's natural-language question.",
        examples=["What is the return policy?"],
    )
    thread_id: Optional[str] = Field(
        default=None,
        description=(
            "Conversation session ID. Pass the same thread_id across "
            "multiple turns to maintain context. Auto-generated if omitted."
        ),
        examples=["user-abc-123"],
    )
    source: str = Field(
        default="api",
        description="Which interface sent this query. Used for LangSmith tagging.",
        examples=["api", "streamlit", "cli"],
    )


class SourceReference(BaseModel):
    """A single document source cited in the answer."""
    source: str = Field(description="PDF filename.")
    page: int   = Field(description="Page number within the PDF.")


class ChatResponse(BaseModel):
    """Response returned by POST /chat."""

    answer: str = Field(
        description="The final answer — from AI or human agent."
    )
    confidence: str = Field(
        description="'high' if AI answered confidently, 'low' if escalated.",
        examples=["high", "low"],
    )
    escalated: bool = Field(
        description="True if this query was handled by a human agent."
    )
    sources: list[SourceReference] = Field(
        description="Documents cited to generate the answer.",
        default=[],
    )
    thread_id: str = Field(
        description="The session ID used for this query (echo back to client)."
    )
    run_id: Optional[str] = Field(
        default=None,
        description=(
            "LangSmith run ID for this trace. Use with POST /feedback "
            "to submit thumbs-up/down ratings."
        ),
    )
    elapsed_ms: Optional[float] = Field(
        default=None,
        description="Total wall-clock time in milliseconds.",
    )


# ── HITL escalation endpoint ──────────────────────────────────

class HITLResumeRequest(BaseModel):
    """POST /hitl/respond — submit a human agent's response to a paused query."""

    thread_id: str = Field(
        ...,
        description="The thread_id of the paused HITL session.",
    )
    human_answer: str = Field(
        ...,
        min_length=1,
        description="The human agent's answer to deliver to the customer.",
    )


class HITLResumeResponse(BaseModel):
    """Response after a human agent submits their answer."""
    thread_id:    str  = Field(description="The resumed session ID.")
    answer:       str  = Field(description="The human answer (echoed back).")
    resolved:     bool = Field(description="True if session completed successfully.")


# ── Feedback endpoint ─────────────────────────────────────────

class FeedbackRequest(BaseModel):
    """POST /feedback — submit user feedback for a completed run."""

    run_id: str = Field(
        ...,
        description="LangSmith run ID from the ChatResponse.",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="1.0 = helpful (👍), 0.0 = not helpful (👎).",
    )
    comment: Optional[str] = Field(
        default="",
        description="Optional free-text comment from the user.",
    )


class FeedbackResponse(BaseModel):
    """Response after feedback submission."""
    submitted: bool = Field(description="True if feedback was sent to LangSmith.")
    message:   str  = Field(description="Human-readable confirmation or error.")


# ── Ingest endpoint ───────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response after POST /ingest — trigger re-ingestion of all PDFs."""
    success:     bool = Field(description="True if ingestion completed.")
    chunks_stored: int  = Field(description="Number of chunks written to ChromaDB.")
    message:     str  = Field(description="Status message.")


# ── Health & status endpoints ─────────────────────────────────

class HealthResponse(BaseModel):
    """GET /health — basic liveness check."""
    status:      str  = Field(description="'ok' if server is running.")
    version:     str  = Field(description="API version string.")


class StatusResponse(BaseModel):
    """GET /status — detailed system component status."""
    api:          str  = Field(description="API server status.")
    chroma_db:    str  = Field(description="'ready' or error message.")
    vector_count: int  = Field(description="Number of vectors in ChromaDB.")
    langsmith:    str  = Field(description="LangSmith connection status.")
    tracing:      bool = Field(description="Is LangSmith tracing active?")
