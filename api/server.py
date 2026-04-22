#!/usr/bin/env python3
# =============================================================
#  api/server.py — FastAPI REST Server
#
#  Endpoints:
#    GET  /health             — liveness check
#    GET  /status             — system component status
#    POST /chat               — send a question, get an answer
#    POST /hitl/respond       — human agent submits answer for paused session
#    POST /feedback           — submit thumbs-up/down for a run
#    POST /ingest             — trigger re-ingestion of all PDFs
#
#  Run:
#    uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
#
#  Interactive docs:
#    http://localhost:8000/docs   (Swagger UI)
#    http://localhost:8000/redoc  (ReDoc)
# =============================================================

import sys
import os
import time
import uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from api.schemas import (
    ChatRequest, ChatResponse, SourceReference,
    HITLResumeRequest, HITLResumeResponse,
    FeedbackRequest, FeedbackResponse,
    IngestResponse,
    HealthResponse, StatusResponse,
)
from api.dependencies import app_state, get_graph_app, initialise_app
from graph import get_graph_config
from tracing import (
    build_run_metadata,
    build_run_tags,
    log_feedback,
    write_query_trace,
    check_langsmith_connection,
)
from ingest import ingest_documents
from config import (
    API_TITLE, API_VERSION,
    CHROMA_DIR, DATA_DIR,
    LANGSMITH_TRACING,
)


# ── Lifespan: startup & shutdown ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the RAG system when the server starts."""
    print("\n" + "=" * 60)
    print(f"  {API_TITLE} v{API_VERSION}")
    print("=" * 60)
    initialise_app()
    yield
    print("\n  [api] Server shutting down.")


# ── FastAPI app ───────────────────────────────────────────────

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "A production-quality RAG Customer Support Assistant powered by "
        "GPT-4o, ChromaDB, and LangGraph. Supports Human-in-the-Loop "
        "escalation and LangSmith observability."
    ),
    lifespan=lifespan,
)

# Allow Streamlit (and any other frontend) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── GET /health ───────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    tags=["System"],
)
async def health():
    """Returns 200 OK if the server is running."""
    return HealthResponse(status="ok", version=API_VERSION)


# ── GET /status ───────────────────────────────────────────────

@app.get(
    "/status",
    response_model=StatusResponse,
    summary="Detailed system status",
    tags=["System"],
)
async def status():
    """
    Returns the status of each system component:
    - API server
    - ChromaDB (vector count)
    - LangSmith connection
    """
    # ChromaDB status
    chroma_status = "not initialised"
    vector_count  = 0
    if app_state.vectorstore:
        try:
            vector_count  = app_state.vectorstore._collection.count()
            chroma_status = "ready"
        except Exception as e:
            chroma_status = f"error: {e}"

    # LangSmith status
    ls = check_langsmith_connection()
    ls_status = ls["message"] if not ls["connected"] else f"connected — project: {ls['project']}"

    return StatusResponse(
        api="running",
        chroma_db=chroma_status,
        vector_count=vector_count,
        langsmith=ls_status,
        tracing=LANGSMITH_TRACING,
    )


# ── POST /chat ────────────────────────────────────────────────

@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question",
    tags=["RAG"],
)
async def chat(
    request: ChatRequest,
    graph_app = Depends(get_graph_app),
):
    """
    Send a customer question through the RAG pipeline.

    The pipeline:
    1. Validates and cleans the query (intake_node)
    2. Retrieves relevant chunks from ChromaDB (rag_node)
    3. Generates a grounded answer via GPT-4o (rag_node)
    4. Routes to answer or HITL escalation (router_node)

    If the query is escalated to HITL, this endpoint returns
    a 202 response with `escalated: true`. The human agent must
    then call POST /hitl/respond to complete the interaction.

    All LangSmith traces are tagged with `source:api`.
    """
    graph_app = get_graph_app()

    # ── Session ID ────────────────────────────────────────────
    thread_id = request.thread_id or f"api-{uuid.uuid4().hex[:12]}"

    # ── LangSmith run config ──────────────────────────────────
    metadata = build_run_metadata(
        query=request.query,
        thread_id=thread_id,
        source=request.source,
    )
    tags = build_run_tags(query=request.query, source=request.source)

    config = get_graph_config(
        thread_id=thread_id,
        metadata=metadata,
        tags=tags,
        run_name=f"chat:{request.query[:40]}",
    )

    start_time = time.perf_counter()
    run_id: str | None = None

    try:
        final_state = graph_app.invoke(
            {"query": request.query},
            config=config,
        )

    except GraphInterrupt as interrupt_exc:
        # ── HITL triggered ────────────────────────────────────
        # Parse the interrupt value from LangGraph
        interrupt_value = {}
        if interrupt_exc.args:
            raw = interrupt_exc.args[0]
            if isinstance(raw, (list, tuple)) and raw:
                first = raw[0]
                interrupt_value = first.value if hasattr(first, "value") else (first if isinstance(first, dict) else {})
            elif isinstance(raw, dict):
                interrupt_value = raw

        # Store the paused session so /hitl/respond can resume it
        app_state.hitl_sessions[thread_id] = {
            "interrupt_value": interrupt_value,
            "query":           request.query,
        }

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Return 202 Accepted — answer will arrive via /hitl/respond
        return JSONResponse(
            status_code=202,
            content={
                "answer":      "This query has been escalated to a human agent. Please wait for a response.",
                "confidence":  "low",
                "escalated":   True,
                "sources":     [],
                "thread_id":   thread_id,
                "run_id":      None,
                "elapsed_ms":  elapsed_ms,
                "hitl_reason": interrupt_value.get("reason", "unknown"),
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {e}")

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # ── Extract result ────────────────────────────────────────
    human_answer = final_state.get("human_answer")
    ai_answer    = final_state.get("answer", "")
    final_answer = human_answer if human_answer else ai_answer
    escalated    = human_answer is not None

    sources = [
        SourceReference(
            source=s.get("source", "unknown"),
            page=s.get("page", 0),
        )
        for s in final_state.get("sources", [])
    ]

    # ── Write local trace ─────────────────────────────────────
    write_query_trace(
        query=request.query,
        answer=final_answer,
        route=final_state.get("route", ""),
        sources=final_state.get("sources", []),
        elapsed_ms=elapsed_ms,
        thread_id=thread_id,
        source=request.source,
        escalated=escalated,
    )

    return ChatResponse(
        answer=final_answer,
        confidence=final_state.get("confidence", "unknown"),
        escalated=escalated,
        sources=sources,
        thread_id=thread_id,
        run_id=run_id,
        elapsed_ms=elapsed_ms,
    )


# ── POST /hitl/respond ────────────────────────────────────────

@app.post(
    "/hitl/respond",
    response_model=HITLResumeResponse,
    summary="Human agent submits answer for a paused HITL session",
    tags=["HITL"],
)
async def hitl_respond(
    request: HITLResumeRequest,
    graph_app = Depends(get_graph_app),
):
    """
    Called by a human agent (or agent dashboard) after receiving an
    escalated query. Resumes the paused LangGraph session with the
    human's answer.

    Workflow:
    1. POST /chat returns 202 with escalated:true and a thread_id
    2. Human agent reads the query from the escalation log or dashboard
    3. Human agent POSTs to /hitl/respond with their answer + thread_id
    4. The graph resumes, stores the human answer, and completes
    """
    thread_id = request.thread_id

    if thread_id not in app_state.hitl_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"No active HITL session found for thread_id: {thread_id}",
        )

    config = get_graph_config(thread_id=thread_id)

    try:
        final_state = graph_app.invoke(
            Command(resume={"human_answer": request.human_answer}),
            config=config,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resume HITL session: {e}",
        )

    # Clean up the paused session
    app_state.hitl_sessions.pop(thread_id, None)

    return HITLResumeResponse(
        thread_id=thread_id,
        answer=request.human_answer,
        resolved=True,
    )


# ── POST /feedback ────────────────────────────────────────────

@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit feedback for a completed run",
    tags=["Observability"],
)
async def feedback(request: FeedbackRequest):
    """
    Submit a thumbs-up (score=1.0) or thumbs-down (score=0.0) rating
    for a specific LangSmith run. The run_id comes from the ChatResponse.

    This feedback appears in the LangSmith dashboard under the run's
    'Feedback' tab and can be used to evaluate system quality over time.
    """
    success = log_feedback(
        run_id=request.run_id,
        score=request.score,
        comment=request.comment or "",
    )

    if success:
        return FeedbackResponse(
            submitted=True,
            message=f"Feedback (score={request.score}) submitted to LangSmith.",
        )
    else:
        return FeedbackResponse(
            submitted=False,
            message=(
                "Feedback not sent. LangSmith tracing may be disabled. "
                "Set LANGSMITH_TRACING=true in .env to enable."
            ),
        )


# ── POST /ingest ──────────────────────────────────────────────

@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Re-ingest all PDF documents",
    tags=["System"],
)
async def ingest(background_tasks: BackgroundTasks):
    """
    Trigger a re-ingestion of all PDF files in the data/ directory.
    This runs in the background and updates ChromaDB.

    Use this endpoint when you add or update PDF documents and want
    the knowledge base to reflect the new content without restarting
    the server.
    """
    def _run_ingest():
        try:
            vs = ingest_documents(source=DATA_DIR, chroma_path=CHROMA_DIR)
            app_state.vectorstore = vs
            from retriever import get_retriever
            app_state.retriever   = get_retriever(vs)
            app_state.graph_app   = build_graph(retriever=app_state.retriever)
            count = vs._collection.count()
            print(f"  [api] ✓ Re-ingestion complete. {count} vectors in ChromaDB.")
        except Exception as e:
            print(f"  [api] ✗ Re-ingestion failed: {e}")

    background_tasks.add_task(_run_ingest)

    # Return immediately; ingestion runs in background
    current_count = 0
    if app_state.vectorstore:
        try:
            current_count = app_state.vectorstore._collection.count()
        except Exception:
            pass

    return IngestResponse(
        success=True,
        chunks_stored=current_count,
        message="Ingestion started in background. Check server logs for progress.",
    )


# ── Run directly ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run("api.server:app", host=API_HOST, port=API_PORT, reload=True)
