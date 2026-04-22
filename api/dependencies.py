# =============================================================
#  api/dependencies.py — FastAPI Dependency Injection
#
#  Provides shared application objects (vectorstore, retriever,
#  graph app) as FastAPI dependencies so they're initialised
#  once at startup and reused across all requests.
# =============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functools import lru_cache
from fastapi import HTTPException

from retriever import load_vectorstore, get_retriever
from graph import build_graph
from config import CHROMA_DIR


class AppState:
    """
    Singleton container for shared application objects.
    Initialised once when the FastAPI app starts.
    """
    vectorstore = None
    retriever   = None
    graph_app   = None
    # Paused HITL sessions: thread_id → paused state
    hitl_sessions: dict = {}


app_state = AppState()


def get_graph_app():
    """
    FastAPI dependency: return the compiled LangGraph app.
    Raises 503 if the app hasn't been initialised (ChromaDB missing).
    """
    if app_state.graph_app is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG system not ready. "
                "Run 'python ingest.py' first to build the vector store."
            ),
        )
    return app_state.graph_app


def get_retriever_dep():
    """FastAPI dependency: return the configured retriever."""
    if app_state.retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised.")
    return app_state.retriever


def get_vectorstore_dep():
    """FastAPI dependency: return the ChromaDB vectorstore."""
    if app_state.vectorstore is None:
        raise HTTPException(status_code=503, detail="Vector store not initialised.")
    return app_state.vectorstore


def initialise_app() -> bool:
    """
    Load ChromaDB + build LangGraph app at startup.
    Called from the FastAPI lifespan handler.

    Returns:
        True if initialisation succeeded, False otherwise.
    """
    try:
        print("  [api] Loading ChromaDB...")
        app_state.vectorstore = load_vectorstore(CHROMA_DIR)
        app_state.retriever   = get_retriever(app_state.vectorstore)

        print("  [api] Building LangGraph workflow...")
        app_state.graph_app   = build_graph(
            retriever=app_state.retriever,
            use_reformulation=False,
        )

        print("  [api] ✓ RAG system ready.")
        return True

    except (FileNotFoundError, ValueError) as e:
        print(f"  [api] ⚠ Warning: {e}")
        print("  [api] Server will start but /chat will return 503 until ingest runs.")
        return False
