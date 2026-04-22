# =============================================================
#  tracing.py — LangSmith Tracing & Observability
#
#  LangSmith automatically traces every LangChain/LangGraph call
#  when LANGCHAIN_TRACING_V2=true is set. This module adds:
#
#    1. Explicit run metadata tagging per query
#    2. Node-level timing wrappers for detailed traces
#    3. Custom feedback logging (thumbs up/down per answer)
#    4. A health-check function to verify the connection
#    5. A local fallback trace log when LangSmith is offline
#
#  How LangSmith auto-traces LangGraph:
#    - Every node execution → one child span in the trace tree
#    - Every ChatOpenAI call → one LLM span with token counts
#    - Every retriever.invoke() → one retriever span with docs
#    - All of this happens AUTOMATICALLY — no code changes needed
#      in nodes.py as long as the env vars are set in config.py
#
#  Dashboard: https://smith.langchain.com
# =============================================================

import json
import os
import time
import functools
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from config import (
    LANGSMITH_TRACING,
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
    LANGSMITH_ENDPOINT,
    LOGS_DIR,
)

# ── Local fallback trace log ──────────────────────────────────
LOCAL_TRACE_LOG = os.path.join(LOGS_DIR, "local_traces.jsonl")


# ── 1. Verify LangSmith connection ───────────────────────────

def check_langsmith_connection() -> dict:
    """
    Verify that LangSmith is reachable and the API key is valid.

    Returns a status dict:
        {
            "enabled":   bool,    # Is tracing configured?
            "connected": bool,    # Did the API respond OK?
            "project":   str,     # Which project traces go to
            "message":   str,     # Human-readable status
        }
    """
    status = {
        "enabled":   LANGSMITH_TRACING,
        "connected": False,
        "project":   LANGSMITH_PROJECT,
        "message":   "",
    }

    if not LANGSMITH_TRACING:
        status["message"] = (
            "LangSmith tracing is DISABLED. "
            "Set LANGSMITH_TRACING=true in .env to enable."
        )
        return status

    if not LANGSMITH_API_KEY:
        status["message"] = (
            "LANGSMITH_API_KEY is missing. "
            "Get one at https://smith.langchain.com and add it to .env"
        )
        return status

    try:
        # LangSmith SDK check
        from langsmith import Client
        client = Client(
            api_url=LANGSMITH_ENDPOINT,
            api_key=LANGSMITH_API_KEY,
        )
        # Try listing projects — lightweight auth check
        projects = list(client.list_projects())
        project_names = [p.name for p in projects]

        status["connected"] = True
        status["message"] = (
            f"LangSmith connected. Project: '{LANGSMITH_PROJECT}'. "
            f"Available projects: {project_names}"
        )

    except ImportError:
        status["message"] = (
            "langsmith package not installed. Run: pip install langsmith"
        )
    except Exception as e:
        status["message"] = f"LangSmith connection failed: {e}"

    return status


# ── 2. Get a LangSmith client (or None if disabled) ──────────

def get_langsmith_client():
    """
    Return a configured LangSmith Client if tracing is enabled,
    otherwise return None.

    Usage:
        client = get_langsmith_client()
        if client:
            client.create_feedback(run_id=..., key="thumbs_up", score=1)
    """
    if not LANGSMITH_TRACING or not LANGSMITH_API_KEY:
        return None
    try:
        from langsmith import Client
        return Client(
            api_url=LANGSMITH_ENDPOINT,
            api_key=LANGSMITH_API_KEY,
        )
    except Exception:
        return None


# ── 3. Metadata builder for each query run ────────────────────

def build_run_metadata(
    query: str,
    thread_id: str,
    source: str = "cli",
) -> dict:
    """
    Build a metadata dict to tag a LangGraph run.
    Pass this into app.invoke() via config["metadata"].

    Args:
        query:     The user's question.
        thread_id: Session/conversation ID.
        source:    Where the query came from: "cli", "api", "streamlit".

    Returns:
        Metadata dict that appears in the LangSmith trace UI.

    Usage:
        config = get_graph_config(thread_id)
        config["metadata"] = build_run_metadata(query, thread_id, "api")
        result = app.invoke({"query": query}, config=config)
    """
    return {
        "query_preview":  query[:80] + ("..." if len(query) > 80 else ""),
        "thread_id":      thread_id,
        "source":         source,        # "cli" | "api" | "streamlit"
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "project":        LANGSMITH_PROJECT,
        "tracing_active": LANGSMITH_TRACING,
    }


def build_run_tags(query: str, source: str = "cli") -> list[str]:
    """
    Build tags to categorise this run in LangSmith.
    Tags appear as filterable labels in the dashboard.

    Returns:
        e.g. ["source:api", "project:rag-customer-support", "env:dev"]
    """
    tags = [
        f"source:{source}",
        f"project:{LANGSMITH_PROJECT}",
    ]
    # Tag sensitive queries for easy filtering in the dashboard
    from config import SENSITIVE_KEYWORDS
    if any(kw in query.lower() for kw in SENSITIVE_KEYWORDS):
        tags.append("sensitive:true")
    return tags


# ── 4. Node-level timing wrapper ──────────────────────────────

def traced_node(node_name: str):
    """
    Decorator that wraps a LangGraph node function with:
      - Wall-clock timing (logged locally)
      - A local JSONL trace entry (fallback when LangSmith is off)
      - Console logging of entry/exit

    LangSmith already auto-traces each node as a child span.
    This decorator adds local observability on top.

    Usage:
        @traced_node("rag_node")
        def rag_node(state): ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(state: dict) -> dict:
            start_time = time.perf_counter()
            print(f"  [trace] ▶ {node_name} started")

            error_msg = None
            result = {}
            try:
                result = func(state)
                return result
            except Exception as e:
                error_msg = str(e)
                raise
            finally:
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                print(f"  [trace] ◀ {node_name} finished in {elapsed_ms}ms")

                # Write local trace entry
                _write_local_trace(
                    node=node_name,
                    elapsed_ms=elapsed_ms,
                    query=state.get("clean_query", state.get("query", "")),
                    route=result.get("route", "") if isinstance(result, dict) else "",
                    confidence=result.get("confidence", "") if isinstance(result, dict) else "",
                    error=error_msg,
                )

        return wrapper
    return decorator


# ── 5. Custom feedback logging ────────────────────────────────

def log_feedback(
    run_id: str,
    score: float,
    comment: str = "",
    key: str = "user_rating",
) -> bool:
    """
    Submit user feedback for a specific LangSmith run.

    Args:
        run_id:  The LangSmith run ID (returned in trace metadata).
        score:   1.0 = positive (thumbs up), 0.0 = negative (thumbs down).
        comment: Optional free-text comment.
        key:     Feedback dimension name. Appears as a column in LangSmith.

    Returns:
        True if feedback was submitted successfully, False otherwise.

    Usage (in Streamlit):
        if st.button("👍"):
            log_feedback(run_id=st.session_state.last_run_id, score=1.0)
    """
    client = get_langsmith_client()
    if not client:
        print(f"  [trace] Feedback not sent (LangSmith disabled). Score: {score}")
        return False

    try:
        client.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
            comment=comment,
        )
        print(f"  [trace] Feedback submitted. run_id={run_id}, score={score}")
        return True
    except Exception as e:
        print(f"  [trace] Feedback failed: {e}")
        return False


# ── 6. Local trace fallback ───────────────────────────────────

def _write_local_trace(
    node: str,
    elapsed_ms: float,
    query: str = "",
    route: str = "",
    confidence: str = "",
    error: Optional[str] = None,
) -> None:
    """
    Write a local trace record to local_traces.jsonl.
    This runs regardless of whether LangSmith is enabled,
    giving you an offline audit trail.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    record = {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "node":        node,
        "elapsed_ms":  elapsed_ms,
        "query":       query[:100],
        "route":       route,
        "confidence":  confidence,
        "error":       error,
        "langsmith":   LANGSMITH_TRACING,
    }
    try:
        with open(LOCAL_TRACE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass   # Never crash the main flow because of logging


def write_query_trace(
    query: str,
    answer: str,
    route: str,
    sources: list,
    elapsed_ms: float,
    thread_id: str,
    source: str = "cli",
    escalated: bool = False,
) -> None:
    """
    Write a complete query-level trace (one entry per user question).
    Useful for building your own analytics dashboard on top of the JSONL.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    record = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "thread_id":  thread_id,
        "source":     source,
        "query":      query,
        "answer":     answer[:300],
        "route":      route,
        "escalated":  escalated,
        "sources":    sources,
        "elapsed_ms": elapsed_ms,
    }
    query_log = os.path.join(LOGS_DIR, "query_traces.jsonl")
    try:
        with open(query_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


# ── 7. Print tracing status on startup ───────────────────────

def print_tracing_status() -> None:
    """Print a human-readable tracing status banner."""
    status = check_langsmith_connection()
    print("\n" + "─" * 60)
    if status["connected"]:
        print(f"  🔍 LangSmith ACTIVE — project: '{status['project']}'")
        print(f"     Dashboard: https://smith.langchain.com")
    elif status["enabled"]:
        print(f"  ⚠  LangSmith ENABLED but not connected:")
        print(f"     {status['message']}")
    else:
        print(f"  📋 LangSmith DISABLED — local traces only")
        print(f"     Enable: set LANGSMITH_TRACING=true in .env")
    print("─" * 60 + "\n")
