# =============================================================
#  nodes.py — LangGraph Node Implementations
#
#  Each function here is a LangGraph node. Every node:
#    - Receives the full GraphState dict as input
#    - Returns a PARTIAL dict with only the keys it updates
#    - Never modifies state in-place
#
#  Nodes defined here:
#    1. intake_node   — validates & cleans the user query
#    2. rag_node      — retrieves context + generates LLM answer
#    3. router_node   — decides: "answer" or "escalate"
#    4. hitl_node     — pauses for human input, resumes with answer
# =============================================================

import json
import os
from datetime import datetime, timezone
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.tracers.context import tracing_v2_enabled

from state import GraphState
from retriever import retrieve
from prompt_builder import build_rag_messages, build_reformulation_prompt
from tracing import traced_node, _write_local_trace
from config import (
    OPENAI_API_KEY,
    CHAT_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    MIN_ANSWER_LENGTH,
    UNCERTAINTY_PHRASES,
    SENSITIVE_KEYWORDS,
    ESCALATION_LOG,
    LOGS_DIR,
)

# ── LLM singleton (shared across nodes) ──────────────────────

def _get_llm() -> ChatOpenAI:
    """Return a configured ChatOpenAI instance."""
    return ChatOpenAI(
        model=CHAT_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        openai_api_key=OPENAI_API_KEY,
    )


# ─────────────────────────────────────────────────────────────
#  NODE 1 — intake_node
# ─────────────────────────────────────────────────────────────

def make_intake_node(use_reformulation: bool = False):
    """
    Factory that returns the intake_node function.

    Args:
        use_reformulation: If True, call GPT to fix typos/abbreviations
                           in the query before retrieval. Costs 1 extra
                           LLM call but improves retrieval accuracy.

    Returns:
        A node function compatible with LangGraph's StateGraph.
    """

    @traced_node("intake_node")
    def intake_node(state: GraphState) -> dict:
        """
        Node 1: Validate and clean the user's raw query.

        - Strips whitespace
        - Checks for empty input
        - Optionally reformulates for better retrieval
        """
        query = state.get("query", "").strip()

        # ── Validation ────────────────────────────────────────
        if not query:
            return {
                "clean_query": "",
                "error": "Query is empty. Please type a question.",
            }

        if len(query) > 1000:
            # Truncate runaway input gracefully
            query = query[:1000]

        # ── Optional reformulation ────────────────────────────
        clean_query = query

        if use_reformulation:
            try:
                llm = _get_llm()
                prompt = build_reformulation_prompt()
                messages = prompt.format_messages(query=query)
                result = llm.invoke(messages)
                reformulated = result.content.strip()
                if reformulated:
                    clean_query = reformulated
                    print(f"  [intake] Reformulated: '{query}' → '{clean_query}'")
            except Exception as e:
                # Reformulation is optional — silently fall back to original
                print(f"  [intake] Reformulation skipped ({e})")

        return {"clean_query": clean_query, "error": None}

    return intake_node


# ─────────────────────────────────────────────────────────────
#  NODE 2 — rag_node
# ─────────────────────────────────────────────────────────────

def make_rag_node(retriever: BaseRetriever):
    """
    Factory that returns the rag_node function, with the retriever
    captured in its closure so LangGraph can call it as a plain function.

    Args:
        retriever: The configured LangChain retriever (from retriever.py).

    Returns:
        A node function compatible with LangGraph's StateGraph.
    """

    @traced_node("rag_node")
    def rag_node(state: GraphState) -> dict:
        """
        Node 2: Core RAG node — retrieve context + generate answer.

        Steps:
          1. Check for upstream errors (from intake_node)
          2. Retrieve top-k relevant chunks from ChromaDB
          3. Build the prompt with retrieved context
          4. Call GPT-4o and get an answer
          5. Assess confidence (pre-routing hint)
        """
        # ── Guard: upstream error ─────────────────────────────
        if state.get("error"):
            return {
                "answer": state["error"],
                "context": "",
                "context_chunks": [],
                "sources": [],
                "confidence": "low",
            }

        query = state.get("clean_query", "")
        print(f"\n  [rag]   Retrieving context for: '{query}'")

        # ── Step 1: Retrieve ──────────────────────────────────
        try:
            context_str, sources, raw_chunks = retrieve(query, retriever)
        except Exception as e:
            return {
                "answer": "I encountered an error while searching the knowledge base.",
                "context": "",
                "context_chunks": [],
                "sources": [],
                "confidence": "low",
                "error": str(e),
            }

        print(f"  [rag]   Retrieved {len(raw_chunks)} chunks.")

        # ── Step 2: Build prompt + call LLM ──────────────────
        try:
            llm = _get_llm()
            messages = build_rag_messages(context=context_str, question=query)
            response = llm.invoke(messages)
            answer = response.content.strip()
        except Exception as e:
            return {
                "answer": "I encountered an error while generating a response.",
                "context": context_str,
                "context_chunks": raw_chunks,
                "sources": sources,
                "confidence": "low",
                "error": str(e),
            }

        print(f"  [rag]   Answer generated ({len(answer)} chars).")

        # ── Step 3: Assess confidence (pre-routing hint) ──────
        answer_lower = answer.lower()
        confidence = "high"
        if (
            len(raw_chunks) == 0
            or len(answer.strip()) < MIN_ANSWER_LENGTH
            or any(phrase in answer_lower for phrase in UNCERTAINTY_PHRASES)
        ):
            confidence = "low"

        return {
            "context": context_str,
            "context_chunks": raw_chunks,
            "sources": sources,
            "answer": answer,
            "confidence": confidence,
            "error": None,
        }

    return rag_node


# ─────────────────────────────────────────────────────────────
#  NODE 3 — router_node
# ─────────────────────────────────────────────────────────────

@traced_node("router_node")
def router_node(state: GraphState) -> dict:
    """
    Node 3: Decide whether the answer is good enough to send
    to the user, or whether human escalation is required.

    This is a PURE function — no API calls, no side effects.
    It only reads state and returns a route decision.

    Routing rules (applied in order):
      1. Upstream error present        → escalate
      2. No chunks retrieved           → escalate
      3. Answer too short              → escalate
      4. Uncertainty phrase in answer  → escalate
      5. Sensitive keyword in query    → escalate
      6. None of the above             → answer
    """
    answer      = state.get("answer", "").lower()
    query       = state.get("clean_query", "").lower()
    chunks      = state.get("context_chunks", [])
    error       = state.get("error")

    # Rule 1: Upstream error
    if error:
        print("  [router] Route: ESCALATE (upstream error)")
        return {"route": "escalate"}

    # Rule 2: No context retrieved
    if not chunks:
        print("  [router] Route: ESCALATE (no chunks retrieved)")
        return {"route": "escalate"}

    # Rule 3: Answer is too short to be useful
    if len(state.get("answer", "").strip()) < MIN_ANSWER_LENGTH:
        print("  [router] Route: ESCALATE (answer too short)")
        return {"route": "escalate"}

    # Rule 4: LLM explicitly signals uncertainty
    if any(phrase in answer for phrase in UNCERTAINTY_PHRASES):
        print("  [router] Route: ESCALATE (uncertainty phrase detected)")
        return {"route": "escalate"}

    # Rule 5: Sensitive topic regardless of answer quality
    if any(kw in query for kw in SENSITIVE_KEYWORDS):
        print(f"  [router] Route: ESCALATE (sensitive keyword in query)")
        return {"route": "escalate"}

    # Rule 6: Default — answer is trustworthy
    print("  [router] Route: ANSWER (confident response)")
    return {"route": "answer"}


# ── Conditional edge function (used in graph wiring) ──────────

def route_decision(state: GraphState) -> str:
    """
    Called by LangGraph's add_conditional_edges().
    Must return a string that matches one of the keys in the
    edge mapping dict.

    Returns:
        "answer"   → graph transitions to END
        "escalate" → graph transitions to hitl_node
    """
    return state.get("route", "escalate")


# ─────────────────────────────────────────────────────────────
#  NODE 4 — hitl_node
# ─────────────────────────────────────────────────────────────

def _log_escalation(query: str, llm_answer: str, reason: str) -> None:
    """Append an escalation record to the JSONL log file."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    record = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "query":      query,
        "llm_answer": llm_answer,
        "reason":     reason,
        "resolved":   False,   # updated to True after human responds
    }
    with open(ESCALATION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _update_escalation_log(query: str, human_answer: str) -> None:
    """Mark the most recent escalation for this query as resolved."""
    if not os.path.exists(ESCALATION_LOG):
        return
    lines = []
    with open(ESCALATION_LOG, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Update the last matching unresolved record
    updated = False
    for i in reversed(range(len(lines))):
        try:
            record = json.loads(lines[i])
            if record.get("query") == query and not record.get("resolved"):
                record["resolved"] = True
                record["human_answer"] = human_answer
                lines[i] = json.dumps(record) + "\n"
                updated = True
                break
        except json.JSONDecodeError:
            continue

    if updated:
        with open(ESCALATION_LOG, "w", encoding="utf-8") as f:
            f.writelines(lines)


@traced_node("hitl_node")
def hitl_node(state: GraphState) -> dict:
    """
    Node 4: Human-in-the-Loop escalation node.

    This node pauses graph execution using LangGraph's interrupt()
    and waits for a human agent to provide a response via the CLI.

    Flow:
      1. Log the escalation to escalation_log.jsonl
      2. Print escalation info to the console
      3. Call interrupt() — graph PAUSES here
      4. Human types a response in the CLI
      5. Graph RESUMES — human response is in state
      6. Update the escalation log as resolved
      7. Return human_answer so it's delivered to the user

    Note: interrupt() is imported here to avoid circular imports
    and to make it easy to mock in unit tests.
    """
    from langgraph.types import interrupt

    query      = state.get("clean_query", state.get("query", ""))
    llm_answer = state.get("answer", "")
    route      = state.get("route", "escalate")

    # ── Step 1: Determine escalation reason ──────────────────
    if state.get("error"):
        reason = "system_error"
    elif not state.get("context_chunks"):
        reason = "no_context"
    elif any(kw in query.lower() for kw in SENSITIVE_KEYWORDS):
        reason = "sensitive_topic"
    else:
        reason = "low_confidence"

    # ── Step 2: Log escalation ────────────────────────────────
    _log_escalation(query=query, llm_answer=llm_answer, reason=reason)

    # ── Step 3: Print escalation notice to console ────────────
    print("\n" + "─" * 60)
    print("  ⚠  ESCALATED TO HUMAN AGENT")
    print("─" * 60)
    print(f"  Query:  {query}")
    print(f"  Reason: {reason}")
    if llm_answer:
        print(f"  AI attempt: {llm_answer[:120]}{'...' if len(llm_answer) > 120 else ''}")
    print("─" * 60)

    # ── Step 4: Pause and wait for human ──────────────────────
    # interrupt() suspends execution. The human's input will be
    # passed back via Command(resume={"human_answer": "..."})
    human_response: Any = interrupt(
        value={
            "message": "Please type your response for the customer:",
            "query":   query,
            "reason":  reason,
        }
    )

    # ── Step 5 & 6: Resume — store and log human answer ───────
    # human_response is whatever was passed to Command(resume=...)
    if isinstance(human_response, dict):
        human_answer = human_response.get("human_answer", str(human_response))
    else:
        human_answer = str(human_response)

    _update_escalation_log(query=query, human_answer=human_answer)

    print(f"\n  [hitl]  Human response recorded.")
    print("─" * 60 + "\n")

    return {
        "human_answer": human_answer,
        "route": "answer",   # Mark as resolved for downstream consumers
    }
