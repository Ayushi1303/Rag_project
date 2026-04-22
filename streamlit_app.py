#!/usr/bin/env python3
# =============================================================
#  streamlit_app.py — Streamlit Web UI
#
#  A polished chat interface for the RAG Customer Support Assistant.
#
#  Features:
#    - Chat bubble UI with conversation history
#    - Live "thinking" spinner during RAG processing
#    - Source citations displayed per answer
#    - Confidence badge (HIGH / LOW) per answer
#    - Thumbs-up / thumbs-down feedback buttons (→ LangSmith)
#    - HITL escalation panel for human agents
#    - Sidebar: system status, LangSmith link, PDF uploader
#    - PDF upload + auto-ingest without leaving the UI
#
#  Run:
#    streamlit run streamlit_app.py
#
#  Requires the FastAPI server to be running:
#    uvicorn api.server:app --reload --port 8000
# =============================================================

import sys
import os
import time
import uuid
import json
import requests
from datetime import datetime

import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────
st.set_page_config(
    page_title="RAG Customer Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
}
[data-testid="stSidebar"] {
    background: #1a1d27;
    border-right: 1px solid #2d3142;
}

/* ── Chat bubbles ── */
.user-bubble {
    background: linear-gradient(135deg, #1f4e79, #2e75b6);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0 6px 15%;
    text-align: right;
    font-size: 15px;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.bot-bubble {
    background: #1e2130;
    color: #e8eaf0;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 15% 6px 0;
    font-size: 15px;
    line-height: 1.6;
    border: 1px solid #2d3142;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.human-bubble {
    background: linear-gradient(135deg, #0e6655, #1e8449);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 15% 6px 0;
    font-size: 15px;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.escalated-bubble {
    background: #2d1f0e;
    color: #f0c080;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 15% 6px 0;
    font-size: 15px;
    border: 1px solid #6d4c0e;
}

/* ── Badges ── */
.badge-high {
    background: #0e6655; color: white;
    padding: 2px 10px; border-radius: 12px;
    font-size: 12px; font-weight: bold;
    display: inline-block; margin-left: 8px;
}
.badge-low {
    background: #7d2020; color: white;
    padding: 2px 10px; border-radius: 12px;
    font-size: 12px; font-weight: bold;
    display: inline-block; margin-left: 8px;
}
.badge-human {
    background: #1e8449; color: white;
    padding: 2px 10px; border-radius: 12px;
    font-size: 12px; font-weight: bold;
    display: inline-block; margin-left: 8px;
}

/* ── Source chips ── */
.source-chip {
    background: #16213e; color: #7eb8f7;
    border: 1px solid #2e75b6;
    padding: 2px 10px; border-radius: 10px;
    font-size: 12px; display: inline-block;
    margin: 2px 4px 2px 0;
}

/* ── Status indicators ── */
.status-dot-green { color: #2ecc71; }
.status-dot-red   { color: #e74c3c; }
.status-dot-gray  { color: #95a5a6; }

/* ── Section headers ── */
.section-header {
    color: #7eb8f7; font-size: 13px;
    font-weight: 600; letter-spacing: 0.05em;
    text-transform: uppercase; margin: 12px 0 4px 0;
}

/* ── HITL panel ── */
.hitl-panel {
    background: #2d1f0e;
    border: 2px solid #d4820a;
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────

def api_get(path: str, timeout: int = 5) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(path: str, payload: dict, timeout: int = 60) -> tuple[int, dict]:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
        return r.status_code, r.json()
    except requests.exceptions.ConnectionError:
        return 0, {"detail": "Cannot connect to API server. Is it running?"}
    except Exception as e:
        return 0, {"detail": str(e)}


# ── Session state initialisation ─────────────────────────────

def init_session():
    defaults = {
        "messages":         [],       # List of chat message dicts
        "thread_id":        f"st-{uuid.uuid4().hex[:10]}",
        "hitl_pending":     False,    # Is there a paused HITL session?
        "hitl_query":       "",       # The escalated query text
        "total_queries":    0,
        "total_escalated":  0,
        "last_run_id":      None,
        "feedback_given":   set(),    # run_ids already rated
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Sidebar ───────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 RAG Assistant")
    st.markdown("*Customer Support · GPT-4o · ChromaDB*")
    st.divider()

    # ── System status ──────────────────────────────────────
    st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)

    status = api_get("/status")
    health = api_get("/health")

    if health and health.get("status") == "ok":
        st.markdown('<span class="status-dot-green">●</span> **API Server** — Online', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot-red">●</span> **API Server** — Offline', unsafe_allow_html=True)
        st.warning("Start the API: `uvicorn api.server:app --reload`", icon="⚠️")

    if status:
        chroma = status.get("chroma_db", "unknown")
        count  = status.get("vector_count", 0)
        if chroma == "ready":
            st.markdown(f'<span class="status-dot-green">●</span> **ChromaDB** — {count:,} vectors', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-dot-red">●</span> **ChromaDB** — {chroma}', unsafe_allow_html=True)

        tracing = status.get("tracing", False)
        if tracing:
            st.markdown('<span class="status-dot-green">●</span> **LangSmith** — Active', unsafe_allow_html=True)
            st.markdown(
                "📊 [Open Dashboard](https://smith.langchain.com)",
                unsafe_allow_html=False,
            )
        else:
            st.markdown('<span class="status-dot-gray">●</span> **LangSmith** — Disabled', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot-gray">●</span> **ChromaDB** — Unknown', unsafe_allow_html=True)

    st.divider()

    # ── Session stats ──────────────────────────────────────
    st.markdown('<div class="section-header">Session Stats</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Queries", st.session_state.total_queries)
    col2.metric("Escalated", st.session_state.total_escalated)

    st.divider()

    # ── PDF uploader ───────────────────────────────────────
    st.markdown('<div class="section-header">Upload Knowledge Base</div>', unsafe_allow_html=True)
    st.caption("Upload PDFs, then click Ingest to update the knowledge base.")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        for uf in uploaded_files:
            save_path = os.path.join(data_dir, uf.name)
            with open(save_path, "wb") as f:
                f.write(uf.read())
        st.success(f"✓ {len(uploaded_files)} file(s) saved to data/")

    if st.button("🔄 Ingest Documents", use_container_width=True):
        with st.spinner("Ingesting PDFs into ChromaDB…"):
            code, resp = api_post("/ingest", {})
            if code == 200:
                st.success("✓ Ingestion started! Refresh status in a moment.")
            else:
                st.error(f"Ingest failed: {resp.get('detail', 'unknown error')}")

    st.divider()

    # ── Settings ───────────────────────────────────────────
    st.markdown('<div class="section-header">Settings</div>', unsafe_allow_html=True)

    show_sources  = st.toggle("Show source citations",  value=True)
    show_timing   = st.toggle("Show response timing",   value=False)
    show_raw_json = st.toggle("Show raw API response",  value=False)

    st.divider()

    if st.button("🗑 Clear Conversation", use_container_width=True):
        st.session_state.messages        = []
        st.session_state.thread_id       = f"st-{uuid.uuid4().hex[:10]}"
        st.session_state.hitl_pending    = False
        st.session_state.total_queries   = 0
        st.session_state.total_escalated = 0
        st.rerun()

    st.caption(f"Session: `{st.session_state.thread_id}`")


# ── Main chat area ────────────────────────────────────────────

st.markdown("## 💬 Customer Support Chat")
st.caption("Powered by GPT-4o · ChromaDB · LangGraph · LangSmith")

# ── Render conversation history ───────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; color:#555; padding: 60px 0;">
            <div style="font-size:48px">🤖</div>
            <div style="font-size:18px; margin-top:12px; color:#7eb8f7">
                RAG Customer Support Assistant
            </div>
            <div style="font-size:14px; margin-top:8px; color:#666">
                Ask me anything about your products, policies, or services.
            </div>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            st.markdown(
                f'<div class="user-bubble">👤 {content}</div>',
                unsafe_allow_html=True,
            )

        elif role == "assistant":
            escalated  = msg.get("escalated", False)
            confidence = msg.get("confidence", "high")
            sources    = msg.get("sources", [])
            elapsed_ms = msg.get("elapsed_ms")
            run_id     = msg.get("run_id")

            # Choose bubble style
            if escalated and msg.get("is_human"):
                bubble_class = "human-bubble"
                icon = "🧑 Human Agent"
                badge = '<span class="badge-human">HUMAN</span>'
            elif escalated and not msg.get("is_human"):
                bubble_class = "escalated-bubble"
                icon = "⚠️ Escalated"
                badge = '<span class="badge-low">ESCALATING…</span>'
            else:
                bubble_class = "bot-bubble"
                icon = "🤖 Assistant"
                badge = (
                    '<span class="badge-high">HIGH CONFIDENCE</span>'
                    if confidence == "high"
                    else '<span class="badge-low">LOW CONFIDENCE</span>'
                )

            # Main bubble
            st.markdown(
                f'<div class="{bubble_class}">'
                f'<strong>{icon}</strong>{badge}<br><br>{content}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Sources ───────────────────────────────────
            if show_sources and sources and not escalated:
                chips = "".join(
                    f'<span class="source-chip">📄 {s["source"]} p.{s["page"]}</span>'
                    for s in sources
                )
                st.markdown(
                    f'<div style="margin: 0 15% 8px 0;">{chips}</div>',
                    unsafe_allow_html=True,
                )

            # ── Timing ────────────────────────────────────
            if show_timing and elapsed_ms:
                st.markdown(
                    f'<div style="color:#555; font-size:12px; margin: 0 15% 4px 0;">⏱ {elapsed_ms:.0f}ms</div>',
                    unsafe_allow_html=True,
                )

            # ── Feedback buttons ──────────────────────────
            if run_id and run_id not in st.session_state.feedback_given and not escalated:
                fb_col1, fb_col2, fb_spacer = st.columns([1, 1, 10])
                with fb_col1:
                    if st.button("👍", key=f"up_{run_id}", help="Helpful"):
                        code, _ = api_post("/feedback", {
                            "run_id": run_id, "score": 1.0, "comment": "thumbs_up"
                        })
                        if code == 200:
                            st.session_state.feedback_given.add(run_id)
                            st.toast("Thanks for the feedback! 👍")
                with fb_col2:
                    if st.button("👎", key=f"down_{run_id}", help="Not helpful"):
                        code, _ = api_post("/feedback", {
                            "run_id": run_id, "score": 0.0, "comment": "thumbs_down"
                        })
                        if code == 200:
                            st.session_state.feedback_given.add(run_id)
                            st.toast("Thanks for the feedback! 👎")

            # ── Raw JSON ──────────────────────────────────
            if show_raw_json and msg.get("raw"):
                with st.expander("Raw API response"):
                    st.json(msg["raw"])


# ── HITL panel (shown when a query is escalated) ─────────────

if st.session_state.hitl_pending:
    st.divider()
    st.markdown("""
    <div class="hitl-panel">
        <strong style="color:#f0a030; font-size:16px;">⚠ Human Agent Required</strong><br>
        <span style="color:#ccc; font-size:14px;">
        The following query could not be answered automatically.
        Please provide a response below.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Escalated query:** *{st.session_state.hitl_query}*")

    human_input = st.text_area(
        "Your response (as human agent):",
        placeholder="Type your response for the customer here…",
        height=100,
        key="hitl_input",
    )

    col_submit, col_cancel = st.columns([2, 1])
    with col_submit:
        if st.button("✅ Submit Response", type="primary", use_container_width=True):
            if human_input.strip():
                code, resp = api_post("/hitl/respond", {
                    "thread_id":    st.session_state.thread_id,
                    "human_answer": human_input.strip(),
                })
                if code == 200:
                    st.session_state.messages.append({
                        "role":      "assistant",
                        "content":   human_input.strip(),
                        "escalated": True,
                        "is_human":  True,
                        "sources":   [],
                        "confidence":"low",
                    })
                    st.session_state.hitl_pending    = False
                    st.session_state.hitl_query      = ""
                    st.session_state.total_escalated += 1
                    st.rerun()
                else:
                    st.error(f"Failed to submit: {resp.get('detail', 'unknown')}")
            else:
                st.warning("Please type a response before submitting.")

    with col_cancel:
        if st.button("✖ Cancel", use_container_width=True):
            st.session_state.hitl_pending = False
            st.session_state.hitl_query   = ""
            st.rerun()


# ── Chat input ────────────────────────────────────────────────

st.divider()

with st.form("chat_form", clear_on_submit=True):
    input_col, send_col = st.columns([10, 1])
    with input_col:
        user_query = st.text_input(
            "Your question",
            placeholder="Ask anything about our products or policies…",
            label_visibility="collapsed",
            disabled=st.session_state.hitl_pending,
        )
    with send_col:
        submitted = st.form_submit_button(
            "Send",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.hitl_pending,
        )

if submitted and user_query.strip():
    query = user_query.strip()

    # Add user bubble immediately
    st.session_state.messages.append({
        "role":    "user",
        "content": query,
    })
    st.session_state.total_queries += 1

    # ── Call the API ──────────────────────────────────────
    with st.spinner("🔍 Searching knowledge base…"):
        code, resp = api_post("/chat", {
            "query":     query,
            "thread_id": st.session_state.thread_id,
            "source":    "streamlit",
        })

    if code == 0:
        # Connection error
        st.session_state.messages.append({
            "role":      "assistant",
            "content":   f"❌ {resp.get('detail', 'Connection error')}",
            "escalated": False,
            "confidence":"low",
            "sources":   [],
        })

    elif code == 202:
        # HITL escalation — query paused
        st.session_state.hitl_pending = True
        st.session_state.hitl_query   = query
        st.session_state.messages.append({
            "role":      "assistant",
            "content":   "⚠️ This query requires human review. Please see the escalation panel below.",
            "escalated": True,
            "is_human":  False,
            "sources":   [],
            "confidence":"low",
            "raw":       resp,
        })

    elif code == 200:
        # Successful answer
        run_id    = resp.get("run_id")
        st.session_state.last_run_id = run_id

        st.session_state.messages.append({
            "role":       "assistant",
            "content":    resp.get("answer", "No answer returned."),
            "escalated":  resp.get("escalated", False),
            "is_human":   resp.get("escalated", False),
            "confidence": resp.get("confidence", "unknown"),
            "sources":    resp.get("sources", []),
            "elapsed_ms": resp.get("elapsed_ms"),
            "run_id":     run_id,
            "raw":        resp,
        })

    else:
        # API error
        detail = resp.get("detail", f"HTTP {code}")
        st.session_state.messages.append({
            "role":      "assistant",
            "content":   f"❌ Error: {detail}",
            "escalated": False,
            "confidence":"low",
            "sources":   [],
        })

    st.rerun()
