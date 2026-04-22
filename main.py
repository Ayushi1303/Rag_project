#!/usr/bin/env python3
# =============================================================
#  main.py — Application Entry Point
#
#  Usage:
#    python main.py
#
#  Prerequisites:
#    1. Copy .env.example to .env and add your OPENAI_API_KEY
#    2. Put at least one PDF in the data/ folder
#    3. Run:  python ingest.py     (only needed once)
#    4. Run:  python main.py       (start chatting)
# =============================================================

import sys
import uuid

from langgraph.errors import GraphInterrupt

from config import CHROMA_DIR, THREAD_ID
from retriever import load_vectorstore, get_retriever
from graph import build_graph, get_graph_config
from display import (
    print_welcome,
    print_answer,
    print_error,
    print_goodbye,
    print_thinking,
)
from hitl_handler import handle_hitl_interrupt


def run():
    """
    Main application loop.

    1. Loads ChromaDB from disk
    2. Builds the LangGraph workflow
    3. Enters an interactive Q&A loop
    4. Handles HITL interrupts transparently
    5. Exits cleanly on 'quit' or Ctrl-C
    """

    # ── Startup ───────────────────────────────────────────────
    print_welcome()

    # Load vector store (raises FileNotFoundError if ingest hasn't run)
    try:
        vectorstore = load_vectorstore(CHROMA_DIR)
    except (FileNotFoundError, ValueError) as e:
        print_error(str(e))
        sys.exit(1)

    retriever = get_retriever(vectorstore)

    # Build the LangGraph app
    # Set use_reformulation=True to enable query spell-correction
    app = build_graph(retriever=retriever, use_reformulation=False)

    # Each session gets its own thread_id for state isolation
    session_id = f"{THREAD_ID}-{uuid.uuid4().hex[:8]}"
    config = get_graph_config(thread_id=session_id)

    # ── Main Q&A loop ─────────────────────────────────────────
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Exit commands
            if user_input.lower() in ("quit", "exit", "q", "bye"):
                break

            # Skip empty input
            if not user_input:
                print("  Please type a question.\n")
                continue

            print_thinking()

            # ── Invoke the graph ──────────────────────────────
            try:
                final_state = app.invoke(
                    {"query": user_input},
                    config=config,
                )

            except GraphInterrupt as interrupt_exc:
                # HITL interrupt: the hitl_node called interrupt()
                # Extract the interrupt value and hand off to hitl_handler
                interrupt_value = {}
                if interrupt_exc.args:
                    # LangGraph packages the interrupt value in args[0]
                    raw = interrupt_exc.args[0]
                    if isinstance(raw, (list, tuple)) and len(raw) > 0:
                        first = raw[0]
                        # LangGraph wraps it in an Interrupt namedtuple
                        if hasattr(first, "value"):
                            interrupt_value = first.value
                        elif isinstance(first, dict):
                            interrupt_value = first
                    elif isinstance(raw, dict):
                        interrupt_value = raw

                final_state = handle_hitl_interrupt(
                    app=app,
                    config=config,
                    interrupt_value=interrupt_value,
                )

            # ── Display the result ────────────────────────────
            if final_state.get("error") and not final_state.get("answer"):
                print_error(final_state["error"])
            else:
                print_answer(final_state)

        except KeyboardInterrupt:
            # Ctrl-C exits gracefully
            print()
            break

        except Exception as e:
            # Catch-all: never crash the loop on unexpected errors
            print_error(f"Unexpected error: {e}")
            print("  Please try again or type 'quit' to exit.\n")

    print_goodbye()


if __name__ == "__main__":
    run()
