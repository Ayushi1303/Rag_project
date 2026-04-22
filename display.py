# =============================================================
#  display.py — CLI Display & Output Formatting
#
#  All print/display logic lives here so main.py stays clean.
# =============================================================

from config import SEPARATOR


def print_welcome():
    """Print the welcome banner when the app starts."""
    print("\n" + SEPARATOR)
    print("  RAG Customer Support Assistant")
    print("  Powered by GPT-4o + ChromaDB + LangGraph")
    print(SEPARATOR)
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to end the session.")
    print(SEPARATOR + "\n")


def print_answer(state: dict):
    """
    Print the final answer to the user.

    Reads from state to determine whether the answer came from
    the AI (state["answer"]) or a human agent (state["human_answer"]).

    Args:
        state: The final GraphState dict returned by app.invoke().
    """
    human_answer = state.get("human_answer")
    ai_answer    = state.get("answer", "")
    sources      = state.get("sources", [])
    confidence   = state.get("confidence", "unknown").upper()
    escalated    = human_answer is not None

    print("\n" + "─" * 60)

    if escalated:
        print("  🧑 HUMAN AGENT RESPONSE:")
        print("─" * 60)
        print(f"\n  {human_answer}\n")
        print("─" * 60)
        print("  [Source: Human Agent]")
    else:
        print("  🤖 ASSISTANT:")
        print("─" * 60)
        # Indent each line of the answer for readability
        for line in ai_answer.split("\n"):
            print(f"  {line}")
        print()
        print("─" * 60)
        # Print sources
        if sources:
            print("  📄 Sources:")
            for src in sources:
                print(f"     • {src.get('source', '?')}  —  Page {src.get('page', '?')}")
        print(f"  📊 Confidence: {confidence}")

    print("─" * 60 + "\n")


def print_error(message: str):
    """Print a formatted error message."""
    print(f"\n  ❌ Error: {message}\n")


def print_goodbye():
    """Print the goodbye message on session exit."""
    print("\n" + SEPARATOR)
    print("  Session ended. Goodbye!")
    print(SEPARATOR + "\n")


def print_thinking():
    """Print a short 'thinking' indicator while waiting for the LLM."""
    print("  ⏳ Searching knowledge base and generating response...\n")
