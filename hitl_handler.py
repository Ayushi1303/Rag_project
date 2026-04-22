# =============================================================
#  hitl_handler.py — CLI Handler for HITL Interrupt/Resume
#
#  When the graph hits hitl_node and calls interrupt(), execution
#  pauses. This module handles:
#    - Detecting that an interrupt occurred
#    - Prompting the human agent for input
#    - Resuming the graph with Command(resume=...)
# =============================================================

from langgraph.types import Command


def handle_hitl_interrupt(app, config: dict, interrupt_value: dict) -> dict:
    """
    Handle a LangGraph interrupt triggered by the HITL node.

    This function is called by main.py when app.invoke() raises
    an interrupt. It:
      1. Extracts the escalation info from the interrupt value
      2. Prompts the human agent for a response (CLI input)
      3. Resumes the graph with the human's answer
      4. Returns the final state

    Args:
        app:             The compiled LangGraph app.
        config:          The same config dict used in the original invoke().
        interrupt_value: The dict passed to interrupt() inside hitl_node.

    Returns:
        The final GraphState after resuming.
    """
    query   = interrupt_value.get("query",   "Unknown query")
    reason  = interrupt_value.get("reason",  "unspecified")
    message = interrupt_value.get("message", "Please provide a response:")

    print("\n" + "═" * 60)
    print("  HUMAN AGENT INPUT REQUIRED")
    print("═" * 60)
    print(f"  Customer query : {query}")
    print(f"  Escalation reason: {reason}")
    print("─" * 60)
    print(f"  {message}")
    print("─" * 60)

    # Prompt the human agent
    human_answer = input("  Your response: ").strip()

    if not human_answer:
        human_answer = (
            "Thank you for your query. A support agent will follow up "
            "with you via email within 24 hours."
        )
        print(f"  (No input provided. Using default: '{human_answer}')")

    print("═" * 60 + "\n")

    # Resume the graph with the human's answer
    final_state = app.invoke(
        Command(resume={"human_answer": human_answer}),
        config=config,
    )
    return final_state
