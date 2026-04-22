# =============================================================
#  prompt_builder.py — Prompt Templates & Construction
#
#  Keeps all prompts in one place so they're easy to iterate on.
#  The three rules enforced by the RAG prompt:
#    1. Answer ONLY from provided context  (grounding)
#    2. Admit uncertainty explicitly       (enables HITL routing)
#    3. Never fabricate facts              (anti-hallucination)
# =============================================================

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate


# ── Main RAG system prompt ────────────────────────────────────

RAG_SYSTEM_TEMPLATE = """\
You are a professional and friendly customer support assistant.

Your job is to answer the customer's question using ONLY the context \
provided below. The context comes from our official company documents.

STRICT RULES:
1. Use ONLY the information in the context. Do not use outside knowledge.
2. If the answer is not clearly present in the context, respond with \
exactly: "I don't have enough information to answer this question."
3. Do NOT make up, guess, or infer facts that are not explicitly stated.
4. Be concise, clear, and professional. Use plain language.
5. If the answer spans multiple points, use a short numbered list.

--- CONTEXT START ---
{context}
--- CONTEXT END ---
"""

RAG_HUMAN_TEMPLATE = "Customer question: {question}"


def build_rag_prompt() -> ChatPromptTemplate:
    """
    Build the LangChain ChatPromptTemplate for the RAG node.

    Usage:
        prompt = build_rag_prompt()
        messages = prompt.format_messages(context="...", question="...")
        response = llm.invoke(messages)

    Returns:
        A ChatPromptTemplate with {context} and {question} placeholders.
    """
    return ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_TEMPLATE),
        ("human",  RAG_HUMAN_TEMPLATE),
    ])


# ── Query reformulation prompt (used in intake_node) ──────────

REFORMULATION_SYSTEM = """\
You are a query pre-processor for a customer support search system.

Your task: rewrite the user's query to make it clearer and more \
searchable, while preserving the original intent exactly.

Rules:
- Fix spelling mistakes.
- Expand obvious abbreviations (e.g. "rtrn policy" → "return policy").
- If the query is already clear, return it unchanged.
- Return ONLY the rewritten query. No explanation. No punctuation changes.
"""

def build_reformulation_prompt() -> ChatPromptTemplate:
    """
    Prompt template for query reformulation in the intake node.
    This uses a cheap, fast call to improve retrieval quality.
    """
    return ChatPromptTemplate.from_messages([
        ("system", REFORMULATION_SYSTEM),
        ("human",  "{query}"),
    ])


# ── Format messages manually (alternative to template) ────────

def build_rag_messages(context: str, question: str) -> list:
    """
    Build the raw message list for direct LLM invocation.
    Alternative to using build_rag_prompt() if you want more control.

    Args:
        context:  The formatted context string from format_context().
        question: The user's question.

    Returns:
        List of [SystemMessage, HumanMessage] for ChatOpenAI.invoke().
    """
    system_content = RAG_SYSTEM_TEMPLATE.format(context=context)
    return [
        SystemMessage(content=system_content),
        HumanMessage(content=RAG_HUMAN_TEMPLATE.format(question=question)),
    ]
