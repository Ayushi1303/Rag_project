# =============================================================
#  retriever.py — Vector Store Loading & Retrieval
#
#  Responsibilities:
#    1. Load an existing ChromaDB from disk
#    2. Return a configured LangChain Retriever
#    3. Provide a helper to format retrieved chunks for prompts
# =============================================================

import os
from typing import List, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_DIR,
    TOP_K,
)


# ── Load vector store ─────────────────────────────────────────

def load_vectorstore(chroma_path: str = CHROMA_DIR) -> Chroma:
    """
    Open an existing ChromaDB from disk.

    IMPORTANT: The embedding model here MUST be the same model used
    during ingestion (text-embedding-ada-002). Mixing models will
    produce meaningless similarity scores.

    Args:
        chroma_path: Directory where ChromaDB files live.

    Returns:
        A Chroma VectorStore object.

    Raises:
        FileNotFoundError: If the ChromaDB directory does not exist.
        EnvironmentError:  If OPENAI_API_KEY is missing.
    """
    if not os.path.isdir(chroma_path):
        raise FileNotFoundError(
            f"[retriever] ChromaDB not found at: {chroma_path}\n"
            "  → Run 'python ingest.py' first to build the vector store."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings,
    )

    count = vectorstore._collection.count()
    if count == 0:
        raise ValueError(
            "[retriever] ChromaDB exists but is empty. "
            "Re-run 'python ingest.py' to populate it."
        )

    print(f"  [retriever] Loaded ChromaDB ({count} vectors) from: {chroma_path}")
    return vectorstore


# ── Build retriever ───────────────────────────────────────────

def get_retriever(vectorstore: Chroma, k: int = TOP_K) -> BaseRetriever:
    """
    Wrap the ChromaDB in a LangChain Retriever configured for
    cosine-similarity search with top-k results.

    Args:
        vectorstore: The Chroma VectorStore to wrap.
        k:           Number of chunks to return per query.

    Returns:
        A BaseRetriever that accepts a query string and returns
        a list of relevant Document objects.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever


# ── Format chunks for prompt ──────────────────────────────────

def format_context(docs: List[Document]) -> Tuple[str, List[dict]]:
    """
    Convert a list of retrieved Documents into:
      - A formatted context string to inject into the LLM prompt
      - A list of source metadata dicts for display to the user

    Args:
        docs: List of Document objects returned by the retriever.

    Returns:
        Tuple of (context_string, sources_list)
        context_string: Ready to paste into {context} in the prompt.
        sources_list:   [{"source": "file.pdf", "page": 3}, ...]
    """
    if not docs:
        return "No relevant context found.", []

    context_parts: List[str] = []
    sources: List[dict] = []

    for i, doc in enumerate(docs, start=1):
        # Build the context block for this chunk
        source_file = os.path.basename(doc.metadata.get("source", "unknown"))
        page_num    = doc.metadata.get("page", "?")

        context_parts.append(
            f"[Chunk {i} — {source_file}, Page {page_num}]\n"
            f"{doc.page_content.strip()}"
        )

        # Collect source info (deduplicate by source+page)
        source_entry = {"source": source_file, "page": page_num}
        if source_entry not in sources:
            sources.append(source_entry)

    # Join chunks with a clear separator so the LLM can distinguish them
    context_string = "\n\n---\n\n".join(context_parts)
    return context_string, sources


# ── Retrieve + format in one call ────────────────────────────

def retrieve(query: str, retriever: BaseRetriever) -> Tuple[str, List[dict], List[str]]:
    """
    Convenience wrapper: retrieve top-k docs and return everything
    the RAG node needs in a single call.

    Args:
        query:     The user's question (already cleaned by intake_node).
        retriever: The configured LangChain retriever.

    Returns:
        Tuple of (context_string, sources_list, raw_chunks_list)
        raw_chunks_list: Plain text list of each chunk (for state storage).
    """
    docs = retriever.invoke(query)
    context_string, sources = format_context(docs)
    raw_chunks = [doc.page_content.strip() for doc in docs]
    return context_string, sources, raw_chunks
