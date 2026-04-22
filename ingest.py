# =============================================================
#  ingest.py — Document Ingestion Pipeline
#
#  Responsibilities:
#    1. Load PDF file(s) from disk
#    2. Split into overlapping chunks
#    3. Embed each chunk via OpenAI ada-002
#    4. Persist embeddings in ChromaDB
#
#  Run directly:  python ingest.py
#  Or call:       ingest_documents("data/my_file.pdf")
# =============================================================

import os
import glob
import sys
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    DATA_DIR,
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# ── 1. Load PDF ───────────────────────────────────────────────

def load_pdf(path: str) -> List[Document]:
    """
    Load a single PDF file and return a list of LangChain Document objects.
    Each Document corresponds to one page of the PDF.

    Args:
        path: Absolute or relative path to the PDF file.

    Returns:
        List of Document objects with page_content and metadata.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the PDF produces no extractable text.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ingest] PDF not found: {path}")
    if not path.lower().endswith(".pdf"):
        raise ValueError(f"[ingest] Expected a .pdf file, got: {path}")

    print(f"  [load]   Loading: {path}")
    loader = PyPDFLoader(path)
    docs = loader.load()

    if not docs:
        raise ValueError(f"[ingest] PDF produced no content: {path}")

    print(f"  [load]   Pages loaded: {len(docs)}")
    return docs


def load_all_pdfs(directory: str) -> List[Document]:
    """
    Load every PDF file found in a directory.

    Args:
        directory: Path to the folder containing PDF files.

    Returns:
        Combined list of Document objects from all PDFs.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no PDF files are found.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"[ingest] Data directory not found: {directory}")

    pdf_paths = glob.glob(os.path.join(directory, "*.pdf"))
    if not pdf_paths:
        raise ValueError(f"[ingest] No PDF files found in: {directory}")

    all_docs: List[Document] = []
    for pdf_path in sorted(pdf_paths):
        all_docs.extend(load_pdf(pdf_path))

    print(f"  [load]   Total pages across all PDFs: {len(all_docs)}")
    return all_docs


# ── 2. Chunk ──────────────────────────────────────────────────

def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split a list of Documents into smaller, overlapping chunks.

    Uses RecursiveCharacterTextSplitter which tries to split on:
    paragraph breaks → line breaks → spaces → individual characters.
    This preserves as much natural language structure as possible.

    Args:
        docs: List of full-page Document objects.

    Returns:
        List of chunk-level Document objects. Each chunk retains
        the original metadata (source, page) plus 'start_index'.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,    # stores char offset in metadata
    )

    chunks = splitter.split_documents(docs)
    print(f"  [chunk]  Total chunks created: {len(chunks)}")
    return chunks


# ── 3 & 4. Embed + Store ─────────────────────────────────────

def embed_and_store(chunks: List[Document], chroma_path: str) -> Chroma:
    """
    Compute embeddings for each chunk and store them in ChromaDB.

    Uses OpenAI text-embedding-ada-002 to produce 1536-dimensional
    dense vectors. ChromaDB persists the vectors to disk at chroma_path
    so they survive process restarts.

    Args:
        chunks:      List of chunk Documents to embed.
        chroma_path: Directory where ChromaDB will write its files.

    Returns:
        A Chroma VectorStore object (ready to query immediately).
    """
    print(f"  [embed]  Embedding {len(chunks)} chunks with HuggingFace model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Chroma.from_documents embeds + stores in one call
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path,
    )

    print(f"  [store]  ChromaDB saved to: {chroma_path}")
    print(f"  [store]  Total vectors stored: {vectorstore._collection.count()}")
    return vectorstore


# ── Public entry point ────────────────────────────────────────

def ingest_documents(source: str = DATA_DIR, chroma_path: str = CHROMA_DIR) -> Chroma:
    """
    Full ingestion pipeline: load → chunk → embed → store.

    Args:
        source:      Path to a single PDF file OR a directory of PDFs.
        chroma_path: Where to persist ChromaDB.

    Returns:
        A ready-to-use Chroma VectorStore.
    """
    print("\n" + "=" * 60)
    print("  RAG INGESTION PIPELINE")
    print("=" * 60)

    # Load
    if os.path.isfile(source):
        docs = load_pdf(source)
    else:
        docs = load_all_pdfs(source)

    # Chunk
    chunks = chunk_documents(docs)

    # Embed & Store
    vectorstore = embed_and_store(chunks, chroma_path)

    print("=" * 60)
    print("  ✓ Ingestion complete. You can now run main.py")
    print("=" * 60 + "\n")
    return vectorstore


# ── CLI entry point ───────────────────────────────────────────

if __name__ == "__main__":
    # Usage:
    #   python ingest.py                          → ingests all PDFs in data/
    #   python ingest.py data/my_document.pdf     → ingests a specific file

    source_path = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR

    try:
        ingest_documents(source=source_path)
    except (FileNotFoundError, ValueError, EnvironmentError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
