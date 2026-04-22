# =============================================================
#  tests/test_ingest.py — Unit Tests for Ingestion Pipeline
#
#  Run with:  pytest tests/ -v
#
#  Tests the load/chunk logic without API calls.
#  Uses a small synthetic Document list to avoid needing a real PDF.
# =============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock
from langchain.schema import Document

from ingest import load_pdf, chunk_documents


# ── Tests: load_pdf ───────────────────────────────────────────

class TestLoadPdf:

    def test_raises_if_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            load_pdf("/nonexistent/path/file.pdf")

    def test_raises_if_not_pdf_extension(self, tmp_path):
        # Create a real file with wrong extension
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("some text")
        with pytest.raises(ValueError, match="Expected a .pdf file"):
            load_pdf(str(txt_file))

    def test_raises_if_pdf_produces_no_content(self, tmp_path):
        # Create a dummy file with .pdf extension
        pdf_file = tmp_path / "empty.pdf"
        pdf_file.write_bytes(b"")
        with patch("ingest.PyPDFLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.load.return_value = []   # Simulates empty/unreadable PDF
            mock_loader_class.return_value = mock_loader
            with pytest.raises(ValueError, match="no content"):
                load_pdf(str(pdf_file))

    def test_returns_documents_on_success(self, tmp_path):
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")  # dummy content
        fake_docs = [
            Document(page_content="Page 1 content", metadata={"source": str(pdf_file), "page": 0}),
            Document(page_content="Page 2 content", metadata={"source": str(pdf_file), "page": 1}),
        ]
        with patch("ingest.PyPDFLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.load.return_value = fake_docs
            mock_loader_class.return_value = mock_loader
            result = load_pdf(str(pdf_file))
        assert len(result) == 2
        assert result[0].page_content == "Page 1 content"


# ── Tests: chunk_documents ────────────────────────────────────

class TestChunkDocuments:

    def _make_docs(self, num_pages: int = 2, page_length: int = 1000) -> list:
        """Helper: create synthetic Document objects."""
        return [
            Document(
                page_content="word " * (page_length // 5),
                metadata={"source": "test.pdf", "page": i},
            )
            for i in range(num_pages)
        ]

    def test_produces_chunks(self):
        docs = self._make_docs(num_pages=2, page_length=1000)
        chunks = chunk_documents(docs)
        # 2 pages × 1000 chars each with chunk_size=500 → expect 4+ chunks
        assert len(chunks) >= 4

    def test_chunks_are_documents(self):
        docs = self._make_docs()
        chunks = chunk_documents(docs)
        assert all(isinstance(c, Document) for c in chunks)

    def test_chunks_have_source_metadata(self):
        docs = self._make_docs()
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert "source" in chunk.metadata

    def test_chunk_content_not_empty(self):
        docs = self._make_docs()
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert len(chunk.page_content.strip()) > 0

    def test_single_short_doc_produces_one_chunk(self):
        short_doc = [Document(
            page_content="This is a short document with less than 500 chars.",
            metadata={"source": "short.pdf", "page": 0},
        )]
        chunks = chunk_documents(short_doc)
        assert len(chunks) == 1
        assert "short document" in chunks[0].page_content
