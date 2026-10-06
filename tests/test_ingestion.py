"""
Unit tests for DocumentLoader, RecursiveSemanticChunker, and IngestionPipeline.
"""

import pytest
from cogni_rag.core.types import Document
from cogni_rag.ingestion.loader import DocumentLoader
from cogni_rag.ingestion.chunker import RecursiveSemanticChunker
from cogni_rag.ingestion.pipeline import IngestionPipeline
from cogni_rag.retrieval.hybrid_search import HybridRetriever
from cogni_rag.retrieval.vector_store import VectorStore
from cogni_rag.retrieval.bm25 import BM25Retriever
from cogni_rag.retrieval.embeddings import EmbeddingEngine


def test_document_loader_from_text():
    doc = DocumentLoader.load_from_text(
        content="This is a test document with technical details.",
        title="Test Document",
        doc_id="test_001",
        metadata={"author": "engineer"}
    )
    assert doc.doc_id == "test_001"
    assert doc.title == "Test Document"
    assert len(doc.content_hash) == 64
    assert doc.metadata["author"] == "engineer"


def test_semantic_chunker():
    long_text = """# Header 1
First paragraph with some explanatory context about enterprise cloud deployments.

## Section A
Second paragraph describing storage performance and NVMe speeds in extensive detail.

## Section B
Third paragraph detailing network latency and edge caching policies for global CDNs.
"""
    doc = Document(doc_id="doc_chunk_test", title="Chunk Spec", content=long_text)
    chunker = RecursiveSemanticChunker(chunk_size=120, chunk_overlap=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    for c in chunks:
        assert c.doc_id == "doc_chunk_test"
        assert len(c.text) > 0
        assert c.chunk_id.startswith("doc_chunk_test_c")


def test_ingestion_pipeline_deduplication():
    emb = EmbeddingEngine()
    v_store = VectorStore(emb)
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(v_store, bm25)
    pipeline = IngestionPipeline(hybrid)

    doc1 = DocumentLoader.load_from_text("Unique knowledge content for testing.", title="Unique 1")
    doc2 = DocumentLoader.load_from_text("Unique knowledge content for testing.", title="Duplicate 1")

    res1 = pipeline.ingest_document(doc1)
    res2 = pipeline.ingest_document(doc2)

    assert res1["status"] == "success"
    assert res2["status"] == "skipped"
    assert pipeline.get_stats()["total_documents"] == 1
