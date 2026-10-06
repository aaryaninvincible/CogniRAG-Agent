"""
Unit tests for BM25, Dense Embeddings, Vector Store, and Hybrid Search.
"""

import pytest
import numpy as np
from cogni_rag.core.types import Document, Chunk
from cogni_rag.retrieval.bm25 import BM25Retriever
from cogni_rag.retrieval.embeddings import EmbeddingEngine
from cogni_rag.retrieval.vector_store import VectorStore
from cogni_rag.retrieval.hybrid_search import HybridRetriever


def test_embedding_engine():
    engine = EmbeddingEngine(dimension=128)
    vec1 = engine.embed_text("Kubernetes cluster autoscaling")
    vec2 = engine.embed_text("Kubernetes cluster pods scaling")
    vec3 = engine.embed_text("Cooking authentic Italian pasta recipe")

    # Unit norm check
    assert np.isclose(np.linalg.norm(vec1), 1.0, atol=1e-3)

    # Cosine similarities
    sim_related = np.dot(vec1, vec2)
    sim_unrelated = np.dot(vec1, vec3)

    assert sim_related > sim_unrelated


def test_bm25_retriever():
    bm25 = BM25Retriever()
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", doc_title="Storage", text="Hot NVMe SSD costs $0.08 per GB.", chunk_index=0, char_length=30),
        Chunk(chunk_id="c2", doc_id="d2", doc_title="Consensus", text="Raft leader heartbeat is 50 milliseconds.", chunk_index=0, char_length=40),
        Chunk(chunk_id="c3", doc_id="d3", doc_title="Compliance", text="Currency Transaction Reporting triggers at $10,000.", chunk_index=0, char_length=50)
    ]
    bm25.index(chunks)

    results = bm25.search("heartbeat leader interval", top_k=2)
    assert len(results) > 0
    assert results[0][0].chunk_id == "c2"


def test_hybrid_rrf_retrieval():
    engine = EmbeddingEngine(dimension=128)
    v_store = VectorStore(engine)
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(v_store, bm25)

    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", doc_title="Cloud", text="Hot tier NVMe SSD pricing is $0.08 per GB monthly.", chunk_index=0, char_length=50),
        Chunk(chunk_id="c2", doc_id="d2", doc_title="Consensus", text="Raft followers randomize election timeouts 150ms to 300ms.", chunk_index=0, char_length=60)
    ]
    hybrid.index_all(chunks)

    results = hybrid.search("What is NVMe SSD pricing?", top_k=1)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].score > 0.5
