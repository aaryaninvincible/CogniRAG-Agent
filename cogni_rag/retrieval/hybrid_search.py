"""
Hybrid Search Retriever combining Dense Cosine Vector Search and Sparse BM25 via Reciprocal Rank Fusion (RRF).
"""

from typing import List, Dict, Optional
from cogni_rag.core.types import Chunk, SearchResult
from cogni_rag.core.config import settings
from cogni_rag.retrieval.vector_store import VectorStore
from cogni_rag.retrieval.bm25 import BM25Retriever


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        rrf_k: int = settings.RRF_K
    ):
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k

    def index_all(self, chunks: List[Chunk]):
        """Indexes or re-indexes full collection across dense vector index and sparse BM25 index."""
        self.vector_store.clear()
        self.vector_store.add_chunks(chunks)
        self.bm25.index(chunks)

    def index_chunks(self, chunks: List[Chunk]):
        """Legacy helper matching index_all."""
        self.index_all(chunks)

    def search(
        self,
        query: str,
        top_k: int = settings.TOP_K_RETRIEVAL,
        filter_doc_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Executes hybrid retrieval:
        1. Dense Vector Cosine Search (Top 2 * k)
        2. Sparse BM25 Keyword Search (Top 2 * k)
        3. Reciprocal Rank Fusion (RRF) combination
        """
        candidate_k = max(top_k * 2, 10)
        dense_results = self.vector_store.search(query, top_k=candidate_k, filter_doc_id=filter_doc_id)
        sparse_results = self.bm25.search(query, top_k=candidate_k)

        # Map candidate chunks by ID
        chunk_map: Dict[str, Chunk] = {}
        dense_ranks: Dict[str, int] = {}
        sparse_ranks: Dict[str, int] = {}

        for chunk, score, rank in dense_results:
            chunk_map[chunk.chunk_id] = chunk
            dense_ranks[chunk.chunk_id] = rank

        for chunk, score, rank in sparse_results:
            chunk_map[chunk.chunk_id] = chunk
            sparse_ranks[chunk.chunk_id] = rank

        # Compute RRF score for all union candidates
        rrf_scores: Dict[str, float] = {}
        for chunk_id in chunk_map.keys():
            score = 0.0
            if chunk_id in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[chunk_id])
            if chunk_id in sparse_ranks:
                score += 1.0 / (self.rrf_k + sparse_ranks[chunk_id])
            rrf_scores[chunk_id] = score

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        # Scale scores to intuitive [0, 1] range
        max_possible_rrf = (1.0 / (self.rrf_k + 1)) * 2.0
        final_results = []

        for cid in sorted_ids[:top_k]:
            normalized_score = min(round(rrf_scores[cid] / max_possible_rrf, 4), 1.0)
            final_results.append(SearchResult(
                chunk=chunk_map[cid],
                score=normalized_score,
                retrieval_method="hybrid_rrf",
                dense_rank=dense_ranks.get(cid),
                sparse_rank=sparse_ranks.get(cid)
            ))

        return final_results
