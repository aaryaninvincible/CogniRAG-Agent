"""
In-memory & persistent Vector Store for dense cosine similarity indexing.
"""

import os
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from cogni_rag.core.types import Chunk
from cogni_rag.retrieval.embeddings import EmbeddingEngine


class VectorStore:
    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None):
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.chunks: List[Chunk] = []
        self.vectors: Optional[np.ndarray] = None

    def add_chunks(self, chunks: List[Chunk]):
        """Embeds and indexes new chunks into the vector store."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        new_vecs = self.embedding_engine.embed_batch(texts)

        if self.vectors is None or len(self.chunks) == 0:
            self.vectors = new_vecs
            self.chunks = list(chunks)
        else:
            self.vectors = np.vstack([self.vectors, new_vecs])
            self.chunks.extend(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_doc_id: Optional[str] = None
    ) -> List[Tuple[Chunk, float, int]]:
        """
        Executes cosine similarity search against query vector.
        Returns list of (Chunk, similarity_score, rank).
        """
        if self.vectors is None or len(self.chunks) == 0:
            return []

        q_vec = self.embedding_engine.embed_text(query)
        # Cosine similarity (vectors are unit normalized)
        sims = np.dot(self.vectors, q_vec)

        # Apply metadata filter if requested
        valid_indices = []
        for i, c in enumerate(self.chunks):
            if filter_doc_id and c.doc_id != filter_doc_id:
                continue
            valid_indices.append(i)

        if not valid_indices:
            return []

        filtered_sims = [(i, float(sims[i])) for i in valid_indices]
        filtered_sims.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(filtered_sims[:top_k], start=1):
            results.append((self.chunks[idx], score, rank))

        return results

    def clear(self):
        """Empties the store."""
        self.chunks = []
        self.vectors = None

    def count(self) -> int:
        return len(self.chunks)
