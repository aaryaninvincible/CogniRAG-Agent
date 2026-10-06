"""
Semantic Dense Embedding Engine.
Generates L2-normalized dense embeddings for similarity search.
"""

import math
import hashlib
import numpy as np
from typing import List
from cogni_rag.core.config import settings


class EmbeddingEngine:
    def __init__(self, dimension: int = settings.VECTOR_DIM):
        self.dimension = dimension

    def _hash_token(self, token: str) -> int:
        """Deterministically hashes token to an index in [0, dimension-1]."""
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        return h % self.dimension

    def _token_weight(self, token: str) -> float:
        """Assigns heavier weight to informative technical tokens."""
        length = len(token)
        if length <= 2:
            return 0.2
        elif length <= 5:
            return 0.8
        else:
            return 1.4

    def embed_text(self, text: str) -> np.ndarray:
        """Generates dense L2-normalized vector for single text snippet."""
        clean = text.lower()
        words = [w.strip(".,;:!?\"'()[]{}") for w in clean.split() if w.strip()]
        
        vector = np.zeros(self.dimension, dtype=np.float32)
        if not words:
            return vector

        # 1. Word unigram projection
        for w in words:
            if not w:
                continue
            h_int = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            idx = h_int % self.dimension
            weight = self._token_weight(w)
            sign = 1.0 if ((h_int >> 8) % 2 == 0) else -1.0
            vector[idx] += weight * sign

        # 2. Bigram contextual projection
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            h_bi = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16)
            idx = h_bi % self.dimension
            sign = 1.0 if ((h_bi >> 8) % 2 == 0) else -1.0
            vector[idx] += 0.9 * sign

        # 3. L2 unit-norm normalization
        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            vector = vector / norm

        return vector

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch vector generation returning shape (len(texts), dimension)."""
        return np.array([self.embed_text(t) for t in texts], dtype=np.float32)
