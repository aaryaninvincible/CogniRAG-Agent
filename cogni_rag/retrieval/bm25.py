"""
Okapi BM25 sparse keyword retrieval engine with configurable k1 and b hyper-parameters.
"""

import math
import re
from typing import List, Dict, Tuple
from cogni_rag.core.types import Chunk, SearchResult
from cogni_rag.core.config import settings


class BM25Retriever:
    def __init__(self, k1: float = settings.BM25_K1, b: float = settings.BM25_B):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.doc_len: List[int] = []
        self.avg_dl: float = 0.0
        self.doc_freq: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.term_freqs: List[Dict[str, int]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple clean alphanumeric tokenization."""
        return re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())

    def index(self, chunks: List[Chunk]):
        """Builds BM25 inverted index and calculates IDF values."""
        self.chunks = chunks
        self.doc_len = []
        self.term_freqs = []
        self.doc_freq = {}
        
        N = len(chunks)
        if N == 0:
            return

        total_words = 0
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            self.doc_len.append(len(tokens))
            total_words += len(tokens)
            
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)

            for t in tf.keys():
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

        self.avg_dl = total_words / max(N, 1)

        # Compute IDF
        self.idf = {}
        for term, df in self.doc_freq.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[term] = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float, int]]:
        """Searches index using BM25 and returns (chunk, score, rank)."""
        tokens = self._tokenize(query)
        if not tokens or not self.chunks:
            return []

        scores: List[float] = [0.0] * len(self.chunks)

        for i, tf_dict in enumerate(self.term_freqs):
            dl = self.doc_len[i]
            score = 0.0
            for t in tokens:
                if t in tf_dict:
                    freq = tf_dict[t]
                    idf = self.idf.get(t, 0.0)
                    denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
                    score += idf * (freq * (self.k1 + 1.0)) / denom
            scores[i] = score

        # Rank candidates
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        results = []
        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            if scores[idx] > 0.0:
                results.append((self.chunks[idx], scores[idx], rank))
                
        return results
