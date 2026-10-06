"""
Recursive semantic chunker with hierarchy preservation and configurable sliding window overlap.
"""

import re
from typing import List
from cogni_rag.core.types import Document, Chunk
from cogni_rag.core.config import settings


class RecursiveSemanticChunker:
    def __init__(
        self,
        chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP,
        min_chunk_len: int = settings.MIN_CHUNK_LENGTH
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_len = min_chunk_len
        self.separators = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text using hierarchical separators."""
        final_chunks = []
        if not separators:
            # Fallback to character slicing
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                final_chunks.append(text[i:i + self.chunk_size])
            return final_chunks

        sep = separators[0]
        splits = text.split(sep)
        
        current_chunk = ""
        for s in splits:
            piece = s if sep.startswith("\n#") else (s + sep)
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                if current_chunk.strip():
                    final_chunks.append(current_chunk.strip())
                if len(piece) > self.chunk_size:
                    sub_chunks = self._split_text(piece, separators[1:])
                    final_chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = piece

        if current_chunk.strip():
            final_chunks.append(current_chunk.strip())

        return final_chunks

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Splits document content into structured Chunk entities with metadata."""
        raw_chunks = self._split_text(document.content, self.separators)
        chunks = []
        
        idx = 0
        for rc in raw_chunks:
            cleaned = rc.strip()
            if len(cleaned) < self.min_chunk_len:
                continue
                
            chunk_id = f"{document.doc_id}_c{idx:03d}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                doc_title=document.title,
                text=cleaned,
                chunk_index=idx,
                char_length=len(cleaned),
                metadata={
                    **document.metadata,
                    "doc_title": document.title,
                    "doc_id": document.doc_id
                }
            ))
            idx += 1
            
        return chunks
