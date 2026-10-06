"""
Ingestion Pipeline Coordinator.
Manages deduplication, chunking, and indexation of document collections.
"""

from typing import List, Dict, Any, Optional
from cogni_rag.core.types import Document, Chunk
from cogni_rag.ingestion.loader import DocumentLoader
from cogni_rag.ingestion.chunker import RecursiveSemanticChunker
from cogni_rag.retrieval.hybrid_search import HybridRetriever


class IngestionPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        chunker: Optional[RecursiveSemanticChunker] = None
    ):
        self.retriever = retriever
        self.chunker = chunker or RecursiveSemanticChunker()
        self.documents: Dict[str, Document] = {}
        self.document_hashes: set = set()
        self.all_chunks: List[Chunk] = []

    def ingest_document(self, document: Document) -> Dict[str, Any]:
        """Ingests a single document with deduplication check and indexing."""
        if document.content_hash in self.document_hashes:
            return {
                "status": "skipped",
                "reason": "Duplicate content detected (SHA-256 match)",
                "doc_id": document.doc_id,
                "title": document.title,
                "chunks_created": 0
            }

        # Chunk document
        chunks = self.chunker.chunk_document(document)
        self.all_chunks.extend(chunks)
        self.retriever.index_all(self.all_chunks)

        self.documents[document.doc_id] = document
        self.document_hashes.add(document.content_hash)

        return {
            "status": "success",
            "doc_id": document.doc_id,
            "title": document.title,
            "chunks_created": len(chunks),
            "char_count": len(document.content)
        }

    def ingest_text(self, title: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ingests raw text string directly."""
        doc = DocumentLoader.load_from_text(text, title=title, metadata=metadata)
        return self.ingest_document(doc)

    def ingest_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """Ingests all documents from a directory."""
        docs = DocumentLoader.load_directory(dir_path)
        results = []
        for doc in docs:
            res = self.ingest_document(doc)
            results.append(res)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on ingested corpus."""
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.all_chunks),
            "unique_hashes": len(self.document_hashes)
        }
