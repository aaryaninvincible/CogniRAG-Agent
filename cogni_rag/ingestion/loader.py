"""
Multi-format document loader and parser supporting Markdown, Text, Code, and JSON files.
"""

import os
import hashlib
from typing import List, Optional, Dict, Any
from cogni_rag.core.types import Document


class DocumentLoader:
    @staticmethod
    def compute_hash(text: str) -> str:
        """Computes SHA-256 hash for document content deduplication."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def load_from_text(
        cls,
        content: str,
        title: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Creates a Document from raw string content."""
        clean_content = content.strip()
        h = cls.compute_hash(clean_content)
        d_id = doc_id or f"doc_{h[:8]}"
        return Document(
            doc_id=d_id,
            title=title,
            content=clean_content,
            metadata=metadata or {},
            content_hash=h
        )

    @classmethod
    def load_from_file(cls, file_path: str) -> Document:
        """Reads a local file and extracts title and contents."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        basename = os.path.basename(file_path)
        title, ext = os.path.splitext(basename)
        title = title.replace("_", " ").title()

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        h = cls.compute_hash(content)
        doc_id = f"{os.path.splitext(basename)[0]}_{h[:6]}"

        return Document(
            doc_id=doc_id,
            title=title,
            content=content,
            metadata={
                "source_file": basename,
                "file_extension": ext,
                "file_size_bytes": len(content.encode("utf-8"))
            },
            content_hash=h
        )

    @classmethod
    def load_directory(cls, dir_path: str) -> List[Document]:
        """Loads all supported text documents from a directory."""
        supported_exts = {".md", ".txt", ".json", ".py", ".sql"}
        docs = []
        if not os.path.exists(dir_path):
            return docs

        for root, _, files in os.walk(dir_path):
            for file in sorted(files):
                _, ext = os.path.splitext(file)
                if ext.lower() in supported_exts:
                    full_p = os.path.join(root, file)
                    docs.append(cls.load_from_file(full_p))

        return docs
