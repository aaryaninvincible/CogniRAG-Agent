"""
REST API request and response schemas for CogniRAG Agent service.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from cogni_rag.core.types import RAGResponse, SearchResult, ToolDefinition


class IngestTextRequest(BaseModel):
    title: str = Field(..., examples=["Payment Gateway Integration Guide"])
    content: str = Field(..., examples=["To initialize the webhook listener, configure secret HMAC-SHA256..."])
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    status: str
    doc_id: str
    title: str
    chunks_created: int
    char_count: int


class SearchRequest(BaseModel):
    query: str = Field(..., examples=["What is the storage cost of Hot NVMe SSD?"])
    top_k: int = Field(default=4, ge=1, le=20)


class ChatRequest(BaseModel):
    query: str = Field(..., examples=["What is the RTO failover SLA for regional databases?"])
    enable_guardrails: bool = True


class HealthResponse(BaseModel):
    status: str
    version: str
    total_documents: int
    total_chunks: int
    tools_registered: int
