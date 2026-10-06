"""
Pydantic data models and type contracts for CogniRAG Agent.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class Document(BaseModel):
    doc_id: str
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content_hash: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    chunk_index: int
    char_length: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    retrieval_method: str = "hybrid_rrf"
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class AgentAction(BaseModel):
    tool_name: str
    tool_input: Dict[str, Any]
    thought: str


class AgentObservation(BaseModel):
    tool_name: str
    output: Any
    success: bool
    execution_time_ms: float


class AgentStep(BaseModel):
    step_num: int
    thought: str
    action: Optional[AgentAction] = None
    observation: Optional[AgentObservation] = None


class RAGResponse(BaseModel):
    query: str
    answer: str
    reasoning_trace: List[AgentStep] = Field(default_factory=list)
    citations: List[SearchResult] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    faithfulness_score: float = Field(default=1.0, description="Verification grounding score [0, 1]")
    is_grounded: bool = True
    total_latency_ms: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
