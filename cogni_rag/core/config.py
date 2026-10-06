"""
Configuration settings for CogniRAG Agent system.
"""

import os
from pydantic import BaseModel, Field


class RAGConfig(BaseModel):
    # System Metadata
    PROJECT_NAME: str = "CogniRAG Agent"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Ingestion & Chunking
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 80
    MIN_CHUNK_LENGTH: int = 40
    
    # Retrieval
    TOP_K_RETRIEVAL: int = 4
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    RRF_K: int = 60
    VECTOR_DIM: int = 128
    MIN_SIMILARITY_SCORE: float = 0.15
    
    # Agent Execution
    MAX_AGENT_STEPS: int = 5
    ENABLE_GUARDRAILS: bool = True
    ANTI_HALLUCINATION_THRESHOLD: float = 0.40
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    STORAGE_DIR: str = os.path.join(BASE_DIR, "storage")
    DATA_DIR: str = os.path.join(BASE_DIR, "sample_data", "docs")
    EVAL_DIR: str = os.path.join(BASE_DIR, "evaluation")
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", "8080"))


settings = RAGConfig()
