"""
FastAPI application server for CogniRAG Agent.
Handles document ingestion, hybrid search, ReAct chat execution, evaluation benchmarks, and failure diagnostics.
"""

import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from cogni_rag.core.config import settings
from cogni_rag.core.types import RAGResponse, SearchResult
from cogni_rag.retrieval.embeddings import EmbeddingEngine
from cogni_rag.retrieval.vector_store import VectorStore
from cogni_rag.retrieval.bm25 import BM25Retriever
from cogni_rag.retrieval.hybrid_search import HybridRetriever
from cogni_rag.ingestion.pipeline import IngestionPipeline
from cogni_rag.tools.registry import registry
from cogni_rag.tools.builtins import register_builtin_tools
from cogni_rag.agent.react_agent import ReActAgent
from cogni_rag.evaluation.evaluator import BenchmarkEvaluator
from cogni_rag.failure_cases.scenarios import FailureCaseScenarios
from cogni_rag.api.schemas import (
    IngestTextRequest,
    IngestResponse,
    SearchRequest,
    ChatRequest,
    HealthResponse
)

# Initialize core services
embedding_engine = EmbeddingEngine()
vector_store = VectorStore(embedding_engine=embedding_engine)
bm25_retriever = BM25Retriever()
hybrid_retriever = HybridRetriever(vector_store=vector_store, bm25_retriever=bm25_retriever)
ingestion_pipeline = IngestionPipeline(retriever=hybrid_retriever)

# Register tools
register_builtin_tools(hybrid_retriever)

# Initialize agent
agent = ReActAgent(retriever=hybrid_retriever)

# Pre-populate default corpus
if os.path.exists(settings.DATA_DIR):
    ingestion_pipeline.ingest_directory(settings.DATA_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    stats = ingestion_pipeline.get_stats()
    print(f"[*] CogniRAG Agent online with {stats['total_documents']} documents ({stats['total_chunks']} chunks).")
    yield
    print("[*] Shutting down CogniRAG service.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production Agentic RAG system with Hybrid Search, Tool Calling, Evaluation, and Failure Diagnostics.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and Asset Mounts
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
assets_dir = os.path.join(settings.BASE_DIR, "docs", "assets")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

if os.path.exists(assets_dir):
    app.mount("/docs/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/", include_in_schema=False)
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "CogniRAG Agent API online", "docs": "/docs"}


@app.get("/api/v1/health", response_model=HealthResponse)
def get_health():
    stats = ingestion_pipeline.get_stats()
    tools = registry.get_definitions()
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        total_documents=stats["total_documents"],
        total_chunks=stats["total_chunks"],
        tools_registered=len(tools)
    )


@app.post("/api/v1/ingest", response_model=IngestResponse)
def ingest_text_document(req: IngestTextRequest):
    res = ingestion_pipeline.ingest_text(
        title=req.title,
        content=req.content,
        metadata=req.metadata
    )
    if res["status"] == "skipped":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document already exists (duplicate hash detected)."
        )
    return IngestResponse(**res)


@app.post("/api/v1/search", response_model=List[SearchResult])
def execute_hybrid_search(req: SearchRequest):
    return hybrid_retriever.search(query=req.query, top_k=req.top_k)


@app.post("/api/v1/chat", response_model=RAGResponse)
def execute_agent_chat(req: ChatRequest):
    agent.enable_guardrails = req.enable_guardrails
    return agent.query(req.query)


@app.get("/api/v1/tools")
def list_available_tools():
    return registry.get_definitions()


@app.post("/api/v1/evaluate")
def run_evaluation_benchmark():
    eval_file = os.path.join(settings.EVAL_DIR, "golden_eval_set.json")
    if not os.path.exists(eval_file):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation dataset file not found."
        )
    evaluator = BenchmarkEvaluator(agent=agent, eval_dataset_path=eval_file)
    return evaluator.run_benchmark()


@app.get("/api/v1/failures")
def get_failure_cases():
    return FailureCaseScenarios.get_all_scenarios()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cogni_rag.api.app:app", host=settings.HOST, port=settings.PORT, reload=True)
