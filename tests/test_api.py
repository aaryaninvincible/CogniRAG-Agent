"""
Integration tests for FastAPI endpoints: /health, /tools, /ingest, /search, /chat, /failures, /evaluate.
"""

import pytest
from fastapi.testclient import TestClient
from cogni_rag.api.app import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert data["tools_registered"] >= 4


def test_tools_endpoint():
    res = client.get("/api/v1/tools")
    assert res.status_code == 200
    tools = res.json()
    assert len(tools) >= 4
    tool_names = [t["name"] for t in tools]
    assert "knowledge_search" in tool_names
    assert "math_calculator" in tool_names


def test_search_endpoint():
    res = client.post("/api/v1/search", json={"query": "RTO recovery failover SLA", "top_k": 2})
    assert res.status_code == 200
    results = res.json()
    assert len(results) > 0
    assert "score" in results[0]


def test_chat_endpoint():
    res = client.post("/api/v1/chat", json={"query": "What is the RTO failover SLA for databases?"})
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "reasoning_trace" in data
    assert len(data["reasoning_trace"]) > 0


def test_failures_endpoint():
    res = client.get("/api/v1/failures")
    assert res.status_code == 200
    scenarios = res.json()
    assert len(scenarios) == 5
    assert scenarios[0]["id"] == "failure_01_chunk_boundary"


def test_evaluate_endpoint():
    res = client.post("/api/v1/evaluate")
    assert res.status_code == 200
    report = res.json()
    assert report["total_test_cases"] == 15
    assert report["hit_rate_at_k"] >= 0.80
