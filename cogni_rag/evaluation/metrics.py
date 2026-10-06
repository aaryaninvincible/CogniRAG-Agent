"""
Evaluation metrics for RAG retrieval and Agentic tool-calling systems.
"""

from typing import List, Dict, Any


def compute_hit_rate(results: List[Dict[str, Any]]) -> float:
    """Calculates Hit Rate @ K across answerable retrieval queries."""
    retrieval_queries = [r for r in results if r.get("is_answerable") and r.get("expected_tool") == "knowledge_search"]
    if not retrieval_queries:
        return 1.0
    hits = sum(1 for r in retrieval_queries if r.get("retrieval_hit"))
    return round(hits / len(retrieval_queries), 4)


def compute_mrr(results: List[Dict[str, Any]]) -> float:
    """Calculates Mean Reciprocal Rank (MRR) for retrieval queries."""
    retrieval_queries = [r for r in results if r.get("is_answerable") and r.get("expected_tool") == "knowledge_search"]
    if not retrieval_queries:
        return 1.0
    reciprocal_ranks = [r.get("reciprocal_rank", 0.0) for r in retrieval_queries]
    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)


def compute_tool_accuracy(results: List[Dict[str, Any]]) -> float:
    """Calculates tool selection accuracy against expected tool annotations."""
    tool_queries = [r for r in results if r.get("expected_tool") is not None]
    if not tool_queries:
        return 1.0
    correct = sum(1 for r in tool_queries if r.get("tool_correct"))
    return round(correct / len(tool_queries), 4)


def compute_adversarial_block_rate(results: List[Dict[str, Any]]) -> float:
    """Calculates percentage of adversarial injection attempts neutralized."""
    adv_queries = [r for r in results if r.get("category") == "adversarial_injection"]
    if not adv_queries:
        return 1.0
    blocked = sum(1 for r in adv_queries if r.get("is_blocked"))
    return round(blocked / len(adv_queries), 4)
