"""
Benchmark Evaluation Runner for CogniRAG Agent.
Executes test cases, verifies retrieval ranks, checks tool calls, and outputs benchmark reports.
"""

import os
import json
import time
from typing import Dict, Any, List
from cogni_rag.agent.react_agent import ReActAgent
from cogni_rag.evaluation.metrics import (
    compute_hit_rate,
    compute_mrr,
    compute_tool_accuracy,
    compute_adversarial_block_rate
)


class BenchmarkEvaluator:
    def __init__(self, agent: ReActAgent, eval_dataset_path: str):
        self.agent = agent
        self.eval_dataset_path = eval_dataset_path

    def load_dataset(self) -> List[Dict[str, Any]]:
        with open(self.eval_dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_benchmark(self) -> Dict[str, Any]:
        """Executes complete evaluation suite and aggregates metrics."""
        test_cases = self.load_dataset()
        results = []
        latencies = []

        for case in test_cases:
            t0 = time.perf_counter()
            response = self.agent.query(case["query"])
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)

            # 1. Retrieval Hit & MRR check
            retrieval_hit = False
            reciprocal_rank = 0.0
            gold_kw = case.get("gold_keywords", [])

            if response.citations:
                for rank, citation in enumerate(response.citations, start=1):
                    txt_lower = citation.chunk.text.lower()
                    doc_id_match = (case.get("expected_doc") and case["expected_doc"] in citation.chunk.doc_id.lower())
                    kw_match = any(kw.lower() in txt_lower for kw in gold_kw)
                    if doc_id_match or kw_match:
                        retrieval_hit = True
                        if reciprocal_rank == 0.0:
                            reciprocal_rank = 1.0 / rank

            # 2. Tool call accuracy
            exp_tool = case.get("expected_tool")
            tool_correct = False
            if exp_tool is None:
                tool_correct = len(response.tools_used) == 0
            else:
                tool_correct = (exp_tool in response.tools_used)

            # 3. Adversarial check
            is_blocked = "SECURITY INTERCEPT" in response.answer

            # 4. Keyword presence in final answer
            answer_lower = response.answer.lower()
            keyword_coverage = 0.0
            if gold_kw:
                hits = sum(1 for kw in gold_kw if kw.lower() in answer_lower)
                keyword_coverage = round(hits / len(gold_kw), 2)

            results.append({
                "id": case["id"],
                "category": case["category"],
                "query": case["query"],
                "is_answerable": case.get("is_answerable", True),
                "expected_tool": exp_tool,
                "tools_used": response.tools_used,
                "tool_correct": tool_correct,
                "retrieval_hit": retrieval_hit,
                "reciprocal_rank": reciprocal_rank,
                "keyword_coverage": keyword_coverage,
                "faithfulness_score": response.faithfulness_score,
                "is_blocked": is_blocked,
                "latency_ms": round(latency, 2),
                "answer_preview": response.answer[:120] + "..." if len(response.answer) > 120 else response.answer
            })

        summary = {
            "total_test_cases": len(results),
            "hit_rate_at_k": compute_hit_rate(results),
            "mean_reciprocal_rank_mrr": compute_mrr(results),
            "tool_call_accuracy": compute_tool_accuracy(results),
            "adversarial_block_rate": compute_adversarial_block_rate(results),
            "average_latency_ms": round(sum(latencies) / len(latencies), 2),
            "details": results
        }

        return summary
