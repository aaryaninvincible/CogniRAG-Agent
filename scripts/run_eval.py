"""
CLI script to execute the golden evaluation benchmark on CogniRAG Agent.
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogni_rag.core.config import settings
from cogni_rag.retrieval.embeddings import EmbeddingEngine
from cogni_rag.retrieval.vector_store import VectorStore
from cogni_rag.retrieval.bm25 import BM25Retriever
from cogni_rag.retrieval.hybrid_search import HybridRetriever
from cogni_rag.ingestion.pipeline import IngestionPipeline
from cogni_rag.tools.builtins import register_builtin_tools
from cogni_rag.agent.react_agent import ReActAgent
from cogni_rag.evaluation.evaluator import BenchmarkEvaluator


def main():
    print("=" * 65)
    print("CogniRAG Agent: Running Golden Evaluation Benchmark")
    print("=" * 65)

    # 1. Initialize Pipeline & Ingest Sample Docs
    emb = EmbeddingEngine()
    v_store = VectorStore(embedding_engine=emb)
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(vector_store=v_store, bm25_retriever=bm25)
    ingestion = IngestionPipeline(retriever=hybrid)
    register_builtin_tools(hybrid)

    docs_dir = settings.DATA_DIR
    print(f"[*] Ingesting enterprise knowledge corpus from: {docs_dir}")
    ingestion.ingest_directory(docs_dir)
    stats = ingestion.get_stats()
    print(f"    Indexed {stats['total_documents']} documents ({stats['total_chunks']} chunks).")

    # 2. Run Benchmark
    agent = ReActAgent(retriever=hybrid)
    eval_file = os.path.join(settings.EVAL_DIR, "golden_eval_set.json")
    evaluator = BenchmarkEvaluator(agent=agent, eval_dataset_path=eval_file)

    print("\n[*] Evaluating 15 Golden Test Cases across factual, math, database & adversarial categories...")
    results = evaluator.run_benchmark()

    # 3. Print Results Summary
    print("\n" + "=" * 65)
    print("EVALUATION BENCHMARK SCORECARD")
    print("=" * 65)
    print(f"Total Test Cases:           {results['total_test_cases']}")
    print(f"Hit Rate @ K:               {results['hit_rate_at_k']:.1%}")
    print(f"Mean Reciprocal Rank (MRR): {results['mean_reciprocal_rank_mrr']:.4f}")
    print(f"Tool Selection Accuracy:    {results['tool_call_accuracy']:.1%}")
    print(f"Adversarial Defense Rate:   {results['adversarial_block_rate']:.1%}")
    print(f"Average Latency:            {results['average_latency_ms']:.2f} ms")
    print("=" * 65)

    # Save to disk
    out_file = os.path.join(settings.EVAL_DIR, "benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Detailed report saved to: {out_file}")


if __name__ == "__main__":
    main()
