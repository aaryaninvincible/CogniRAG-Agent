"""
CLI script to demonstrate RAG failure modes, unmitigated behaviors, and engineered mitigations.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogni_rag.failure_cases.scenarios import FailureCaseScenarios


def main():
    scenarios = FailureCaseScenarios.get_all_scenarios()
    print("=" * 70)
    print("CogniRAG Agent: RAG & AI Agent Failure Modes & Mitigations")
    print("=" * 70)

    for i, s in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] {s['title'].upper()}")
        print(f"      Category: {s['category']} | Severity: {s['severity']}")
        print(f"\n      [Vulnerability]:")
        print(f"      {s['vulnerability']}")
        print(f"\n      [Unmitigated Behavior (Failure)]:")
        for line in s['unmitigated_behavior'].split("\n"):
            print(f"      {line}")
        print(f"\n      [Engineered Mitigation]:")
        for line in s['mitigation_strategy'].split("\n"):
            print(f"      {line}")
        print(f"\n      [Mitigated Result]:")
        print(f"      {s['mitigated_result']}")
        print("-" * 70)


if __name__ == "__main__":
    main()
