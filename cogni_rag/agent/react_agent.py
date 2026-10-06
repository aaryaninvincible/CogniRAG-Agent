"""
ReAct Autonomous Agent Execution Engine.
Coordinates Thought -> Action -> Observation cycle, guardrail validation, and citation tracking.
"""

import time
from typing import List, Dict, Any, Optional
from cogni_rag.core.types import RAGResponse, AgentStep, SearchResult
from cogni_rag.core.config import settings
from cogni_rag.tools.registry import registry
from cogni_rag.retrieval.hybrid_search import HybridRetriever
from cogni_rag.agent.guardrails import GuardrailEngine
from cogni_rag.agent.planner import AgentPlanner


class ReActAgent:
    def __init__(
        self,
        retriever: HybridRetriever,
        max_steps: int = settings.MAX_AGENT_STEPS,
        enable_guardrails: bool = settings.ENABLE_GUARDRAILS
    ):
        self.retriever = retriever
        self.max_steps = max_steps
        self.enable_guardrails = enable_guardrails

    def query(self, user_query: str) -> RAGResponse:
        """Executes full ReAct reasoning trajectory against user query."""
        t_start = time.perf_counter()
        
        # 1. Pre-flight Guardrail: Adversarial Prompt Injection Defense
        if self.enable_guardrails:
            is_injected, alert_msg = GuardrailEngine.check_prompt_injection(user_query)
            if is_injected:
                return RAGResponse(
                    query=user_query,
                    answer=f"[SECURITY INTERCEPT] {alert_msg}",
                    reasoning_trace=[
                        AgentStep(
                            step_num=1,
                            thought="Analyzing input for adversarial prompt injection signatures.",
                            action=None,
                            observation=None
                        )
                    ],
                    citations=[],
                    tools_used=[],
                    faithfulness_score=0.0,
                    is_grounded=False,
                    total_latency_ms=round((time.perf_counter() - t_start) * 1000, 2)
                )

        steps: List[AgentStep] = []
        tools_invoked: List[str] = []
        citations: List[SearchResult] = []
        raw_observations: List[Dict[str, Any]] = []

        step_num = 1
        while step_num <= self.max_steps:
            # Plan next action
            action = AgentPlanner.plan_next_step(
                query=user_query,
                step_num=step_num,
                past_observations=raw_observations
            )

            if not action:
                # No further action needed, formulate answer
                break

            # Execute tool
            tools_invoked.append(action.tool_name)
            observation = registry.execute(action.tool_name, action.tool_input)

            # If tool was knowledge search, record retrieved citations
            if action.tool_name == "knowledge_search" and observation.success:
                q_text = action.tool_input.get("query", user_query)
                citations = self.retriever.search(q_text, top_k=action.tool_input.get("top_k", 4))

            raw_observations.append({
                "tool_name": action.tool_name,
                "input": action.tool_input,
                "output": observation.output,
                "success": observation.success
            })

            steps.append(AgentStep(
                step_num=step_num,
                thought=action.thought,
                action=action,
                observation=observation
            ))

            step_num += 1

        # Synthesize final grounded answer
        answer = AgentPlanner.synthesize_answer(
            query=user_query,
            observations=raw_observations,
            citations=citations
        )

        # Post-flight Guardrail: Grounding & Anti-Hallucination verification
        faithfulness = 1.0
        is_grounded = True
        if self.enable_guardrails and citations:
            faithfulness, is_grounded = GuardrailEngine.verify_grounding(answer, citations)

        total_latency = round((time.perf_counter() - t_start) * 1000, 2)

        return RAGResponse(
            query=user_query,
            answer=answer,
            reasoning_trace=steps,
            citations=citations,
            tools_used=tools_invoked,
            faithfulness_score=faithfulness,
            is_grounded=is_grounded,
            total_latency_ms=total_latency
        )
