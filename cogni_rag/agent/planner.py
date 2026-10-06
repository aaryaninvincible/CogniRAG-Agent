"""
Autonomous Agent Planner and Decision Dispatcher.
Parses query intent, resolves tool selection, and synthesizes grounded answers.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from cogni_rag.core.types import AgentAction, SearchResult


class AgentPlanner:
    @staticmethod
    def plan_next_step(
        query: str,
        step_num: int,
        past_observations: List[Dict[str, Any]]
    ) -> Optional[AgentAction]:
        """Determines the next action in the ReAct reasoning trajectory."""
        q_lower = query.lower()

        # Step 1: Analyze initial intent
        if step_num == 1:
            # Date/Time query
            if any(k in q_lower for k in ["what time is it", "current time", "what day is today", "what year is this", "today's date", "current date"]):
                return AgentAction(
                    thought="User is asking for the current real-world timestamp or date. I should invoke the datetime_service.",
                    tool_name="datetime_service",
                    tool_input={}
                )

            # Database Lookup
            usr_match = re.search(r"\b(usr-\d{3})\b", query, re.IGNORECASE)
            if usr_match:
                usr_id = usr_match.group(1).upper()
                return AgentAction(
                    thought=f"Query references structured entity '{usr_id}'. I will query the users database table.",
                    tool_name="database_lookup",
                    tool_input={"table": "users", "record_id": usr_id}
                )

            audit_match = re.search(r"\b(audit-\d{4}-[a-z])\b", query, re.IGNORECASE)
            if audit_match:
                audit_id = audit_match.group(1).upper()
                return AgentAction(
                    thought=f"Query references audit record '{audit_id}'. I will query the compliance_records database table.",
                    tool_name="database_lookup",
                    tool_input={"table": "compliance_records", "record_id": audit_id}
                )

            # Pure Math query
            math_expr = re.search(r"(\d+[\s\+\-\*\/\%\^\(\)\.]+\d+)", query)
            if math_expr and any(word in q_lower for word in ["calculate", "compute", "what is", "solve", "math"]):
                expr = math_expr.group(1).replace("^", "**")
                return AgentAction(
                    thought=f"Query contains mathematical formula '{expr}'. I will invoke math_calculator.",
                    tool_name="math_calculator",
                    tool_input={"expression": expr}
                )

            # Default: Knowledge Base Search
            search_query = re.sub(r"^(what is|how do i|tell me about|explain|find|search for)\s+", "", q_lower, flags=re.IGNORECASE)
            return AgentAction(
                thought=f"Query requires factual domain knowledge. I will execute hybrid knowledge_search with query '{search_query}'.",
                tool_name="knowledge_search",
                tool_input={"query": search_query.strip(), "top_k": 4}
            )

        # Step 2: Handle Multi-step chaining (e.g. retrieved numeric data that now needs calculation)
        if step_num == 2 and past_observations:
            last_obs = past_observations[-1]
            # If we just did a search and user asked to calculate something with it
            if last_obs.get("tool_name") == "knowledge_search" and any(w in q_lower for w in ["calculate", "compute", "how much", "total"]):
                # Look for numbers in retrieved text
                raw_chunks = last_obs.get("output", {}).get("chunks", [])
                combined_txt = " ".join([c.get("text_snippet", "") for c in raw_chunks])
                nums = re.findall(r"\$?\b\d+(?:\.\d+)?\b", combined_txt)
                
                # Check for explicit formula in query
                m = re.search(r"(\d+[\s\+\-\*\/\%\^\(\)\.]+\d+)", query)
                if m:
                    expr = m.group(1).replace("^", "**")
                    return AgentAction(
                        thought=f"Using retrieved domain facts, I now need to compute formula '{expr}'.",
                        tool_name="math_calculator",
                        tool_input={"expression": expr}
                    )

        # No further tool invocation needed
        return None

    @staticmethod
    def synthesize_answer(
        query: str,
        observations: List[Dict[str, Any]],
        citations: List[SearchResult]
    ) -> str:
        """Synthesizes the final answer using tool observations and retrieved passages."""
        if not observations:
            return "Unable to determine an answer: No search operations or tools were executed."

        # Check for error observations
        for obs in observations:
            if not obs.get("success"):
                return f"Agent encountered an execution failure while invoking '{obs.get('tool_name')}': {obs.get('output')}"

        last_obs = observations[-1]
        tool_name = last_obs.get("tool_name")
        output = last_obs.get("output")

        # 1. DateTime tool
        if tool_name == "datetime_service" and isinstance(output, dict):
            return f"Authoritative System Time: Current UTC time is {output.get('time_utc')} on {output.get('day_of_week')}, {output.get('current_utc_iso')}."

        # 2. Math Calculator
        if tool_name == "math_calculator" and isinstance(output, dict):
            if "result" in output:
                return f"Mathematical Computation Result: The evaluation of `{output.get('expression')}` equals **{output.get('result')}**."
            elif "error" in output:
                return f"Computation failed: {output.get('error')}"

        # 3. Database lookup
        if tool_name == "database_lookup" and isinstance(output, dict):
            if output.get("record_found"):
                data = output.get("data", {})
                fields = ", ".join([f"{k}: {v}" for k, v in data.items()])
                return f"Database Record [{output.get('table').upper()} - {output.get('record_id')}]: {fields}."
            return f"Entity Lookup Result: {output.get('message', 'Record not found in database.')}"

        # 4. Knowledge search synthesis
        if tool_name == "knowledge_search" or citations:
            if not citations:
                return f"Knowledge Base Query for '{query}': No relevant documents or matching sections were found in the current corpus."

            # Extract best matching chunks
            top_chunk = citations[0].chunk
            
            # Formulate structured answer with factual extracts
            summary_sentences = []
            for c in citations[:2]:
                first_lines = [line.strip() for line in c.chunk.text.split("\n") if line.strip() and not line.strip().startswith("#")]
                if first_lines:
                    summary_sentences.append(first_lines[0])

            joined_summary = " ".join(summary_sentences)
            return (
                f"Based on **{top_chunk.doc_title}** (relevance score: {citations[0].score:.2%}):\n\n"
                f"{joined_summary}\n\n"
                f"*Source: [{top_chunk.doc_title} - Chunk {top_chunk.chunk_index}]*"
            )

        return f"Completed query with result: {str(output)}"
