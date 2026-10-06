"""
Built-in Tools for AI Agent: Knowledge Search, Math Calculator, Mock Database, and Time Service.
"""

import ast
import operator
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from cogni_rag.tools.registry import registry
from cogni_rag.retrieval.hybrid_search import HybridRetriever

# Simulated enterprise relational database
MOCK_DATABASE = {
    "users": {
        "USR-101": {"name": "Alice Chen", "tier": "Enterprise", "region": "US-East", "status": "Active"},
        "USR-102": {"name": "Bob Martin", "tier": "Pro", "region": "EU-Central", "status": "Active"},
        "USR-103": {"name": "Carol Danvers", "tier": "Basic", "region": "AP-South", "status": "Suspended"}
    },
    "compliance_records": {
        "AUDIT-2026-A": {"entity": "FinTech Core", "status": "Compliant", "standard": "SOC2-TypeII", "score": 98.4},
        "AUDIT-2026-B": {"entity": "Cloud Ingress", "status": "Minor Finding", "standard": "PCI-DSS-4.0", "score": 89.1}
    }
}


def register_builtin_tools(retriever: HybridRetriever):
    """Registers standard tools with the global registry."""

    # 1. Knowledge Base Search
    def knowledge_search(query: str, top_k: int = 4) -> Dict[str, Any]:
        """Searches ingested enterprise documentation using hybrid vector and BM25 retrieval."""
        results = retriever.search(query=query, top_k=top_k)
        if not results:
            return {"results_found": 0, "message": "No matching knowledge documents found for query."}
            
        items = []
        for r in results:
            items.append({
                "chunk_id": r.chunk.chunk_id,
                "doc_title": r.chunk.doc_title,
                "relevance_score": round(r.score, 4),
                "text_snippet": r.chunk.text
            })
        return {"results_found": len(items), "chunks": items}

    registry.register_func(
        name="knowledge_search",
        description="Searches indexed technical documentation, manuals, specs, and knowledge bases for factual verification.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query keywords or question"},
                "top_k": {"type": "integer", "description": "Number of top passages to retrieve (default: 4)"}
            },
            "required": ["query"]
        },
        func=knowledge_search
    )

    # 2. Safe Mathematical Calculator
    def math_calculator(expression: str) -> Dict[str, Any]:
        """Safely evaluates arithmetic expressions (+, -, *, /, **, %) without arbitrary code execution."""
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def eval_node(node):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError(f"Unsupported constant type: {type(node.value)}")
            elif isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type in allowed_operators:
                    return allowed_operators[op_type](eval_node(node.left), eval_node(node.right))
                raise ValueError(f"Unsupported binary operator: {op_type}")
            elif isinstance(node, ast.UnaryOp):
                op_type = type(node.op)
                if op_type in allowed_operators:
                    return allowed_operators[op_type](eval_node(node.operand))
                raise ValueError(f"Unsupported unary operator: {op_type}")
            else:
                raise ValueError(f"Unsupported syntax expression: {type(node)}")

        try:
            parsed = ast.parse(expression.strip(), mode="eval")
            res = eval_node(parsed.body)
            return {"expression": expression, "result": round(float(res), 6)}
        except Exception as e:
            return {"error": f"Failed to compute expression '{expression}': {str(e)}"}

    registry.register_func(
        name="math_calculator",
        description="Performs exact numerical calculations, ratios, percentages, and arithmetic operations.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical formula to evaluate (e.g. '(1500 * 0.08) + 45')"}
            },
            "required": ["expression"]
        },
        func=math_calculator
    )

    # 3. Database Entity Lookup
    def database_lookup(table: str, record_id: str) -> Dict[str, Any]:
        """Queries structured customer, user, and compliance records by ID."""
        table_key = table.lower()
        if table_key not in MOCK_DATABASE:
            return {"error": f"Table '{table}' does not exist. Available tables: {list(MOCK_DATABASE.keys())}"}

        tbl = MOCK_DATABASE[table_key]
        if record_id in tbl:
            return {"record_found": True, "table": table, "record_id": record_id, "data": tbl[record_id]}
        return {"record_found": False, "message": f"No record found for ID '{record_id}' in '{table}'."}

    registry.register_func(
        name="database_lookup",
        description="Queries structured relational database records for users and compliance audit reports.",
        parameters={
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Table to search: 'users' or 'compliance_records'"},
                "record_id": {"type": "string", "description": "Identifier key (e.g., 'USR-101' or 'AUDIT-2026-A')"}
            },
            "required": ["table", "record_id"]
        },
        func=database_lookup
    )

    # 4. System DateTime Service
    def datetime_service() -> Dict[str, Any]:
        """Provides authoritative current UTC datetime, day of week, and ISO timestamp."""
        now = datetime.now(timezone.utc)
        return {
            "current_utc_iso": now.isoformat(),
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "day_of_week": now.strftime("%A"),
            "time_utc": now.strftime("%H:%M:%S UTC")
        }

    registry.register_func(
        name="datetime_service",
        description="Returns current authoritative UTC date, time, and day of week.",
        parameters={
            "type": "object",
            "properties": {},
            "required": []
        },
        func=datetime_service
    )
