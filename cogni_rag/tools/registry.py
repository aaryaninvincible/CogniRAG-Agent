"""
Tool Registry and schema generator for AI Agent function calling.
"""

import time
import inspect
from typing import Callable, Dict, Any, List, Optional
from cogni_rag.core.types import ToolDefinition, AgentObservation


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._definitions: Dict[str, ToolDefinition] = {}

    def register(self, name: str, description: str, parameters: Dict[str, Any]):
        """Decorator or direct method to register a callable tool."""
        def decorator(func: Callable):
            self._tools[name] = func
            self._definitions[name] = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters
            )
            return func
        return decorator

    def register_func(self, name: str, description: str, parameters: Dict[str, Any], func: Callable):
        """Direct registration helper."""
        self._tools[name] = func
        self._definitions[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters
        )

    def get_definitions(self) -> List[ToolDefinition]:
        """Returns schemas for all registered tools."""
        return list(self._definitions.values())

    def get_tool(self, name: str) -> Optional[Callable]:
        return self._tools.get(name)

    def execute(self, name: str, args: Dict[str, Any]) -> AgentObservation:
        """Executes tool safely with execution timing and error capture."""
        t0 = time.perf_counter()
        func = self.get_tool(name)
        
        if not func:
            elapsed = (time.perf_counter() - t0) * 1000
            return AgentObservation(
                tool_name=name,
                output=f"Error: Unknown tool '{name}'. Available tools: {list(self._tools.keys())}",
                success=False,
                execution_time_ms=round(elapsed, 2)
            )

        try:
            # Execute tool callable
            res = func(**args)
            elapsed = (time.perf_counter() - t0) * 1000
            return AgentObservation(
                tool_name=name,
                output=res,
                success=True,
                execution_time_ms=round(elapsed, 2)
            )
        except TypeError as te:
            elapsed = (time.perf_counter() - t0) * 1000
            return AgentObservation(
                tool_name=name,
                output=f"Argument Mismatch Error: {str(te)}. Expected schema: {self._definitions[name].parameters}",
                success=False,
                execution_time_ms=round(elapsed, 2)
            )
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            return AgentObservation(
                tool_name=name,
                output=f"Tool Execution Exception: {str(e)}",
                success=False,
                execution_time_ms=round(elapsed, 2)
            )


# Global tool registry
registry = ToolRegistry()
