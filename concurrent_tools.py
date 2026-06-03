"""
lambdagent.concurrent_tools — Tool concurrency safety declarations

Tools can declare whether they are safe for concurrent execution.
The executor uses this to parallelize safe tools and serialize unsafe ones.
"""
from __future__ import annotations

import time
from typing import Any, Callable, List
from .core import Term, Context
from .primitives import Tool


class ConcurrentTool(Tool):
    """Tool with explicit concurrency safety declaration."""

    def __init__(self, name: str, fn: Callable, concurrent_safe: bool = False,
                 description: str = ""):
        super().__init__(name, fn)
        self.concurrent_safe = concurrent_safe
        self.description = description


def partition_by_safety(tools: List[ConcurrentTool]) -> tuple:
    """Split tools into (safe, unsafe) groups.

    Returns:
        (safe_tools, unsafe_tools) - safe can run in parallel, unsafe must be sequential
    """
    safe = [t for t in tools if getattr(t, 'concurrent_safe', False)]
    unsafe = [t for t in tools if not getattr(t, 'concurrent_safe', False)]
    return safe, unsafe


# Pre-built concurrent-safe tool constructors
def read_tool(name: str, fn: Callable) -> ConcurrentTool:
    """Create a read-only (concurrent-safe) tool."""
    return ConcurrentTool(name, fn, concurrent_safe=True)

def write_tool(name: str, fn: Callable) -> ConcurrentTool:
    """Create a write (not concurrent-safe) tool."""
    return ConcurrentTool(name, fn, concurrent_safe=False)
