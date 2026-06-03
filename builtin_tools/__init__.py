"""
lambdagent.builtin_tools — Built-in tool implementations

Each tool is a Lambda term: λx. tool_fn(x)
All tools use ValidatedTool with Pydantic-style schema validation.
"""
from __future__ import annotations
