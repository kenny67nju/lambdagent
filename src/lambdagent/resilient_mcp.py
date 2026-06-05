"""
lambdagent.resilient_mcp — Resilient MCP client with connection pooling

Wraps MCP calls with circuit breaker, retry, and tool discovery caching.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .retry import RetryPolicy, CircuitBreaker, with_retry_sync


@dataclass
class MCPToolSchema:
    """Cached tool schema from MCP discovery."""

    name: str
    description: str = ""
    input_schema: Dict = field(default_factory=dict)


class ResilientMCPClient:
    """MCP client with circuit breaker, retry, and tool discovery caching.

    Features:
      - Circuit breaker: fast-fail when server is down
      - Retry with exponential backoff for transient errors
      - Tool discovery caching: avoid repeated list_tools calls
      - Connection reuse via urllib keep-alive
    """

    def __init__(
        self,
        url: str,
        endpoint: str = "",
        headers: Dict[str, str] = None,
        timeout: int = 30,
        retry_policy: RetryPolicy = None,
        circuit_breaker: CircuitBreaker = None,
        name: str = "",
    ):
        self.url = url.rstrip("/")
        self.endpoint = endpoint
        self.headers = headers or {}
        self.timeout = timeout
        self.name = name or url
        self.retry_policy = retry_policy or RetryPolicy(
            max_attempts=2,
            base_delay=1.0,
            retryable_errors=(TimeoutError, ConnectionError, OSError),
        )
        self.circuit = circuit_breaker or CircuitBreaker(
            failure_threshold=3, reset_timeout=30.0, name=f"mcp:{name or url}"
        )
        self._tool_cache: Optional[List[MCPToolSchema]] = None
        self._cache_time: float = 0
        self._cache_ttl: float = 300  # 5 min cache for tool discovery
        self._call_count: int = 0
        self._error_count: int = 0

    @property
    def full_url(self) -> str:
        return f"{self.url}{self.endpoint}"

    def call_tool(self, tool_name: str, arguments: Any, timeout: int = None) -> str:
        """Call an MCP tool with circuit breaker + retry."""
        timeout = timeout or self.timeout

        def _do_call():
            return self.circuit.call_sync(
                lambda: self._raw_call(tool_name, arguments, timeout)
            )

        try:
            result = with_retry_sync(_do_call, self.retry_policy)
            self._call_count += 1
            return result
        except Exception as e:
            self._error_count += 1
            return f"[MCP_ERROR: {self.name}/{tool_name}] {e}"

    def list_tools(self, force_refresh: bool = False) -> List[MCPToolSchema]:
        """List available tools. Uses cache unless expired or forced."""
        now = time.time()
        if not force_refresh and self._tool_cache is not None:
            if now - self._cache_time < self._cache_ttl:
                return self._tool_cache

        try:
            body = json.dumps(
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
            ).encode("utf-8")

            req_headers = {"Content-Type": "application/json"}
            req_headers.update(self.headers)

            req = urllib.request.Request(
                self.full_url, data=body, headers=req_headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            tools_data = data.get("result", {}).get("tools", [])
            self._tool_cache = [
                MCPToolSchema(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                )
                for t in tools_data
            ]
            self._cache_time = now
            return self._tool_cache
        except Exception:
            return self._tool_cache or []

    def _raw_call(self, tool_name: str, arguments: Any, timeout: int) -> str:
        """Raw MCP tool call via HTTP."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, ValueError):
                arguments = {"input": arguments}

        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
        ).encode("utf-8")

        req_headers = {"Content-Type": "application/json"}
        req_headers.update(self.headers)

        req = urllib.request.Request(
            self.full_url, data=body, headers=req_headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if "result" in data:
            result = data["result"]
            if isinstance(result, dict):
                content = result.get("content", [])
                if content and isinstance(content, list):
                    return content[0].get("text", str(result))
                return str(result)
            return str(result)
        elif "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return str(data)

    def summary(self) -> str:
        return (
            f"ResilientMCPClient({self.name}): "
            f"{self._call_count} calls, {self._error_count} errors, "
            f"circuit={self.circuit.state}"
        )
