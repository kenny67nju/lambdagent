"""agentruntime.mcp_client — MCP protocol HTTP client"""

from __future__ import annotations
import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ToolSchema:
    name: str
    description: str = ""
    input_schema: Dict = None


class MCPClient:
    """
    MCP protocol client.
    Lambda: MCPClient = lambda (server, tool, input). HTTP_POST(url, {tool, input})
    """

    def __init__(self, nodes: Dict[str, Any] = None):
        self.nodes = nodes or {}

    def invoke(self, server: str, tool: str, input_data: Any, timeout: int = 30) -> str:
        """
        Call an MCP tool via HTTP POST (JSON-RPC 2.0).

        Returns: tool execution result (string)
        Raises: MCPError on failure
        """
        node = self.nodes.get(server)
        if not node:
            return f"[MCP_ERROR: Unknown server '{server}']"

        url = node.url if hasattr(node, "url") else node.get("url", "")
        endpoint = (
            node.endpoint if hasattr(node, "endpoint") else node.get("endpoint", "")
        )
        headers = node.headers if hasattr(node, "headers") else node.get("headers", {})
        retry = node.retry if hasattr(node, "retry") else node.get("retry", 0)

        if not url:
            return f"[MCP_NOT_CONFIGURED: {server}]"

        full_url = f"{url.rstrip('/')}{endpoint}"
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool,
                    "arguments": input_data
                    if isinstance(input_data, dict)
                    else {"input": str(input_data)},
                },
            }
        ).encode("utf-8")

        req_headers = {"Content-Type": "application/json"}
        req_headers.update(headers)

        for attempt in range(1 + retry):
            try:
                req = urllib.request.Request(
                    full_url, data=body, headers=req_headers, method="POST"
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
                        return f"[MCP_ERROR: {data['error']}]"
                    return str(data)
            except urllib.error.URLError as e:
                if attempt < retry:
                    time.sleep(min(2**attempt, 30))
                    continue
                return f"[MCP_TIMEOUT: {server}/{tool}] {e}"
            except Exception as e:
                return f"[MCP_ERROR: {server}/{tool}] {e}"

        return f"[MCP_FAILED: {server}/{tool}]"

    def discover(self, server: str) -> List[ToolSchema]:
        """Discover available tools on an MCP server."""
        node = self.nodes.get(server)
        if not node:
            return []

        url = node.url if hasattr(node, "url") else node.get("url", "")
        endpoint = (
            node.endpoint if hasattr(node, "endpoint") else node.get("endpoint", "")
        )
        headers = node.headers if hasattr(node, "headers") else node.get("headers", {})

        if not url:
            return []

        full_url = f"{url.rstrip('/')}{endpoint}"
        body = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        ).encode("utf-8")

        req_headers = {"Content-Type": "application/json"}
        req_headers.update(headers)

        try:
            req = urllib.request.Request(
                full_url, data=body, headers=req_headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                tools = data.get("result", {}).get("tools", [])
                return [
                    ToolSchema(
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        input_schema=t.get("inputSchema", {}),
                    )
                    for t in tools
                ]
        except Exception:
            return []
