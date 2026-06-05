"""
lambdagent.mcp_client — MCP (Model Context Protocol) Client 集成

将 MCP Server 的工具无缝接入 lambdagent 的 Lambda 演算体系。

Lambda 语义:
    MCPTool(server, tool_name) = Tool(tool_name, λx. mcp_call(server, tool_name, x))
    MCPServer(url)             = {tool_name → MCPTool(url, tool_name)}  一组 Tool
    mcp_tools(url)             = [Tool₁, Tool₂, ...]  自动发现所有工具

MCP 协议: JSON-RPC 2.0 over HTTP (Streamable HTTP transport)
    https://modelcontextprotocol.io/specification/2025-11-25

支持:
    - Streamable HTTP transport (2025-11-25 spec)
    - stdio transport (本地 MCP Server)
    - 工具自动发现 (tools/list)
    - 工具调用 (tools/call)
    - 资源读取 (resources/read)
    - 连接池 + 重试
    - 超时控制
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .core import Term, Context, LambdagentError


# ════════════════════════════════════════════════════════════
# 异常
# ════════════════════════════════════════════════════════════


class MCPError(LambdagentError):
    """MCP 协议错误"""

    pass


class MCPConnectionError(MCPError):
    """MCP 连接失败"""

    pass


class MCPToolError(MCPError):
    """MCP 工具调用失败"""

    pass


# ════════════════════════════════════════════════════════════
# MCP JSON-RPC 通信层
# ════════════════════════════════════════════════════════════


@dataclass
class MCPResponse:
    """MCP JSON-RPC 响应"""

    id: str
    result: Any = None
    error: Optional[Dict] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class MCPTransport:
    """
    MCP 传输层基类。

    两种传输方式:
        HTTP:  MCPHttpTransport(url)   — 远程 MCP Server
        stdio: MCPStdioTransport(cmd)  — 本地 MCP Server (子进程)
    """

    def send(self, method: str, params: Optional[Dict] = None) -> MCPResponse:
        raise NotImplementedError


class MCPHttpTransport(MCPTransport):
    """
    Streamable HTTP transport (MCP 2025-11-25 spec).

    JSON-RPC 2.0 over HTTP POST.
    """

    def __init__(
        self, url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0
    ):
        self.url = url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self._session_id: Optional[str] = None

    def send(self, method: str, params: Optional[Dict] = None) -> MCPResponse:
        """发送 JSON-RPC 请求"""
        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            payload["params"] = params

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.headers,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=body, headers=headers, method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                # 记录 session ID
                sid = resp.headers.get("Mcp-Session-Id")
                if sid:
                    self._session_id = sid

                data = json.loads(resp.read().decode("utf-8"))

                # 处理批量响应
                if isinstance(data, list):
                    # 找到匹配我们 request_id 的响应
                    for item in data:
                        if item.get("id") == request_id:
                            data = item
                            break
                    else:
                        data = data[0] if data else {}

                return MCPResponse(
                    id=data.get("id", request_id),
                    result=data.get("result"),
                    error=data.get("error"),
                )
        except urllib.error.HTTPError as e:
            raise MCPConnectionError(
                f"MCP HTTP error {e.code}: {e.reason} (url={self.url})"
            )
        except urllib.error.URLError as e:
            raise MCPConnectionError(
                f"MCP connection failed: {e.reason} (url={self.url})"
            )
        except Exception as e:
            raise MCPConnectionError(f"MCP transport error: {e}")


class MCPStdioTransport(MCPTransport):
    """
    stdio transport — 本地 MCP Server (子进程通信).

    启动一个子进程，通过 stdin/stdout 进行 JSON-RPC 通信。
    对应 lambdagent CLI 协议: stdin=输入, stdout=输出, stderr=日志。
    """

    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ):
        self.command = command
        self.args = args or []
        self.env = env
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None

    def _ensure_started(self):
        """确保子进程已启动"""
        if self._process is None or self._process.poll() is not None:
            cmd = [self.command] + self.args
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                text=True,
            )

    def send(self, method: str, params: Optional[Dict] = None) -> MCPResponse:
        """通过 stdin/stdout 发送 JSON-RPC 请求"""
        self._ensure_started()

        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            payload["params"] = params

        try:
            line = json.dumps(payload) + "\n"
            self._process.stdin.write(line)
            self._process.stdin.flush()

            # 读取响应（一行 JSON）
            response_line = self._process.stdout.readline()
            if not response_line:
                raise MCPConnectionError("MCP stdio: no response from server")

            data = json.loads(response_line)
            return MCPResponse(
                id=data.get("id", request_id),
                result=data.get("result"),
                error=data.get("error"),
            )
        except json.JSONDecodeError as e:
            raise MCPError(f"MCP stdio: invalid JSON response: {e}")
        except Exception as e:
            raise MCPConnectionError(f"MCP stdio error: {e}")

    def close(self):
        """关闭子进程"""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=5)


# ════════════════════════════════════════════════════════════
# MCPServer: MCP 服务器连接
# ════════════════════════════════════════════════════════════


@dataclass
class MCPToolInfo:
    """MCP 工具元信息"""

    name: str
    description: str = ""
    input_schema: Optional[Dict] = None


class MCPServer:
    """
    MCP Server 连接管理器。

    Lambda 语义:
        MCPServer(url) = Γ_mcp : tool_name → Tool(tool_name, mcp_call)
        一个 MCP Server 就是一组 Tool 的环境

    用法:
        server = MCPServer.http("http://localhost:3000/mcp")
        tools = server.list_tools()
        result = server.call_tool("search", {"query": "AI agents"})

        # 或者直接转为 lambdagent Term:
        search = server.to_tool("search")  # → Tool
        search("AI agents")                 # → β-规约
    """

    def __init__(self, transport: MCPTransport, name: str = ""):
        self.transport = transport
        self.name = name or "mcp_server"
        self._tools_cache: Optional[List[MCPToolInfo]] = None
        self._initialized = False

    @classmethod
    def http(
        cls,
        url: str,
        headers: Optional[Dict] = None,
        timeout: float = 30.0,
        name: str = "",
    ) -> MCPServer:
        """连接远程 MCP Server (HTTP)"""
        transport = MCPHttpTransport(url, headers=headers, timeout=timeout)
        return cls(transport, name=name or url.split("/")[-1])

    @classmethod
    def stdio(
        cls,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict] = None,
        name: str = "",
    ) -> MCPServer:
        """连接本地 MCP Server (stdio)"""
        transport = MCPStdioTransport(command, args=args, env=env)
        return cls(transport, name=name or command)

    def initialize(self) -> Dict:
        """
        MCP 协议初始化握手。

        发送 initialize 请求，协商协议版本和能力。
        """
        resp = self.transport.send(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {
                    "name": "lambdagent",
                    "version": "1.0.0",
                },
            },
        )
        if not resp.ok:
            raise MCPConnectionError(f"MCP initialize failed: {resp.error}")
        self._initialized = True

        # 发送 initialized 通知
        try:
            self.transport.send("notifications/initialized")
        except Exception:
            pass  # 通知失败不影响功能

        return resp.result or {}

    def list_tools(self, force_refresh: bool = False) -> List[MCPToolInfo]:
        """
        发现 MCP Server 上的所有工具。

        MCP 方法: tools/list
        """
        if self._tools_cache and not force_refresh:
            return self._tools_cache

        resp = self.transport.send("tools/list")
        if not resp.ok:
            raise MCPError(f"tools/list failed: {resp.error}")

        tools = []
        for t in (resp.result or {}).get("tools", []):
            tools.append(
                MCPToolInfo(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema"),
                )
            )
        self._tools_cache = tools
        return tools

    def call_tool(self, tool_name: str, arguments: Optional[Dict] = None) -> str:
        """
        调用 MCP Server 上的工具。

        MCP 方法: tools/call

        返回工具的文本输出。
        """
        resp = self.transport.send(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments or {},
            },
        )
        if not resp.ok:
            raise MCPToolError(f"MCP tool '{tool_name}' failed: {resp.error}")

        # 提取文本内容
        result = resp.result or {}
        content = result.get("content", [])
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts) if texts else json.dumps(result)

    def read_resource(self, uri: str) -> str:
        """
        读取 MCP Server 上的资源。

        MCP 方法: resources/read
        """
        resp = self.transport.send("resources/read", {"uri": uri})
        if not resp.ok:
            raise MCPError(f"resources/read failed: {resp.error}")

        result = resp.result or {}
        contents = result.get("contents", [])
        texts = []
        for item in contents:
            if isinstance(item, dict):
                texts.append(item.get("text", item.get("blob", "")))
        return "\n".join(texts)

    # ── lambdagent Term 集成 ──

    def to_tool(self, tool_name: str) -> MCPTool:
        """
        将单个 MCP 工具转换为 lambdagent Term。

        Lambda 语义:
            to_tool("search") = Tool("search", λx. mcp_call(server, "search", x))
        """
        # 查找工具元信息
        tools = self.list_tools()
        info = None
        for t in tools:
            if t.name == tool_name:
                info = t
                break

        return MCPTool(
            server=self,
            tool_name=tool_name,
            description=info.description if info else "",
            input_schema=info.input_schema if info else None,
        )

    def to_tools(self) -> List[MCPTool]:
        """
        将所有 MCP 工具转换为 lambdagent Term 列表。

        Lambda 语义:
            to_tools() = [Tool₁, Tool₂, ..., Toolₙ]
        """
        return [self.to_tool(t.name) for t in self.list_tools()]

    def to_route_dict(self) -> Dict[str, MCPTool]:
        """
        将所有工具转换为 Route 可用的字典。

        用法:
            Route(classifier, server.to_route_dict())
        """
        return {t.name: self.to_tool(t.name) for t in self.list_tools()}

    def __repr__(self):
        n_tools = len(self._tools_cache) if self._tools_cache else "?"
        return f"MCPServer({self.name!r}, {n_tools} tools)"


# ════════════════════════════════════════════════════════════
# MCPTool: MCP 工具的 lambdagent Term 封装
# ════════════════════════════════════════════════════════════


class MCPTool(Term):
    """
    MCP 工具封装为 lambdagent Term。

    Lambda 语义:
        MCPTool(server, name) = Tool(name, λx. mcp_call(server, name, parse(x)))
        = tool[f] where f(x) = server.call_tool(name, parse(x))

    输入解析:
        - 字符串 → {"input": x} 或尝试 JSON 解析
        - 字典   → 直接作为 arguments
    """

    def __init__(
        self,
        server: MCPServer,
        tool_name: str,
        description: str = "",
        input_schema: Optional[Dict] = None,
        retry: int = 0,
        timeout: float = 30.0,
    ):
        super().__init__(f"MCP:{tool_name}")
        self.server = server
        self.tool_name = tool_name
        self.description = description
        self.input_schema = input_schema
        self.retry = retry
        self.timeout = timeout

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        """
        调用 MCP 工具 = β-规约。

        输入可以是:
            - str:  尝试 JSON 解析，失败则包装为 {"input": str}
            - dict: 直接作为 arguments
            - tuple/list: 包装为 {"items": list}
        """
        ctx = ctx or Context()
        t0 = time.time()

        # 解析输入为 MCP arguments
        arguments = self._parse_input(input)

        # 调用（带重试）
        last_error = None
        for attempt in range(1 + self.retry):
            try:
                result = self.server.call_tool(self.tool_name, arguments)
                elapsed = (time.time() - t0) * 1000
                ctx.log(
                    f"MCP:{self.tool_name}",
                    self._trace_id,
                    str(input)[:100],
                    str(result)[:100],
                    elapsed,
                )
                return result
            except MCPToolError as e:
                last_error = e
                if attempt < self.retry:
                    time.sleep(0.5 * (attempt + 1))

        raise MCPToolError(
            f"MCP tool '{self.tool_name}' failed after {1 + self.retry} attempts: {last_error}"
        )

    def _parse_input(self, input: Any) -> Dict:
        """将输入解析为 MCP arguments 字典"""
        if isinstance(input, dict):
            return input
        if isinstance(input, str):
            # 尝试 JSON 解析
            try:
                parsed = json.loads(input)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            # 使用 input_schema 推断参数名
            if self.input_schema and "properties" in self.input_schema:
                props = self.input_schema["properties"]
                if len(props) == 1:
                    key = list(props.keys())[0]
                    return {key: input}
            return {"input": input}
        if isinstance(input, (list, tuple)):
            return {"items": list(input)}
        return {"input": str(input)}

    def __repr__(self):
        return f"MCPTool({self.tool_name!r}, server={self.server.name!r})"


# ════════════════════════════════════════════════════════════
# 便利函数
# ════════════════════════════════════════════════════════════


def mcp_tools(url: str, headers: Optional[Dict] = None) -> List[MCPTool]:
    """
    一行代码获取 MCP Server 的所有工具。

    用法:
        tools = mcp_tools("http://localhost:3000/mcp")
        search = tools[0]
        search("AI agents")
    """
    server = MCPServer.http(url, headers=headers)
    try:
        server.initialize()
    except MCPConnectionError:
        pass  # 有些 server 不需要 initialize
    return server.to_tools()


def mcp_tool(url: str, tool_name: str, headers: Optional[Dict] = None) -> MCPTool:
    """
    一行代码获取单个 MCP 工具。

    用法:
        search = mcp_tool("http://localhost:3000/mcp", "search")
        search("AI agents")
    """
    server = MCPServer.http(url, headers=headers)
    return server.to_tool(tool_name)
