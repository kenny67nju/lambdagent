"""
lambdagent.a2a — Google A2A (Agent-to-Agent) Protocol 集成

将 lambdagent 的 Skill/Agent 发布为 A2A Agent Card，
使其可被 Google A2A 生态中的其他 Agent 发现和调用。

A2A 协议: https://a2a-protocol.org/latest/specification/
版本: 基于 A2A v0.3 规范

Lambda 语义:
    to_agent_card(skill)  = serialize(skill.metadata) → A2A AgentCard JSON
    A2AServer(agent)      = HTTP server 接受 A2A task 请求，执行 β-规约，返回结果
    A2AClient(card_url)   = Tool(agent_name, λx. a2a_call(url, x))

核心概念:
    AgentCard    — Agent 的能力描述（JSON），用于发现
    Task         — A2A 的工作单元（submitted→working→completed/failed）
    Message/Part — Task 中的消息和内容片段（text/file/data）

lambdagent 与 A2A 的映射:
    Skill          → AgentCard (能力描述)
    Skill.apply()  → Task execution (任务执行)
    SkillRegistry  → Agent discovery (技能发现)
    Skill.tags     → AgentCard.skills (能力标签)
    SkillSignature → AgentCard input/output schemas
"""

from __future__ import annotations

import json
import time
import uuid
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

from .core import Term, Context, LambdagentError


# ════════════════════════════════════════════════════════════
# A2A 数据结构 (基于 A2A v0.3 规范)
# ════════════════════════════════════════════════════════════


@dataclass
class AgentCard:
    """
    A2A Agent Card — Agent 的能力描述文档。

    对应 A2A spec 的 /.well-known/agent.json

    字段说明:
        name:          Agent 名称
        description:   自然语言描述（供其他 Agent 理解用途）
        url:           Agent 的 A2A endpoint URL
        version:       Agent 版本
        skills:        能力列表（A2A Skill 描述）
        input_modes:   支持的输入模式 (text, file, data)
        output_modes:  支持的输出模式 (text, file, data)
        auth:          认证方式
    """

    name: str
    description: str = ""
    url: str = ""
    version: str = "1.0.0"
    provider: str = "lambdagent"
    documentation_url: str = ""
    skills: List[Dict] = field(default_factory=list)
    input_modes: List[str] = field(default_factory=lambda: ["text"])
    output_modes: List[str] = field(default_factory=lambda: ["text"])
    authentication: Optional[Dict] = None
    # lambdagent 扩展字段
    lambda_type: str = "Str → Str"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """序列化为 A2A Agent Card JSON"""
        card = {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "provider": {
                "organization": self.provider,
                "url": self.documentation_url or "",
            },
            "capabilities": {
                "streaming": False,
                "pushNotifications": False,
                "stateTransitionHistory": True,
            },
            "defaultInputModes": self.input_modes,
            "defaultOutputModes": self.output_modes,
            "skills": self.skills,
        }
        if self.authentication:
            card["authentication"] = self.authentication
        # lambdagent 扩展
        card["x-lambdagent"] = {
            "lambda_type": self.lambda_type,
            "tags": self.tags,
            "framework": "lambdagent",
            "framework_version": "1.0.0",
        }
        return card

    def to_json(self, indent: int = 2) -> str:
        """序列化为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str) -> str:
        """保存到文件"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return path

    @classmethod
    def from_dict(cls, data: Dict) -> AgentCard:
        """从字典反序列化"""
        ext = data.get("x-lambdagent", {})
        provider = data.get("provider", {})
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            url=data.get("url", ""),
            version=data.get("version", "1.0.0"),
            provider=provider.get("organization", "")
            if isinstance(provider, dict)
            else str(provider),
            documentation_url=provider.get("url", "")
            if isinstance(provider, dict)
            else "",
            skills=data.get("skills", []),
            input_modes=data.get("defaultInputModes", ["text"]),
            output_modes=data.get("defaultOutputModes", ["text"]),
            authentication=data.get("authentication"),
            lambda_type=ext.get("lambda_type", "Str → Str"),
            tags=ext.get("tags", []),
        )

    @classmethod
    def from_json(cls, json_str: str) -> AgentCard:
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: str) -> AgentCard:
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())

    def __repr__(self):
        return f"AgentCard({self.name!r}, {len(self.skills)} skills)"


# ════════════════════════════════════════════════════════════
# Skill → AgentCard 转换
# ════════════════════════════════════════════════════════════


def skill_to_agent_card(skill, url: str = "", port: int = 8000) -> AgentCard:
    """
    将 lambdagent Skill 转换为 A2A Agent Card。

    Lambda → A2A 映射:
        skill._name        → card.name
        skill.description  → card.description
        skill.tags         → card.skills[].tags + card.x-lambdagent.tags
        skill.signature    → card.skills[].inputModes/outputModes
        skill.examples     → card.skills[].examples
    """
    from .skills import Skill

    if not isinstance(skill, Skill):
        # 普通 Term → 简单 AgentCard
        return AgentCard(
            name=skill._name,
            description=f"lambdagent agent: {skill._name}",
            url=url or f"http://localhost:{port}",
            tags=[],
        )

    # Skill 的完整转换
    a2a_skills = [
        {
            "id": skill.skill_id,
            "name": skill._name,
            "description": skill.description,
            "tags": skill.tags,
            "examples": [{"input": inp, "output": out} for inp, out in skill.examples]
            if skill.examples
            else [],
            "inputModes": ["text"],
            "outputModes": ["text"],
        }
    ]

    return AgentCard(
        name=skill._name,
        description=skill.description,
        url=url or f"http://localhost:{port}",
        version=skill.version,
        provider=skill.author or "lambdagent",
        skills=a2a_skills,
        lambda_type=f"{skill.signature.input_type} → {skill.signature.output_type}",
        tags=skill.tags,
    )


def registry_to_agent_card(registry, url: str = "", port: int = 8000) -> AgentCard:
    """
    将整个 SkillRegistry 转换为一个 A2A Agent Card。

    一个 registry 中的所有 skill 成为 card 的 skills 列表。
    """
    from .skills import SkillRegistry, Skill

    all_skills = []
    all_tags = set()

    for name in registry.list_all():
        s = registry.get(name)
        if s and isinstance(s, Skill):
            all_skills.append(
                {
                    "id": s.skill_id,
                    "name": s._name,
                    "description": s.description,
                    "tags": s.tags,
                    "inputModes": ["text"],
                    "outputModes": ["text"],
                }
            )
            all_tags.update(s.tags)

    return AgentCard(
        name="lambdagent-agent",
        description=f"lambdagent agent with {len(all_skills)} skills",
        url=url or f"http://localhost:{port}",
        skills=all_skills,
        tags=sorted(all_tags),
    )


# ════════════════════════════════════════════════════════════
# A2A Task 数据结构
# ════════════════════════════════════════════════════════════


@dataclass
class A2ATask:
    """
    A2A Task — 一次任务请求。

    生命周期: submitted → working → completed | failed | canceled
    """

    id: str = ""
    status: str = "submitted"  # submitted | working | completed | failed | canceled
    input_text: str = ""
    output_text: str = ""
    skill_id: Optional[str] = None
    error: Optional[str] = None
    created_at: float = 0.0
    completed_at: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> Dict:
        result: Dict[str, Any] = {
            "id": self.id,
            "status": {"state": self.status},
            "history": [],
        }
        if self.input_text:
            result["history"].append(
                {
                    "role": "user",
                    "parts": [{"type": "text", "text": self.input_text}],
                }
            )
        if self.output_text:
            result["history"].append(
                {
                    "role": "agent",
                    "parts": [{"type": "text", "text": self.output_text}],
                }
            )
        if self.error:
            result["status"]["message"] = self.error
        return result


# ════════════════════════════════════════════════════════════
# A2AServer: 将 lambdagent Agent 发布为 A2A 服务
# ════════════════════════════════════════════════════════════


class A2AServer:
    """
    将 lambdagent Agent/Skill 发布为 A2A 服务器。

    Lambda 语义:
        A2AServer(agent) = HTTP 服务器，接受 JSON-RPC 请求，
                           执行 agent(input) = β-规约，
                           返回 A2A Task 结果

    端点:
        GET  /.well-known/agent.json  → AgentCard
        POST /                         → JSON-RPC (tasks/send, tasks/get)

    用法:
        server = A2AServer(my_skill, port=8000)
        server.start()  # 启动 HTTP 服务器
    """

    def __init__(
        self,
        agent: Term,
        card: Optional[AgentCard] = None,
        port: int = 8000,
        host: str = "0.0.0.0",
    ):
        self.agent = agent
        self.port = port
        self.host = host
        self.card = card or skill_to_agent_card(agent, port=port)
        self.card.url = f"http://{host}:{port}"
        self._tasks: Dict[str, A2ATask] = {}
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, background: bool = True):
        """启动 A2A 服务器"""
        server_self = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/.well-known/agent.json":
                    body = server_self.card.to_json().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                try:
                    request = json.loads(body)
                    response = server_self._handle_jsonrpc(request)
                    resp_body = json.dumps(response).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(resp_body)
                except Exception as e:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())

            def log_message(self, format, *args):
                pass  # 静默日志

        self._server = HTTPServer((self.host, self.port), Handler)

        if background:
            self._thread = threading.Thread(
                target=self._server.serve_forever, daemon=True
            )
            self._thread.start()
            print(f"  A2A Server started at http://{self.host}:{self.port}")
            print(
                f"  Agent Card: http://{self.host}:{self.port}/.well-known/agent.json"
            )
        else:
            print(f"  A2A Server starting at http://{self.host}:{self.port}")
            self._server.serve_forever()

    def stop(self):
        """停止服务器"""
        if self._server:
            self._server.shutdown()

    def _handle_jsonrpc(self, request: Dict) -> Dict:
        """处理 JSON-RPC 请求"""
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id", str(uuid.uuid4()))

        if method == "tasks/send":
            return self._handle_task_send(params, req_id)
        elif method == "tasks/get":
            return self._handle_task_get(params, req_id)
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

    def _handle_task_send(self, params: Dict, req_id: str) -> Dict:
        """处理 tasks/send — 执行任务"""
        # 提取输入文本
        message = params.get("message", {})
        parts = message.get("parts", [])
        input_text = ""
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                input_text += part.get("text", "")

        # 创建 task
        task = A2ATask(input_text=input_text)
        task.status = "working"
        self._tasks[task.id] = task

        # 执行 β-规约
        try:
            ctx = Context()
            result = self.agent.apply(input_text, ctx)
            task.output_text = str(result)
            task.status = "completed"
            task.completed_at = time.time()
        except Exception as e:
            task.error = str(e)
            task.status = "failed"
            task.completed_at = time.time()

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": task.to_dict(),
        }

    def _handle_task_get(self, params: Dict, req_id: str) -> Dict:
        """处理 tasks/get — 查询任务状态"""
        task_id = params.get("id", "")
        task = self._tasks.get(task_id)
        if not task:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": f"Task not found: {task_id}"},
            }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": task.to_dict(),
        }

    def __repr__(self):
        return f"A2AServer({self.card.name!r}, port={self.port})"


# ════════════════════════════════════════════════════════════
# A2AClient: 调用远程 A2A Agent
# ════════════════════════════════════════════════════════════


class A2AClient(Term):
    """
    A2A Client — 调用远程 A2A Agent 的 lambdagent Term。

    Lambda 语义:
        A2AClient(url) = Tool(agent_name, λx. a2a_send_task(url, x))
        = 远程 Agent 封装为本地 Term，可参与 >> / Route / Loop 等组合

    用法:
        remote = A2AClient("http://remote-agent:8000")
        remote("翻译这段话")  # → β-规约 → HTTP → 远程执行 → 返回结果

        # 与本地 Agent 组合
        pipeline = local_agent >> remote >> another_local
    """

    def __init__(self, url: str, name: Optional[str] = None, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self._card: Optional[AgentCard] = None
        self.timeout = timeout

        # 尝试获取 Agent Card
        try:
            self._card = self._fetch_card()
            agent_name = self._card.name
        except Exception:
            agent_name = name or url.split("/")[-1] or "a2a_agent"

        super().__init__(f"A2A:{agent_name}")

    def _fetch_card(self) -> AgentCard:
        """获取远程 Agent Card"""
        card_url = f"{self.url}/.well-known/agent.json"
        req = urllib.request.Request(card_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return AgentCard.from_dict(data)

    @property
    def card(self) -> Optional[AgentCard]:
        return self._card

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        """
        发送 A2A task 请求 = 远程 β-规约。
        """
        ctx = ctx or Context()
        t0 = time.time()

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": str(input)}],
                },
            },
        }

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise LambdagentError(f"A2A call failed: {e}")

        # 提取结果
        result_data = data.get("result", {})
        status = result_data.get("status", {}).get("state", "unknown")

        if status == "failed":
            error_msg = result_data.get("status", {}).get("message", "unknown error")
            raise LambdagentError(f"A2A task failed: {error_msg}")

        # 提取输出文本
        history = result_data.get("history", [])
        output_text = ""
        for msg in reversed(history):
            if msg.get("role") == "agent":
                for part in msg.get("parts", []):
                    if part.get("type") == "text":
                        output_text += part.get("text", "")
                break

        elapsed = (time.time() - t0) * 1000
        ctx.log(
            self._name, self._trace_id, str(input)[:100], output_text[:100], elapsed
        )
        return output_text

    def __repr__(self):
        card_name = self._card.name if self._card else "?"
        return f"A2AClient({card_name!r}, url={self.url!r})"
