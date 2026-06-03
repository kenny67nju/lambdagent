"""
lambdagent.tool_gateway — Tool 调用权限网关
=============================================

所有工具调用在执行前必须通过 ToolGateway 的权限检查。

Lambda 语义:
    ⟦GatedTool(n, f, policy)⟧ = λx. IF policy.allows(n, x) THEN f(x) ELSE ⊥

安全层级:
    L0: 无限制（开发模式，仅日志）
    L1: 命令分类 + 黑名单拦截（默认）
    L2: 白名单模式，只允许声明的操作
    L3: 人机确认模式，高危操作需人工批准

修复的问题:
    - guard.dangerousCommandBlock 之前声明但从未检查 → 现在由 Gateway 执行
    - guard.highRiskConfirmation 之前声明但从未检查 → 现在触发确认流程
    - guard.maxOutputLength 之前声明但从未检查 → 现在截断输出
    - 工具调用无审计日志 → 现在每次调用写入 audit trail
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from .core import Term, Context, ValidationError


# ════════════════════════════════════════════════════════════
# 命令风险分类
# ════════════════════════════════════════════════════════════

class RiskLevel(Enum):
    SAFE = "safe"             # 只读操作，无副作用
    LOW = "low"               # 轻微副作用（写文件到工作目录）
    MEDIUM = "medium"         # 中等风险（安装包、修改配置）
    HIGH = "high"             # 高危操作（删除文件、系统命令）
    CRITICAL = "critical"     # 极危操作（rm -rf、格式化、sudo）


class Action(Enum):
    ALLOW = "allow"           # 放行
    BLOCK = "block"           # 拦截
    CONFIRM = "confirm"       # 需要人工确认
    LOG_ONLY = "log_only"     # 仅记录，不拦截


# ════════════════════════════════════════════════════════════
# 危险命令模式库
# ════════════════════════════════════════════════════════════

# CRITICAL: 无条件拦截
CRITICAL_PATTERNS = [
    # 文件系统破坏
    (r"rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/\s*$", "rm -rf /"),
    (r"rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?~", "rm home directory"),
    (r"rm\s+-[a-zA-Z]*r[a-zA-Z]*\s+/(?:usr|etc|var|boot|sys|proc)", "rm system directory"),
    (r"find\s+/\s+.*-delete", "find / -delete"),
    (r"find\s+/\s+.*-exec\s+rm", "find / -exec rm"),
    (r"mkfs\.", "format filesystem"),
    (r"dd\s+if=.*of=/dev/", "dd to device"),
    # Fork bomb & 资源耗尽
    (r":\(\)\s*\{.*\};\s*:", "fork bomb"),
    (r"while\s+true.*fork", "fork loop"),
    # 权限提升
    (r"chmod\s+(-[a-zA-Z]*R[a-zA-Z]*)?\s*[0-7]*7[0-7]*\s+/", "chmod 777 system"),
    (r"chown\s+(-[a-zA-Z]*R[a-zA-Z]*)?\s+.*\s+/", "chown system root"),
    # 远程代码执行
    (r"curl\s+.*\|\s*(?:ba)?sh", "curl pipe to shell"),
    (r"wget\s+.*\|\s*(?:ba)?sh", "wget pipe to shell"),
    (r"curl\s+.*\|\s*python", "curl pipe to python"),
    # 凭证窃取
    (r"cat\s+.*\.ssh/", "read SSH keys"),
    (r"cat\s+.*\.aws/credentials", "read AWS credentials"),
    (r"cat\s+.*\.env\b", "read .env file"),
    (r"cat\s+.*/etc/shadow", "read shadow file"),
    # 网络后门
    (r"nc\s+-[a-zA-Z]*l[a-zA-Z]*\s+-p", "netcat listener"),
    (r"ncat\s+.*--exec", "ncat exec"),
    (r"python.*-m\s+http\.server", "python HTTP server"),
]

# HIGH: 默认拦截，highRiskConfirmation=true 时可确认放行
HIGH_RISK_PATTERNS = [
    (r"sudo\s+", "sudo command"),
    (r"rm\s+-[a-zA-Z]*r", "recursive delete"),
    (r"rm\s+-[a-zA-Z]*f", "force delete"),
    (r"chmod\s+", "change permissions"),
    (r"chown\s+", "change ownership"),
    (r"kill\s+-9", "force kill process"),
    (r"killall\s+", "kill all processes"),
    (r"pkill\s+", "kill by pattern"),
    (r"systemctl\s+(?:stop|disable|mask)", "stop system service"),
    (r"launchctl\s+(?:unload|remove)", "unload macOS service"),
    (r"pip\s+install", "install Python package"),
    (r"npm\s+install\s+-g", "global npm install"),
    (r"brew\s+(?:install|uninstall|remove)", "homebrew operation"),
    (r"apt\s+(?:install|remove|purge)", "apt operation"),
    (r"git\s+push\s+.*--force", "force push"),
    (r"git\s+reset\s+--hard", "hard reset"),
    (r"docker\s+(?:rm|rmi|system\s+prune)", "docker cleanup"),
    (r"crontab\s+-[re]", "modify crontab"),
    (r"shutdown|reboot|halt", "system shutdown/reboot"),
]

# MEDIUM: 记录 + 允许（dangerousCommandBlock=true 时拦截）
MEDIUM_RISK_PATTERNS = [
    (r"mv\s+", "move/rename file"),
    (r"cp\s+-[a-zA-Z]*r", "recursive copy"),
    (r"mkdir\s+-p\s+/", "create system directory"),
    (r"touch\s+/", "create file in system path"),
    (r"git\s+checkout\s+", "git checkout"),
    (r"git\s+merge\s+", "git merge"),
    (r"git\s+rebase\s+", "git rebase"),
    (r"sed\s+-i", "in-place file edit"),
    (r"tee\s+/", "write to system path"),
    (r">\s*/", "redirect to system path"),
]

# 编译正则（一次编译多次使用）
_COMPILED_CRITICAL = [(re.compile(p, re.IGNORECASE), desc) for p, desc in CRITICAL_PATTERNS]
_COMPILED_HIGH = [(re.compile(p, re.IGNORECASE), desc) for p, desc in HIGH_RISK_PATTERNS]
_COMPILED_MEDIUM = [(re.compile(p, re.IGNORECASE), desc) for p, desc in MEDIUM_RISK_PATTERNS]


def classify_command(command: str) -> tuple[RiskLevel, str]:
    """
    对 shell 命令进行风险分类。

    Returns:
        (risk_level, reason)
    """
    cmd = command.strip()

    for pattern, desc in _COMPILED_CRITICAL:
        if pattern.search(cmd):
            return RiskLevel.CRITICAL, desc

    for pattern, desc in _COMPILED_HIGH:
        if pattern.search(cmd):
            return RiskLevel.HIGH, desc

    for pattern, desc in _COMPILED_MEDIUM:
        if pattern.search(cmd):
            return RiskLevel.MEDIUM, desc

    # 只读命令 → SAFE
    safe_prefixes = (
        "ls", "cat", "head", "tail", "grep", "rg", "find", "which", "where",
        "echo", "printf", "date", "pwd", "whoami", "hostname", "uname",
        "wc", "sort", "uniq", "diff", "file", "stat", "du", "df",
        "ps", "top", "htop", "env", "printenv", "id", "groups",
        "git status", "git log", "git diff", "git show", "git branch",
    )
    first_word = cmd.split()[0] if cmd.split() else ""
    if first_word in [p.split()[0] for p in safe_prefixes]:
        return RiskLevel.SAFE, "read-only command"

    return RiskLevel.LOW, "unknown command, assumed low risk"


def classify_tool_call(tool_name: str, tool_input: Any) -> tuple[RiskLevel, str]:
    """
    对任意工具调用进行风险分类。

    Args:
        tool_name: 工具名称（shell, file, browser, ...）
        tool_input: 工具输入参数

    Returns:
        (risk_level, reason)
    """
    input_str = str(tool_input) if tool_input else ""

    # shell 工具 → 命令级分类
    if tool_name in ("shell", "bash", "exec", "command"):
        # 从 JSON 或纯文本中提取命令
        cmd = input_str
        try:
            data = json.loads(input_str)
            cmd = data.get("command", data.get("cmd", input_str))
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        return classify_command(str(cmd))

    # file 工具
    if tool_name == "file":
        try:
            data = json.loads(input_str)
            action = data.get("action", "")
        except (json.JSONDecodeError, TypeError, AttributeError):
            action = ""

        if action in ("read", "list", "stat", "exists"):
            return RiskLevel.SAFE, f"file {action}"
        elif action in ("write", "append", "create"):
            path = str(data.get("path", ""))
            if path.startswith("/etc") or path.startswith("/usr") or path.startswith("/sys"):
                return RiskLevel.HIGH, f"file write to system path: {path}"
            return RiskLevel.LOW, f"file {action}"
        elif action in ("delete", "remove"):
            return RiskLevel.HIGH, f"file delete"
        return RiskLevel.LOW, f"file operation"

    # browser 工具
    if tool_name == "browser":
        return RiskLevel.LOW, "browser operation"

    # screenshot 工具
    if tool_name == "screenshot":
        return RiskLevel.SAFE, "screenshot capture"

    # system 工具
    if tool_name in ("system", "sys"):
        return RiskLevel.SAFE, "system info query"

    # terminate 工具
    if tool_name in ("terminate", "done"):
        return RiskLevel.SAFE, "agent termination"

    # MCP 远程工具
    if tool_name.startswith("mcp_") or "." in tool_name:
        return RiskLevel.LOW, f"MCP remote tool: {tool_name}"

    return RiskLevel.LOW, f"unknown tool: {tool_name}"


# ════════════════════════════════════════════════════════════
# 安全策略
# ════════════════════════════════════════════════════════════

@dataclass
class GatewayPolicy:
    """
    工具调用权限策略，从 YAML guard 配置映射而来。

    guard:
      dangerousCommandBlock: true   → block_dangerous = True
      highRiskConfirmation: true    → confirm_high_risk = True
      maxOutputLength: 3000         → max_output_length = 3000
    """
    # 拦截策略
    block_dangerous: bool = True          # 拦截 CRITICAL + HIGH 级命令
    confirm_high_risk: bool = False       # HIGH 级操作需要确认（非 block）
    block_medium: bool = False            # 拦截 MEDIUM 级操作

    # 输出限制
    max_output_length: int = 0            # 0 = 不限制

    # 白名单/黑名单
    allowed_tools: Set[str] = field(default_factory=set)    # 空 = 允许所有
    blocked_tools: Set[str] = field(default_factory=set)    # 始终拦截
    allowed_paths: List[str] = field(default_factory=list)  # 允许访问的路径前缀
    blocked_paths: List[str] = field(default_factory=list)  # 禁止访问的路径前缀
    allowed_hosts: List[str] = field(default_factory=list)  # S19: 允许访问的主机名
    blocked_hosts: List[str] = field(default_factory=list)  # S19: 禁止访问的主机名

    # 审计
    audit_log: bool = True                # 记录所有调用
    audit_file: Optional[str] = None      # 审计日志文件路径

    # 确认回调
    confirm_callback: Optional[Callable[[str, str, str], bool]] = None

    @classmethod
    def from_guard_config(cls, guard_cfg: Dict) -> "GatewayPolicy":
        """从 YAML guard 配置创建策略。"""
        if not guard_cfg:
            return cls()

        return cls(
            block_dangerous=guard_cfg.get("dangerousCommandBlock", True),
            confirm_high_risk=guard_cfg.get("highRiskConfirmation", False),
            max_output_length=guard_cfg.get("maxOutputLength", 0),
        )

    @classmethod
    def permissive(cls) -> "GatewayPolicy":
        """宽松策略：只拦截 CRITICAL，仅日志。"""
        return cls(block_dangerous=True, confirm_high_risk=False,
                   block_medium=False, audit_log=True)

    @classmethod
    def strict(cls) -> "GatewayPolicy":
        """严格策略：拦截所有危险操作。"""
        return cls(block_dangerous=True, confirm_high_risk=True,
                   block_medium=True, audit_log=True)


# ════════════════════════════════════════════════════════════
# 审计日志
# ════════════════════════════════════════════════════════════

class AuditLog:
    """工具调用审计日志。"""

    def __init__(self, log_file: Optional[str] = None):
        self._entries: List[Dict] = []
        self._log_file = log_file

    def record(self, tool_name: str, tool_input: str,
               risk_level: RiskLevel, action: Action,
               reason: str, result: Optional[str] = None,
               duration_ms: float = 0):
        entry = {
            "timestamp": time.time(),
            "tool": tool_name,
            "input": tool_input[:500],
            "risk": risk_level.value,
            "action": action.value,
            "reason": reason,
            "result_preview": (result[:200] if result else None),
            "duration_ms": round(duration_ms, 1),
        }
        self._entries.append(entry)

        # 写文件
        if self._log_file:
            try:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError:
                pass

    @property
    def entries(self) -> List[Dict]:
        return list(self._entries)

    @property
    def blocked_count(self) -> int:
        return sum(1 for e in self._entries if e["action"] == "block")

    @property
    def stats(self) -> Dict:
        total = len(self._entries)
        if total == 0:
            return {"total": 0}
        risk_counts = {}
        action_counts = {}
        for e in self._entries:
            risk_counts[e["risk"]] = risk_counts.get(e["risk"], 0) + 1
            action_counts[e["action"]] = action_counts.get(e["action"], 0) + 1
        return {
            "total": total,
            "by_risk": risk_counts,
            "by_action": action_counts,
        }


# ════════════════════════════════════════════════════════════
# ToolGateway — 核心网关
# ════════════════════════════════════════════════════════════

class ToolGateway:
    """
    工具调用权限网关。

    所有 Tool 调用在执行前经过 Gateway 检查：
    1. 命令分类（SAFE/LOW/MEDIUM/HIGH/CRITICAL）
    2. 策略匹配（block/allow/confirm）
    3. 执行或拦截
    4. 输出截断（maxOutputLength）
    5. 审计日志

    用法:
        gateway = ToolGateway(policy=GatewayPolicy.from_guard_config(guard_cfg))
        gated_tool = gateway.wrap(tool)  # 返回 GatedTool
    """

    def __init__(self, policy: Optional[GatewayPolicy] = None):
        self.policy = policy or GatewayPolicy()
        self.audit = AuditLog(self.policy.audit_file)

    def check(self, tool_name: str, tool_input: Any) -> tuple[Action, str]:
        """
        检查工具调用是否被允许。

        Returns:
            (action, reason)
        """
        policy = self.policy

        # 工具级黑名单
        if policy.blocked_tools and tool_name in policy.blocked_tools:
            return Action.BLOCK, f"tool '{tool_name}' is in blocklist"

        # 工具级白名单
        if policy.allowed_tools and tool_name not in policy.allowed_tools:
            return Action.BLOCK, f"tool '{tool_name}' not in allowlist"

        # 风险分类
        risk, reason = classify_tool_call(tool_name, tool_input)

        # S18: Path ACL enforcement — resolve paths to prevent traversal
        input_str = str(tool_input)
        file_paths = self._extract_file_paths(input_str)
        for fp in file_paths:
            resolved = os.path.abspath(fp)
            # Check blocked paths first
            for blocked_path in policy.blocked_paths:
                blocked_abs = os.path.abspath(blocked_path)
                if resolved.startswith(blocked_abs):
                    return Action.BLOCK, f"blocked path: {resolved} (matches {blocked_path})"
            # Check allowed paths (if set, path must be within at least one)
            if policy.allowed_paths:
                allowed = False
                for allowed_path in policy.allowed_paths:
                    allowed_abs = os.path.abspath(allowed_path)
                    if resolved.startswith(allowed_abs):
                        allowed = True
                        break
                if not allowed:
                    return Action.BLOCK, f"path not in allowed_paths: {resolved}"

        # S19: Network ACL enforcement — check hostnames in URLs
        hostnames = self._extract_hostnames(input_str)
        for hostname in hostnames:
            for blocked_host in policy.blocked_hosts:
                if hostname == blocked_host or hostname.endswith(f".{blocked_host}"):
                    return Action.BLOCK, f"blocked host: {hostname}"
            if policy.allowed_hosts:
                allowed = False
                for allowed_host in policy.allowed_hosts:
                    if hostname == allowed_host or hostname.endswith(f".{allowed_host}"):
                        allowed = True
                        break
                if not allowed:
                    return Action.BLOCK, f"host not in allowed_hosts: {hostname}"

        # 策略匹配
        if risk == RiskLevel.CRITICAL:
            # CRITICAL 无条件拦截
            return Action.BLOCK, f"CRITICAL: {reason}"

        if risk == RiskLevel.HIGH:
            if policy.block_dangerous:
                if policy.confirm_high_risk:
                    return Action.CONFIRM, f"HIGH risk: {reason}"
                return Action.BLOCK, f"HIGH risk blocked: {reason}"
            return Action.LOG_ONLY, f"HIGH risk (policy=permissive): {reason}"

        if risk == RiskLevel.MEDIUM:
            if policy.block_medium:
                return Action.BLOCK, f"MEDIUM risk blocked: {reason}"
            return Action.LOG_ONLY, f"MEDIUM risk: {reason}"

        return Action.ALLOW, reason

    def wrap(self, tool: Term) -> "GatedTool":
        """将普通 Tool 包裹为 GatedTool。"""
        from .primitives import Tool as BaseTool
        if isinstance(tool, GatedTool):
            return tool  # 已经包裹过
        if isinstance(tool, BaseTool):
            return GatedTool(
                name=tool._name,
                fn=tool.fn,
                gateway=self,
            )
        return tool

    def truncate_output(self, output: str) -> str:
        """根据 maxOutputLength 截断输出。"""
        max_len = self.policy.max_output_length
        if max_len > 0 and len(output) > max_len:
            return output[:max_len] + f"\n... [truncated at {max_len} chars]"
        return output

    @staticmethod
    def _extract_file_paths(input_str: str) -> List[str]:
        """S18: Extract potential file paths from tool input."""
        paths = []
        # Try JSON first
        try:
            data = json.loads(input_str)
            for key in ("path", "file", "filepath", "file_path", "directory", "dir"):
                if key in data and isinstance(data[key], str):
                    paths.append(data[key])
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        # Regex fallback: match absolute paths
        for match in re.finditer(r'(?:^|\s|["\'])(/[^\s"\']+)', input_str):
            candidate = match.group(1)
            if len(candidate) > 1:
                paths.append(candidate)
        return paths

    @staticmethod
    def _extract_hostnames(input_str: str) -> List[str]:
        """S19: Extract hostnames from URLs in tool input."""
        from urllib.parse import urlparse
        hostnames = []
        for match in re.finditer(r'https?://[^\s"\'<>]+', input_str):
            try:
                parsed = urlparse(match.group(0))
                if parsed.hostname:
                    hostnames.append(parsed.hostname)
            except Exception:
                pass
        return hostnames


# ════════════════════════════════════════════════════════════
# GatedTool — 经过权限检查的 Tool
# ════════════════════════════════════════════════════════════

class ToolBlockedError(ValidationError):
    """工具调用被安全策略拦截。"""
    def __init__(self, tool_name: str, reason: str):
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Tool '{tool_name}' blocked: {reason}")


class GatedTool(Term):
    """
    经过 ToolGateway 权限检查的 Tool。

    Lambda 语义:
        ⟦GatedTool(n, f, G)⟧ = λx. IF G.allows(n, x) THEN f(x) ELSE ⊥

    与 Tool 的 apply 接口完全一致，但在调用 fn 前先过 Gateway。
    """

    def __init__(self, name: str, fn: Callable, gateway: ToolGateway):
        super().__init__(name)
        self.fn = fn
        self.gateway = gateway

    def apply(self, input: Any, ctx: Optional[Context] = None) -> Any:
        ctx = ctx or Context()
        t0 = time.time()
        input_str = str(input)

        # ── Step 1: 权限检查 ──
        action, reason = self.gateway.check(self._name, input)

        if action == Action.BLOCK:
            duration = (time.time() - t0) * 1000
            self.gateway.audit.record(
                self._name, input_str, classify_tool_call(self._name, input)[0],
                Action.BLOCK, reason, duration_ms=duration,
            )
            blocked_msg = f"[BLOCKED] Tool '{self._name}': {reason}"
            ctx.log(self._name, self._trace_id, input, blocked_msg, duration)
            # 返回错误消息而非抛异常，让 agent 知道被拦截了，可以换个方式
            return blocked_msg

        if action == Action.CONFIRM:
            # 确认回调
            confirmed = False
            if self.gateway.policy.confirm_callback:
                try:
                    confirmed = self.gateway.policy.confirm_callback(
                        self._name, input_str, reason
                    )
                except Exception:
                    confirmed = False

            if not confirmed:
                duration = (time.time() - t0) * 1000
                self.gateway.audit.record(
                    self._name, input_str, RiskLevel.HIGH,
                    Action.BLOCK, f"confirmation denied: {reason}",
                    duration_ms=duration,
                )
                blocked_msg = f"[BLOCKED] Tool '{self._name}' requires confirmation: {reason}"
                ctx.log(self._name, self._trace_id, input, blocked_msg, duration)
                return blocked_msg

        # ── Step 2: 执行工具 ──
        try:
            result = self.fn(input)
        except Exception as e:
            duration = (time.time() - t0) * 1000
            error_msg = f"[ERROR] {type(e).__name__}: {e}"
            self.gateway.audit.record(
                self._name, input_str, classify_tool_call(self._name, input)[0],
                Action.ALLOW, reason, result=error_msg, duration_ms=duration,
            )
            ctx.log(self._name, self._trace_id, input, error_msg, duration)
            raise

        # ── Step 3: 输出截断 ──
        result_str = str(result)
        result_str = self.gateway.truncate_output(result_str)

        duration = (time.time() - t0) * 1000

        # ── Step 4: 审计日志 ──
        if self.gateway.policy.audit_log:
            risk_level = classify_tool_call(self._name, input)[0]
            self.gateway.audit.record(
                self._name, input_str, risk_level,
                action, reason, result=result_str, duration_ms=duration,
            )

        ctx.log(self._name, self._trace_id, input, result_str, duration)
        return result_str
