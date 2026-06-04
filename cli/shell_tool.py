"""
lambdagent.cli.shell_tool — 将 shell 命令包装为 Lambda 项 (Tool)

--tool grep="grep -c ERROR /var/log/app.log"
  → Tool("grep", fn=ShellTool("grep -c ERROR /var/log/app.log"))

--tool search="grep -r {} src/"
  → Tool("search", fn=ShellTool("grep -r {} src/", parameterized=True))

Lambda 语义:
  ShellTool(cmd) = λx. exec(cmd, x)
  即: 将外部 shell 命令提升为 Lambda 项中的 Oracle / primitive
"""

import shlex
import subprocess
from typing import Optional

from .._shell_compat import resolve_shell as _resolve_shell


DANGEROUS_PATTERNS = [
    "rm -rf /",
    "rm -rf ~",
    "rm -rf .",
    "dd if=",
    "mkfs",
    "format c:",
    "> /dev/sd",
    "> /dev/null",
    ":(){ :|:& };:",  # fork bomb
    "chmod -R 777 /",
    "wget -O- | sh",
    "curl | sh",
]


class ShellToolError(Exception):
    pass


class ShellTool:
    """
    将 shell 命令包装为可调用对象，用于 Tool 注入。

    Lambda 对应:
        ShellTool("wc -l file.txt") = λx. exec("wc -l file.txt")
        ShellTool("grep -r {} src/") = λx. exec("grep -r x src/")

    参数化模式（command 含 {}）:
        {} 会被 Agent 的输出替换（经 shlex.quote 转义）
    非参数化模式:
        忽略输入，直接执行命令
    """

    def __init__(self, command: str, timeout: int = 30):
        self.command = command
        self.timeout = timeout
        self.parameterized = "{}" in command
        self._safety_check()

    def _safety_check(self):
        """安全检查：拦截危险命令"""
        cmd_lower = self.command.lower().strip()
        for pattern in DANGEROUS_PATTERNS:
            if pattern.lower() in cmd_lower:
                raise ShellToolError(
                    f"Dangerous command blocked: '{self.command}'\n"
                    f"Matched pattern: '{pattern}'"
                )

    def __call__(self, input_text: str) -> str:
        """执行 shell 命令 = 一次 β-规约"""
        if self.parameterized:
            safe_input = shlex.quote(str(input_text))
            cmd = self.command.replace("{}", safe_input)
        else:
            cmd = self.command

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                executable=_resolve_shell(),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode != 0 and result.stderr:
                return f"[EXIT:{result.returncode}] {result.stderr.strip()}"
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return f"[TIMEOUT after {self.timeout}s] {cmd}"
        except Exception as e:
            return f"[ERROR] {e}"

    def __repr__(self):
        mode = "parameterized" if self.parameterized else "static"
        return f"ShellTool({self.command!r}, {mode})"


class CLIAgent:
    """
    通过 CLI 调用另一个 Agent = 跨进程 β-规约。

    Lambda 语义:
        CLIAgent(cmd) = λx. decode(exec(cmd, encode(x)))

    通信协议:
        1. 将 input 写入子进程的 stdin
        2. 从子进程的 stdout 读取 output
        3. stderr 用于日志/错误（不影响结果）

    Unix 管道 = Lambda 应用:
        (f x) 在 Lambda 中 = 把 x 喂给 f，拿到结果
        cmd < input 在 Unix 中 = 把 input 喂给 cmd，拿到结果
    """

    def __init__(self, command: str, format: str = "text", timeout: int = 60, retry: int = 0):
        self.command = command
        self.format = format
        self.timeout = timeout
        self.retry = retry

    def __call__(self, input_text: str) -> str:
        """跨进程 β-规约"""
        last_error = None
        for attempt in range(1 + self.retry):
            try:
                proc = subprocess.run(
                    self.command,
                    shell=True,
                    executable=_resolve_shell(),
                    input=str(input_text),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if proc.returncode != 0:
                    last_error = f"[CLI_ERROR:{proc.returncode}] {proc.stderr.strip()}"
                    continue

                output = proc.stdout.strip()
                if self.format == "json":
                    import json
                    data = json.loads(output)
                    return data.get("result", output)
                return output

            except subprocess.TimeoutExpired:
                last_error = f"[CLI_TIMEOUT after {self.timeout}s] {self.command}"
            except Exception as e:
                last_error = f"[CLI_ERROR] {e}"

        return last_error or "[CLI_ERROR] unknown"

    def __repr__(self):
        return f"CLIAgent({self.command!r}, format={self.format!r})"


def parse_tool_args(tool_specs):
    """
    解析 --tool NAME=COMMAND 参数列表。

    输入: ["grep=grep -c ERROR log.txt", "search=grep -r {} src/"]
    输出: {"grep": ShellTool("grep -c ERROR log.txt"),
           "search": ShellTool("grep -r {} src/")}
    """
    tools = {}
    for spec in tool_specs:
        if "=" not in spec:
            raise ShellToolError(
                f"Invalid tool spec: '{spec}'\n"
                f"Expected format: NAME=COMMAND (e.g., grep='grep -c ERROR log.txt')"
            )
        name, command = spec.split("=", 1)
        name = name.strip()
        command = command.strip().strip("'\"")
        tools[name] = ShellTool(command)
    return tools
