"""
lambdagent.providers.claude_code_provider — Claude Code CLI as LLM provider.

No API Key required — uses Claude Code Max Plan authentication.

Optimizations:
  - First call: creates session with system prompt, captures session_id
  - Subsequent calls: --resume <session_id> (Claude retains full memory)
  - Falls back to messages-in-prompt if --resume unavailable
  - Auto-detects a working claude binary across nvm versions
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Dict, List, Optional

from .base import LLMProvider, ProviderConfig, ProviderError


def _find_working_claude(preferred: str = "claude") -> Optional[str]:
    """
    Find a claude binary that actually supports modern flags (--tools, --mcp-config).

    Strategy:
      1. If preferred is an absolute path, use it directly.
      2. Test whatever shutil.which finds first.
      3. Scan ~/.nvm/versions/node in reverse order (newest first).

    A binary is "working" if:
      - It's executable and exits 0 for --version
      - Its version output contains "Claude Code"
      - Its --help output mentions "--tools" (rules out old v1.x CLI)
    """

    def _works(path: str) -> bool:
        try:
            r = subprocess.run(
                [path, "--version"], capture_output=True, text=True, timeout=10
            )
            if r.returncode != 0 or "Claude Code" not in r.stdout:
                return False
            # v1.x (old) doesn't have --tools; v2.x does. Use --help as a proxy check.
            h = subprocess.run(
                [path, "--help"], capture_output=True, text=True, timeout=10
            )
            help_text = h.stdout + h.stderr
            return "--tools" in help_text
        except Exception:
            return False

    # 1. Absolute path given — trust it.
    if os.path.isabs(preferred):
        return preferred if os.path.isfile(preferred) else None

    # 2. Whatever is first in PATH.
    found = shutil.which(preferred)
    if found and _works(found):
        return found

    # 3. Scan nvm versions, newest first.
    nvm_base = os.path.expanduser("~/.nvm/versions/node")
    if os.path.isdir(nvm_base):
        try:
            versions = sorted(os.listdir(nvm_base), reverse=True)
        except OSError:
            versions = []
        for ver in versions:
            candidate = os.path.join(nvm_base, ver, "bin", "claude")
            if os.path.isfile(candidate) and _works(candidate):
                return candidate

    # Fall back to whatever shutil found, even if it might not work.
    return found or preferred


class ClaudeCodeProvider(LLMProvider):
    """
    LLM Provider backed by Claude Code CLI.

    Uses `claude -p` subprocess. Session persistence via `--resume`.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        preferred = (
            config.extra.get("claude_bin", "claude") if config.extra else "claude"
        )
        self.claude_bin = _find_working_claude(preferred) or preferred
        self._session_id = None

        if not shutil.which(self.claude_bin) and not os.path.isfile(self.claude_bin):
            raise ProviderError(
                f"claude command not found at '{self.claude_bin}'. "
                f"Install: npm install -g @anthropic-ai/claude-code",
                "claude-code",
            )

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if self._session_id:
            return self._call_resume(messages)
        else:
            return self._call_new(messages)

    # ReAct constraint appended to every user message.
    # Prevents claude-code from using its native tools (Bash, WebSearch, etc.)
    # and forces it to output a single JSON tool-call decision instead.
    _REACT_CONSTRAINT = (
        "\n\n[REACT_MODE] 你是决策模块，不是执行模块。"
        '只输出下一步的JSON工具调用，格式：{"tool":"工具名","input":"..."}，'
        "或输出 terminate 表示任务完成。"
        "严禁直接执行任何搜索、Bash命令或文件读写操作。"
    )

    def _call_new(self, messages: List[Dict[str, str]]) -> str:
        """First call: create session with system prompt, capture session_id."""
        system_prompt = ""
        user_content = ""
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
            elif m["role"] == "user":
                user_content = m["content"]

        # Inject ReAct constraint so claude outputs a JSON decision, not execution.
        prompt_arg = user_content + self._REACT_CONSTRAINT

        cmd = [
            self.claude_bin,
            "-p",
            prompt_arg,
            "--output-format",
            "json",
            "--model",
            self.config.model,
            "--system-prompt",
            system_prompt,
            "--dangerously-skip-permissions",
        ]

        try:
            result = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,  # avoid 3-s stdin-wait warning
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )
        except subprocess.TimeoutExpired:
            raise ProviderError(
                f"Claude Code timeout ({self.config.timeout}s) on first turn",
                "claude-code",
                retryable=True,
            )

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout_hint = (result.stdout or "").strip()[:200]
            detail = stderr or stdout_hint or "(no output)"
            raise ProviderError(
                f"Claude Code error (exit {result.returncode}): {detail}",
                "claude-code",
            )

        raw = result.stdout.strip()
        if not raw:
            raise ProviderError(
                f"Claude Code returned empty output (exit {result.returncode}). "
                f"stderr: {(result.stderr or '').strip()[:200] or '(empty)'}",
                "claude-code",
                retryable=True,
            )

        try:
            data = json.loads(raw)
            self._session_id = data.get("session_id")
            return data.get("result", raw)
        except (json.JSONDecodeError, KeyError):
            return raw

    def _call_resume(self, messages: List[Dict[str, str]]) -> str:
        """Subsequent calls: resume session (Claude has full memory)."""
        last_user = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user = m["content"]
                break

        # Same ReAct constraint to keep each resume turn as a single JSON decision.
        prompt_arg = last_user + self._REACT_CONSTRAINT

        cmd = [
            self.claude_bin,
            "-p",
            prompt_arg,
            "--output-format",
            "text",
            "--model",
            self.config.model,
            "--resume",
            self._session_id,
            "--dangerously-skip-permissions",
        ]

        try:
            result = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )
        except subprocess.TimeoutExpired:
            self._session_id = None
            raise ProviderError(
                f"Claude Code timeout ({self.config.timeout}s) on resume",
                "claude-code",
                retryable=True,
            )

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout_hint = (result.stdout or "").strip()[:200]
            detail = stderr or stdout_hint or "(no output)"
            self._session_id = None
            raise ProviderError(
                f"Claude Code resume error (exit {result.returncode}): {detail}",
                "claude-code",
            )

        output = result.stdout.strip()
        return output if output else "[no output]"

    def reset_session(self):
        """Reset session — next call creates a new conversation."""
        self._session_id = None
