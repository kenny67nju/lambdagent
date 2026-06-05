"""
lambdagent.providers.claude_code — Claude Code CLI as LLM backend.

No API Key required — uses Claude Code Max Plan authentication.

Lambda semantics:
    ClaudeLam("name", "prompt") = lambda x. claude(prompt, x)
    apply() = beta-reduction = `claude -p` decoding

Session persistence:
    First call creates a session with system prompt.
    Subsequent calls use `--resume <session_id>` to maintain full context.
    This eliminates the "amnesia" problem of stateless `-p` calls.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Callable, Optional

from lambdagent.core import Term, Context


def _find_working_claude(preferred: str = "claude") -> str:
    """Find the first claude CLI binary that passes --version (handles nvm Node version mismatch)."""
    import shutil

    def _works(path: str) -> bool:
        try:
            r = subprocess.run(
                [path, "--version"], capture_output=True, text=True, timeout=10
            )
            return r.returncode == 0 and "Claude Code" in r.stdout
        except Exception:
            return False

    if os.path.isabs(preferred):
        return preferred

    found = shutil.which(preferred)
    if found and _works(found):
        return found

    nvm_base = os.path.expanduser("~/.nvm/versions/node")
    if os.path.isdir(nvm_base):
        for ver in sorted(os.listdir(nvm_base), reverse=True):
            candidate = os.path.join(nvm_base, ver, "bin", "claude")
            if os.path.isfile(candidate) and _works(candidate):
                return candidate

    return found or preferred


class ClaudeLam(Term):
    """
    Lambda abstraction backed by Claude Code CLI with session persistence.

    .. deprecated:: 0.1.0
        Use :class:`lambdagent.providers.ClaudeCodeProvider` together with
        :class:`lambdagent.Lam` instead. ``ClaudeLam`` will be removed in 0.3.0.
        Migration::

            # before
            agent = ClaudeLam("name", "prompt")

            # after
            from lambdagent.providers import ClaudeCodeProvider
            agent = Lam("name", "prompt", provider=ClaudeCodeProvider(...))

    Key improvement over stateless `-p`:
      Each apply() continues the same conversation session.
      Claude retains full memory of previous steps, file contents, etc.
    """

    def __init__(
        self,
        name: str,
        prompt: str,
        model: str = "sonnet",
        max_tokens: int = 4096,
        temperature: float = 0.3,
        output_parser: Callable[[str], Any] | None = None,
        claude_bin: str = "claude",
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
        inject_override: bool = True,
    ):
        import warnings

        warnings.warn(
            "ClaudeLam is deprecated and will be removed in lambdagent 0.3.0. "
            "Use Lam(provider=ClaudeCodeProvider(...)) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(name)
        self.prompt = prompt
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.output_parser = output_parser or (lambda x: x)
        self.claude_bin = _find_working_claude(claude_bin)
        self.stream = stream
        self.on_chunk = on_chunk
        self.inject_override = inject_override
        # Session persistence
        self._session_id = None  # Set after first call

    # Injected at the end of the system prompt for from_config path.
    _TOOL_OVERRIDE = (
        "\n\n[RUNTIME ENVIRONMENT]\n"
        "You are running inside a ReAct agent runtime with a tool execution engine. "
        "All tools listed in your system prompt ARE available and fully functional. "
        "To call a tool, output exactly one JSON code block:\n"
        "```json\n"
        '{"action": "ToolName", "input": {parameters}}\n'
        "```\n"
        "The runtime will execute it and return the result as your next observation. "
        "Do NOT say tools are unavailable. Do NOT ask the user to do it manually. "
        "Just call the tool directly.\n\n"
        "[CRITICAL RULES]\n"
        "1. NEVER claim you have read, written, or executed something unless you see "
        "the ACTUAL tool result confirming it succeeded.\n"
        "2. If a tool returns [VALIDATION_ERROR], [ERROR], or [Permission denied], "
        "the operation FAILED. Fix the parameters and retry.\n"
        "3. For multi-step tasks (read → modify → test → commit), complete EACH step "
        "with a real tool call. Do NOT skip steps or summarize unexecuted work.\n"
        "4. Tool parameter names: use 'file_path' (not 'path'), 'command' (not 'cmd').\n"
        "5. PATHS: NEVER guess the home directory. Use ~/... for paths, or extract "
        "the real absolute path from previous Bash output. NEVER use /Users/user/, "
        "/Users/power/, /root/ etc. If unsure, run `echo $HOME` first.\n"
        "6. FIRST STEP: Your first action must be reading the project (ls, cat README). "
        "NEVER commit, push, or modify code before reading the project structure."
    )

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        """beta-reduction: (lambda_D x) -> claude -p (prompt + x)"""
        ctx = ctx or Context()
        t0 = time.time()

        # Append tool override only on first call (when no session exists yet).
        # With --resume, Claude already has the override in its session memory.
        should_inject = self.inject_override and self._session_id is None
        augmented_input = (
            str(input) + self._TOOL_OVERRIDE if should_inject else str(input)
        )

        if self.stream:
            raw = self._call_stream(augmented_input)
        else:
            raw = self._call_sync(augmented_input)

        duration = (time.time() - t0) * 1000
        result = self.output_parser(raw)
        ctx.log(
            self._name,
            self._trace_id,
            input,
            result,
            duration,
            f"claude-code/{self.model}",
        )
        return result

    def _build_cmd(self, for_json: bool = False) -> list[str]:
        """Build the claude CLI command.

        First call: creates new session with --system-prompt.
        Subsequent calls: --resume <session_id> to continue conversation.
        """
        output_format = "json" if for_json else "text"
        cmd = [
            self.claude_bin,
            "-p",
            "--output-format",
            output_format,
            "--model",
            self.model,
            "--tools",
            "",
        ]

        if self._session_id:
            # Resume existing session (Claude has full memory of previous steps)
            cmd += ["--resume", self._session_id]
        else:
            # First call: create new session with system prompt
            cmd += ["--system-prompt", self.prompt]

        # MCP isolation: prevent Claude from seeing session MCP tools
        cmd += ["--mcp-config", '{"mcpServers":{}}', "--strict-mcp-config"]
        return cmd

    def _call_stream(self, input_text: str) -> str:
        """Streaming call — reads stdout char-by-char."""
        try:
            # First call uses JSON to capture session_id, then switch to text
            is_first = self._session_id is None
            proc = subprocess.Popen(
                self._build_cmd(for_json=is_first),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            proc.stdin.write(input_text)
            proc.stdin.close()

            output_chars = []
            start = time.time()
            first_token = True
            chunk_buf = []
            CHUNK_FLUSH_SIZE = 20

            while True:
                char = proc.stdout.read(1)
                if not char:
                    break
                if time.time() - start > 600:
                    proc.kill()
                    return "[Claude Code timeout] request exceeded 600s"

                output_chars.append(char)

                if not is_first:
                    if self.on_chunk:
                        chunk_buf.append(char)
                        if len(chunk_buf) >= CHUNK_FLUSH_SIZE or char == "\n":
                            self.on_chunk("".join(chunk_buf))
                            chunk_buf = []
                    else:
                        if first_token:
                            sys.stdout.write("  \U0001f40f ")
                            first_token = False
                        sys.stdout.write(char)
                        sys.stdout.flush()

            if chunk_buf and self.on_chunk:
                self.on_chunk("".join(chunk_buf))

            proc.wait(timeout=5)

            if not first_token and not self.on_chunk:
                sys.stdout.write("\n")
                sys.stdout.flush()

            if proc.returncode != 0:
                stderr = proc.stderr.read().strip()
                if stderr:
                    return f"[Claude Code error] {stderr[:500]}"

            raw_output = "".join(output_chars).strip()

            # Extract session_id from first JSON call
            if is_first and raw_output:
                try:
                    data = json.loads(raw_output)
                    self._session_id = data.get("session_id")
                    return data.get("result", raw_output)
                except (json.JSONDecodeError, KeyError):
                    pass

            return raw_output if raw_output else "[no output]"

        except FileNotFoundError:
            return "[error] claude command not found."
        except Exception as e:
            return f"[error] {e}"

    def _call_sync(self, input_text: str) -> str:
        """Non-streaming call with session persistence."""
        try:
            # First call: use JSON output to capture session_id
            is_first = self._session_id is None

            result = subprocess.run(
                self._build_cmd(for_json=is_first),
                input=input_text,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                if stderr:
                    return f"[Claude Code error] {stderr[:500]}"
                return "[Claude Code error] unknown error"

            raw_output = result.stdout.strip()

            # Extract session_id from first JSON response
            if is_first and raw_output:
                try:
                    data = json.loads(raw_output)
                    self._session_id = data.get("session_id")
                    return data.get("result", raw_output)
                except (json.JSONDecodeError, KeyError):
                    pass

            return raw_output if raw_output else "[no output]"

        except subprocess.TimeoutExpired:
            return "[Claude Code timeout] request exceeded 600s"
        except FileNotFoundError:
            return "[error] claude command not found."
        except Exception as e:
            return f"[error] {e}"

    def reset_session(self):
        """Reset session — next call creates a new conversation."""
        self._session_id = None

    def __rshift__(self, other):
        from lambdagent.primitives import Compose

        return Compose(self, other)
