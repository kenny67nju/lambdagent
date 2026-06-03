"""
lambdagent.builtin_tools.shell_tools — Enhanced shell execution

Bash     λx. exec(command)  — persistent CWD, background, timeout
Git*     λx. git(args)      — status, diff, log, commit, branch
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
from typing import Any, Dict, Optional


# Shared session CWD (persists across calls)
_session_cwd = os.getcwd()
_cwd_lock = threading.Lock()

# Interactive commands that should be rejected
_INTERACTIVE_CMDS = {"vim", "vi", "nano", "emacs", "less", "more", "top", "htop",
                     "ssh", "telnet", "ftp", "python", "node", "irb", "ghci"}


def _get_cwd() -> str:
    with _cwd_lock:
        return _session_cwd


def _set_cwd(path: str):
    global _session_cwd
    with _cwd_lock:
        _session_cwd = path


# ════════════════════════════════════════════════════════════
# A06: Bash
# ════════════════════════════════════════════════════════════

class BashSchema:
    def __init__(self, command: str, timeout: int = 120, working_dir: str = "",
                 run_in_background: bool = False):
        if not command or not isinstance(command, str):
            raise ValueError("command is required")
        if timeout < 1 or timeout > 600:
            raise ValueError("timeout must be between 1 and 600 seconds")
        # Reject interactive commands
        first_word = command.strip().split()[0].split("/")[-1] if command.strip() else ""
        if first_word in _INTERACTIVE_CMDS:
            raise ValueError(f"Interactive command '{first_word}' is not supported. Use non-interactive alternatives.")
        self.command = command
        self.timeout = timeout
        self.working_dir = working_dir
        self.run_in_background = run_in_background

    def dict(self):
        return {"command": self.command, "timeout": self.timeout,
                "working_dir": self.working_dir, "run_in_background": self.run_in_background}


def run_bash(input_val: Any) -> str:
    """Execute shell command with persistent CWD, timeout, background support."""
    params = _parse_input(input_val)
    command = params.get("command", "")
    if not command:
        raise ValueError("command is required")
    # Reject interactive commands
    first_word = command.strip().split()[0].split("/")[-1] if command.strip() else ""
    if first_word in _INTERACTIVE_CMDS:
        raise ValueError(f"Interactive command '{first_word}' is not supported. Use non-interactive alternatives.")
    timeout = params.get("timeout", 120)
    working_dir = params.get("working_dir", "") or _get_cwd()
    background = params.get("run_in_background", False)

    # Handle pure cd commands — update session CWD
    # Only intercept simple "cd <path>", not chained commands like "cd x && git init"
    stripped = command.strip()
    is_pure_cd = stripped.startswith("cd ") and not any(op in stripped for op in ["&&", "||", ";", "|", "\n"])
    if is_pure_cd:
        target = stripped[3:].strip().strip("'\"")
        if not os.path.isabs(target):
            target = os.path.join(working_dir, target)
        target = os.path.realpath(target)
        if os.path.isdir(target):
            _set_cwd(target)
            return f"[OK] Changed directory to {target}"
        else:
            return f"[ERROR] Directory not found: {target}"

    env = os.environ.copy()
    env["HOME"] = os.path.expanduser("~")

    if background:
        # Background execution
        try:
            proc = subprocess.Popen(
                command, shell=True, cwd=working_dir, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            return f"[BACKGROUND] PID={proc.pid}, command='{command[:60]}'"
        except Exception as e:
            return f"[ERROR] Failed to start background process: {e}"

    # Foreground execution
    try:
        result = subprocess.run(
            command, shell=True, cwd=working_dir, env=env,
            capture_output=True, text=True, timeout=timeout,
        )
        stdout = result.stdout
        stderr = result.stderr

        # Smart truncation: keep head + tail
        max_len = 50000
        if len(stdout) > max_len:
            head = stdout[:max_len // 2]
            tail = stdout[-(max_len // 2):]
            stdout = f"{head}\n\n... [{len(result.stdout) - max_len} chars truncated] ...\n\n{tail}"

        output = stdout
        if result.returncode != 0:
            if stderr:
                output = f"{stdout}\n[STDERR] {stderr.strip()}" if stdout else f"[STDERR] {stderr.strip()}"
            output = f"[EXIT:{result.returncode}] {output}"

        return output.rstrip() if output.strip() else f"[OK] (no output, exit code {result.returncode})"

    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] Command timed out after {timeout}s: {command[:80]}"
    except Exception as e:
        return f"[ERROR] {e}"


# ════════════════════════════════════════════════════════════
# A07: Git tools
# ════════════════════════════════════════════════════════════

def _git(args: list, cwd: str = "") -> str:
    """Run git command and return output."""
    cwd = cwd or _get_cwd()
    try:
        result = subprocess.run(
            ["git"] + args, cwd=cwd,
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()
            return f"[GIT_ERROR] {err}" if err else f"[GIT_ERROR] exit code {result.returncode}"
        return output or "[OK] (no output)"
    except FileNotFoundError:
        return "[ERROR] git not found. Install git first."
    except subprocess.TimeoutExpired:
        return "[TIMEOUT] git command timed out"


def git_status(input_val: Any) -> str:
    """Git status + untracked files."""
    return _git(["status", "--short"])


def git_diff(input_val: Any) -> str:
    """Git diff (staged + unstaged)."""
    params = _parse_input(input_val) if isinstance(input_val, (dict, str)) and input_val else {}
    staged = params.get("staged", False)
    file_path = params.get("file_path", "")

    args = ["diff"]
    if staged:
        args.append("--cached")
    if file_path:
        args.extend(["--", file_path])
    return _git(args)


def git_log(input_val: Any) -> str:
    """Git log (recent commits)."""
    params = _parse_input(input_val) if isinstance(input_val, (dict, str)) and input_val else {}
    n = params.get("n", 10)
    oneline = params.get("oneline", True)

    args = ["log", f"-{n}"]
    if oneline:
        args.append("--oneline")
    return _git(args)


def git_commit(input_val: Any) -> str:
    """Stage changed files and commit."""
    params = _parse_input(input_val)
    message = params.get("message", "")
    files = params.get("files", [])

    if not message:
        return "[ERROR] commit message is required"

    # Stage files
    if files:
        for f in files:
            _git(["add", f])
    else:
        # Stage all modified (not untracked)
        _git(["add", "-u"])

    return _git(["commit", "-m", message])


def git_branch(input_val: Any) -> str:
    """List, create, or switch branches."""
    params = _parse_input(input_val) if isinstance(input_val, (dict, str)) and input_val else {}
    action = params.get("action", "list")
    name = params.get("name", "")

    if action == "list":
        return _git(["branch", "-a"])
    elif action == "create" and name:
        return _git(["checkout", "-b", name])
    elif action == "switch" and name:
        return _git(["checkout", name])
    else:
        return _git(["branch", "-a"])


# ════════════════════════════════════════════════════════════
# Shared
# ════════════════════════════════════════════════════════════

def _parse_input(input_val: Any) -> dict:
    if isinstance(input_val, dict):
        # Extract nested "input" from ReAct format: {"action":"Bash","input":{...}}
        if "input" in input_val and ("action" in input_val or "tool" in input_val):
            inner = input_val["input"]
            return json.loads(inner) if isinstance(inner, str) else inner if isinstance(inner, dict) else {"command": str(inner)}
        return input_val
    if isinstance(input_val, str):
        try:
            parsed = json.loads(input_val)
            if isinstance(parsed, dict):
                return _parse_input(parsed)  # recurse to handle nested format
            return {"command": input_val}
        except (json.JSONDecodeError, ValueError):
            return {"command": input_val}
    return {}
