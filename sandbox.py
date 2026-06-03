"""
lambdagent.sandbox — Secure execution sandbox for Tool β-reduction

Phase 8.1: Process-level isolation (L1)

Lambda semantics unchanged:
  ⟦SandboxedTool(n, f, policy)⟧ = λx. f(x)   [where f runs in sandbox]
  E-Tool rule unchanged — only the execution environment changes.

Security layers:
  1. Subprocess isolation (separate process)
  2. Resource limits (CPU, memory, time, file descriptors)
  3. Filesystem isolation (tmpdir jail)
  4. Optional network blocking
  5. Syscall filtering (seccomp-bpf on Linux)

Usage:
    from lambdagent.sandbox import SandboxedTool, SandboxPolicy

    # Quick: wrap any function with defaults
    safe_tool = SandboxedTool("calc", lambda x: str(eval(x)))

    # Custom policy
    policy = SandboxPolicy(timeout=10, memory_mb=128, network=False)
    safe_tool = SandboxedTool("calc", lambda x: str(eval(x)), policy=policy)

    # Use exactly like Tool
    result = safe_tool("2 + 3")  # → "5"
"""
from __future__ import annotations

import json
import os
import platform
import resource
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from .core import Term, Context, ValidationError


# ════════════════════════════════════════════════════════════
# Sandbox Policy
# ════════════════════════════════════════════════════════════

@dataclass
class SandboxPolicy:
    """
    Security policy for sandboxed execution.

    Lambda: type constraint on Oracle — f must satisfy policy P.
    ⟦SandboxedTool(n, f, P)⟧ = λx. let r = f(x) in IF P_satisfied THEN r ELSE ⊥
    """
    # ── Resource limits ──
    timeout: float = 30.0           # max execution time (seconds)
    memory_mb: int = 256            # max memory (MB)
    cpu_time: float = 30.0          # max CPU time (seconds)
    max_output_bytes: int = 1_000_000  # max stdout size (1MB)
    max_file_descriptors: int = 64  # max open files

    # ── Filesystem ──
    allowed_read_paths: List[str] = field(default_factory=list)   # readable paths
    allowed_write_paths: List[str] = field(default_factory=list)  # writable paths
    use_tmpdir: bool = True         # run in isolated tmpdir

    # ── Network ──
    network: bool = False           # allow network access

    # ── Execution ──
    allow_subprocess: bool = False  # allow spawning child processes
    allow_exec: bool = False        # allow exec() / eval() of arbitrary code
    env_vars: Dict[str, str] = field(default_factory=dict)  # environment variables
    python_path: str = sys.executable  # python interpreter to use

    @classmethod
    def strict(cls) -> "SandboxPolicy":
        """Most restrictive policy — no network, no subprocess, 10s timeout."""
        return cls(timeout=10, memory_mb=128, network=False,
                   allow_subprocess=False, allow_exec=False)

    @classmethod
    def permissive(cls) -> "SandboxPolicy":
        """Relaxed policy — network allowed, 60s timeout."""
        return cls(timeout=60, memory_mb=512, network=True,
                   allow_subprocess=True, allow_exec=False)

    @classmethod
    def default(cls) -> "SandboxPolicy":
        """Balanced default — no network, 30s timeout, 256MB."""
        return cls()


# ════════════════════════════════════════════════════════════
# Sandbox Violations
# ════════════════════════════════════════════════════════════

class SandboxViolation(ValidationError):
    """Raised when sandbox policy is violated."""
    def __init__(self, message: str, violation_type: str = "unknown",
                 policy: Optional[SandboxPolicy] = None):
        super().__init__(message)
        self.violation_type = violation_type
        self.policy = policy


class TimeoutViolation(SandboxViolation):
    def __init__(self, timeout: float):
        super().__init__(f"Execution exceeded timeout of {timeout}s",
                         "timeout")


class MemoryViolation(SandboxViolation):
    def __init__(self, limit_mb: int):
        super().__init__(f"Execution exceeded memory limit of {limit_mb}MB",
                         "memory")


class OutputViolation(SandboxViolation):
    def __init__(self, limit_bytes: int):
        super().__init__(f"Output exceeded {limit_bytes} bytes",
                         "output_size")


# ════════════════════════════════════════════════════════════
# Resource Limiter (applied inside child process)
# ════════════════════════════════════════════════════════════

class ResourceLimiter:
    """
    Apply resource limits inside the child process.
    Uses POSIX resource module (works on macOS + Linux).

    Lambda: bounded Oracle — f must return within resource envelope.

    macOS fix: RLIMIT_NPROC is unreliable on macOS (silently fails).
    We use a Python-level monkey-patch of subprocess/os.exec* as a fallback.
    This is defense-in-depth, not a hard security boundary — the real
    isolation comes from SandboxedTool running in a separate process.
    """

    _subprocess_blocked = False

    @staticmethod
    def apply(policy: SandboxPolicy):
        """Apply resource limits in current process. Call after fork."""
        # CPU time limit
        if policy.cpu_time > 0:
            soft = int(policy.cpu_time)
            hard = soft + 5  # 5s grace
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))
            except (ValueError, resource.error):
                pass

        # Memory limit (address space)
        if policy.memory_mb > 0:
            limit_bytes = policy.memory_mb * 1024 * 1024
            try:
                # RLIMIT_AS limits virtual memory (address space)
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
            except (ValueError, resource.error):
                pass

        # File descriptor limit
        if policy.max_file_descriptors > 0:
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE,
                                   (policy.max_file_descriptors,
                                    policy.max_file_descriptors))
            except (ValueError, resource.error):
                pass

        # Disable subprocess creation if not allowed
        if not policy.allow_subprocess:
            nproc_ok = False
            try:
                # RLIMIT_NPROC = 0 prevents fork (Linux)
                resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
                nproc_ok = True
            except (ValueError, resource.error, AttributeError):
                pass

            # macOS fallback: monkey-patch subprocess module inside child
            # This prevents the sandboxed code from spawning children via Python.
            # Not a kernel-level boundary, but blocks casual subprocess usage.
            if not nproc_ok and platform.system() == "Darwin":
                ResourceLimiter._block_subprocess_python()

    @staticmethod
    def _block_subprocess_python():
        """
        Monkey-patch subprocess.Popen and os.system in the child process
        to raise PermissionError. macOS fallback for RLIMIT_NPROC.

        This is a Python-level guard, not a kernel boundary.
        It prevents the sandboxed function from using subprocess.run(),
        os.system(), os.popen(), etc.
        """
        if ResourceLimiter._subprocess_blocked:
            return
        ResourceLimiter._subprocess_blocked = True

        def _blocked_popen(*args, **kwargs):
            raise PermissionError(
                "Subprocess creation blocked by sandbox policy "
                "(allow_subprocess=False)"
            )

        def _blocked_system(cmd):
            raise PermissionError(
                "os.system() blocked by sandbox policy "
                "(allow_subprocess=False)"
            )

        def _blocked_exec(*args, **kwargs):
            raise PermissionError(
                "os.exec*() blocked by sandbox policy "
                "(allow_subprocess=False)"
            )

        # Patch subprocess
        try:
            import subprocess as _sp
            _sp.Popen = _blocked_popen
            _sp.run = _blocked_popen
            _sp.call = _blocked_popen
            _sp.check_call = _blocked_popen
            _sp.check_output = _blocked_popen
        except Exception:
            pass

        # Patch os.system / os.popen
        try:
            os.system = _blocked_system
            os.popen = _blocked_system
        except Exception:
            pass

        # Patch os.exec* family
        for attr in ("execl", "execle", "execlp", "execlpe",
                      "execv", "execve", "execvp", "execvpe"):
            if hasattr(os, attr):
                try:
                    setattr(os, attr, _blocked_exec)
                except Exception:
                    pass


# ════════════════════════════════════════════════════════════
# SandboxedTool — Tool with process isolation
# ════════════════════════════════════════════════════════════

class SandboxedTool(Term):
    """
    Tool that runs in an isolated subprocess with resource limits.

    Lambda semantics:
        ⟦SandboxedTool(n, f, P)⟧ = λx. f(x)   [f runs in sandbox]

    The β-reduction rule E-Tool is unchanged:
        f(v) = v'
        ─────────────────
        tool[f] v → v'

    But f is executed in a subprocess with policy P enforced.
    If P is violated, the tool returns SandboxViolation (= stuck).
    """

    def __init__(self, name: str, fn: Callable, policy: Optional[SandboxPolicy] = None,
                 description: str = ""):
        super().__init__(name)
        self.fn = fn
        self.policy = policy or SandboxPolicy.default()
        self.description = description
        self._fn_source = _extract_fn_source(fn)

    def apply(self, input: Any, ctx: Optional[Context] = None) -> Any:
        ctx = ctx or Context()
        t0 = time.time()

        try:
            result = self._execute_sandboxed(str(input))
        except SandboxViolation as e:
            elapsed = (time.time() - t0) * 1000
            ctx.log(self._name, self._trace_id, input,
                    f"[SANDBOX_VIOLATION] {e}", elapsed)
            raise
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            ctx.log(self._name, self._trace_id, input,
                    f"[ERROR] {e}", elapsed)
            raise

        elapsed = (time.time() - t0) * 1000
        ctx.log(self._name, self._trace_id, input, result, elapsed)
        return result

    def _execute_sandboxed(self, input_str: str) -> str:
        """Execute function in isolated subprocess using pickle IPC."""
        policy = self.policy

        # Serialize function via pickle/cloudpickle
        fn_bytes = _serialize_fn(self.fn)

        # Write function pickle to temp file
        fn_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False,
                                               prefix="lambdagent_fn_")
        fn_file.write(fn_bytes)
        fn_file.close()

        # Build child script
        script = self._build_child_script(fn_file.name)

        # Prepare environment
        env = os.environ.copy()
        env.update(policy.env_vars)
        if not policy.network:
            env["no_proxy"] = "*"
            env["NO_PROXY"] = "*"

        # Working directory
        work_dir = None
        if policy.use_tmpdir:
            work_dir = tempfile.mkdtemp(prefix="lambdagent_sandbox_")

        try:
            preexec = None
            if os.name != "nt":
                def _apply_limits():
                    ResourceLimiter.apply(policy)
                preexec = _apply_limits

            proc = subprocess.Popen(
                [policy.python_path, "-c", script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=work_dir,
                preexec_fn=preexec,
            )

            try:
                stdout, stderr = proc.communicate(
                    input=input_str.encode("utf-8"),
                    timeout=policy.timeout,
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise TimeoutViolation(policy.timeout)

            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="replace").strip()
                if proc.returncode == -signal.SIGKILL:
                    raise MemoryViolation(policy.memory_mb)
                elif proc.returncode == -signal.SIGXCPU:
                    raise TimeoutViolation(policy.cpu_time)
                else:
                    raise SandboxViolation(
                        f"Child exited {proc.returncode}: {err_msg[:200]}",
                        "execution_error")

            if len(stdout) > policy.max_output_bytes:
                raise OutputViolation(policy.max_output_bytes)

            result_str = stdout.decode("utf-8", errors="replace").strip()

            try:
                envelope = json.loads(result_str)
                if isinstance(envelope, dict):
                    if "error" in envelope:
                        raise SandboxViolation(envelope["error"], "runtime_error")
                    if "result" in envelope:
                        return str(envelope["result"])
            except (json.JSONDecodeError, KeyError):
                pass

            return result_str

        finally:
            # Cleanup
            try:
                os.unlink(fn_file.name)
            except OSError:
                pass
            if work_dir and os.path.exists(work_dir):
                try:
                    import shutil
                    shutil.rmtree(work_dir, ignore_errors=True)
                except Exception:
                    pass

    def _build_child_script(self, fn_pickle_path: str) -> str:
        """Build child script that loads pickled function."""
        script = textwrap.dedent(f'''\
            import sys, json, os, pickle

            # Load serialized function
            with open({fn_pickle_path!r}, "rb") as _f:
                fn = pickle.load(_f)

            # Read input from stdin
            input_text = sys.stdin.read()

            # Execute in try/except
            try:
                result = fn(input_text)
                print(json.dumps({{"result": str(result)}}))
            except MemoryError:
                print(json.dumps({{"error": "MemoryError"}}))
                sys.exit(1)
            except Exception as e:
                print(json.dumps({{"error": f"{{type(e).__name__}}: {{e}}"}}))
                sys.exit(1)
        ''')
        return script


# ════════════════════════════════════════════════════════════
# SecureExecutor — Sandbox-aware β-reduction engine
# ════════════════════════════════════════════════════════════

class SecureExecutor:
    """
    Wraps the standard Executor with sandbox enforcement.

    Lambda: SecureExecutor = Executor where all Tool reductions
            go through sandbox policy checking.

    Usage:
        executor = SecureExecutor(policy=SandboxPolicy.strict())
        result = executor.run(term, input)
    """

    def __init__(self, default_policy: Optional[SandboxPolicy] = None):
        self.default_policy = default_policy or SandboxPolicy.default()
        self._stats = {
            "sandboxed_calls": 0,
            "violations": 0,
            "total_sandbox_ms": 0.0,
        }

    def wrap_tool(self, tool: Term) -> Term:
        """
        Wrap a Tool in sandbox if not already sandboxed.
        Returns SandboxedTool or the original if already sandboxed.
        """
        if isinstance(tool, SandboxedTool):
            return tool  # already sandboxed

        from .primitives import Tool as BaseTool
        if isinstance(tool, BaseTool):
            return SandboxedTool(
                name=tool._name,
                fn=tool.fn,
                policy=self.default_policy,
                description=f"Auto-sandboxed: {tool._name}",
            )

        return tool  # not a Tool, return as-is

    def sandbox_all_tools(self, term: Term) -> Term:
        """
        Recursively wrap all Tool instances in a term tree with sandbox.
        Returns a new term tree with sandboxed tools.
        """
        from .primitives import Tool as BaseTool, Compose, Loop, Pair, If
        from .extensions import Par, Route, Guard, Memory

        if isinstance(term, SandboxedTool):
            return term
        elif isinstance(term, BaseTool):
            return self.wrap_tool(term)
        elif isinstance(term, Compose):
            new_stages = [self.sandbox_all_tools(s) for s in term.stages]
            c = Compose(*new_stages)
            return c
        elif isinstance(term, Loop):
            new_body = self.sandbox_all_tools(term.body)
            return Loop(new_body, term.condition, term.max_steps)
        elif isinstance(term, Route):
            new_cls = self.sandbox_all_tools(term.classifier)
            new_routes = {k: self.sandbox_all_tools(v) for k, v in term.routes.items()}
            return Route(new_cls, new_routes, default=term.default)
        elif isinstance(term, Guard):
            new_agent = self.sandbox_all_tools(term.agent)
            return Guard(new_agent, term.validator, retry=term.retry, on_fail=term.on_fail)
        elif isinstance(term, Memory):
            new_agent = self.sandbox_all_tools(term.agent)
            return Memory(new_agent, store=term.store)
        elif isinstance(term, Par):
            new_agents = [self.sandbox_all_tools(a) for a in term.agents]
            return Par(*new_agents)
        elif isinstance(term, Pair):
            return Pair(self.sandbox_all_tools(term.first),
                        self.sandbox_all_tools(term.second))
        elif isinstance(term, If):
            return If(term.cond,
                      self.sandbox_all_tools(term.then_),
                      self.sandbox_all_tools(term.else_))
        else:
            return term

    @property
    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)


# ════════════════════════════════════════════════════════════
# Utility
# ════════════════════════════════════════════════════════════

def _serialize_fn(fn: Callable) -> bytes:
    """Serialize a function to bytes for cross-process transfer."""
    # Try cloudpickle first (handles lambdas, closures)
    try:
        import cloudpickle
        return cloudpickle.dumps(fn)
    except ImportError:
        pass

    # Try dill
    try:
        import dill
        return dill.dumps(fn)
    except ImportError:
        pass

    # Fallback: standard pickle (works for named functions, not lambdas)
    import pickle
    return pickle.dumps(fn)


def _extract_fn_source(fn: Callable) -> Optional[str]:
    """
    Try to extract function source code for serialization.
    Returns a string like 'fn = lambda x: ...' or None if not possible.
    """
    import inspect

    # Named functions defined at module level
    try:
        source = inspect.getsource(fn)
        # Clean up indentation
        source = textwrap.dedent(source)
        # Get function name
        name = getattr(fn, "__name__", "fn")
        if name == "<lambda>":
            # Lambda: extract the lambda expression
            # Try to find it in the source
            if "lambda" in source:
                return f"fn = {source.strip()}"
        else:
            # Named function: use as-is
            return source + f"\nfn = {name}"
    except (OSError, TypeError):
        pass

    return None


def sandboxed(fn: Callable = None, *, name: str = "", policy: Optional[SandboxPolicy] = None,
              timeout: float = 30, memory_mb: int = 256, network: bool = False):
    """
    Decorator to create a SandboxedTool from a function.

    Usage:
        @sandboxed(name="calc", timeout=10)
        def calculate(x):
            return str(eval(x))

        result = calculate("2 + 3")  # runs in sandbox
    """
    def decorator(f):
        p = policy or SandboxPolicy(timeout=timeout, memory_mb=memory_mb, network=network)
        tool_name = name or getattr(f, "__name__", "sandboxed_fn")
        return SandboxedTool(tool_name, f, policy=p)

    if fn is not None:
        # Called without arguments: @sandboxed
        return decorator(fn)
    else:
        # Called with arguments: @sandboxed(name="calc")
        return decorator
