"""
lambdagent.hooks — Three-layer Hook system for agent execution

Layer 3: HookRegistry  — Machine-level observer (global, cross-cutting)
Layer 2: HookTerm      — Term-level wrapper (local, reusable policy)
Layer 1: pre_hook/post_hook/guard_hook — Decorators (inline, algebraic)

The three layers are orthogonal and can coexist:
  Layer 3 observes execution (infrastructure: audit, OTel, rate limiting)
  Layer 2 wraps specific terms (policy: PII filter, domain retry)
  Layer 1 composes inline (business: one-off transforms)

Formal basis (Paper10 §4):
  Hook = Term transformer H : Term → Term
  Pre-hook  = Compose(Tool(pre_fn), agent)       [C-Comp + C-Tool]
  Post-hook = Compose(agent, Tool(post_fn))       [C-CompRet + C-Tool]
  Guard-hook = Guard(agent, validator, k)         [C-Guard*]
  Event-hook = CEK transition observer            [all rules]
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .core import Term, Context
from ._shell_compat import run_shell as _run_shell


# ════════════════════════════════════════════════════════════
# Layer 3: HookRegistry — Machine-level event system
# ════════════════════════════════════════════════════════════


@dataclass
class HookRegistry:
    """
    Global hook registry. Callbacks fire at CEK transition points.

    Events (mapped to CEK rules):
      pre_llm        C-Lam before      (term, input, ctx)
      post_llm       C-Lam after       (term, input, output, usage, ctx)
      pre_tool       C-Tool before     (term, input, ctx)
      post_tool      C-Tool after      (term, input, output, duration_ms, ctx)
      on_guard_fail  C-GuardFail       (term, input, last_result, attempts, ctx)
      on_error       any exception     (term, input, error, ctx)
      on_step        every β-reduction (step, term_name, duration_ms, ctx)
      on_cancel      cancellation      (reason, ctx)

    post_llm and post_tool can modify output via output["value"] dict pattern.
    """

    pre_llm: List[Callable] = field(default_factory=list)
    post_llm: List[Callable] = field(default_factory=list)
    pre_tool: List[Callable] = field(default_factory=list)
    post_tool: List[Callable] = field(default_factory=list)
    on_guard_fail: List[Callable] = field(default_factory=list)
    on_error: List[Callable] = field(default_factory=list)
    on_step: List[Callable] = field(default_factory=list)
    on_cancel: List[Callable] = field(default_factory=list)

    def register(self, event: str, fn: Callable):
        """Register a hook callback for an event."""
        hooks = getattr(self, event, None)
        if hooks is None:
            raise ValueError(
                f"Unknown hook event: {event}. "
                f"Available: {', '.join(self._event_names())}"
            )
        hooks.append(fn)

    def unregister(self, event: str, fn: Callable):
        """Remove a hook callback."""
        hooks = getattr(self, event, None)
        if hooks and fn in hooks:
            hooks.remove(fn)

    def fire(self, event: str, **kwargs) -> None:
        """Fire all callbacks for an event. Errors in hooks are logged but not raised."""
        hooks = getattr(self, event, [])
        for fn in hooks:
            try:
                fn(**kwargs)
            except Exception as e:
                # Hook errors should not break agent execution
                import sys

                print(f"[HookError] {event}: {e}", file=sys.stderr)

    async def afire(self, event: str, **kwargs) -> None:
        """Async fire — runs sync hooks in order, awaits async hooks."""
        hooks = getattr(self, event, [])
        for fn in hooks:
            try:
                if asyncio.iscoroutinefunction(fn):
                    await fn(**kwargs)
                else:
                    fn(**kwargs)
            except Exception as e:
                import sys

                print(f"[HookError] {event}: {e}", file=sys.stderr)

    def _event_names(self) -> list:
        return [f.name for f in self.__dataclass_fields__.values()]

    def clear(self):
        """Remove all registered hooks."""
        for name in self._event_names():
            getattr(self, name).clear()

    def summary(self) -> str:
        parts = []
        for name in self._event_names():
            hooks = getattr(self, name)
            if hooks:
                parts.append(f"  {name}: {len(hooks)} hook(s)")
        if not parts:
            return "HookRegistry: (empty)"
        return "HookRegistry:\n" + "\n".join(parts)


# Singleton default registry
_default_registry = HookRegistry()


def get_default_registry() -> HookRegistry:
    return _default_registry


# ════════════════════════════════════════════════════════════
# Layer 2: HookTerm — First-class Lambda term
# ════════════════════════════════════════════════════════════


class HookTerm(Term):
    """
    Hook as a first-class Lambda term.

    Small-step rules (Paper10 extension):
      E-HookPre:  ⟨Hook(a,pre,post) v, E, K, σ⟩
                   →τ ⟨a v, E, HookK(post,v) :: K, σ⟩
                   (premise: pre(v) evaluated for side effects)

      E-HookPost: ⟨v', E, HookK(post,orig) :: K, σ⟩
                   →τ ⟨post(v'), E, K, σ⟩

    Usage:
        hooked = HookTerm(my_agent,
            pre=lambda x: print(f"Input: {x}"),
            post=lambda x: x.strip(),
            on_error=lambda e, x: f"Fallback for {e}",
        )
    """

    def __init__(
        self,
        agent: Term,
        pre: Callable | None = None,
        post: Callable | None = None,
        on_error: Callable | None = None,
        name: str = "",
    ):
        super().__init__(name or f"Hook({agent._name})")
        self.agent = agent
        self.pre = pre
        self.post = post
        self.on_error = on_error

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        """Synchronous β-reduction with pre/post hooks."""
        ctx = ctx or Context()
        t0 = time.time()

        # E-HookPre
        if self.pre:
            self.pre(input)

        # Agent execution
        try:
            result = self.agent.apply(input, ctx)
        except Exception as e:
            if self.on_error:
                return self.on_error(e, input)
            raise

        # E-HookPost
        if self.post:
            result = self.post(result)

        duration = (time.time() - t0) * 1000
        ctx.log(self._name, self._trace_id, input, result, duration)
        return result

    async def aapply(self, input: Any, ctx: Context | None = None, cancel=None) -> Any:
        """Async β-reduction with pre/post hooks."""
        from .cancellation import NullCancellationToken

        cancel = cancel or NullCancellationToken()
        ctx = ctx or Context()
        t0 = time.time()

        # E-HookPre
        if self.pre:
            if asyncio.iscoroutinefunction(self.pre):
                await self.pre(input)
            else:
                self.pre(input)

        cancel.check()

        # Agent execution
        try:
            result = await self.agent.aapply(input, ctx, cancel)
        except Exception as e:
            if self.on_error:
                return self.on_error(e, input)
            raise

        cancel.check()

        # E-HookPost
        if self.post:
            if asyncio.iscoroutinefunction(self.post):
                result = await self.post(result)
            else:
                result = self.post(result)

        duration = (time.time() - t0) * 1000
        ctx.log(self._name, self._trace_id, input, result, duration)
        return result


# ════════════════════════════════════════════════════════════
# Layer 1: Decorators — Syntactic sugar over Compose/Tool/Guard
# ════════════════════════════════════════════════════════════


def pre_hook(fn: Callable) -> Callable:
    """
    Decorator: run fn before agent execution (pass-through).

    Lambda: pre_hook(fn)(agent) = Compose(Tool(fn_passthrough), agent)
    """

    def wrapper(agent: Term) -> Term:
        from .primitives import Compose, Tool

        passthrough = Tool(f"pre_hook:{fn.__name__}", lambda x: (fn(x), x)[1])
        return Compose(passthrough, agent)

    return wrapper


def post_hook(fn: Callable) -> Callable:
    """
    Decorator: run fn after agent execution (can modify output).

    Lambda: post_hook(fn)(agent) = Compose(agent, Tool(fn))
    """

    def wrapper(agent: Term) -> Term:
        from .primitives import Compose, Tool

        transformer = Tool(f"post_hook:{fn.__name__}", fn)
        return Compose(agent, transformer)

    return wrapper


def guard_hook(
    predicate: Callable, retry: int = 0, on_fail: Callable = None
) -> Callable:
    """
    Decorator: validate output, retry on failure.

    Lambda: guard_hook(P, k)(agent) = Guard(agent, P, k)
    """

    def wrapper(agent: Term) -> Term:
        from .extensions import Guard

        return Guard(agent, predicate, retry=retry, on_fail=on_fail)

    return wrapper


# ════════════════════════════════════════════════════════════
# YAML Hook compilation helper
# ════════════════════════════════════════════════════════════


def compile_shell_hook(command: str, event: str) -> Callable:
    """
    Compile a shell command into a hook callback.

    The shell command receives context via environment variables:
      HOOK_EVENT, HOOK_TERM, HOOK_INPUT (first 500 chars)
    """
    import subprocess

    def hook_fn(**kwargs):
        env = {
            "HOOK_EVENT": event,
            "HOOK_TERM": str(getattr(kwargs.get("term"), "_name", "unknown")),
            "HOOK_INPUT": str(kwargs.get("input", ""))[:500],
        }
        if "output" in kwargs:
            val = kwargs["output"]
            if isinstance(val, dict) and "value" in val:
                env["HOOK_OUTPUT"] = str(val["value"])[:500]
            else:
                env["HOOK_OUTPUT"] = str(val)[:500]

        import os

        full_env = {**os.environ, **env}
        try:
            result = _run_shell(
                command,
                env=full_env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0 and result.stderr:
                import sys

                print(f"[ShellHook:{event}] {result.stderr.strip()}", file=sys.stderr)
        except subprocess.TimeoutExpired:
            import sys

            print(f"[ShellHook:{event}] timeout after 10s", file=sys.stderr)
        except Exception as e:
            import sys

            print(f"[ShellHook:{event}] error: {e}", file=sys.stderr)

    return hook_fn


def compile_hooks_from_config(hooks_cfg: dict) -> HookRegistry:
    """
    Compile YAML hooks config into a HookRegistry.

    YAML format:
      hooks:
        pre_tool:
          - command: "python validate.py"
        post_llm:
          - command: "python audit.py"
    """
    registry = HookRegistry()
    if not hooks_cfg:
        return registry

    for event, hook_list in hooks_cfg.items():
        if not isinstance(hook_list, list):
            continue
        for hook_def in hook_list:
            if isinstance(hook_def, dict) and "command" in hook_def:
                fn = compile_shell_hook(hook_def["command"], event)
                registry.register(event, fn)

    return registry
