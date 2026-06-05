"""
lambdagent.async_core — Async extensions for Lambda calculus Agent DSL

Adds async aapply() to all Term types for non-blocking β-reduction.
Preserves full backward compatibility with synchronous Term.apply().

Usage:
    # Sync (unchanged)
    result = agent.apply(input, ctx)

    # Async (new)
    result = await agent.aapply(input, ctx)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from .core import Term, Context, TraceEntry
from .cancellation import CancellationToken, NullCancellationToken


# ════════════════════════════════════════════════════════════
# Stream event types
# ════════════════════════════════════════════════════════════


class StreamEvent:
    """Base class for streaming events during async reduction."""

    __slots__ = ("type", "data", "term_name", "step")

    def __init__(
        self, type: str, data: Any = None, term_name: str = "", step: int = -1
    ):
        self.type = type
        self.data = data
        self.term_name = term_name
        self.step = step

    def __repr__(self):
        return f"StreamEvent({self.type!r}, {str(self.data)[:40]!r})"


class TokenEvent(StreamEvent):
    """A single token from LLM streaming."""

    def __init__(self, token: str, term_name: str = ""):
        super().__init__("token", token, term_name)
        self.token = token


class StepEvent(StreamEvent):
    """A completed reduction step."""

    def __init__(
        self, result: Any, term_name: str = "", step: int = -1, duration_ms: float = 0
    ):
        super().__init__("step", result, term_name, step)
        self.result = result
        self.duration_ms = duration_ms


class ErrorEvent(StreamEvent):
    """An error during reduction."""

    def __init__(self, error: Exception, term_name: str = ""):
        super().__init__("error", error, term_name)
        self.error = error


# ════════════════════════════════════════════════════════════
# Async mixin for Term — monkey-patched onto base Term class
# ════════════════════════════════════════════════════════════


async def _term_aapply(
    self,
    input: Any,
    ctx: Context | None = None,
    cancel: CancellationToken | None = None,
) -> Any:
    """
    Async β-reduction. Default implementation wraps sync apply() in a thread.
    Subclass-specific async versions are patched below.
    """
    cancel = cancel or NullCancellationToken()
    cancel.check()
    ctx = ctx or Context()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self.apply, input, ctx)


# Patch onto Term base class
Term.aapply = _term_aapply


# ════════════════════════════════════════════════════════════
# Async implementations for each Term type
# ════════════════════════════════════════════════════════════


async def _lam_aapply(
    self,
    input: Any,
    ctx: Context | None = None,
    cancel: CancellationToken | None = None,
) -> Any:
    """Async Lam: LLM call in thread pool (SDK clients are sync)."""
    cancel = cancel or NullCancellationToken()
    cancel.check()
    ctx = ctx or Context()
    t0 = time.time()
    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(None, self._call_llm, str(input))
    cancel.check()
    duration = (time.time() - t0) * 1000
    result = self.output_parser(raw)
    ctx.log(self._name, self._trace_id, input, result, duration, self.model)
    return result


async def _compose_aapply(
    self,
    input: Any,
    ctx: Context | None = None,
    cancel: CancellationToken | None = None,
) -> Any:
    """Async Compose: sequential async reduction."""
    cancel = cancel or NullCancellationToken()
    ctx = ctx or Context()
    result = input
    for stage in self.stages:
        cancel.check()
        result = await stage.aapply(result, ctx, cancel)
    return result


async def _if_aapply(
    self,
    input: Any,
    ctx: Context | None = None,
    cancel: CancellationToken | None = None,
) -> Any:
    """Async If: async condition + branch."""
    cancel = cancel or NullCancellationToken()
    ctx = ctx or Context()
    if isinstance(self.cond, Term):
        cond_result = await self.cond.aapply(input, ctx, cancel)
        from .primitives import If

        branch = If._is_truthy(cond_result)
    else:
        branch = self.cond(input)
    if branch:
        return await self.then_.aapply(input, ctx, cancel)
    else:
        return await self.else_.aapply(input, ctx, cancel)


async def _loop_aapply(
    self,
    input: Any,
    ctx: Context | None = None,
    cancel: CancellationToken | None = None,
) -> Any:
    """Async Loop: Y combinator with cancellation."""
    cancel = cancel or NullCancellationToken()
    ctx = ctx or Context()
    result = input
    for step in range(self.max_steps):
        cancel.check()
        result = await self.body.aapply(result, ctx, cancel)
        if self.condition(result, step):
            break
    return result


async def _par_aapply(
    self,
    input: Any,
    ctx: Context | None = None,
    cancel: CancellationToken | None = None,
) -> tuple:
    """Async Par: true parallel via asyncio.gather."""
    cancel = cancel or NullCancellationToken()
    cancel.check()
    ctx = ctx or Context()
    tasks = [agent.aapply(input, ctx, cancel.child()) for agent in self.agents]
    return tuple(await asyncio.gather(*tasks))


async def _pair_aapply(
    self,
    input: Any,
    ctx: Context | None = None,
    cancel: CancellationToken | None = None,
) -> tuple:
    """Async Pair: parallel execution of both sides."""
    cancel = cancel or NullCancellationToken()
    ctx = ctx or Context()
    a, b = await asyncio.gather(
        self.first.aapply(input, ctx, cancel.child()),
        self.second.aapply(input, ctx, cancel.child()),
    )
    return (a, b)


async def _tool_aapply(
    self,
    input: Any,
    ctx: Context | None = None,
    cancel: CancellationToken | None = None,
) -> Any:
    """Async Tool: run fn in thread pool."""
    cancel = cancel or NullCancellationToken()
    cancel.check()
    ctx = ctx or Context()
    t0 = time.time()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, self.fn, input)
    duration = (time.time() - t0) * 1000
    ctx.log(self._name, self._trace_id, input, result, duration)
    return result


async def _route_aapply(
    self,
    input: Any,
    ctx: Context | None = None,
    cancel: CancellationToken | None = None,
) -> Any:
    """Async Route: async classify then async dispatch."""
    cancel = cancel or NullCancellationToken()
    ctx = ctx or Context()
    label = str(await self.classifier.aapply(input, ctx, cancel)).strip().lower()
    agent = self.routes.get(label)
    if agent is None:
        for key, val in self.routes.items():
            if key.lower() in label:
                agent = val
                break
    if agent is None:
        agent = self.default
    if agent is None:
        from .core import RouteError

        raise RouteError(
            f"No route for '{label}'. Available: {list(self.routes.keys())}"
        )
    return await agent.aapply(input, ctx, cancel)


async def _memory_aapply(
    self,
    input: Any,
    ctx: Context | None = None,
    cancel: CancellationToken | None = None,
) -> Any:
    """Async Memory: inject memory then async reduce."""
    cancel = cancel or NullCancellationToken()
    ctx = ctx or Context()
    if self.store:
        memory_str = "\n".join(f"- {k}: {v}" for k, v in self.store.items())
        augmented = f"[Memory]\n{memory_str}\n\n[Input]\n{input}"
    else:
        augmented = str(input)
    return await self.agent.aapply(augmented, ctx, cancel)


async def _guard_aapply(
    self,
    input: Any,
    ctx: Context | None = None,
    cancel: CancellationToken | None = None,
) -> Any:
    """Async Guard: async execute + validate + retry."""
    cancel = cancel or NullCancellationToken()
    ctx = ctx or Context()
    last_result = None
    for attempt in range(1 + self.retry):
        cancel.check()
        result = await self.agent.aapply(input, ctx, cancel)
        last_result = result
        if isinstance(self.validator, Term):
            valid = await self.validator.aapply(result, ctx, cancel)
            from .primitives import If

            valid = If._is_truthy(valid) if isinstance(valid, str) else bool(valid)
        else:
            valid = self.validator(result)
        if valid:
            return result
    if self.on_fail:
        return self.on_fail(last_result)
    from .core import ValidationError

    raise ValidationError(
        f"Guard({self.agent._name}) failed after {1 + self.retry} attempts. "
        f"Last output: {last_result}"
    )


async def _groupchat_aapply(
    self,
    input: Any,
    ctx: Context | None = None,
    cancel: CancellationToken | None = None,
) -> Any:
    """Async GroupChat: async speaker selection + execution."""
    cancel = cancel or NullCancellationToken()
    ctx = ctx or Context()
    state = str(input)
    conversation = []
    t0_total = time.time()

    for round_idx in range(self.max_rounds):
        cancel.check()
        speaker = self._select_speaker(round_idx, state, ctx)

        if conversation:
            history = "\n".join(
                f"[{msg['speaker']}]: {msg['content']}" for msg in conversation
            )
            speaker_input = (
                f"[Conversation History]\n{history}\n\n"
                f"[Original Task]\n{input}\n\nYour turn to speak:"
            )
        else:
            speaker_input = state

        t0 = time.time()
        response = await speaker.aapply(speaker_input, ctx, cancel)
        elapsed = (time.time() - t0) * 1000

        conversation.append(
            {
                "speaker": speaker._name,
                "content": str(response),
                "round": round_idx,
            }
        )
        state = f"{state}\n[{speaker._name}]: {response}"
        ctx.log(
            f"GroupChat.round[{round_idx}]:{speaker._name}",
            self._trace_id,
            speaker_input[:100],
            str(response)[:100],
            elapsed,
        )
        if self.termination(state, round_idx + 1):
            break

    if self.summary_agent:
        summary = await self.summary_agent.aapply(state, ctx, cancel)
        return summary

    return conversation[-1]["content"] if conversation else state


async def _asyncpar_aapply(
    self,
    input: Any,
    ctx: Context | None = None,
    cancel: CancellationToken | None = None,
) -> tuple:
    """Async AsyncPar: true parallel via asyncio.gather (replaces ThreadPoolExecutor)."""
    cancel = cancel or NullCancellationToken()
    cancel.check()
    ctx = ctx or Context()
    t0 = time.time()

    tasks = [agent.aapply(input, ctx, cancel.child()) for agent in self.agents]

    if self.timeout:
        results = tuple(
            await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=self.timeout,
            )
        )
    else:
        results = tuple(await asyncio.gather(*tasks))

    elapsed = (time.time() - t0) * 1000
    ctx.log(
        self._name,
        self._trace_id,
        str(input)[:100],
        f"{len(results)} results in {elapsed:.0f}ms",
        elapsed,
    )
    return results


async def _handoff_aapply(
    self,
    input: Any,
    ctx: Context | None = None,
    cancel: CancellationToken | None = None,
) -> Any:
    """Async Handoff: async selector + async dispatch."""
    cancel = cancel or NullCancellationToken()
    ctx = ctx or Context()
    t0 = time.time()

    if isinstance(self.selector, Term):
        target_name = str(await self.selector.aapply(input, ctx, cancel)).strip()
    else:
        target_name = self.selector(str(input))

    with self._lock:
        agent = self.registry.get(target_name)
        if agent is None:
            for name, a in self.registry.items():
                if (
                    name.lower() in target_name.lower()
                    or target_name.lower() in name.lower()
                ):
                    agent = a
                    target_name = name
                    break

    if agent is None:
        if self.fallback:
            agent = self.fallback
            target_name = "fallback"
        else:
            from .multiagent import HandoffError

            raise HandoffError(
                f"Handoff target '{target_name}' not found. "
                f"Available: {list(self.registry.keys())}"
            )

    result = await agent.aapply(input, ctx, cancel)
    elapsed = (time.time() - t0) * 1000
    ctx.log(
        f"Handoff→{target_name}",
        self._trace_id,
        str(input)[:100],
        str(result)[:100],
        elapsed,
    )
    return result


# ════════════════════════════════════════════════════════════
# Apply patches to all Term subclasses
# ════════════════════════════════════════════════════════════


def _patch_async():
    """Monkey-patch aapply onto all Term subclasses."""
    from .primitives import Lam, Compose, If, Loop, Pair, Fst, Snd, Tool
    from .extensions import Par, Route, Memory, Guard
    from .multiagent import GroupChat, AsyncPar, Handoff, Send, Receive

    Lam.aapply = _lam_aapply
    Compose.aapply = _compose_aapply
    If.aapply = _if_aapply
    Loop.aapply = _loop_aapply
    Par.aapply = _par_aapply
    Pair.aapply = _pair_aapply
    Tool.aapply = _tool_aapply
    Route.aapply = _route_aapply
    Memory.aapply = _memory_aapply
    Guard.aapply = _guard_aapply
    GroupChat.aapply = _groupchat_aapply
    AsyncPar.aapply = _asyncpar_aapply
    Handoff.aapply = _handoff_aapply
    # Fst/Snd use the default Term.aapply (thread pool wrapper)


_patch_async()
