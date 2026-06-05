"""
agentruntime.async_executor — Async beta-reduction engine

Async version of Executor. Uses aapply() on all Term types,
supports CancellationToken and configurable timeouts.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncGenerator, Optional

from lambdagent.core import Term, Context
from lambdagent.primitives import Lam, Compose, If, Loop, Pair, Fst, Snd, Tool
from lambdagent.extensions import Par, Route, Memory, Guard
from lambdagent.cancellation import CancellationToken, NullCancellationToken
from lambdagent.async_core import StreamEvent, TokenEvent, StepEvent, ErrorEvent

from .config import RuntimeConfig
from .llm_adapter import LLMAdapter, LLMResponse
from .trace_store import TraceStore, TraceRecord

from lambdagent.hooks import HookRegistry, HookTerm


class AsyncExecutor:
    """
    Async beta-reduction engine.

    Lambda: AsyncExecutor = async Evaluator
        reduce(term, input) = await (term input) ->beta* result

    Supports:
      - Full async/await for all Term types
      - Streaming output via reduce_stream()
      - CancellationToken for hierarchical cancellation
      - Configurable timeouts per operation
      - HookRegistry for global pre/post callbacks (Layer 3)
    """

    def __init__(self, config: RuntimeConfig, hooks: HookRegistry | None = None):
        self.config = config
        self.llm = LLMAdapter(config.llm)
        self.trace = TraceStore()
        self.hooks = hooks or HookRegistry()
        self._step_counter = 0
        self._default_timeout = getattr(config, "timeout", 120)

    async def reduce(
        self,
        term: Term,
        input_val: Any,
        ctx: Context,
        cancel: CancellationToken | None = None,
        timeout: float | None = None,
    ) -> Any:
        """
        Async reduce: dispatches to type-specific reducers.
        Falls back to term.aapply() for unknown types.
        """
        cancel = cancel or NullCancellationToken()
        cancel.check()
        timeout = timeout or self._default_timeout

        if isinstance(term, HookTerm):
            return await term.aapply(input_val, ctx, cancel)
        elif isinstance(term, Lam):
            return await self._reduce_lam(term, input_val, ctx, cancel, timeout)
        elif isinstance(term, Compose):
            return await self._reduce_compose(term, input_val, ctx, cancel, timeout)
        elif isinstance(term, Loop):
            return await self._reduce_loop(term, input_val, ctx, cancel)
        elif isinstance(term, Tool):
            return await self._reduce_tool(term, input_val, ctx, cancel, timeout)
        elif isinstance(term, Route):
            return await self._reduce_route(term, input_val, ctx, cancel, timeout)
        elif isinstance(term, Par):
            return await self._reduce_par(term, input_val, ctx, cancel, timeout)
        elif isinstance(term, Pair):
            return await self._reduce_pair(term, input_val, ctx, cancel, timeout)
        elif isinstance(term, If):
            return await self._reduce_if(term, input_val, ctx, cancel, timeout)
        elif isinstance(term, Memory):
            return await self._reduce_memory(term, input_val, ctx, cancel, timeout)
        elif isinstance(term, Guard):
            return await self._reduce_guard(term, input_val, ctx, cancel, timeout)
        elif isinstance(term, (Fst, Snd)):
            return term.apply(input_val, ctx)
        else:
            return await term.aapply(input_val, ctx, cancel)

    async def reduce_stream(
        self,
        term: Term,
        input_val: Any,
        ctx: Context,
        cancel: CancellationToken | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Streaming reduce: yields events as reduction progresses.
        For Lam terms, yields individual tokens.
        For Compose, yields step completion events.
        """
        cancel = cancel or NullCancellationToken()
        cancel.check()

        if isinstance(term, Lam):
            async for event in self._stream_lam(term, input_val, ctx, cancel):
                yield event
        elif isinstance(term, Compose):
            result = input_val
            for i, stage in enumerate(term.stages):
                cancel.check()
                t0 = time.time()
                if isinstance(stage, Lam):
                    tokens = []
                    async for event in self._stream_lam(stage, result, ctx, cancel):
                        yield event
                        if isinstance(event, TokenEvent):
                            tokens.append(event.token)
                    result = "".join(tokens)
                    result = stage.output_parser(result)
                else:
                    result = await self.reduce(stage, result, ctx, cancel)
                elapsed = (time.time() - t0) * 1000
                yield StepEvent(result, stage._name, i, elapsed)
        else:
            result = await self.reduce(term, input_val, ctx, cancel)
            yield StepEvent(result, term._name)

    # ── Type-specific async reducers ──

    async def _reduce_lam(
        self,
        lam: Lam,
        input_val: Any,
        ctx: Context,
        cancel: CancellationToken,
        timeout: float,
    ) -> Any:
        # Hook Layer 3: pre_llm
        await self.hooks.afire("pre_llm", term=lam, input=input_val, ctx=ctx)

        t0 = time.time()
        response = await asyncio.wait_for(
            self.llm.acall(
                model=lam.model,
                system=lam.prompt,
                user=str(input_val),
                temperature=lam.temperature,
                max_tokens=lam.max_tokens,
            ),
            timeout=timeout,
        )
        cancel.check()
        duration = (time.time() - t0) * 1000
        result = lam.output_parser(response.text)

        # Hook Layer 3: post_llm (can modify output via output["value"])
        output_wrapper = {"value": result}
        await self.hooks.afire(
            "post_llm",
            term=lam,
            input=input_val,
            output=output_wrapper,
            usage=response.usage,
            ctx=ctx,
        )
        result = output_wrapper["value"]

        self._record(lam, input_val, result, duration, response)
        ctx.log(
            lam._name,
            lam._trace_id,
            input_val,
            result,
            duration,
            lam.model,
            response.usage.total_tokens,
        )
        return result

    async def _stream_lam(
        self, lam: Lam, input_val: Any, ctx: Context, cancel: CancellationToken
    ) -> AsyncGenerator[StreamEvent, None]:
        t0 = time.time()
        full_text = []
        async for token in self.llm.stream(
            model=lam.model,
            system=lam.prompt,
            user=str(input_val),
            temperature=lam.temperature,
            max_tokens=lam.max_tokens,
        ):
            cancel.check()
            full_text.append(token)
            yield TokenEvent(token, lam._name)

        duration = (time.time() - t0) * 1000
        raw = "".join(full_text)
        result = lam.output_parser(raw)
        ctx.log(lam._name, lam._trace_id, input_val, result, duration, lam.model)
        yield StepEvent(result, lam._name, duration_ms=duration)

    async def _reduce_compose(
        self,
        comp: Compose,
        input_val: Any,
        ctx: Context,
        cancel: CancellationToken,
        timeout: float,
    ) -> Any:
        result = input_val
        for stage in comp.stages:
            cancel.check()
            result = await self.reduce(stage, result, ctx, cancel, timeout)
        return result

    async def _reduce_loop(
        self, loop: Loop, input_val: Any, ctx: Context, cancel: CancellationToken
    ) -> Any:
        result = input_val
        for step in range(loop.max_steps):
            cancel.check()
            result = await self.reduce(loop.body, result, ctx, cancel)
            if loop.condition(result, step):
                break
        return result

    async def _reduce_tool(
        self,
        tool: Tool,
        input_val: Any,
        ctx: Context,
        cancel: CancellationToken,
        timeout: float,
    ) -> Any:
        # Hook Layer 3: pre_tool
        await self.hooks.afire("pre_tool", term=tool, input=input_val, ctx=ctx)

        t0 = time.time()
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, tool.fn, input_val),
            timeout=timeout,
        )
        duration = (time.time() - t0) * 1000

        # Hook Layer 3: post_tool (can modify output)
        output_wrapper = {"value": result}
        await self.hooks.afire(
            "post_tool",
            term=tool,
            input=input_val,
            output=output_wrapper,
            duration_ms=duration,
            ctx=ctx,
        )
        result = output_wrapper["value"]

        self.trace.append(
            TraceRecord(
                step=self._step_counter,
                term_name=tool._name,
                term_type="Tool",
                duration_ms=duration,
                input=str(input_val)[:200],
                output=str(result)[:200],
            )
        )
        ctx.log(tool._name, tool._trace_id, input_val, result, duration)
        self._step_counter += 1
        return result

    async def _reduce_route(
        self,
        route: Route,
        input_val: Any,
        ctx: Context,
        cancel: CancellationToken,
        timeout: float,
    ) -> Any:
        label = (
            str(await self.reduce(route.classifier, input_val, ctx, cancel, timeout))
            .strip()
            .lower()
        )
        agent = route.routes.get(label)
        if agent is None:
            for key, val in route.routes.items():
                if key.lower() in label:
                    agent = val
                    break
        if agent is None:
            agent = route.default
        if agent is None:
            from lambdagent.core import RouteError

            raise RouteError(f"No route for '{label}'")
        return await self.reduce(agent, input_val, ctx, cancel, timeout)

    async def _reduce_par(
        self,
        par: Par,
        input_val: Any,
        ctx: Context,
        cancel: CancellationToken,
        timeout: float,
    ) -> tuple:
        tasks = [
            self.reduce(a, input_val, ctx, cancel.child(), timeout) for a in par.agents
        ]
        return tuple(await asyncio.gather(*tasks))

    async def _reduce_pair(
        self,
        pair: Pair,
        input_val: Any,
        ctx: Context,
        cancel: CancellationToken,
        timeout: float,
    ) -> tuple:
        a, b = await asyncio.gather(
            self.reduce(pair.first, input_val, ctx, cancel.child(), timeout),
            self.reduce(pair.second, input_val, ctx, cancel.child(), timeout),
        )
        return (a, b)

    async def _reduce_if(
        self,
        if_term: If,
        input_val: Any,
        ctx: Context,
        cancel: CancellationToken,
        timeout: float,
    ) -> Any:
        if isinstance(if_term.cond, Term):
            cond_result = await self.reduce(
                if_term.cond, input_val, ctx, cancel, timeout
            )
            branch = (
                If._is_truthy(cond_result)
                if isinstance(cond_result, str)
                else bool(cond_result)
            )
        else:
            branch = if_term.cond(input_val)
        if branch:
            return await self.reduce(if_term.then_, input_val, ctx, cancel, timeout)
        else:
            return await self.reduce(if_term.else_, input_val, ctx, cancel, timeout)

    async def _reduce_memory(
        self,
        mem: Memory,
        input_val: Any,
        ctx: Context,
        cancel: CancellationToken,
        timeout: float,
    ) -> Any:
        if mem.store:
            memory_str = "\n".join(
                f"- {k}: {v}" for k, v in mem.store.items() if not k.startswith("_")
            )
            augmented = (
                f"[Memory]\n{memory_str}\n\n[Input]\n{input_val}"
                if memory_str
                else str(input_val)
            )
        else:
            augmented = str(input_val)
        return await self.reduce(mem.agent, augmented, ctx, cancel, timeout)

    async def _reduce_guard(
        self,
        guard: Guard,
        input_val: Any,
        ctx: Context,
        cancel: CancellationToken,
        timeout: float,
    ) -> Any:
        last_result = None
        for attempt in range(1 + guard.retry):
            cancel.check()
            result = await self.reduce(guard.agent, input_val, ctx, cancel, timeout)
            last_result = result
            if callable(guard.validator) and not isinstance(guard.validator, Term):
                valid = guard.validator(result)
            elif isinstance(guard.validator, Term):
                valid = bool(
                    await self.reduce(guard.validator, result, ctx, cancel, timeout)
                )
            else:
                valid = True
            if valid:
                return result
        # Hook Layer 3: on_guard_fail
        await self.hooks.afire(
            "on_guard_fail",
            term=guard,
            input=input_val,
            last_result=last_result,
            attempts=1 + guard.retry,
            ctx=ctx,
        )
        if guard.on_fail:
            return guard.on_fail(last_result)
        from lambdagent.core import ValidationError

        raise ValidationError(f"Guard failed after {1 + guard.retry} attempts")

    # ── Helpers ──

    def _record(self, lam: Lam, input_val, result, duration_ms, response: LLMResponse):
        self.trace.append(
            TraceRecord(
                step=self._step_counter,
                term_name=lam._name,
                term_type="Lam",
                duration_ms=duration_ms,
                input=str(input_val)[:200],
                output=str(result)[:200],
                model=lam.model,
                temperature=lam.temperature,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        )
        self._step_counter += 1
