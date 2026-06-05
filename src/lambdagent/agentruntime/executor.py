"""agentruntime.executor — Beta-reduction engine"""
from __future__ import annotations
import time
from typing import Any

from lambdagent.core import Term, Context
from lambdagent.primitives import Lam, Compose, If, Loop, Pair, Fst, Snd, Tool
from lambdagent.extensions import Par, Route, Memory, Guard

from .config import RuntimeConfig
from .llm_adapter import LLMAdapter
from .mcp_client import MCPClient
from .memory_backend import MemoryBackend
from .trace_store import TraceStore, TraceRecord
from .action_parser import ActionParser
from .termination import TerminationOracle
from .react_engine import ReActEngine


class Executor:
    """
    Beta-reduction engine.

    Lambda: Executor = Evaluator
        reduce(term, input) = (term input) ->beta* result

    Evaluation strategy: Call-by-Value (strict)
        Arguments evaluated before passed to function body.
        Reason: LLM calls have side effects (API call).
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.llm = LLMAdapter(config.llm)
        self.mcp = MCPClient(config.mcp)
        self.memory = MemoryBackend.create(config.memory)
        self.trace = TraceStore()
        self.termination = TerminationOracle(
            signals=config.termination.signals,
            implicit_detection=config.termination.implicit_detection,
        )
        self._step_counter = 0

    def reduce(self, term: Term, input_val: Any, ctx: Context) -> Any:
        """
        Reduce term with input via beta-reduction.
        Dispatches based on Term type.
        """
        if isinstance(term, Memory):
            return self._reduce_memory(term, input_val, ctx)
        elif isinstance(term, Guard):
            return self._reduce_guard(term, input_val, ctx)
        elif isinstance(term, Lam):
            return self._reduce_lam(term, input_val, ctx)
        elif isinstance(term, Compose):
            return self._reduce_compose(term, input_val, ctx)
        elif isinstance(term, Loop):
            return self._reduce_loop(term, input_val, ctx)
        elif isinstance(term, Tool):
            return self._reduce_tool(term, input_val, ctx)
        elif isinstance(term, Route):
            return self._reduce_route(term, input_val, ctx)
        elif isinstance(term, Par):
            return self._reduce_par(term, input_val, ctx)
        elif isinstance(term, Pair):
            return self._reduce_pair(term, input_val, ctx)
        elif isinstance(term, If):
            return self._reduce_if(term, input_val, ctx)
        elif isinstance(term, (Fst, Snd)):
            return term.apply(input_val, ctx)
        else:
            # Fallback: use term's own apply
            return term.apply(input_val, ctx)

    def _reduce_lam(self, lam: Lam, input_val: Any, ctx: Context) -> Any:
        """(lambda_D x) ->beta F_{M,D}(x) = one LLM API call."""
        t0 = time.time()

        response = self.llm.call(
            model=lam.model,
            system=lam.prompt,
            user=str(input_val),
            temperature=lam.temperature,
            max_tokens=lam.max_tokens,
        )

        duration = (time.time() - t0) * 1000
        result = lam.output_parser(response.text)

        self.trace.append(TraceRecord(
            step=self._step_counter, term_name=lam._name, term_type="Lam",
            duration_ms=duration, input=str(input_val)[:200], output=str(result)[:200],
            model=lam.model, temperature=lam.temperature,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        ))
        ctx.log(lam._name, lam._trace_id, input_val, result, duration, lam.model,
                response.usage.total_tokens)
        self._step_counter += 1
        return result

    def _reduce_compose(self, comp: Compose, input_val: Any, ctx: Context) -> Any:
        """f >> g = lambda x. g(f(x)). Sequential reduction."""
        result = input_val
        for stage in comp.stages:
            result = self.reduce(stage, result, ctx)
        return result

    def _reduce_loop(self, loop: Loop, input_val: Any, ctx: Context) -> Any:
        """Y combinator = ReAct loop."""
        # Check if loop body is a react step (Tool wrapping react_step)
        result = input_val
        for step in range(loop.max_steps):
            result = self.reduce(loop.body, result, ctx)
            if loop.condition(result, step):
                break
        return result

    def _reduce_tool(self, tool: Tool, input_val: Any, ctx: Context) -> Any:
        """Primitive call."""
        t0 = time.time()
        result = tool.fn(input_val)
        duration = (time.time() - t0) * 1000
        self.trace.append(TraceRecord(
            step=self._step_counter, term_name=tool._name, term_type="Tool",
            duration_ms=duration, input=str(input_val)[:200], output=str(result)[:200],
        ))
        ctx.log(tool._name, tool._trace_id, input_val, result, duration)
        self._step_counter += 1
        return result

    def _reduce_route(self, route: Route, input_val: Any, ctx: Context) -> Any:
        """CASE dispatch."""
        label = str(self.reduce(route.classifier, input_val, ctx)).strip().lower()
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
        return self.reduce(agent, input_val, ctx)

    def _reduce_memory(self, mem: Memory, input_val: Any, ctx: Context) -> Any:
        """Environment extension: inject memory, execute, write back."""
        if mem.store:
            memory_str = "\n".join(f"- {k}: {v}" for k, v in mem.store.items()
                                   if not k.startswith("_"))
            if memory_str:
                augmented = f"[Memory]\n{memory_str}\n\n[Input]\n{input_val}"
            else:
                augmented = str(input_val)
        else:
            augmented = str(input_val)
        return self.reduce(mem.agent, augmented, ctx)

    def _reduce_guard(self, guard: Guard, input_val: Any, ctx: Context) -> Any:
        """Type constraint: execute + validate + retry."""
        last_result = None
        for attempt in range(1 + guard.retry):
            result = self.reduce(guard.agent, input_val, ctx)
            last_result = result
            if callable(guard.validator) and not isinstance(guard.validator, Term):
                valid = guard.validator(result)
            elif isinstance(guard.validator, Term):
                valid = bool(self.reduce(guard.validator, result, ctx))
            else:
                valid = True
            if valid:
                return result
        if guard.on_fail:
            return guard.on_fail(last_result)
        from lambdagent.core import ValidationError
        raise ValidationError(f"Guard failed after {1 + guard.retry} attempts")

    def _reduce_par(self, par: Par, input_val: Any, ctx: Context) -> tuple:
        """Church pair = parallel execution."""
        return tuple(self.reduce(a, input_val, ctx) for a in par.agents)

    def _reduce_pair(self, pair: Pair, input_val: Any, ctx: Context) -> tuple:
        a = self.reduce(pair.first, input_val, ctx)
        b = self.reduce(pair.second, input_val, ctx)
        return (a, b)

    def _reduce_if(self, if_term: If, input_val: Any, ctx: Context) -> Any:
        if isinstance(if_term.cond, Term):
            cond_result = self.reduce(if_term.cond, input_val, ctx)
            branch = If._is_truthy(cond_result) if isinstance(cond_result, str) else bool(cond_result)
        else:
            branch = if_term.cond(input_val)
        if branch:
            return self.reduce(if_term.then_, input_val, ctx)
        else:
            return self.reduce(if_term.else_, input_val, ctx)
