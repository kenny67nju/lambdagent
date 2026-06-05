"""
agentruntime.async_react_engine — Async ReAct loop engine

Async version of ReActEngine with streaming support,
cancellation tokens, and tool call timeouts.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Optional

from lambdagent.core import Context
from lambdagent.primitives import Lam, Tool
from lambdagent.cancellation import CancellationToken, NullCancellationToken
from lambdagent.async_core import StreamEvent, TokenEvent, StepEvent

from .action_parser import ActionParser, Action, ParseError
from .termination import TerminationOracle
from .memory_backend import MemoryBackend
from .trace_store import TraceStore, TraceRecord
from .llm_adapter import LLMAdapter


@dataclass
class AsyncStepResult:
    """Result of one async Y combinator unfolding."""

    terminated: bool
    answer: Optional[str]
    next_state: Optional[str]
    thought: str
    action: Optional[Action]
    observation: Optional[str]
    step: int
    duration_ms: float = 0


class AsyncReActEngine:
    """
    Async ReAct loop = Y combinator runtime with streaming + cancellation.

    Lambda: react = Y_n(lambda self. lambda state.
        let (thought, action) = think_and_parse(state) in
        IF (action = terminate) THEN extract_answer(thought)
        ELSE let obs = invoke(action) in self(state + format(thought, obs))
    )
    """

    def __init__(
        self,
        think: Lam,
        tools: Dict[str, Tool],
        action_parser: ActionParser,
        memory: MemoryBackend,
        termination: TerminationOracle,
        trace: TraceStore,
        llm: LLMAdapter | None = None,
        max_steps: int = 10,
        tool_timeout: int = 30,
        think_timeout: int = 120,
        observation_enabled: bool = True,
        system_prompt: str = "",
    ):
        self.think = think
        self.tools = tools
        self.action_parser = action_parser
        self.memory = memory
        self.termination = termination
        self.trace = trace
        self.llm = llm
        self.max_steps = max_steps
        self.tool_timeout = tool_timeout
        self.think_timeout = think_timeout
        self.observation_enabled = observation_enabled
        self.system_prompt = system_prompt

    async def run(
        self,
        input_text: str,
        ctx: Context,
        cancel: CancellationToken | None = None,
    ) -> str:
        """Execute full async ReAct loop."""
        cancel = cancel or NullCancellationToken()
        state = input_text
        final_answer = None

        for step in range(self.max_steps):
            cancel.check()
            result = await self._step(state, step, ctx, cancel)
            if result.terminated:
                final_answer = result.answer
                break
            state = result.next_state

        if final_answer is None:
            final_answer = await self._force_terminate(state, ctx, cancel)

        return final_answer

    async def run_stream(
        self,
        input_text: str,
        ctx: Context,
        cancel: CancellationToken | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Streaming ReAct loop — yields tokens and step events."""
        cancel = cancel or NullCancellationToken()
        state = input_text

        for step in range(self.max_steps):
            cancel.check()

            # Stream the THINK phase
            prompt = self._build_prompt(state, step)
            t0 = time.time()
            thought_tokens = []

            if self.llm:
                async for token in self.llm.stream(
                    model=self.think.model,
                    system=self.think.prompt,
                    user=prompt,
                    temperature=self.think.temperature,
                    max_tokens=self.think.max_tokens,
                ):
                    cancel.check()
                    thought_tokens.append(token)
                    yield TokenEvent(token, f"think[{step}]")
            else:
                # Fall back to non-streaming
                loop = asyncio.get_event_loop()
                thought_raw = await loop.run_in_executor(None, self.think, prompt, ctx)
                thought_tokens.append(str(thought_raw))
                yield TokenEvent(str(thought_raw), f"think[{step}]")

            thought = "".join(thought_tokens)
            think_ms = (time.time() - t0) * 1000
            ctx.log(
                f"think[{step}]",
                self.think._trace_id,
                state[:100],
                thought[:100],
                think_ms,
                self.think.model,
            )

            # PARSE + ROUTE + INVOKE (non-streaming)
            result = await self._process_thought(state, thought, step, ctx, cancel)
            yield StepEvent(
                {
                    "thought": thought,
                    "action": result.action,
                    "observation": result.observation,
                },
                f"step[{step}]",
                step,
                result.duration_ms,
            )

            if result.terminated:
                return
            state = result.next_state

        # Force terminate
        answer = await self._force_terminate(state, ctx, cancel)
        yield StepEvent(answer, "force_terminate")

    async def _step(
        self, state: str, step: int, ctx: Context, cancel: CancellationToken
    ) -> AsyncStepResult:
        """One async ReAct step = 7 phases."""
        t0_step = time.time()

        # Phase 1: THINK
        prompt = self._build_prompt(state, step)
        t0 = time.time()

        loop = asyncio.get_event_loop()
        thought = await asyncio.wait_for(
            loop.run_in_executor(None, self.think, prompt, ctx),
            timeout=self.think_timeout,
        )
        thought = str(thought)
        think_ms = (time.time() - t0) * 1000
        cancel.check()

        self.trace.append(
            TraceRecord(
                step=step,
                term_name="think",
                term_type="Lam",
                duration_ms=think_ms,
                input=state[:200],
                output=thought[:200],
            )
        )

        result = await self._process_thought(state, thought, step, ctx, cancel)
        result.duration_ms = (time.time() - t0_step) * 1000
        return result

    async def _process_thought(
        self,
        state: str,
        thought: str,
        step: int,
        ctx: Context,
        cancel: CancellationToken,
    ) -> AsyncStepResult:
        """Phases 2-7: parse, route, invoke, observe, update, check."""

        # Phase 2: PARSE
        try:
            action = self.action_parser.parse(thought)
        except ParseError as e:
            observation = f"[FORMAT_ERROR] {e}. Please output a valid action."
            next_state = self._append_observation(state, thought, observation)
            return AsyncStepResult(
                terminated=False,
                answer=None,
                next_state=next_state,
                thought=thought,
                action=None,
                observation=observation,
                step=step,
            )

        # Phase 3: ROUTE
        if action.tool not in self.tools:
            observation = f"[ROUTE_ERROR] Unknown tool: {action.tool}. Available: {list(self.tools.keys())}"
            next_state = self._append_observation(state, thought, observation)
            return AsyncStepResult(
                terminated=False,
                answer=None,
                next_state=next_state,
                thought=thought,
                action=action,
                observation=observation,
                step=step,
            )

        tool = self.tools[action.tool]

        # Phase 4: INVOKE
        if action.tool == "terminate":
            answer = self._extract_final_answer(thought, action)
            self.trace.append(
                TraceRecord(
                    step=step,
                    term_name="terminate",
                    term_type="Tool",
                    input=thought[:200],
                    output=answer[:200],
                    terminated=True,
                    action="terminate",
                )
            )
            return AsyncStepResult(
                terminated=True,
                answer=answer,
                next_state=None,
                thought=thought,
                action=action,
                observation=None,
                step=step,
            )

        t0 = time.time()
        try:
            tool_input = (
                action.input if isinstance(action.input, str) else str(action.input)
            )
            loop = asyncio.get_event_loop()
            tool_result = await asyncio.wait_for(
                loop.run_in_executor(None, tool, tool_input),
                timeout=self.tool_timeout,
            )
            cancel.check()
            tool_ms = (time.time() - t0) * 1000
            self.trace.append(
                TraceRecord(
                    step=step,
                    term_name=f"Tool:{action.tool}",
                    term_type="Tool",
                    duration_ms=tool_ms,
                    input=tool_input[:200],
                    output=str(tool_result)[:200],
                    action=action.tool,
                    action_input=action.input,
                )
            )
        except asyncio.TimeoutError:
            tool_result = (
                f"[TOOL_TIMEOUT] {action.tool} timed out after {self.tool_timeout}s"
            )
            self.trace.append(
                TraceRecord(
                    step=step,
                    term_name=f"Tool:{action.tool}",
                    term_type="Tool",
                    input=str(action.input)[:200],
                    output=tool_result,
                    error="timeout",
                    action=action.tool,
                )
            )
        except Exception as e:
            tool_result = f"[TOOL_ERROR] {e}"
            self.trace.append(
                TraceRecord(
                    step=step,
                    term_name=f"Tool:{action.tool}",
                    term_type="Tool",
                    input=str(action.input)[:200],
                    output=tool_result,
                    error=str(e),
                    action=action.tool,
                )
            )

        # Phase 5: OBSERVE
        observation = self._format_observation(action.tool, str(tool_result))

        # Phase 6: UPDATE
        self.memory.auto_save(
            key=f"step_{step}",
            thought=thought,
            action=action.tool,
            observation=observation,
        )

        # Phase 7: CHECK
        if self.termination.should_stop(thought, observation, step):
            answer = self._extract_final_answer(thought, action)
            return AsyncStepResult(
                terminated=True,
                answer=answer,
                next_state=None,
                thought=thought,
                action=action,
                observation=observation,
                step=step,
            )

        next_state = self._append_observation(state, thought, observation)
        return AsyncStepResult(
            terminated=False,
            answer=None,
            next_state=next_state,
            thought=thought,
            action=action,
            observation=observation,
            step=step,
        )

    # ── Prompt building (shared with sync version) ──

    def _build_prompt(self, state: str, step: int) -> str:
        parts = []
        memory_items = self.memory.read_recent(n=10)
        if memory_items:
            parts.append("[Memory Context]")
            for key, value, age in memory_items:
                parts.append(f"  - {key}: {value} ({age})")
            parts.append("")

        parts.append("[Available Tools]")
        parts.append(
            'Call a tool by outputting JSON: {"action": "tool_name", "input": {...}}'
        )
        parts.append(
            'To finish, call: {"action": "terminate", "answer": "your final answer"}'
        )
        parts.append("")
        for name in self.tools:
            if name == "terminate":
                parts.append(f"  - terminate: Signal task completion")
            else:
                parts.append(f"  - {name}: MCP tool")
        parts.append("")

        remaining = self.max_steps - step - 1
        parts.append(f"[Step {step + 1}/{self.max_steps}, remaining: {remaining}]")
        if remaining <= 3:
            parts.append("Warning: Running low on steps. Consider calling terminate.")
        parts.append("")
        parts.append(state)
        return "\n".join(parts)

    def _append_observation(self, state: str, thought: str, observation: str) -> str:
        step_count = state.count("[Step ")
        return (
            f"{state}\n\n"
            f"[Step {step_count + 1}]\n"
            f"Thought: {thought}\n"
            f"Observation: {observation}"
        )

    def _extract_final_answer(self, thought: str, action: Action) -> str:
        if action and action.input:
            answer = action.input.get("answer", "")
            if answer:
                return str(answer)
        return thought

    def _format_observation(self, tool_name: str, result: str) -> str:
        max_len = 2000
        if len(result) > max_len:
            result = result[:max_len] + f"\n... [truncated, {len(result)} chars total]"
        return f"[{tool_name}] {result}"

    async def _force_terminate(
        self, state: str, ctx: Context, cancel: CancellationToken
    ) -> str:
        lines = state.split("\n")
        for line in reversed(lines):
            if line.startswith("Thought:"):
                return line[8:].strip()
        return state[-500:]
