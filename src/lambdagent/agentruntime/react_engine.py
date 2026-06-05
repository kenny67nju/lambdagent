"""agentruntime.react_engine — ReAct loop engine (Y combinator runtime)"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lambdagent.core import Context
from lambdagent.primitives import Lam, Tool
from .action_parser import ActionParser, Action, ParseError
from .termination import TerminationOracle
from .memory_backend import MemoryBackend
from .trace_store import TraceStore, TraceRecord


# Step event types for streaming callbacks
STEP_THINK = "think"
STEP_TOOL_CALL = "tool_call"
STEP_TOOL_RESULT = "tool_result"
STEP_ERROR = "error"
STEP_ANSWER = "answer"


@dataclass
class StepEvent:
    """One streaming event emitted during ReAct execution."""

    type: str  # STEP_THINK, STEP_TOOL_CALL, STEP_TOOL_RESULT, STEP_ERROR, STEP_ANSWER
    step: int
    content: str  # The text payload
    tool: str = ""  # Tool name (for tool_call / tool_result)
    duration_ms: float = 0


@dataclass
class StepResult:
    """Result of one Y combinator unfolding."""

    terminated: bool
    answer: Optional[str]
    next_state: Optional[str]
    thought: str
    action: Optional[Action]
    observation: Optional[str]
    step: int


class ReActEngine:
    """
    ReAct loop engine = Y combinator runtime implementation.

    Lambda: react = Y_n(lambda self. lambda state.
        let (thought, action) = think_and_parse(state) in
        IF (action = terminate) THEN extract_answer(thought)
        ELSE let obs = invoke(action) in self(state + format(thought, obs))
    )

    Each step = 7 sub-phases: THINK, PARSE, ROUTE, INVOKE, OBSERVE, UPDATE, CHECK
    """

    def __init__(
        self,
        think: Lam,
        tools: Dict[str, Tool],
        action_parser: ActionParser,
        memory: MemoryBackend,
        termination: TerminationOracle,
        trace: TraceStore,
        max_steps: int = 10,
        tool_timeout: int = 30,
        observation_enabled: bool = True,
        system_prompt: str = "",
        on_step: Any = None,
    ):
        self.think = think
        self.tools = tools
        self.action_parser = action_parser
        self.memory = memory
        self.termination = termination
        self.trace = trace
        self.max_steps = max_steps
        self.tool_timeout = tool_timeout
        self.observation_enabled = observation_enabled
        self.system_prompt = system_prompt
        self.on_step = on_step  # callback(StepEvent) -> None

    def _emit(self, event: StepEvent):
        """Emit a step event to the streaming callback if registered."""
        if self.on_step:
            try:
                self.on_step(event)
            except Exception:
                pass  # Don't let callback errors break the loop

    def _record(self, record: TraceRecord, ctx: Context):
        """
        FIX-05: Record trace to BOTH self.trace (TraceStore) AND ctx.trace.
        This ensures PaaS layer and Engine layer both see the trace.
        """
        self.trace.append(record)
        ctx.log(
            term_name=record.term_name,
            term_id=f"react-{record.step}",
            inp=record.input,
            out=record.output,
            duration_ms=record.duration_ms,
            model=record.model,
            tokens=record.input_tokens + record.output_tokens,
        )

    def run(self, input_text: str, ctx: Context) -> str:
        """Execute full ReAct loop. Equivalent to Y_n(react_body)(input)."""
        state = input_text
        final_answer = None

        for step in range(self.max_steps):
            result = self._step(state, step, ctx)

            if result.terminated:
                final_answer = result.answer
                self._emit(
                    StepEvent(
                        type=STEP_ANSWER,
                        step=step,
                        content=final_answer or "",
                    )
                )
                break

            state = result.next_state

        if final_answer is None:
            final_answer = self._force_terminate(state, ctx)
            self._emit(
                StepEvent(
                    type=STEP_ANSWER,
                    step=self.max_steps,
                    content=final_answer,
                )
            )

        return final_answer

    def _step(self, state: str, step: int, ctx: Context) -> StepResult:
        """One ReAct step = one Y combinator unfolding (7 phases)."""

        # ═══ Phase 1: THINK (beta-reduction: LLM inference) ═══
        prompt = self._build_prompt(state, step)
        t0 = time.time()
        thought = self.think(prompt, ctx)
        think_ms = (time.time() - t0) * 1000

        self._record(
            TraceRecord(
                step=step,
                term_name="think",
                term_type="Lam",
                duration_ms=think_ms,
                input=state[:200],
                output=str(thought)[:200],
            ),
            ctx,
        )
        self._emit(
            StepEvent(
                type=STEP_THINK,
                step=step,
                content=str(thought),
                duration_ms=think_ms,
            )
        )

        # ═══ Phase 2: PARSE (extract structured action) ═══
        try:
            action = self.action_parser.parse(str(thought))
        except ParseError as e:
            observation = f"[FORMAT_ERROR] {e}. Please output a valid action."
            next_state = self._append_observation(state, str(thought), observation)
            self._record(
                TraceRecord(
                    step=step,
                    term_name="parse_error",
                    term_type="Error",
                    input=str(thought)[:200],
                    output=observation,
                    error=str(e),
                ),
                ctx,
            )
            return StepResult(
                terminated=False,
                answer=None,
                next_state=next_state,
                thought=str(thought),
                action=None,
                observation=observation,
                step=step,
            )

        # ═══ Phase 3: ROUTE (select tool = CASE dispatch) ═══
        if action.tool not in self.tools:
            observation = f"[ROUTE_ERROR] Unknown tool: {action.tool}. Available: {list(self.tools.keys())}"
            next_state = self._append_observation(state, str(thought), observation)
            return StepResult(
                terminated=False,
                answer=None,
                next_state=next_state,
                thought=str(thought),
                action=action,
                observation=observation,
                step=step,
            )

        tool = self.tools[action.tool]

        # ═══ Phase 4: INVOKE (execute tool = beta-reduction) ═══

        # 4a. Base case: terminate = lambda x.x
        if action.tool == "terminate":
            answer = self._extract_final_answer(str(thought), action)
            self._record(
                TraceRecord(
                    step=step,
                    term_name="terminate",
                    term_type="Tool",
                    input=str(thought)[:200],
                    output=answer[:200],
                    terminated=True,
                    action="terminate",
                ),
                ctx,
            )
            return StepResult(
                terminated=True,
                answer=answer,
                next_state=None,
                thought=str(thought),
                action=action,
                observation=None,
                step=step,
            )

        # 4b. Tool call
        self._emit(
            StepEvent(
                type=STEP_TOOL_CALL,
                step=step,
                content=str(action.input)[:500],
                tool=action.tool,
            )
        )
        t0 = time.time()
        try:
            tool_input = (
                action.input if isinstance(action.input, str) else str(action.input)
            )
            tool_result = tool(tool_input)
            tool_ms = (time.time() - t0) * 1000
            self._record(
                TraceRecord(
                    step=step,
                    term_name=f"Tool:{action.tool}",
                    term_type="Tool",
                    duration_ms=tool_ms,
                    input=tool_input[:200],
                    output=str(tool_result)[:200],
                    action=action.tool,
                    action_input=action.input,
                ),
                ctx,
            )
            self._emit(
                StepEvent(
                    type=STEP_TOOL_RESULT,
                    step=step,
                    content=str(tool_result)[:1000],
                    tool=action.tool,
                    duration_ms=tool_ms,
                )
            )
        except Exception as e:
            tool_result = f"[TOOL_ERROR] {e}"
            self._record(
                TraceRecord(
                    step=step,
                    term_name=f"Tool:{action.tool}",
                    term_type="Tool",
                    input=str(action.input)[:200],
                    output=tool_result,
                    error=str(e),
                    action=action.tool,
                ),
                ctx,
            )
            self._emit(
                StepEvent(
                    type=STEP_ERROR,
                    step=step,
                    content=tool_result,
                    tool=action.tool,
                )
            )

        # ═══ Phase 5: OBSERVE (format observation) ═══
        observation = self._format_observation(action.tool, str(tool_result))

        # ═══ Phase 6: UPDATE (write to memory = Gamma update) ═══
        self.memory.auto_save(
            key=f"step_{step}",
            thought=str(thought),
            action=action.tool,
            observation=observation,
        )

        # ═══ Phase 7: CHECK (termination detection) ═══
        if self.termination.should_stop(str(thought), observation, step):
            answer = self._extract_final_answer(str(thought), action)
            return StepResult(
                terminated=True,
                answer=answer,
                next_state=None,
                thought=str(thought),
                action=action,
                observation=observation,
                step=step,
            )

        # Prepare next state (recursion: self(state + obs))
        next_state = self._append_observation(state, str(thought), observation)

        return StepResult(
            terminated=False,
            answer=None,
            next_state=next_state,
            thought=str(thought),
            action=action,
            observation=observation,
            step=step,
        )

    def _build_prompt(self, state: str, step: int) -> str:
        """Build complete prompt for LLM."""
        parts = []

        # Memory injection
        memory_items = self.memory.read_recent(n=10)
        if memory_items:
            parts.append("[Memory Context]")
            for key, value, age in memory_items:
                parts.append(f"  - {key}: {value} ({age})")
            parts.append("")

        # Tool descriptions
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

        # Step info
        remaining = self.max_steps - step - 1
        parts.append(f"[Step {step + 1}/{self.max_steps}, remaining: {remaining}]")
        if remaining <= 3:
            parts.append("Warning: Running low on steps. Consider calling terminate.")
        parts.append("")

        # Current state
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
        # Truncate long observations
        max_len = 2000
        if len(result) > max_len:
            result = result[:max_len] + f"\n... [truncated, {len(result)} chars total]"
        return f"[{tool_name}] {result}"

    def _force_terminate(self, state: str, ctx: Context) -> str:
        """Max steps exhausted. Extract best answer."""
        # Try to extract from the last thought
        lines = state.split("\n")
        for line in reversed(lines):
            if line.startswith("Thought:"):
                return line[8:].strip()
        return state[-500:]  # Last 500 chars as fallback
