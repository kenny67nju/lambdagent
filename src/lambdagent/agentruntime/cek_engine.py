"""
agentruntime.cek_engine — Wraps AgentCEKMachine as an Engine.

Phase 6.5 (E03): The CEKEngine provides step-by-step execution with:
  - Per-step cost monitoring and budget enforcement
  - Infinite loop detection via state hash history
  - Pause/resume via serializable CEKState
  - Full transition trace with continuation stack visibility

This is the "Agent CEK Machine" execution model from Paper II §5.

References:
  - Paper II §5: Agent CEK Machine with Yield mechanism
  - Paper II Proposition 23: Cost monotonicity
  - Paper III §6: Effect handler integration
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, Dict, List, Optional

from lambdagent.core import Term, Context, TraceEntry
from lambdagent.cek_machine import (
    AgentCEKMachine,
    CEKState,
    CostVector,
    ZERO_COST,
    Transition,
    LabelKind,
    Kont,
    HaltK,
    CompK,
    LoopK,
    PairLK,
    PairRK,
    GuardK,
    MemK,
    IfK,
    RouteK,
)

from .engine import (
    Engine,
    EngineMode,
    EngineResult,
    UnifiedTraceRecord,
    CostBudgetExceeded,
    InfiniteLoopDetected,
    MaxStepsExceeded,
)


class CEKEngine(Engine):
    """
    Execution engine backed by Agent CEK Machine.

    Advantages:
      - Per-step cost monitoring with budget ceiling
      - Infinite loop detection (repeated state hash)
      - Serializable state for pause/resume
      - Full continuation stack visibility for debugging
      - Formal correspondence with Paper II operational semantics

    Disadvantages:
      - ~5-15% overhead vs RecursiveEngine (CEK dispatch per step)
      - More complex error handling
    """

    def __init__(
        self,
        cost_budget: float = float("inf"),
        max_steps: int = 10000,
        loop_detection_window: int = 5,
        loop_detection_threshold: int = 3,
        handler=None,
        check_cost_monotonicity: bool = True,
    ):
        """
        Args:
            cost_budget: Max cost in USD. Raises CostBudgetExceeded if exceeded.
            max_steps: Max CEK transitions. Raises MaxStepsExceeded if exceeded.
            loop_detection_window: How many recent state hashes to keep.
            loop_detection_threshold: How many identical hashes trigger detection.
            handler: Effect handler (Paper III §6) for test/prod switching.
            check_cost_monotonicity: Enforce Paper II Proposition 23.
        """
        self._cost_budget = cost_budget
        self._max_steps = max_steps
        self._loop_window = loop_detection_window
        self._loop_threshold = loop_detection_threshold
        self._handler = handler
        self._check_monotonicity = check_cost_monotonicity

    def execute(self, term: Term, input_val: Any, ctx: Context, **opts) -> EngineResult:
        """Synchronous step-by-step execution via CEK Machine."""
        machine = AgentCEKMachine(
            store=dict(ctx.memory),
            handler=self._handler or opts.get("handler"),
            check_cost_monotonicity=self._check_monotonicity,
        )

        machine.load(term, input_val)
        state_hashes: List[str] = []

        while not machine.state.is_terminal():
            # Max steps check
            if machine.step_count >= self._max_steps:
                raise MaxStepsExceeded(self._max_steps, machine.state)

            # Execute one transition
            transition = machine.step()

            # Cost budget check (Paper III graded types enforcement)
            if machine.state.cost.money > self._cost_budget:
                raise CostBudgetExceeded(
                    machine.state.cost, self._cost_budget, machine.state
                )

            # Loop detection: hash of (control, kont) to detect stuck states
            if transition.label.kind != LabelKind.TAU:
                h = _state_hash(machine.state)
                state_hashes.append(h)
                if len(state_hashes) > self._loop_window:
                    state_hashes = state_hashes[-self._loop_window :]
                recent_same = state_hashes.count(h)
                if recent_same >= self._loop_threshold:
                    raise InfiniteLoopDetected(
                        f"State repeated {recent_same} times in last "
                        f"{self._loop_window} non-silent steps. "
                        f"Agent is not making progress.",
                        machine.state,
                    )

        # Populate ctx.trace for backward compatibility
        _sync_ctx_trace(ctx, machine.trace)

        # Sync memory back to ctx
        ctx.memory.update(machine.state.store)

        # Build result
        trace = _convert_transitions(machine.trace)
        result = EngineResult(
            value=machine.state.control,
            trace=trace,
            cost=machine.state.cost,
            steps=machine.step_count,
            engine_mode=EngineMode.CEK,
            final_state=machine.state,
            transitions=machine.trace,
        )

        # DESIGN-08: Cross-validate if cost prediction available
        if opts.get("validate_cost"):
            from lambdagent.cost_grade import estimate_cost, validate_cost

            try:
                predicted = estimate_cost(term)
                validation = validate_cost(
                    predicted, result.cost.tokens, result.cost.money
                )
                if not validation["valid"]:
                    import warnings

                    warnings.warn(validation["alert"])
            except Exception:
                pass  # Cost validation is optional, never break execution

        return result

    async def execute_async(
        self, term: Term, input_val: Any, ctx: Context, **opts
    ) -> EngineResult:
        """Async execution — yields to event loop at each Yield point."""
        machine = AgentCEKMachine(
            store=dict(ctx.memory),
            handler=self._handler or opts.get("handler"),
            check_cost_monotonicity=self._check_monotonicity,
        )

        machine.load(term, input_val)
        state_hashes: List[str] = []

        while not machine.state.is_terminal():
            if machine.step_count >= self._max_steps:
                raise MaxStepsExceeded(self._max_steps, machine.state)

            # Yield to event loop before each step (non-blocking)
            await asyncio.sleep(0)

            transition = machine.step()

            # Cost budget
            if machine.state.cost.money > self._cost_budget:
                raise CostBudgetExceeded(
                    machine.state.cost, self._cost_budget, machine.state
                )

            # Loop detection
            if transition.label.kind != LabelKind.TAU:
                h = _state_hash(machine.state)
                state_hashes.append(h)
                if len(state_hashes) > self._loop_window:
                    state_hashes = state_hashes[-self._loop_window :]
                if state_hashes.count(h) >= self._loop_threshold:
                    raise InfiniteLoopDetected(
                        f"State repeated in last {self._loop_window} steps.",
                        machine.state,
                    )

        _sync_ctx_trace(ctx, machine.trace)
        ctx.memory.update(machine.state.store)

        trace = _convert_transitions(machine.trace)
        return EngineResult(
            value=machine.state.control,
            trace=trace,
            cost=machine.state.cost,
            steps=machine.step_count,
            engine_mode=EngineMode.CEK,
            final_state=machine.state,
            transitions=machine.trace,
        )


# ============================================================
# State Hashing (loop detection)
# ============================================================


def _state_hash(state: CEKState) -> str:
    """Hash of (control repr, kont repr) for loop detection."""
    key = f"{repr(state.control)[:500]}|{repr(state.kont)[:500]}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ============================================================
# Trace Conversion (E04 — CEKEngine side)
# ============================================================


def _convert_transitions(transitions: List[Transition]) -> List[UnifiedTraceRecord]:
    """Convert CEK Transition list to unified trace format."""
    records = []
    cumulative = ZERO_COST
    step_idx = 0

    for t in transitions:
        # Skip silent (τ) transitions for unified trace
        if t.label.kind == LabelKind.TAU:
            continue

        cumulative = cumulative + t.cost_delta
        step_idx += 1

        records.append(
            UnifiedTraceRecord(
                step=step_idx,
                term_name=t.label.name,
                action=t.label.kind.name.lower(),  # "llm" | "tool" | "mem"
                input_summary=str(t.label.input)[:200] if t.label.input else "",
                output_summary=str(t.label.output)[:200] if t.label.output else "",
                duration_ms=t.duration_ms,
                cost=t.cost_delta,
                cumulative_cost=cumulative,
                continuation=_format_kont_from_repr(t.state_after),
                yield_type=(
                    "llm"
                    if t.label.kind == LabelKind.LLM
                    else "tool"
                    if t.label.kind == LabelKind.TOOL
                    else None
                ),
            )
        )

    return records


def _format_kont_from_repr(state_after_repr: str) -> str:
    """Extract K stack description from state repr string."""
    # state_after is stored as repr string in Transition
    # Return it as-is for debugging; full parsing would require the actual state
    return state_after_repr[:200] if state_after_repr else ""


def _format_kont(kont: Kont) -> str:
    """Human-readable continuation stack description."""
    parts = []
    current = kont
    while current is not None:
        if isinstance(current, HaltK):
            parts.append("halt")
            break
        elif isinstance(current, CompK):
            name = getattr(current.g_term, "_name", "?")
            parts.append(f"compK({name})")
            current = current.prev
        elif isinstance(current, LoopK):
            parts.append(f"loopK(n={current.remaining})")
            current = current.prev
        elif isinstance(current, PairLK):
            name = getattr(current.g_term, "_name", "?")
            parts.append(f"pairLK({name})")
            current = current.prev
        elif isinstance(current, PairRK):
            parts.append("pairRK")
            current = current.prev
        elif isinstance(current, GuardK):
            parts.append(f"guardK(k={current.retries})")
            current = current.prev
        elif isinstance(current, MemK):
            parts.append(f"memK({current.store_key})")
            current = current.prev
        elif isinstance(current, IfK):
            parts.append("ifK")
            current = current.prev
        elif isinstance(current, RouteK):
            parts.append("routeK")
            current = current.prev
        else:
            parts.append(repr(current)[:30])
            break
    return " :: ".join(parts)


# ============================================================
# Backward Compatibility: sync to ctx.trace
# ============================================================


def _sync_ctx_trace(ctx: Context, transitions: List[Transition]):
    """
    Populate ctx.trace from CEK transitions for backward compatibility.

    Code that reads ctx.trace (e.g., trace_store, PaaS API) expects
    TraceEntry objects. We convert non-silent transitions to TraceEntry.
    """
    for t in transitions:
        if t.label.kind == LabelKind.TAU:
            continue
        ctx.log(
            term_name=t.label.name,
            term_id="cek",
            inp=t.label.input,
            out=t.label.output,
            duration_ms=t.duration_ms,
            model="" if t.label.kind != LabelKind.LLM else t.label.name,
            tokens=t.cost_delta.tokens,
        )
