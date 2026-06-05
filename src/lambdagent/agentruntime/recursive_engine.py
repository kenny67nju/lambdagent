"""
agentruntime.recursive_engine — Wraps the existing Executor as an Engine.

Phase 6.5 (E02): The RecursiveEngine is the default execution backend.
It delegates to Executor.reduce() (sync) and term.aapply() (async),
converting their output to the unified EngineResult format.

This is the "Python call stack" execution model — simple, fast, but
lacking CEK's pause/resume and step-by-step cost monitoring.
"""

from __future__ import annotations

import time
from typing import Any, List

from lambdagent.core import Term, Context, TraceEntry
from lambdagent.cek_machine import CostVector, ZERO_COST

from .engine import (
    Engine,
    EngineMode,
    EngineResult,
    UnifiedTraceRecord,
)
from .config import RuntimeConfig
from .executor import Executor


# Model cost table (USD per 1K tokens) — shared with cost_grade.py
_MODEL_COSTS = {
    "claude-sonnet": 0.003,
    "claude-opus": 0.015,
    "claude-haiku": 0.00025,
    "gpt-4o": 0.005,
    "gpt-4": 0.03,
    "qwen3-max": 0.002,
    "deepseek-chat": 0.001,
}


def _estimate_money(model: str, tokens: int) -> float:
    """Estimate cost from model name and token count."""
    for key, price_per_k in _MODEL_COSTS.items():
        if key in model.lower():
            return tokens * price_per_k / 1000
    return 0.0


class RecursiveEngine(Engine):
    """
    Execution engine backed by Python call stack (Executor.reduce).

    This is the original lambdagent execution model:
      term.apply(input, ctx) → result (recursive β-reduction)

    Advantages: simple, fast, minimal overhead.
    Disadvantages: no pause/resume, no per-step cost monitoring,
                   no explicit continuation stack.
    """

    def __init__(self, config: RuntimeConfig | None = None):
        self._config = config

    def execute(self, term: Term, input_val: Any, ctx: Context, **opts) -> EngineResult:
        """Synchronous execution via Executor.reduce()."""
        t0 = time.time()

        if self._config:
            executor = Executor(self._config)
            result = executor.reduce(term, input_val, ctx)
        else:
            # Direct apply if no config (lightweight mode)
            result = term.apply(input_val, ctx)

        duration_ms = (time.time() - t0) * 1000

        # Convert ctx.trace to unified format
        trace = _convert_trace(ctx.trace)
        cost = _compute_cost(ctx.trace)

        return EngineResult(
            value=result,
            trace=trace,
            cost=cost,
            steps=len(ctx.trace),
            engine_mode=EngineMode.RECURSIVE,
            final_state=None,
            transitions=None,
        )

    async def execute_async(
        self, term: Term, input_val: Any, ctx: Context, **opts
    ) -> EngineResult:
        """Asynchronous execution via term.aapply()."""
        t0 = time.time()

        cancel = opts.get("cancel", None)
        result = await term.aapply(input_val, ctx, cancel)

        duration_ms = (time.time() - t0) * 1000

        trace = _convert_trace(ctx.trace)
        cost = _compute_cost(ctx.trace)

        return EngineResult(
            value=result,
            trace=trace,
            cost=cost,
            steps=len(ctx.trace),
            engine_mode=EngineMode.RECURSIVE,
            final_state=None,
            transitions=None,
        )


# ============================================================
# Trace Conversion (E04 — RecursiveEngine side)
# ============================================================


def _convert_trace(entries: List[TraceEntry]) -> List[UnifiedTraceRecord]:
    """Convert ctx.trace (List[TraceEntry]) to unified format."""
    records = []
    cumulative = ZERO_COST
    for i, e in enumerate(entries):
        step_cost = CostVector(
            tokens=e.tokens_used,
            latency=e.duration_ms / 1000,
            money=_estimate_money(e.model, e.tokens_used),
        )
        cumulative = cumulative + step_cost
        records.append(
            UnifiedTraceRecord(
                step=i,
                term_name=e.term_name,
                action="llm_call" if e.model else "tool_call",
                input_summary=str(e.input)[:200],
                output_summary=str(e.output)[:200],
                duration_ms=e.duration_ms,
                cost=step_cost,
                cumulative_cost=cumulative,
                continuation=None,  # recursive engine has no K stack
                yield_type=None,
            )
        )
    return records


def _compute_cost(entries: List[TraceEntry]) -> CostVector:
    """Sum total cost from trace entries."""
    total = ZERO_COST
    for e in entries:
        total = total + CostVector(
            tokens=e.tokens_used,
            latency=e.duration_ms / 1000,
            money=_estimate_money(e.model, e.tokens_used),
        )
    return total
