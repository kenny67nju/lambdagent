"""
agentruntime.engine — Unified Engine abstraction for dual execution backends.

Phase 6.5 (E01): Provides a switchable interface between:
  - RecursiveEngine: Python call stack (Executor.reduce + term.apply)
  - CEKEngine: Agent CEK Machine (step-by-step, pause/resume, cost monitoring)
  - AdaptiveEngine: Auto-selects based on term complexity

Both engines take the same input (Term + input + Context) and produce the
same output (EngineResult), ensuring behavioral equivalence.

References:
  - Paper II §5: Agent CEK Machine
  - Paper II Proposition 23: Cost monotonicity
  - Paper III §4.3: Graded types for cost prediction
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from lambdagent.core import Term, Context, TraceEntry
from lambdagent.cek_machine import CostVector, ZERO_COST


# ============================================================
# Engine Mode
# ============================================================


class EngineMode(Enum):
    """Execution engine selection."""

    RECURSIVE = "recursive"  # Python call stack (Executor.reduce)
    CEK = "cek"  # Agent CEK Machine (step-by-step)
    ADAPTIVE = "adaptive"  # Auto-select based on term complexity


# ============================================================
# Unified Trace Record
# ============================================================


@dataclass
class UnifiedTraceRecord:
    """
    Unified trace record — bridge between TraceEntry (recursive) and
    Transition (CEK). Both engines output this format.

    Recursive engine: maps from TraceEntry fields.
    CEK engine: maps from Transition fields, skipping silent (τ) steps.
    """

    step: int
    term_name: str
    action: str  # "llm_call" | "tool_call" | "compose" | "route" | ...
    input_summary: str  # truncated input (≤200 chars)
    output_summary: str  # truncated output (≤200 chars)
    duration_ms: float
    cost: CostVector  # cost of this step
    cumulative_cost: CostVector  # total cost so far

    # CEK-only fields (None for recursive engine)
    continuation: Optional[str] = None  # human-readable K stack
    yield_type: Optional[str] = None  # "llm" | "tool" | None


# ============================================================
# Engine Result
# ============================================================


@dataclass
class EngineResult:
    """
    Unified execution result from any engine.

    Both RecursiveEngine and CEKEngine produce this.
    The `engine_mode` field indicates which engine was used.
    """

    value: Any  # final output value
    trace: List[UnifiedTraceRecord]  # unified trace records
    cost: CostVector  # total cost
    steps: int  # number of meaningful steps
    engine_mode: EngineMode  # which engine ran

    # CEK-only fields (None for recursive engine)
    final_state: Optional[Any] = None  # CEKState — serializable
    transitions: Optional[List[Any]] = None  # List[Transition] — full small-step log


# ============================================================
# Engine Exceptions
# ============================================================


class CostBudgetExceeded(RuntimeError):
    """Raised when execution cost exceeds the configured budget ceiling."""

    def __init__(self, cost: CostVector, budget: float, state: Optional[Any] = None):
        self.cost = cost
        self.budget = budget
        self.state = state  # CEKState if available, for resume
        super().__init__(
            f"Cost budget exceeded: ${cost.money:.4f} > ${budget:.4f} "
            f"(tokens={cost.tokens})"
        )


class InfiniteLoopDetected(RuntimeError):
    """Raised when repeated identical states suggest no progress."""

    def __init__(self, message: str, state: Optional[Any] = None):
        self.state = state
        super().__init__(message)


class MaxStepsExceeded(RuntimeError):
    """Raised when CEK machine exceeds maximum step count."""

    def __init__(self, max_steps: int, state: Optional[Any] = None):
        self.state = state
        super().__init__(f"Exceeded {max_steps} CEK steps")


# ============================================================
# Engine ABC
# ============================================================


class Engine(ABC):
    """
    Abstract execution engine.

    Both RecursiveEngine and CEKEngine implement this interface.
    Runtime selects which engine to use based on configuration.

    Contract:
      - execute(term, input_val, ctx) → EngineResult
      - ctx.trace is populated with TraceEntry records (backward compat)
      - EngineResult.trace has UnifiedTraceRecord records (new format)
      - Both engines produce identical EngineResult.value for the same input
    """

    @abstractmethod
    def execute(self, term: Term, input_val: Any, ctx: Context, **opts) -> EngineResult:
        """Synchronous execution."""
        ...

    @abstractmethod
    async def execute_async(
        self, term: Term, input_val: Any, ctx: Context, **opts
    ) -> EngineResult:
        """Asynchronous execution (non-blocking LLM/Tool calls)."""
        ...
