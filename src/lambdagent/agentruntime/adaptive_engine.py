"""
agentruntime.adaptive_engine — Auto-selects Recursive or CEK based on term complexity.

Phase 6.5 (E06): Analyzes the compiled Term tree to decide which engine
is appropriate. Simple agents use Recursive (low overhead), complex agents
use CEK (pause/resume, cost monitoring, loop detection).

Decision criteria:
  - max_possible_steps > 10  → CEK (long loops need cost monitoring)
  - has_parallel              → CEK (Pair confluence needs tracking)
  - has_guard with retry > 0  → CEK (retries can explode cost)
  - estimated_cost > $1.00    → CEK (expensive runs need budget control)
  - Otherwise                 → Recursive (minimal overhead)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from lambdagent.core import Term, Context
from lambdagent.primitives import Lam, Compose, If, Loop, Pair, Tool
from lambdagent.extensions import Par, Route, Memory, Guard

from .engine import Engine, EngineMode, EngineResult
from .recursive_engine import RecursiveEngine
from .cek_engine import CEKEngine
from .config import RuntimeConfig


@dataclass
class Complexity:
    """Term complexity assessment for engine selection."""

    max_possible_steps: int = 1
    has_parallel: bool = False
    has_guard: bool = False
    guard_max_retries: int = 0
    estimated_cost_usd: float = 0.0
    depth: int = 0

    @property
    def should_use_cek(self) -> bool:
        """Decision: should we use CEK for this term?"""
        if self.max_possible_steps > 10:
            return True
        if self.has_parallel:
            return True
        if self.has_guard and self.guard_max_retries > 1:
            return True
        if self.estimated_cost_usd > 1.0:
            return True
        return False


def assess_complexity(term: Term) -> Complexity:
    """
    Walk the term tree and compute complexity metrics.

    This is a lightweight static analysis — no LLM calls, O(n) in term size.
    """
    c = Complexity()
    _walk(term, c, depth=0)

    # Estimate cost if cost_grade module is available
    try:
        from lambdagent.cost_grade import estimate_cost

        grade = estimate_cost(term)
        c.estimated_cost_usd = grade.money
    except Exception:
        pass

    return c


def _walk(term: Term, c: Complexity, depth: int):
    """Recursive tree walk to gather complexity metrics."""
    c.depth = max(c.depth, depth)

    if isinstance(term, Loop):
        max_steps = getattr(term, "max_steps", 10)
        c.max_possible_steps += max_steps
        if hasattr(term, "body"):
            _walk(term.body, c, depth + 1)

    elif isinstance(term, Compose):
        stages = getattr(term, "stages", [])
        for stage in stages:
            _walk(stage, c, depth + 1)

    elif isinstance(term, (Pair, Par)):
        c.has_parallel = True
        agents = getattr(term, "agents", [])
        if not agents:
            # Pair has first/second
            first = getattr(term, "first", None)
            second = getattr(term, "second", None)
            if first:
                _walk(first, c, depth + 1)
            if second:
                _walk(second, c, depth + 1)
        else:
            for agent in agents:
                _walk(agent, c, depth + 1)

    elif isinstance(term, Guard):
        c.has_guard = True
        retry = getattr(term, "retry", 0)
        c.guard_max_retries = max(c.guard_max_retries, retry)
        c.max_possible_steps += retry
        if hasattr(term, "agent"):
            _walk(term.agent, c, depth + 1)

    elif isinstance(term, Route):
        routes = getattr(term, "routes", {})
        for route_term in routes.values():
            if isinstance(route_term, Term):
                _walk(route_term, c, depth + 1)

    elif isinstance(term, If):
        then_ = getattr(term, "then_", None)
        else_ = getattr(term, "else_", None)
        if then_ and isinstance(then_, Term):
            _walk(then_, c, depth + 1)
        if else_ and isinstance(else_, Term):
            _walk(else_, c, depth + 1)

    elif isinstance(term, Memory):
        if hasattr(term, "agent"):
            _walk(term.agent, c, depth + 1)

    # Check for multiagent constructs
    try:
        from lambdagent.multiagent import GroupChat, AsyncPar, Handoff

        if isinstance(term, (GroupChat, AsyncPar)):
            c.has_parallel = True
            agents = getattr(term, "agents", [])
            max_rounds = getattr(term, "max_rounds", 10)
            c.max_possible_steps += max_rounds * len(agents)
            for agent in agents:
                _walk(agent, c, depth + 1)
        elif isinstance(term, Handoff):
            registry = getattr(term, "registry", {})
            for agent in registry.values():
                if isinstance(agent, Term):
                    _walk(agent, c, depth + 1)
    except ImportError:
        pass


class AdaptiveEngine(Engine):
    """
    Auto-selects RecursiveEngine or CEKEngine based on term complexity.

    Simple agents (type: simple, short pipelines) → Recursive (fast).
    Complex agents (long loops, parallel, guards, expensive) → CEK (safe).
    """

    def __init__(self, config: RuntimeConfig | None = None, **engine_opts):
        self._config = config
        self._engine_opts = engine_opts

    def execute(self, term: Term, input_val: Any, ctx: Context, **opts) -> EngineResult:
        complexity = assess_complexity(term)
        engine = self._select(complexity)
        result = engine.execute(term, input_val, ctx, **opts)
        return result

    async def execute_async(
        self, term: Term, input_val: Any, ctx: Context, **opts
    ) -> EngineResult:
        complexity = assess_complexity(term)
        engine = self._select(complexity)
        result = await engine.execute_async(term, input_val, ctx, **opts)
        return result

    def _select(self, complexity: Complexity) -> Engine:
        """Select engine based on complexity assessment."""
        if complexity.should_use_cek:
            return CEKEngine(**self._engine_opts)
        return RecursiveEngine(config=self._config)
