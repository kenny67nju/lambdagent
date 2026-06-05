"""
lambdagent.cek_machine — Agent CEK Machine

An abstract machine implementation of the operational semantics
defined in Paper II. The CEK machine provides:

  1. Step-by-step execution with full state inspection
  2. YIELD mechanism for LLM/Tool oracle calls
  3. Complete trace of all transitions (dispatch, yield, return)
  4. Cost vector tracking at every step
  5. Effect handler integration (Paper III §6)
  6. Cost monotonicity invariant (Paper II Proposition 23)
  7. Async run support via run_async()

The machine corresponds 1-to-1 with the small-step reduction rules
in Paper II §4, and the transition rules in §5.

References:
  - Felleisen & Friedman (1986): CEK Machine
  - Paper II §5: Agent CEK Machine
  - Paper III §6: Algebraic Effect Handlers
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum, auto


# ============================================================
# Values and Cost
# ============================================================


@dataclass(frozen=True)
class CostVector:
    """Cost vector c = (tokens, latency_s, money_usd)"""

    tokens: int = 0
    latency: float = 0.0
    money: float = 0.0

    def __add__(self, other: CostVector) -> CostVector:
        return CostVector(
            self.tokens + other.tokens,
            self.latency + other.latency,
            self.money + other.money,
        )

    def __repr__(self):
        return f"Cost(tok={self.tokens}, lat={self.latency:.2f}s, ${self.money:.4f})"


ZERO_COST = CostVector()


class CostMonotonicityViolation(RuntimeError):
    """Paper II Proposition 23: cost must be monotonically non-decreasing."""

    pass


# ============================================================
# Labels (observable actions)
# ============================================================


class LabelKind(Enum):
    TAU = auto()  # silent / internal
    LLM = auto()  # LLM oracle call
    TOOL = auto()  # tool oracle call
    MEM = auto()  # memory store update


@dataclass(frozen=True)
class Label:
    kind: LabelKind
    name: str = ""
    input: Any = None
    output: Any = None

    def __repr__(self):
        if self.kind == LabelKind.TAU:
            return "τ"
        return f"{self.kind.name.lower()}({self.name})"


TAU = Label(LabelKind.TAU)


# ============================================================
# Continuation Frames (K)
# ============================================================


class Kont:
    """Base class for continuation frames."""

    pass


class HaltK(Kont):
    """Top-level: machine is done."""

    def __repr__(self):
        return "halt"


class CompK(Kont):
    """Composition: apply g to result of f."""

    def __init__(self, g_term, prev: Kont):
        self.g_term = g_term
        self.prev = prev

    def __repr__(self):
        return f"compK({self.g_term}) :: {self.prev}"


class LoopK(Kont):
    """Loop: check condition after body evaluation."""

    def __init__(self, body, cond, remaining: int, prev: Kont):
        self.body = body
        self.cond = cond
        self.remaining = remaining
        self.prev = prev

    def __repr__(self):
        return f"loopK(n={self.remaining}) :: {self.prev}"


class PairLK(Kont):
    """Pair: left branch done, need to start right."""

    def __init__(self, g_term, input_val, prev: Kont):
        self.g_term = g_term
        self.input_val = input_val
        self.prev = prev

    def __repr__(self):
        return f"pairLK({self.g_term}) :: {self.prev}"


class PairRK(Kont):
    """Pair: right branch evaluating, left value stored."""

    def __init__(self, left_val, prev: Kont):
        self.left_val = left_val
        self.prev = prev

    def __repr__(self):
        return f"pairRK :: {self.prev}"


class GuardK(Kont):
    """Guard: check predicate on result."""

    def __init__(self, predicate, retries: int, agent_term, input_val, prev: Kont):
        self.predicate = predicate
        self.retries = retries
        self.agent_term = agent_term
        self.input_val = input_val
        self.prev = prev

    def __repr__(self):
        return f"guardK(k={self.retries}) :: {self.prev}"


class MemK(Kont):
    """Memory: store result after inner computation."""

    def __init__(self, store_key: str, prev: Kont):
        self.store_key = store_key
        self.prev = prev

    def __repr__(self):
        return f"memK({self.store_key}) :: {self.prev}"


class IfK(Kont):
    """If: condition evaluated, dispatch to then/else based on result."""

    def __init__(self, then_term, else_term, input_val, prev: Kont):
        self.then_term = then_term
        self.else_term = else_term
        self.input_val = input_val
        self.prev = prev

    def __repr__(self):
        return f"ifK :: {self.prev}"


class RouteK(Kont):
    """Route: classifier evaluated, dispatch to route based on label."""

    def __init__(self, routes: dict, default, input_val, prev: Kont):
        self.routes = routes
        self.default = default
        self.input_val = input_val
        self.prev = prev

    def __repr__(self):
        return f"routeK :: {self.prev}"


# ============================================================
# CEK Machine State
# ============================================================


@dataclass
class CEKState:
    """
    Machine state = <C, E, K, σ, c>

    C: control (current term — a lambdagent Term or a value)
    E: environment (dict of bindings)
    K: continuation (stack of frames)
    store: memory store (dict)
    cost: cumulative cost vector
    """

    control: Any  # Term or value
    env: Dict[str, Any]  # environment
    kont: Kont  # continuation stack
    store: Dict[str, Any]  # memory store
    cost: CostVector  # cumulative cost

    def is_terminal(self) -> bool:
        """Terminal if control is a value and continuation is halt."""
        return isinstance(self.kont, HaltK) and self._is_value(self.control)

    @staticmethod
    def _is_value(x) -> bool:
        """Values are non-Term Python objects (str, int, dict, tuple, etc.)
        Auxiliary CEK terms (_AppTerm, _FstTerm, _SndTerm, _LoopState) are NOT values."""
        from .core import Term

        if isinstance(x, (Term, _AppTerm, _FstTerm, _SndTerm, _LoopState)):
            return False
        return True

    @property
    def per_step_costs(self) -> List[CostVector]:
        """Per-step cost breakdown (Paper II Definition 5)"""
        return []  # populated from machine trace externally


# ============================================================
# Transition Trace Entry
# ============================================================


@dataclass
class Transition:
    """Record of one CEK transition."""

    step: int
    rule: str  # e.g. "C-Comp", "C-Lam", "C-CompRet"
    label: Label
    state_before: str  # string repr of state before
    state_after: str  # string repr of state after
    cost_delta: CostVector  # cost added in this step
    duration_ms: float  # wall-clock time


# ============================================================
# Agent CEK Machine
# ============================================================


class AgentCEKMachine:
    """
    Agent CEK Machine — step-by-step executor for lambdagent programs.

    Usage:
        from lambdagent.cek_machine import AgentCEKMachine

        machine = AgentCEKMachine()
        result = machine.run(agent, input_val)

        # Or step-by-step:
        machine.load(agent, input_val)
        while not machine.state.is_terminal():
            transition = machine.step()
            print(transition.rule, transition.label)
        result = machine.state.control
    """

    def __init__(
        self,
        store: Dict[str, Any] | None = None,
        handler=None,
        check_cost_monotonicity: bool = True,
    ):
        """
        Args:
            store: initial memory store
            handler: EffectHandler (Paper III §6) — if provided, LLM/Tool
                     calls are routed through the handler
            check_cost_monotonicity: if True, assert Paper II Proposition 23
                     (cost never decreases) after every step
        """
        self.state: Optional[CEKState] = None
        self.trace: List[Transition] = []
        self.step_count: int = 0
        self._initial_store = store or {}
        self._handler = handler
        self._check_cost_monotonicity = check_cost_monotonicity

    def load(self, term, input_val: Any) -> CEKState:
        """Load a term into the machine, creating the initial state.

        The initial control is the application (term input_val).
        """
        from .core import Term

        # Create application term: term applied to input_val
        app = _AppTerm(term, input_val)

        self.state = CEKState(
            control=app,
            env={},
            kont=HaltK(),
            store=dict(self._initial_store),
            cost=ZERO_COST,
        )
        self.trace = []
        self.step_count = 0
        return self.state

    def _is_stuck(self) -> bool:
        """Check if machine is stuck (control is non-value, non-dispatchable)."""
        s = self.state
        if s is None:
            return True
        ctrl = s.control
        if s._is_value(ctrl):
            return False
        if isinstance(ctrl, (_AppTerm, _FstTerm, _SndTerm)):
            return False
        return True

    def step(self) -> Transition:
        """Execute one machine transition. Returns the transition record."""
        if self.state is None:
            raise RuntimeError("Machine not loaded. Call load() first.")
        if self.state.is_terminal():
            raise RuntimeError("Machine already halted.")

        t0 = time.time()
        state_before = self._state_repr()
        cost_before = self.state.cost

        rule, label = self._dispatch()

        duration_ms = (time.time() - t0) * 1000
        cost_delta = CostVector(
            self.state.cost.tokens - cost_before.tokens,
            self.state.cost.latency - cost_before.latency,
            self.state.cost.money - cost_before.money,
        )

        # Paper II Proposition 23: cost monotonicity — c' ≥ c component-wise
        if self._check_cost_monotonicity and (
            cost_delta.tokens < 0 or cost_delta.latency < 0 or cost_delta.money < 0
        ):
            raise CostMonotonicityViolation(
                f"Cost decreased at step {self.step_count + 1} (rule {rule}): "
                f"delta={cost_delta}. Paper II Prop. 23 requires c' ≥ c."
            )

        self.step_count += 1
        transition = Transition(
            step=self.step_count,
            rule=rule,
            label=label,
            state_before=state_before,
            state_after=self._state_repr(),
            cost_delta=cost_delta,
            duration_ms=duration_ms,
        )
        self.trace.append(transition)
        return transition

    def run(self, term, input_val: Any, max_steps: int = 10000) -> Any:
        """Run to completion. Returns the final value."""
        self.load(term, input_val)
        while not self.state.is_terminal():
            if self.step_count >= max_steps:
                raise RuntimeError(f"CEK machine exceeded {max_steps} steps")
            self.step()
        return self.state.control

    async def run_async(self, term, input_val: Any, max_steps: int = 10000) -> Any:
        """
        Async version of run() — yields control at LLM/Tool calls.

        Paper II §5: The Yield mechanism maps naturally to async/await.
        Each C-Lam and C-Tool step is an await point.
        """
        import asyncio

        self.load(term, input_val)
        while not self.state.is_terminal():
            if self.step_count >= max_steps:
                raise RuntimeError(f"CEK machine exceeded {max_steps} steps")
            # Yield to event loop at each step (non-blocking)
            await asyncio.sleep(0)
            self.step()
        return self.state.control

    def cost_summary(self) -> Dict[str, Any]:
        """
        Cost summary for the entire execution.

        Paper II Definition 5: c = (tokens, latency_s, money_usd)
        """
        if self.state is None:
            return {"tokens": 0, "latency_s": 0.0, "money_usd": 0.0, "steps": 0}
        c = self.state.cost
        # Per-step cost breakdown
        per_step = [
            (t.step, t.rule, t.cost_delta)
            for t in self.trace
            if t.cost_delta.tokens > 0 or t.cost_delta.latency > 0
        ]
        return {
            "tokens": c.tokens,
            "latency_s": c.latency,
            "money_usd": c.money,
            "steps": self.step_count,
            "llm_calls": sum(1 for t in self.trace if t.label.kind == LabelKind.LLM),
            "tool_calls": sum(1 for t in self.trace if t.label.kind == LabelKind.TOOL),
            "per_step_costs": per_step,
        }

    def print_trace(self):
        """Print the full transition trace."""
        for t in self.trace:
            cost_str = f" +{t.cost_delta}" if t.cost_delta.tokens > 0 else ""
            print(
                f"  [{t.step:3d}] {t.rule:<16s} {t.label}{cost_str}  ({t.duration_ms:.1f}ms)"
            )

    # ── Internal dispatch ─────────────────────────────────────

    def _dispatch(self) -> Tuple[str, Label]:
        """Main dispatch: examine control and continuation, apply transition."""
        from .primitives import Lam, Compose, If, Loop, Pair, Tool
        from .extensions import Route, Guard, Memory, Par
        from .core import Term

        s = self.state
        ctrl = s.control

        # ── If control is a value, pop continuation ──
        if s._is_value(ctrl):
            return self._return(ctrl)

        # ── Application: term applied to value ──
        if isinstance(ctrl, _AppTerm):
            return self._dispatch_app(ctrl.operator, ctrl.operand)

        # ── Fst / Snd ──
        if isinstance(ctrl, _FstTerm):
            if isinstance(ctrl.inner, tuple) and len(ctrl.inner) == 2:
                s.control = ctrl.inner[0]
                return "C-Fst", TAU
            # inner not yet a pair value — should not happen in well-formed programs
            raise RuntimeError(f"Fst applied to non-pair: {ctrl.inner}")

        if isinstance(ctrl, _SndTerm):
            if isinstance(ctrl.inner, tuple) and len(ctrl.inner) == 2:
                s.control = ctrl.inner[1]
                return "C-Snd", TAU
            raise RuntimeError(f"Snd applied to non-pair: {ctrl.inner}")

        # ── Term that needs to be applied but has no operand ──
        # This shouldn't happen in well-formed programs
        raise RuntimeError(
            f"CEK stuck: control is {type(ctrl).__name__}, not a value or application"
        )

    def _dispatch_app(self, op, arg) -> Tuple[str, Label]:
        """Dispatch an application (op arg)."""
        from .primitives import Lam, Compose, If, Loop, Pair, Tool
        from .extensions import Route, Guard, Memory, Par
        from .core import Term, Context

        s = self.state

        # ── C-Lam: YIELD to LLM oracle ──
        # Paper III §6: route through handler if available
        if isinstance(op, Lam) or _is_claude_lam(op):
            t0 = time.time()
            handler = self._handler
            if handler is not None and not isinstance(
                handler, _passthrough_handler_types()
            ):
                # Use handler for LLM effect
                result = handler.handle_llm(
                    getattr(op, "prompt", ""),
                    str(arg),
                    getattr(op, "model", "unknown"),
                    temperature=getattr(op, "temperature", 0.0),
                    max_tokens=getattr(op, "max_tokens", 1024),
                )
                if hasattr(op, "output_parser"):
                    result = op.output_parser(result)
            else:
                ctx = Context(bindings=s.env, memory=s.store)
                result = op.apply(arg, ctx)
            elapsed = time.time() - t0

            model_name = getattr(op, "model", "claude-code")
            tokens = 0  # handler may not report tokens; real calls update via trace
            cost_llm = CostVector(
                tokens=tokens,
                latency=elapsed,
                money=tokens * _price_per_token(model_name),
            )
            label = Label(LabelKind.LLM, op._name, arg, result)
            s.control = result
            s.cost = s.cost + cost_llm
            return "C-Lam", label

        # ── C-Tool: YIELD to tool oracle ──
        # Paper III §6: route through handler if available
        if isinstance(op, Tool):
            t0 = time.time()
            handler = self._handler
            if handler is not None and not isinstance(
                handler, _passthrough_handler_types()
            ):
                result = handler.handle_tool(op._name, op.fn, arg)
            else:
                ctx = Context(bindings=s.env, memory=s.store)
                result = op.apply(arg, ctx)
            elapsed = time.time() - t0

            cost_tool = CostVector(latency=elapsed)
            label = Label(LabelKind.TOOL, op._name, arg, result)
            s.control = result
            s.cost = s.cost + cost_tool
            return "C-Tool", label

        # ── C-Comp: push compK, evaluate f(v) ──
        if isinstance(op, Compose):
            stages = op.stages if hasattr(op, "stages") else [op.f, op.g]
            if len(stages) == 2:
                f, g = stages
                s.control = _AppTerm(f, arg)
                s.kont = CompK(g, s.kont)
                return "C-Comp", TAU
            else:
                # Multi-stage: (f >> g >> h)(v) → push all but first
                first = stages[0]
                rest = stages[1:]
                # Build nested CompK: compK(h, compK(g, ...))
                for stage in reversed(rest):
                    s.kont = CompK(stage, s.kont)
                s.control = _AppTerm(first, arg)
                return "C-Comp", TAU

        # ── C-If: push IfK if cond is a Term, else inline ──
        if isinstance(op, If):
            cond = op.cond
            if isinstance(cond, Term):
                # Proper CEK: evaluate condition first, then dispatch
                s.control = _AppTerm(cond, arg)
                s.kont = IfK(op.then_, op.else_, arg, s.kont)
                return "C-If", TAU
            else:
                # Python callable condition — inline dispatch
                cond_result = cond(arg)
                if cond_result:
                    s.control = _AppTerm(op.then_, arg)
                else:
                    s.control = _AppTerm(op.else_, arg)
                return "C-If", TAU

        # ── C-Route: push RouteK, evaluate classifier ──
        if isinstance(op, Route):
            s.control = _AppTerm(op.classifier, arg)
            s.kont = RouteK(op.routes, op.default, arg, s.kont)
            return "C-Route", TAU

        # ── C-Loop: check condition or unfold ──
        if isinstance(op, (Loop, _LoopState)):
            cond = op.condition
            remaining = op.max_steps

            if remaining == 0:
                # E-LoopBound: return value
                s.control = arg
                return "C-LoopBound", TAU

            # Loop.condition takes (result, step) — we pass step=0 here
            # since the CEK machine tracks iteration via remaining count
            step_idx = (op.max_steps if isinstance(op, Loop) else remaining) - remaining
            try:
                cond_result = cond(arg, step_idx)
            except TypeError:
                # Some conditions only take one arg
                cond_result = cond(arg)

            if cond_result:
                # E-LoopBase: terminate
                s.control = arg
                return "C-LoopBase", TAU
            else:
                # E-LoopUnfold: evaluate body, push loopK
                body = op.body
                new_remaining = remaining - 1
                s.control = _AppTerm(body, arg)
                s.kont = LoopK(body, cond, new_remaining, s.kont)
                return "C-LoopUnfold", TAU

        # ── C-Pair: evaluate left branch, push pairLK ──
        if isinstance(op, (Pair, Par)):
            f = op.first if hasattr(op, "first") else op.f
            g = op.second if hasattr(op, "second") else op.g
            s.control = _AppTerm(f, arg)
            s.kont = PairLK(g, arg, s.kont)
            return "C-Pair", TAU

        # ── C-Guard: evaluate inner agent, push guardK ──
        if isinstance(op, Guard):
            s.control = _AppTerm(op.agent, arg)
            retries = op.retry
            s.kont = GuardK(op.validator, retries, op.agent, arg, s.kont)
            return "C-Guard", TAU

        # ── C-Mem: evaluate inner agent, push memK ──
        if isinstance(op, Memory):
            s.control = _AppTerm(op.agent, arg)
            store_key = op._name  # use Memory's name as store key
            s.kont = MemK(store_key, s.kont)
            return "C-Mem", TAU

        # ── Fallback: try direct application ──
        if isinstance(op, Term):
            ctx = Context(bindings=s.env, memory=s.store)
            result = op.apply(arg, ctx)
            s.control = result
            # Copy any trace/memory updates
            s.store.update(ctx.memory)
            return "C-Generic", TAU

        raise RuntimeError(f"CEK dispatch: unknown operator type {type(op).__name__}")

    def _return(self, val) -> Tuple[str, Label]:
        """Return a value to the continuation."""
        s = self.state
        k = s.kont

        # ── C-Halt ──
        if isinstance(k, HaltK):
            # Already terminal — should not be called
            return "C-Halt", TAU

        # ── C-CompRet: pass value to next stage ──
        if isinstance(k, CompK):
            s.control = _AppTerm(k.g_term, val)
            s.kont = k.prev
            return "C-CompRet", TAU

        # ── C-LoopRet: feed body result back into loop ──
        if isinstance(k, LoopK):
            # Re-enter loop with the body's result as new value
            new_loop = _LoopState(k.body, k.cond, k.remaining)
            s.control = _AppTerm(new_loop, val)
            s.kont = k.prev
            return "C-LoopRet", TAU

        # ── C-PairMid: left done, start right ──
        if isinstance(k, PairLK):
            s.control = _AppTerm(k.g_term, k.input_val)
            s.kont = PairRK(val, k.prev)
            return "C-PairMid", TAU

        # ── C-PairRet: both done, construct pair ──
        if isinstance(k, PairRK):
            s.control = (k.left_val, val)
            s.kont = k.prev
            return "C-PairRet", TAU

        # ── C-GuardRet: check predicate ──
        if isinstance(k, GuardK):
            try:
                passed = k.predicate(val)
            except Exception:
                passed = False

            if passed:
                s.control = val
                s.kont = k.prev
                return "C-GuardOK", TAU
            elif k.retries > 0:
                # Retry: re-evaluate inner agent
                s.control = _AppTerm(k.agent_term, k.input_val)
                s.kont = GuardK(
                    k.predicate, k.retries - 1, k.agent_term, k.input_val, k.prev
                )
                return "C-GuardRetry", TAU
            else:
                s.control = None  # unit
                s.kont = k.prev
                return "C-GuardFail", TAU

        # ── C-MemRet: store result ──
        if isinstance(k, MemK):
            s.store[k.store_key] = val
            label = Label(LabelKind.MEM, k.store_key, None, val)
            s.control = val
            s.kont = k.prev
            return "C-MemRet", label

        # ── C-IfRet: condition evaluated, dispatch to then/else ──
        if isinstance(k, IfK):
            from .primitives import If

            is_true = If._is_truthy(val) if isinstance(val, str) else bool(val)
            if is_true:
                s.control = _AppTerm(k.then_term, k.input_val)
            else:
                s.control = _AppTerm(k.else_term, k.input_val)
            s.kont = k.prev
            return "C-IfRet", TAU

        # ── C-RouteRet: classifier evaluated, dispatch to route ──
        if isinstance(k, RouteK):
            label_str = str(val).strip().lower()
            agent = k.routes.get(label_str)
            if agent is None:
                # Fuzzy match
                for key, a in k.routes.items():
                    if key.lower() in label_str or label_str in key.lower():
                        agent = a
                        break
            if agent is None:
                agent = k.default
            if agent is None:
                raise RuntimeError(
                    f"Route: no branch for label '{val}'. "
                    f"Available: {list(k.routes.keys())}"
                )
            s.control = _AppTerm(agent, k.input_val)
            s.kont = k.prev
            return "C-RouteRet", TAU

        raise RuntimeError(f"CEK return: unknown continuation {type(k).__name__}")

    def _state_repr(self) -> str:
        """Compact string representation of current state."""
        s = self.state
        ctrl_str = _term_repr(s.control)
        return f"<{ctrl_str}, K={s.kont}, σ={len(s.store)} keys, {s.cost}>"


# ============================================================
# Internal helper terms (not part of source syntax)
# ============================================================


class _AppTerm:
    """Application: operator applied to operand. Auxiliary term for CEK."""

    def __init__(self, operator, operand):
        self.operator = operator
        self.operand = operand

    def __repr__(self):
        return f"({_term_repr(self.operator)} {_term_repr(self.operand)})"


class _FstTerm:
    """Fst projection. Auxiliary term for CEK."""

    def __init__(self, inner):
        self.inner = inner


class _SndTerm:
    """Snd projection. Auxiliary term for CEK."""

    def __init__(self, inner):
        self.inner = inner


class _LoopState:
    """Represents a Loop with reduced remaining count. Auxiliary for CEK."""

    def __init__(self, body, cond, remaining: int):
        self.body = body
        self.condition = cond
        self.max_steps = remaining


def _term_repr(t) -> str:
    """Compact representation of a term or value."""
    if isinstance(t, _AppTerm):
        return f"({_term_repr(t.operator)} {_term_repr(t.operand)})"
    if isinstance(t, str):
        s = t[:40] + "..." if len(t) > 40 else t
        return f'"{s}"'
    if isinstance(t, tuple):
        return f"({_term_repr(t[0])}, {_term_repr(t[1])})"
    if hasattr(t, "_name"):
        return t._name
    return repr(t)[:40]


def _is_claude_lam(op) -> bool:
    """Check if op is a ClaudeLam instance (without importing it directly,
    to avoid hard dependency on agentexample)."""
    return type(op).__name__ == "ClaudeLam"


def _passthrough_handler_types():
    """Handler types that should use direct apply() — lazy import.

    DESIGN-01: Use PassthroughHandler base class instead of listing individual types.
    """
    from .handlers import PassthroughHandler

    return (PassthroughHandler,)


def _price_per_token(model: str) -> float:
    """Rough per-token price for cost tracking."""
    prices = {
        "claude-sonnet-4-20250514": 3e-6,
        "claude-opus-4-20250514": 15e-6,
        "claude-haiku-4-5-20251001": 0.25e-6,
        "gpt-4": 30e-6,
        "gpt-4o": 2.5e-6,
        "qwen3-max": 1e-6,
    }
    for key, price in prices.items():
        if key in model:
            return price
    return 1e-6  # default
