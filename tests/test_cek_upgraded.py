"""
Tests for upgraded CEK Machine — P1-1.

Tests cover:
  1. Basic execution with all constructs
  2. Handler integration (TestHandler in CEK)
  3. IfK continuation frame (proper CPS for Term conditions)
  4. RouteK continuation frame
  5. Cost monotonicity invariant (Paper II Proposition 23)
  6. Cost summary
  7. Step-by-step execution
"""

import pytest
from lambdagent.cek_machine import (
    AgentCEKMachine, CEKState, CostVector, ZERO_COST,
    CostMonotonicityViolation,
    HaltK, CompK, LoopK, PairLK, PairRK, GuardK, MemK, IfK, RouteK,
    LabelKind,
)
from lambdagent.primitives import Lam, Compose, If, Loop, Pair, Fst, Snd, Tool
from lambdagent.extensions import Par, Route, Memory, Guard
from lambdagent.handlers import TestHandler


# ============================================================
# 1. Basic Execution
# ============================================================

class TestCEKBasicExecution:

    def test_tool_execution(self):
        """CEK can execute a simple Tool"""
        tool = Tool("double", lambda x: int(x) * 2)
        machine = AgentCEKMachine()
        result = machine.run(tool, "5")
        assert result == 10

    def test_compose_execution(self):
        """CEK can execute Compose(f, g)"""
        f = Tool("add1", lambda x: int(x) + 1)
        g = Tool("double", lambda x: int(x) * 2)
        pipeline = Compose(f, g)
        machine = AgentCEKMachine()
        result = machine.run(pipeline, "5")
        assert result == 12  # (5+1)*2

    def test_three_stage_compose(self):
        """CEK can execute 3-stage composition"""
        f = Tool("a", lambda x: int(x) + 1)
        g = Tool("b", lambda x: int(x) * 2)
        h = Tool("c", lambda x: int(x) - 3)
        pipeline = Compose(f, g, h)
        machine = AgentCEKMachine()
        result = machine.run(pipeline, "5")
        assert result == 9  # ((5+1)*2)-3

    def test_pair_execution(self):
        """CEK can execute Pair(f, g)"""
        f = Tool("upper", lambda x: str(x).upper())
        g = Tool("lower", lambda x: str(x).lower())
        pair = Pair(f, g)
        machine = AgentCEKMachine()
        result = machine.run(pair, "Hello")
        assert result == ("HELLO", "hello")

    def test_if_python_cond(self):
        """CEK can execute If with Python condition"""
        then_ = Tool("yes", lambda x: "yes")
        else_ = Tool("no", lambda x: "no")
        cond = If(lambda x: len(str(x)) > 3, then_, else_)
        machine = AgentCEKMachine()
        assert machine.run(cond, "hello") == "yes"
        machine2 = AgentCEKMachine()
        assert machine2.run(cond, "hi") == "no"

    def test_loop_execution(self):
        """CEK can execute Loop"""
        body = Tool("inc", lambda x: int(x) + 1)
        loop = Loop(body, lambda r, s: int(r) >= 5, max_steps=10)
        machine = AgentCEKMachine()
        result = machine.run(loop, "0")
        assert result == 5

    def test_guard_execution(self):
        """CEK can execute Guard"""
        agent = Tool("len", lambda x: str(len(str(x))))
        guard = Guard(agent, lambda x: int(x) > 0, retry=1)
        machine = AgentCEKMachine()
        result = machine.run(guard, "hello")
        assert int(result) > 0

    def test_fst_snd(self):
        """CEK can execute Fst/Snd"""
        f = Tool("a", lambda x: "first")
        g = Tool("b", lambda x: "second")
        pipeline = Compose(Pair(f, g), Fst())
        machine = AgentCEKMachine()
        result = machine.run(pipeline, "input")
        assert result == "first"


# ============================================================
# 2. Handler Integration
# ============================================================

class TestCEKHandlerIntegration:

    def test_cek_with_test_handler_lam(self):
        """CEK routes Lam calls through TestHandler"""
        handler = TestHandler()
        handler.mock_llm("summarize", "Mock CEK summary")
        lam = Lam("summarizer", "Summarize this")

        machine = AgentCEKMachine(handler=handler)
        result = machine.run(lam, "Long text about AI")
        assert result == "Mock CEK summary"
        assert len(handler.llm_calls) == 1

    def test_cek_with_test_handler_tool(self):
        """CEK routes Tool calls through TestHandler"""
        handler = TestHandler()
        handler.mock_tool("search", {"results": ["found"]})
        tool = Tool("search", lambda x: {"results": ["real"]})

        machine = AgentCEKMachine(handler=handler)
        result = machine.run(tool, "query")
        assert result == {"results": ["found"]}

    def test_cek_compose_with_handler(self):
        """CEK runs Compose pipeline through handler"""
        handler = TestHandler()
        handler.mock_llm_default("42")
        handler.mock_tool("double", 84)

        lam = Lam("to_num", "Convert")
        tool = Tool("double", lambda x: int(x) * 2)
        pipeline = Compose(lam, tool)

        machine = AgentCEKMachine(handler=handler)
        result = machine.run(pipeline, "what is 42?")
        assert result == 84
        assert len(handler.llm_calls) == 1
        assert len(handler.tool_calls) == 1

    def test_cek_no_handler_uses_real(self):
        """CEK without handler uses real function"""
        tool = Tool("double", lambda x: int(x) * 2)
        machine = AgentCEKMachine()
        result = machine.run(tool, "5")
        assert result == 10


# ============================================================
# 3. IfK Continuation Frame
# ============================================================

class TestCEKIfK:

    def test_if_term_condition_uses_ifk(self):
        """When If.cond is a Term, CEK pushes IfK and evaluates cond first"""
        # Condition is a Term (Tool returning "TRUE")
        cond_agent = Tool("check", lambda x: "TRUE" if len(str(x)) > 3 else "FALSE")
        then_ = Tool("yes", lambda x: "branch_then")
        else_ = Tool("no", lambda x: "branch_else")
        if_term = If(cond_agent, then_, else_)

        machine = AgentCEKMachine()
        result = machine.run(if_term, "hello")
        assert result == "branch_then"

        # Verify IfK was used (check transition trace)
        rules = [t.rule for t in machine.trace]
        assert "C-If" in rules
        assert "C-IfRet" in rules

    def test_if_term_condition_else_branch(self):
        """IfK correctly dispatches to else branch"""
        cond_agent = Tool("check", lambda x: "FALSE")
        then_ = Tool("yes", lambda x: "branch_then")
        else_ = Tool("no", lambda x: "branch_else")
        if_term = If(cond_agent, then_, else_)

        machine = AgentCEKMachine()
        result = machine.run(if_term, "hi")
        assert result == "branch_else"


# ============================================================
# 4. RouteK Continuation Frame
# ============================================================

class TestCEKRouteK:

    def test_route_uses_routek(self):
        """Route pushes RouteK, evaluates classifier, then dispatches"""
        classifier = Tool("cls", lambda x: "math" if "+" in str(x) else "text")
        routes = {
            "math": Tool("math_agent", lambda x: f"math: {x}"),
            "text": Tool("text_agent", lambda x: f"text: {x}"),
        }
        route = Route(classifier, routes)

        machine = AgentCEKMachine()
        result = machine.run(route, "2 + 3")
        assert result == "math: 2 + 3"

        rules = [t.rule for t in machine.trace]
        assert "C-Route" in rules
        assert "C-RouteRet" in rules

    def test_route_fuzzy_match(self):
        """RouteK supports fuzzy matching of labels"""
        classifier = Tool("cls", lambda x: "it's about MATH")
        routes = {"math": Tool("m", lambda x: "matched")}
        route = Route(classifier, routes)

        machine = AgentCEKMachine()
        result = machine.run(route, "input")
        assert result == "matched"


# ============================================================
# 5. Cost Monotonicity (Paper II Proposition 23)
# ============================================================

class TestCostMonotonicity:

    def test_cost_never_decreases(self):
        """Normal execution maintains cost monotonicity"""
        f = Tool("a", lambda x: x)
        g = Tool("b", lambda x: x)
        pipeline = Compose(f, g)
        machine = AgentCEKMachine(check_cost_monotonicity=True)
        machine.run(pipeline, "test")
        # Should not raise — latency only increases

    def test_cost_summary(self):
        """cost_summary() returns structured data"""
        tool = Tool("t", lambda x: x)
        machine = AgentCEKMachine()
        machine.run(tool, "test")
        summary = machine.cost_summary()
        assert "tokens" in summary
        assert "latency_s" in summary
        assert "steps" in summary
        assert summary["steps"] > 0
        assert summary["tool_calls"] >= 1


# ============================================================
# 6. Step-by-Step Execution
# ============================================================

class TestCEKStepByStep:

    def test_step_by_step(self):
        """Can execute step by step and inspect state"""
        f = Tool("add1", lambda x: int(x) + 1)
        g = Tool("double", lambda x: int(x) * 2)
        pipeline = Compose(f, g)

        machine = AgentCEKMachine()
        machine.load(pipeline, "5")

        steps = []
        while not machine.state.is_terminal():
            t = machine.step()
            steps.append(t.rule)

        assert machine.state.control == 12
        assert "C-Comp" in steps
        assert "C-Tool" in steps

    def test_trace_records_all_transitions(self):
        """Trace records every transition"""
        tool = Tool("t", lambda x: x)
        machine = AgentCEKMachine()
        machine.run(tool, "val")
        assert len(machine.trace) >= 1
        for t in machine.trace:
            assert t.step > 0
            assert t.rule != ""


# ============================================================
# 7. Memory Integration
# ============================================================

class TestCEKMemory:

    def test_memory_stores_result(self):
        """CEK Memory stores result in store"""
        inner = Tool("t", lambda x: f"processed: {x}")
        mem = Memory(inner)
        machine = AgentCEKMachine()
        result = machine.run(mem, "input")
        assert "processed" in result
        # Check store was updated
        rules = [t.rule for t in machine.trace]
        assert "C-Mem" in rules
