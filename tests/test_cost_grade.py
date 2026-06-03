"""
Tests for Paper III §4.3 Graded Cost Prediction.

Tests cover:
  1. CostGrade construction and properties
  2. Grade composition rules (serial, parallel, iterate, guard)
  3. estimate_cost() for all constructs
  4. Pipeline cost estimation
  5. format_cost_estimate()
"""

import pytest
from lambdagent.cost_grade import (
    CostGrade,
    grade_serial, grade_parallel, grade_iterate, grade_guard,
    estimate_cost, format_cost_estimate,
)
from lambdagent.primitives import Lam, Compose, If, Loop, Pair, Fst, Snd, Tool
from lambdagent.extensions import Par, Route, Memory, Guard


# ============================================================
# 1. CostGrade Construction
# ============================================================

class TestCostGrade:

    def test_default_grade(self):
        g = CostGrade()
        assert g.probability == 1.0
        assert g.tokens == 0
        assert g.is_free

    def test_repr(self):
        g = CostGrade(probability=0.95, tokens=800, latency=2.0, money=0.0024)
        r = repr(g)
        assert "95.00%" in r
        assert "800" in r


# ============================================================
# 2. Grade Composition Rules (Paper III Definition 12)
# ============================================================

class TestGradeComposition:

    def test_serial(self):
        """g1 · g2: probabilities multiply, everything else adds"""
        g1 = CostGrade(0.9, 100, 1.0, 0.01)
        g2 = CostGrade(0.8, 200, 2.0, 0.02)
        r = grade_serial(g1, g2)
        assert r.probability == pytest.approx(0.72)
        assert r.tokens == 300
        assert r.latency == pytest.approx(3.0)
        assert r.money == pytest.approx(0.03)

    def test_parallel(self):
        """g1 ∥ g2: latency is max, rest same as serial"""
        g1 = CostGrade(0.9, 100, 1.0, 0.01)
        g2 = CostGrade(0.8, 200, 2.0, 0.02)
        r = grade_parallel(g1, g2)
        assert r.probability == pytest.approx(0.72)
        assert r.tokens == 300
        assert r.latency == pytest.approx(2.0)  # max(1, 2)
        assert r.money == pytest.approx(0.03)

    def test_iterate(self):
        """g^n (修正版): 概率只算致命失败的连乘, 成本仍然 ×n"""
        g = CostGrade(0.9, 100, 1.0, 0.01, p_fatal=0.02)
        r = grade_iterate(g, 5)
        # 修正模型: p = (1 - p_fatal)^n = 0.98^5
        assert r.probability == pytest.approx(0.98 ** 5)
        # 成本不变
        assert r.tokens == 500
        assert r.latency == pytest.approx(5.0)
        assert r.money == pytest.approx(0.05)

    def test_guard(self):
        """Guard: p = 1-(1-p)^k, rest multiplied by k"""
        g = CostGrade(0.8, 100, 1.0, 0.01)
        r = grade_guard(g, retries=2)  # 3 total attempts
        assert r.probability == pytest.approx(1.0 - 0.2**3)
        assert r.tokens == 300
        assert r.latency == pytest.approx(3.0)
        assert r.money == pytest.approx(0.03)


# ============================================================
# 3. estimate_cost() for All Constructs
# ============================================================

class TestEstimateCost:

    def test_lam(self):
        """Lam has LLM cost"""
        lam = Lam("test", "prompt", model="qwen3-max")
        g = estimate_cost(lam)
        assert g.tokens > 0
        assert g.latency > 0
        assert g.money > 0
        assert 0 < g.probability < 1

    def test_tool(self):
        """Tool has zero token cost"""
        tool = Tool("t", lambda x: x)
        g = estimate_cost(tool)
        assert g.tokens == 0
        assert g.money == 0.0
        assert g.latency > 0  # still has latency

    def test_fst_snd(self):
        """Fst/Snd are free"""
        assert estimate_cost(Fst()).is_free
        assert estimate_cost(Snd()).is_free

    def test_compose(self):
        """Compose costs add up"""
        a = Lam("a", "p", model="qwen3-max")
        b = Tool("b", lambda x: x)
        g = estimate_cost(Compose(a, b))
        ga = estimate_cost(a)
        gb = estimate_cost(b)
        assert g.tokens == ga.tokens + gb.tokens
        assert g.latency == pytest.approx(ga.latency + gb.latency)

    def test_pair(self):
        """Pair: latency is max"""
        a = Lam("a", "p", model="qwen3-max")
        b = Lam("b", "p", model="qwen3-max")
        g = estimate_cost(Pair(a, b))
        ga = estimate_cost(a)
        assert g.tokens == ga.tokens * 2
        assert g.latency == pytest.approx(ga.latency)  # max(l, l) = l

    def test_loop(self):
        """Loop: cost * max_steps"""
        body = Lam("b", "p", model="qwen3-max")
        loop = Loop(body, lambda r, s: False, max_steps=5)
        g = estimate_cost(loop)
        gb = estimate_cost(body)
        assert g.tokens == gb.tokens * 5
        assert g.latency == pytest.approx(gb.latency * 5)

    def test_guard(self):
        """Guard: retries increase cost but improve probability"""
        inner = Lam("a", "p", model="qwen3-max")
        guard = Guard(inner, lambda x: True, retry=2)
        g = estimate_cost(guard)
        gi = estimate_cost(inner)
        assert g.probability > gi.probability  # retries help
        assert g.tokens == gi.tokens * 3

    def test_route(self):
        """Route: classifier + worst route"""
        cls = Lam("cls", "classify", model="qwen3-max")
        routes = {
            "cheap": Tool("t", lambda x: x),
            "expensive": Lam("exp", "p", model="claude-opus-4-20250514"),
        }
        route = Route(cls, routes)
        g = estimate_cost(route)
        gc = estimate_cost(cls)
        assert g.tokens > gc.tokens  # classifier + route

    def test_memory(self):
        """Memory wraps inner cost"""
        inner = Tool("t", lambda x: x)
        mem = Memory(inner)
        g = estimate_cost(mem)
        gi = estimate_cost(inner)
        assert g.tokens == gi.tokens


# ============================================================
# 4. Pipeline Cost Estimation
# ============================================================

class TestPipelineCost:

    def test_realistic_pipeline(self):
        """Realistic pipeline: summarize >> validate >> format"""
        summarizer = Lam("summarize", "Summarize", model="qwen3-max")
        validator = Guard(
            Lam("validate", "Validate", model="qwen3-max"),
            lambda x: len(x) > 10,
            retry=2,
        )
        formatter = Tool("format", lambda x: f"<p>{x}</p>")
        pipeline = Compose(summarizer, validator, formatter)

        g = estimate_cost(pipeline)
        assert g.tokens > 0
        assert g.money > 0
        assert g.probability < 1.0
        # Should be sum of components
        print(format_cost_estimate(g))

    def test_parallel_reduces_latency(self):
        """Parallel should have lower latency than serial"""
        a = Lam("a", "p", model="qwen3-max")
        b = Lam("b", "p", model="qwen3-max")
        serial = grade_serial(estimate_cost(a), estimate_cost(b))
        parallel = grade_parallel(estimate_cost(a), estimate_cost(b))
        assert parallel.latency < serial.latency
        assert parallel.money == serial.money  # same total cost


# ============================================================
# 5. format_cost_estimate()
# ============================================================

class TestFormatCostEstimate:

    def test_format(self):
        g = CostGrade(0.95, 800, 2.0, 0.0024)
        s = format_cost_estimate(g)
        assert "95.0%" in s
        assert "800" in s
        assert "$0.0024" in s
