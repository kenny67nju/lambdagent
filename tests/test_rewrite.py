"""
Tests for Paper II Theorems 36-41 — algebraic laws as rewrite rules.

Tests cover:
  1. Identity elimination (Thm 37/38)
  2. Route distribution (Thm 40)
  3. Loop simplification (Thm 39)
  4. Guard anti-pattern warning (Prop 42)
  5. Recursive rewriting
  6. Semantic preservation
"""

import pytest
from lambdagent.rewrite import optimize, RewriteLog, _IdentityTerm, _is_identity
from lambdagent.primitives import Compose, If, Loop, Pair, Tool
from lambdagent.extensions import Route, Guard, Memory
from lambdagent.core import Context


# ============================================================
# 1. Identity Elimination (Thm 37/38)
# ============================================================


class TestIdentityElimination:
    def test_left_unit(self):
        """Id >> f ≡ f"""
        id_term = _IdentityTerm()
        f = Tool("f", lambda x: f"f({x})")
        pipeline = Compose(id_term, f)
        result, log = optimize(pipeline)
        assert not isinstance(result, Compose)
        assert result._name == "f"
        assert log.count >= 1

    def test_right_unit(self):
        """f >> Id ≡ f"""
        f = Tool("f", lambda x: f"f({x})")
        id_term = _IdentityTerm()
        pipeline = Compose(f, id_term)
        result, log = optimize(pipeline)
        assert not isinstance(result, Compose)
        assert result._name == "f"

    def test_middle_unit(self):
        """f >> Id >> g ≡ f >> g"""
        f = Tool("f", lambda x: f"f({x})")
        g = Tool("g", lambda x: f"g({x})")
        id_term = _IdentityTerm()
        pipeline = Compose(f, id_term, g)
        result, log = optimize(pipeline)
        assert isinstance(result, Compose)
        assert len(result.stages) == 2

    def test_named_identity(self):
        """Tool named 'Id' is detected as identity"""
        id_tool = Tool("Id", lambda x: x)
        assert _is_identity(id_tool)

    def test_no_identity_no_rewrite(self):
        """No identity → no rewrite"""
        f = Tool("f", lambda x: x)
        g = Tool("g", lambda x: x)
        pipeline = Compose(f, g)
        result, log = optimize(pipeline)
        assert isinstance(result, Compose)
        assert log.count == 0


# ============================================================
# 2. Route Distribution (Thm 40)
# ============================================================


class TestRouteDistribution:
    def test_route_then_tool(self):
        """Route(c, {li: fi}) >> g ≡ Route(c, {li: fi >> g})"""
        classifier = Tool("cls", lambda x: "a")
        routes = {
            "a": Tool("agent_a", lambda x: f"a:{x}"),
            "b": Tool("agent_b", lambda x: f"b:{x}"),
        }
        route = Route(classifier, routes)
        postprocess = Tool("format", lambda x: f"<{x}>")
        pipeline = Compose(route, postprocess)

        result, log = optimize(pipeline)
        assert log.count >= 1
        assert any("Route Distribution" in e.law for e in log.entries)
        # Result should be Route with postprocess pushed into branches
        assert isinstance(result, Route)

    def test_route_distribution_preserves_semantics(self):
        """Rewritten route produces same output"""
        classifier = Tool("cls", lambda x: "a")
        routes = {
            "a": Tool("agent_a", lambda x: f"a:{x}"),
        }
        route = Route(classifier, routes)
        postprocess = Tool("fmt", lambda x: f"[{x}]")
        pipeline = Compose(route, postprocess)

        # Original
        orig_result = pipeline("input")

        # Optimized
        optimized, _ = optimize(pipeline)
        opt_result = optimized("input")

        assert orig_result == opt_result


# ============================================================
# 3. Loop Simplification (Thm 39)
# ============================================================


class TestLoopSimplification:
    def test_loop_1_to_if(self):
        """Loop(body, cond, 1) simplified to If(cond, Id, body)"""
        body = Tool("inc", lambda x: int(x) + 1)
        loop = Loop(body, lambda r, s: int(r) > 5, max_steps=1)
        result, log = optimize(loop)
        assert isinstance(result, If)
        assert any("Loop Unfolding" in e.law for e in log.entries)

    def test_loop_n_no_simplify(self):
        """Loop(body, cond, n>1) not simplified"""
        body = Tool("inc", lambda x: int(x) + 1)
        loop = Loop(body, lambda r, s: int(r) > 5, max_steps=5)
        result, log = optimize(loop)
        assert isinstance(result, Loop)


# ============================================================
# 4. Guard Anti-Pattern Warning (Prop 42)
# ============================================================


class TestGuardAntiPattern:
    def test_guard_compose_warning(self):
        """Guard wrapping Compose triggers Prop 42 warning"""
        inner = Compose(
            Tool("a", lambda x: x),
            Tool("b", lambda x: x),
        )
        guard = Guard(inner, lambda x: True)
        pipeline = Compose(guard, Tool("c", lambda x: x))

        _, log = optimize(pipeline)
        assert len(log.warnings) >= 1
        assert "Proposition 42" in log.warnings[0]

    def test_no_warning_for_simple_guard(self):
        """Simple Guard (non-Compose inner) → no warning"""
        guard = Guard(Tool("a", lambda x: x), lambda x: True)
        pipeline = Compose(guard, Tool("b", lambda x: x))
        _, log = optimize(pipeline)
        assert len(log.warnings) == 0


# ============================================================
# 5. Recursive Rewriting
# ============================================================


class TestRecursiveRewriting:
    def test_nested_identity_elimination(self):
        """Identity inside Pair is eliminated"""
        id_term = _IdentityTerm()
        f = Tool("f", lambda x: x)
        pair = Pair(
            Compose(id_term, f),
            f,
        )
        result, log = optimize(pair)
        assert log.count >= 1

    def test_deeply_nested(self):
        """Rewrites work at multiple depth levels"""
        id_term = _IdentityTerm()
        f = Tool("f", lambda x: x)
        deep = Compose(
            Compose(id_term, f),
            Compose(f, id_term),
        )
        result, log = optimize(deep)
        assert log.count >= 1


# ============================================================
# 6. Semantic Preservation
# ============================================================


class TestSemanticPreservation:
    def test_identity_elim_preserves_result(self):
        """Removing Id doesn't change output"""
        id_term = _IdentityTerm()
        f = Tool("double", lambda x: int(x) * 2)
        pipeline = Compose(id_term, f, id_term)

        orig = pipeline("5")
        optimized, _ = optimize(pipeline)
        opt = optimized("5")
        assert orig == opt == 10
