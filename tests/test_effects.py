"""
Tests for Paper III Effect Algebra — effect annotations on all 11 constructs.

Tests cover:
  1. Basic effect construction and properties
  2. Effect composition: serial (·), parallel (∥), iterate (εⁿ)
  3. Effect subtype lattice (Definition 9)
  4. Effect inference for all 11 constructs
  5. Effect annotation parsing from YAML
  6. Term.effect property integration
"""

import pytest
from lambdagent.effects import (
    Effect,
    EffectKind,
    ComposedEffect,
    PURE,
    IO,
    LLM,
    STATE,
    serial,
    parallel,
    iterate,
    effect_leq,
    max_effect,
    parse_effect_annotation,
    infer_effect_for_term,
)
from lambdagent.primitives import Lam, Compose, If, Loop, Pair, Tool
from lambdagent.extensions import Par, Route, Memory, Guard


# ============================================================
# 1. Basic Effect Construction
# ============================================================


class TestEffectConstruction:
    def test_pure(self):
        assert PURE.kind == EffectKind.PURE
        assert repr(PURE) == "pure"

    def test_llm(self):
        e = LLM("claude-sonnet")
        assert e.kind == EffectKind.LLM
        assert e.model == "claude-sonnet"
        assert repr(e) == "llm(claude-sonnet)"

    def test_io(self):
        assert IO.kind == EffectKind.IO
        assert repr(IO) == "io"

    def test_state(self):
        e = STATE("memory", "counter")
        assert e.kind == EffectKind.STATE
        assert "memory" in e.state_keys
        assert "counter" in e.state_keys

    def test_state_repr(self):
        e = STATE("mem")
        assert "state(mem)" == repr(e)


# ============================================================
# 2. Effect Composition
# ============================================================


class TestEffectComposition:
    def test_serial_composition(self):
        """ε1 · ε2"""
        result = serial(LLM("claude"), IO)
        assert result.mode == "serial"
        assert len(result.effects) == 2
        assert "llm(claude) · io" == repr(result)

    def test_serial_pure_elimination(self):
        """pure · ε = ε"""
        result = serial(PURE, LLM("claude"))
        assert len(result.effects) == 1
        assert result.effects[0].kind == EffectKind.LLM

    def test_serial_all_pure(self):
        """pure · pure = pure"""
        result = serial(PURE, PURE)
        assert result.is_pure

    def test_parallel_composition(self):
        """ε1 ∥ ε2"""
        result = parallel(LLM("claude"), LLM("gpt-4"))
        assert result.mode == "parallel"
        assert len(result.effects) == 2

    def test_iterate(self):
        """εⁿ"""
        result = iterate(LLM("claude"), 5)
        assert result.mode == "iterate"
        assert result.iterations == 5
        assert "llm(claude)^5" == repr(result)

    def test_composed_properties(self):
        """Test ComposedEffect property accessors"""
        result = serial(LLM("claude"), IO, STATE("mem"))
        assert result.has_llm
        assert result.has_io
        assert result.has_state
        assert not result.is_pure
        assert "claude" in result.models_used
        assert "mem" in result.all_state_keys


# ============================================================
# 3. Effect Subtype Lattice (Paper III Definition 9)
# ============================================================


class TestEffectLattice:
    def test_pure_bottom(self):
        """pure ≤ ε for all ε"""
        assert effect_leq(PURE, PURE)
        assert effect_leq(PURE, IO)
        assert effect_leq(PURE, LLM())
        assert effect_leq(PURE, STATE())

    def test_reflexivity(self):
        """ε ≤ ε"""
        assert effect_leq(IO, IO)
        assert effect_leq(LLM(), LLM())
        assert effect_leq(STATE(), STATE())

    def test_ordering(self):
        """pure ≤ state ≤ io ≤ llm"""
        assert effect_leq(STATE(), IO)
        assert effect_leq(IO, LLM())

    def test_max_effect(self):
        """join in the lattice"""
        result = max_effect(PURE, IO, LLM("claude"))
        assert result.kind == EffectKind.LLM


# ============================================================
# 4. Effect Inference for All 11 Constructs
# ============================================================


class TestEffectInference:
    def test_lam_effect(self):
        """Lam → llm(model)"""
        lam = Lam("test", "prompt", model="claude-sonnet")
        eff = infer_effect_for_term(lam)
        assert isinstance(eff, Effect)
        assert eff.kind == EffectKind.LLM
        assert eff.model == "claude-sonnet"

    def test_tool_effect(self):
        """Tool → io"""
        tool = Tool("double", lambda x: int(x) * 2)
        eff = infer_effect_for_term(tool)
        assert isinstance(eff, Effect)
        assert eff.kind == EffectKind.IO

    def test_compose_effect(self):
        """Compose → ε1 · ε2"""
        f = Lam("a", "prompt", model="m1")
        g = Tool("b", lambda x: x)
        comp = Compose(f, g)
        eff = infer_effect_for_term(comp)
        assert isinstance(eff, ComposedEffect)
        assert eff.mode == "serial"
        assert eff.has_llm
        assert eff.has_io

    def test_pair_effect(self):
        """Pair → ε1 ∥ ε2"""
        f = Lam("a", "prompt", model="m1")
        g = Lam("b", "prompt", model="m2")
        pair = Pair(f, g)
        eff = infer_effect_for_term(pair)
        assert isinstance(eff, ComposedEffect)
        assert eff.mode == "parallel"
        assert len(eff.models_used) == 2

    def test_par_effect(self):
        """Par → ε1 ∥ ε2 ∥ ... ∥ εn"""
        agents = [Tool(f"t{i}", lambda x: x) for i in range(3)]
        par = Par(*agents)
        eff = infer_effect_for_term(par)
        assert isinstance(eff, ComposedEffect)
        assert eff.mode == "parallel"

    def test_loop_effect(self):
        """Loop → ε_body^n"""
        body = Lam("b", "prompt", model="m")
        loop = Loop(body, lambda r, s: False, max_steps=5)
        eff = infer_effect_for_term(loop)
        assert isinstance(eff, ComposedEffect)
        assert eff.mode == "iterate"
        assert eff.iterations == 5

    def test_memory_effect(self):
        """Memory → state · inner_effect"""
        inner = Tool("t", lambda x: x)
        mem = Memory(inner)
        eff = infer_effect_for_term(mem)
        assert isinstance(eff, ComposedEffect)
        assert eff.has_state
        assert eff.has_io

    def test_guard_effect(self):
        """Guard → ε_agent^(1+retry)"""
        inner = Lam("a", "prompt", model="m")
        guard = Guard(inner, lambda x: True, retry=2)
        eff = infer_effect_for_term(guard)
        assert isinstance(eff, ComposedEffect)
        assert eff.mode == "iterate"
        assert eff.iterations == 3

    def test_route_effect(self):
        """Route → ε_classifier · max(ε_routes)"""
        classifier = Lam("cls", "classify", model="m")
        routes = {
            "a": Tool("ta", lambda x: x),
            "b": Lam("lb", "prompt", model="m2"),
        }
        route = Route(classifier, routes)
        eff = infer_effect_for_term(route)
        assert isinstance(eff, ComposedEffect)
        assert eff.has_llm


# ============================================================
# 5. Effect Annotation Parsing
# ============================================================


class TestEffectParsing:
    def test_parse_pure(self):
        result = parse_effect_annotation("pure")
        assert result == PURE

    def test_parse_io(self):
        result = parse_effect_annotation("io")
        assert result == IO

    def test_parse_llm(self):
        result = parse_effect_annotation("llm(claude-sonnet)")
        assert isinstance(result, Effect)
        assert result.kind == EffectKind.LLM
        assert result.model == "claude-sonnet"

    def test_parse_state(self):
        result = parse_effect_annotation("state(memory)")
        assert isinstance(result, Effect)
        assert result.kind == EffectKind.STATE

    def test_parse_serial(self):
        result = parse_effect_annotation("llm(claude) · io")
        assert isinstance(result, ComposedEffect)
        assert result.mode == "serial"

    def test_parse_parallel(self):
        result = parse_effect_annotation("llm(m1) ∥ llm(m2)")
        assert isinstance(result, ComposedEffect)
        assert result.mode == "parallel"

    def test_parse_none(self):
        result = parse_effect_annotation(None)
        assert isinstance(result, ComposedEffect)
        assert result.is_pure


# ============================================================
# 6. Term.effect Property Integration
# ============================================================


class TestTermEffectProperty:
    def test_lam_effect_property(self):
        """Term.effect should auto-infer for Lam"""
        lam = Lam("test", "prompt", model="claude")
        eff = lam.effect
        assert isinstance(eff, Effect)
        assert eff.kind == EffectKind.LLM

    def test_tool_effect_property(self):
        """Term.effect should auto-infer for Tool"""
        tool = Tool("t", lambda x: x)
        eff = tool.effect
        assert isinstance(eff, Effect)
        assert eff.kind == EffectKind.IO

    def test_effect_setter(self):
        """Manual effect annotation overrides inference"""
        tool = Tool("t", lambda x: x)
        tool.effect = PURE
        assert tool.effect == PURE

    def test_agent_type_includes_effect(self):
        """AgentType includes effect in repr"""
        lam = Lam("test", "prompt", model="claude")
        at = lam.agent_type
        assert "llm(claude)" in repr(at)
