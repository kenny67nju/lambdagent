"""
Tests for lambdagent.core (Context, Term) and lambdagent.primitives
(Lam, Compose, If, Loop, Pair, Fst, Snd, Tool) plus extensions
(Route, Guard, Memory).

All tests are deterministic -- no real LLM calls. Lam tests use
TestHandler for mocking; everything else uses Tool(name, lambda).
"""

from __future__ import annotations

import pytest

from lambdagent.core import Context, UnboundVariable, RouteError, ValidationError
from lambdagent.primitives import Lam, Compose, If, Loop, Pair, Fst, Snd, Tool
from lambdagent.extensions import Route, Guard, Memory
from lambdagent.handlers import TestHandler, set_current_handler


# ── helpers ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_handler():
    """Ensure no handler leaks between tests."""
    set_current_handler(None)
    yield
    set_current_handler(None)


def _make_tool(name: str, fn):
    """Shorthand for a deterministic Tool."""
    return Tool(name, fn)


# ════════════════════════════════════════════════════════════
# Context tests
# ════════════════════════════════════════════════════════════


class TestContext:
    def test_empty_context(self):
        ctx = Context()
        assert ctx.bindings == {}
        assert ctx.trace == []
        assert ctx.memory == {}
        assert ctx.parent is None

    def test_extend_creates_child_with_binding(self):
        ctx = Context()
        child = ctx.extend(x=1)
        assert child.lookup("x") == 1
        assert child.parent is ctx

    def test_extend_inherits_parent_bindings(self):
        ctx = Context(bindings={"a": 10})
        child = ctx.extend(b=20)
        assert child.lookup("a") == 10
        assert child.lookup("b") == 20

    def test_lookup_raises_on_missing(self):
        ctx = Context()
        with pytest.raises(UnboundVariable, match="missing"):
            ctx.lookup("missing")

    def test_log_appends_to_trace(self):
        ctx = Context()
        ctx.log("test_term", "id1", "in", "out", 1.0, "model", 10)
        assert len(ctx.trace) == 1
        entry = ctx.trace[0]
        assert entry.term_name == "test_term"
        assert entry.input == "in"
        assert entry.output == "out"
        assert entry.duration_ms == 1.0
        assert entry.model == "model"
        assert entry.tokens_used == 10

    def test_fork_creates_deep_copy(self):
        """FIX-01: fork must deep-copy nested dicts so branches are independent."""
        ctx = Context(
            bindings={"nested": {"key": [1, 2, 3]}},
            memory={"data": {"inner": "value"}},
        )
        forked = ctx.fork()

        # Verify values are equal
        assert forked.bindings["nested"]["key"] == [1, 2, 3]
        assert forked.memory["data"]["inner"] == "value"

        # Verify they are independent objects
        forked.bindings["nested"]["key"].append(4)
        assert ctx.bindings["nested"]["key"] == [1, 2, 3]  # original unchanged

        forked.memory["data"]["inner"] = "changed"
        assert ctx.memory["data"]["inner"] == "value"  # original unchanged

    def test_fork_has_independent_trace(self):
        ctx = Context()
        ctx.log("parent", "id1", "in", "out", 1.0)
        forked = ctx.fork()
        forked.log("child", "id2", "in2", "out2", 2.0)

        assert len(ctx.trace) == 1
        assert len(forked.trace) == 1
        assert forked.trace[0].term_name == "child"

    def test_fork_parallel_writes_no_interference(self):
        """FIX-01 key test: parallel writes to forked contexts must not interfere."""
        ctx = Context(
            bindings={"shared_list": [1, 2]},
            memory={"counter": 0},
        )
        fork_a = ctx.fork()
        fork_b = ctx.fork()

        # Simulate parallel writes
        fork_a.bindings["shared_list"].append(3)
        fork_a.memory["counter"] = 10

        fork_b.bindings["shared_list"].append(99)
        fork_b.memory["counter"] = 20

        # Each fork sees only its own writes
        assert fork_a.bindings["shared_list"] == [1, 2, 3]
        assert fork_b.bindings["shared_list"] == [1, 2, 99]
        assert fork_a.memory["counter"] == 10
        assert fork_b.memory["counter"] == 20

        # Original is untouched
        assert ctx.bindings["shared_list"] == [1, 2]
        assert ctx.memory["counter"] == 0

    def test_merge_trace(self):
        parent = Context()
        child = parent.fork()
        child.log("child_step", "id1", "a", "b", 1.0)
        child.log("child_step2", "id2", "c", "d", 2.0)

        parent.merge_trace(child)
        assert len(parent.trace) == 2
        assert parent.trace[0].term_name == "child_step"
        assert parent.trace[1].term_name == "child_step2"


# ════════════════════════════════════════════════════════════
# Lam tests (with TestHandler mock)
# ════════════════════════════════════════════════════════════


class TestLam:
    def test_lam_apply_with_test_handler(self):
        handler = TestHandler()
        handler.mock_llm("summarize", "Short summary.")
        set_current_handler(handler)

        agent = Lam("summarizer", "Summarize the input.")
        result = agent.apply("summarize this long text")
        assert result == "Short summary."

    def test_lam_default_mock(self):
        handler = TestHandler()
        set_current_handler(handler)

        agent = Lam("generic", "Do something.")
        result = agent.apply("anything")
        assert result == "Mock LLM response"

    def test_lam_name_is_set(self):
        agent = Lam("my_agent", "prompt")
        assert agent._name == "my_agent"

    def test_lam_has_type_properties(self):
        agent = Lam("typed", "prompt")
        # Default types are T_ANY
        from lambdagent.lam_types import T_ANY

        assert agent.input_type == T_ANY
        assert agent.output_type == T_ANY

    def test_lam_output_parser(self):
        handler = TestHandler()
        handler.mock_llm_default("42")
        set_current_handler(handler)

        agent = Lam("parser_test", "return a number", output_parser=int)
        result = agent.apply("give me a number")
        assert result == 42
        assert isinstance(result, int)

    def test_lam_logs_trace(self):
        handler = TestHandler()
        set_current_handler(handler)

        ctx = Context()
        agent = Lam("tracer", "prompt")
        agent.apply("input", ctx)

        assert len(ctx.trace) == 1
        assert ctx.trace[0].term_name == "tracer"


# ════════════════════════════════════════════════════════════
# Compose tests
# ════════════════════════════════════════════════════════════


class TestCompose:
    def test_compose_two(self):
        f = _make_tool("double", lambda x: int(x) * 2)
        g = _make_tool("add1", lambda x: int(x) + 1)
        pipeline = Compose(f, g)
        # g(f(3)) = (3*2)+1 = 7
        assert pipeline("3") == 7

    def test_compose_three(self):
        f = _make_tool("a", lambda x: str(x) + "A")
        g = _make_tool("b", lambda x: str(x) + "B")
        h = _make_tool("c", lambda x: str(x) + "C")
        pipeline = Compose(f, g, h)
        assert pipeline("_") == "_ABC"

    def test_rshift_creates_compose(self):
        f = _make_tool("f", lambda x: str(x) + "1")
        g = _make_tool("g", lambda x: str(x) + "2")
        pipeline = f >> g
        assert isinstance(pipeline, Compose)
        assert pipeline("x") == "x12"

    def test_rshift_flattens_stages(self):
        f = _make_tool("f", lambda x: x)
        g = _make_tool("g", lambda x: x)
        h = _make_tool("h", lambda x: x)
        pipeline = (f >> g) >> h
        assert isinstance(pipeline, Compose)
        assert len(pipeline.stages) == 3

    def test_compose_logs_each_stage(self):
        f = _make_tool("step1", lambda x: "a")
        g = _make_tool("step2", lambda x: "b")
        ctx = Context()
        Compose(f, g).apply("input", ctx)
        assert len(ctx.trace) == 2
        assert ctx.trace[0].term_name == "step1"
        assert ctx.trace[1].term_name == "step2"


# ════════════════════════════════════════════════════════════
# If tests
# ════════════════════════════════════════════════════════════


class TestIf:
    def test_true_branch(self):
        cond = _make_tool("cond", lambda x: True)
        then = _make_tool("then", lambda x: "YES")
        else_ = _make_tool("else", lambda x: "NO")
        result = If(cond, then, else_)("anything")
        assert result == "YES"

    def test_false_branch(self):
        cond = _make_tool("cond", lambda x: False)
        then = _make_tool("then", lambda x: "YES")
        else_ = _make_tool("else", lambda x: "NO")
        result = If(cond, then, else_)("anything")
        assert result == "NO"

    def test_callable_condition(self):
        then = _make_tool("then", lambda x: "positive")
        else_ = _make_tool("else", lambda x: "non-positive")
        result = If(lambda x: int(x) > 0, then, else_)("5")
        assert result == "positive"

        result2 = If(lambda x: int(x) > 0, then, else_)("-1")
        assert result2 == "non-positive"

    def test_string_truthy_values(self):
        """If._is_truthy recognizes TRUE, YES, 1 as truthy strings."""
        cond_true = _make_tool("cond", lambda x: "TRUE")
        cond_yes = _make_tool("cond", lambda x: "yes")
        cond_one = _make_tool("cond", lambda x: "1")
        cond_no = _make_tool("cond", lambda x: "NO")

        then = _make_tool("then", lambda x: "T")
        else_ = _make_tool("else", lambda x: "F")

        assert If(cond_true, then, else_)("x") == "T"
        assert If(cond_yes, then, else_)("x") == "T"
        assert If(cond_one, then, else_)("x") == "T"
        assert If(cond_no, then, else_)("x") == "F"


# ════════════════════════════════════════════════════════════
# Loop tests
# ════════════════════════════════════════════════════════════


class TestLoop:
    def test_loop_terminates_on_condition(self):
        body = _make_tool("incr", lambda x: int(x) + 1)
        # Stop when result >= 5
        loop = Loop(body, condition=lambda r, step: int(r) >= 5, max_steps=100)
        result = loop("0")
        assert result == 5

    def test_loop_respects_max_steps(self):
        body = _make_tool("incr", lambda x: int(x) + 1)
        # Condition never met, but max_steps=3
        loop = Loop(body, condition=lambda r, step: False, max_steps=3)
        result = loop("0")
        assert result == 3  # 3 iterations: 0->1->2->3

    def test_loop_immediate_termination(self):
        body = _make_tool("id", lambda x: x)
        loop = Loop(body, condition=lambda r, step: True, max_steps=100)
        result = loop("hello")
        assert result == "hello"


# ════════════════════════════════════════════════════════════
# Pair / Fst / Snd tests
# ════════════════════════════════════════════════════════════


class TestPairFstSnd:
    def test_pair(self):
        f = _make_tool("upper", lambda x: str(x).upper())
        g = _make_tool("lower", lambda x: str(x).lower())
        result = Pair(f, g)("Hello")
        assert result == ("HELLO", "hello")

    def test_fst(self):
        assert Fst()(("a", "b")) == "a"

    def test_snd(self):
        assert Snd()(("a", "b")) == "b"

    def test_fst_on_list(self):
        assert Fst()([10, 20, 30]) == 10

    def test_snd_on_list(self):
        assert Snd()([10, 20, 30]) == 20

    def test_fst_on_non_tuple_raises(self):
        """FIX-03: Fst must reject non-tuple/list inputs."""
        with pytest.raises(TypeError, match="Fst expects"):
            Fst()("not a tuple")

    def test_fst_on_empty_raises(self):
        with pytest.raises(TypeError, match="Fst expects"):
            Fst()(())

    def test_snd_on_single_element_raises(self):
        """FIX-03: Snd must reject tuples with fewer than 2 elements."""
        with pytest.raises(TypeError, match="Snd expects"):
            Snd()(("only_one",))

    def test_snd_on_non_tuple_raises(self):
        with pytest.raises(TypeError, match="Snd expects"):
            Snd()(42)


# ════════════════════════════════════════════════════════════
# Tool tests
# ════════════════════════════════════════════════════════════


class TestTool:
    def test_tool_apply(self):
        tool = Tool("double", lambda x: int(x) * 2)
        assert tool("5") == 10

    def test_tool_name(self):
        tool = Tool("my_tool", lambda x: x)
        assert tool._name == "my_tool"

    def test_tool_logs_trace(self):
        ctx = Context()
        tool = Tool("tracer", lambda x: x)
        tool.apply("input", ctx)
        assert len(ctx.trace) == 1
        assert ctx.trace[0].term_name == "tracer"

    def test_tool_with_effect_handler(self):
        """When a non-Production handler is active, Tool delegates to handle_tool."""
        handler = TestHandler()
        handler.mock_tool("my_tool", "mocked_result")
        set_current_handler(handler)

        tool = Tool("my_tool", lambda x: "real_result")
        result = tool("input")
        assert result == "mocked_result"


# ════════════════════════════════════════════════════════════
# Route tests
# ════════════════════════════════════════════════════════════


class TestRoute:
    def test_route_dispatches_correctly(self):
        classifier = _make_tool("classify", lambda x: "math")
        math_agent = _make_tool("math", lambda x: f"math({x})")
        code_agent = _make_tool("code", lambda x: f"code({x})")
        route = Route(classifier, {"math": math_agent, "code": code_agent})
        assert route("2+2") == "math(2+2)"

    def test_route_with_default(self):
        classifier = _make_tool("classify", lambda x: "unknown_label")
        fallback = _make_tool("fallback", lambda x: f"fallback({x})")
        route = Route(
            classifier,
            {"math": _make_tool("math", lambda x: "m")},
            default=fallback,
        )
        assert route("hello") == "fallback(hello)"

    def test_route_raises_on_unknown_label(self):
        classifier = _make_tool("classify", lambda x: "nonexistent")
        route = Route(classifier, {"a": _make_tool("a", lambda x: x)})
        with pytest.raises(RouteError, match="No route"):
            route("input")

    def test_route_fuzzy_match(self):
        """Route tries substring matching when exact match fails."""
        classifier = _make_tool("classify", lambda x: "I think this is math")
        math_agent = _make_tool("math", lambda x: "matched_math")
        route = Route(classifier, {"math": math_agent})
        assert route("2+2") == "matched_math"


# ════════════════════════════════════════════════════════════
# Guard tests
# ════════════════════════════════════════════════════════════


class TestGuard:
    def test_guard_passes_valid(self):
        agent = _make_tool("gen", lambda x: 42)
        guard = Guard(agent, validator=lambda r: isinstance(r, int))
        assert guard("x") == 42

    def test_guard_retries_and_succeeds(self):
        call_count = {"n": 0}

        def flaky(x):
            call_count["n"] += 1
            if call_count["n"] < 3:
                return "bad"
            return "good"

        agent = _make_tool("flaky", flaky)
        guard = Guard(agent, validator=lambda r: r == "good", retry=3)
        assert guard("x") == "good"

    def test_guard_exhausts_retries_raises(self):
        agent = _make_tool("always_bad", lambda x: "bad")
        guard = Guard(agent, validator=lambda r: r == "good", retry=2)
        with pytest.raises(ValidationError, match="failed after 3 attempts"):
            guard("x")

    def test_guard_on_fail_callback(self):
        agent = _make_tool("bad", lambda x: "invalid")
        guard = Guard(
            agent,
            validator=lambda r: False,
            retry=1,
            on_fail=lambda r: f"fallback({r})",
        )
        result = guard("x")
        assert result == "fallback(invalid)"


# ════════════════════════════════════════════════════════════
# Memory tests
# ════════════════════════════════════════════════════════════


class TestMemory:
    def test_memory_injects_store(self):
        """Memory prepends store contents to the agent input."""
        captured = {}

        def capture(x):
            captured["input"] = x
            return x

        agent = _make_tool("echo", capture)
        mem = Memory(agent, store={"user": "Alice", "role": "admin"})
        mem("hello")
        assert "[Memory]" in captured["input"]
        assert "Alice" in captured["input"]
        assert "hello" in captured["input"]

    def test_memory_empty_store_passthrough(self):
        agent = _make_tool("echo", lambda x: x)
        mem = Memory(agent)
        assert mem("plain input") == "plain input"

    def test_remember_and_forget(self):
        agent = _make_tool("echo", lambda x: x)
        mem = Memory(agent)

        mem.remember("key1", "value1")
        assert mem.store["key1"] == "value1"

        mem.forget("key1")
        assert "key1" not in mem.store

    def test_forget_nonexistent_is_safe(self):
        agent = _make_tool("echo", lambda x: x)
        mem = Memory(agent)
        mem.forget("does_not_exist")  # should not raise
