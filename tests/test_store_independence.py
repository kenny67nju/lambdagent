"""
Tests for Paper II Proposition 30 — store-independence enforcement.

Tests cover:
  1. writes() analysis for all constructs
  2. Store-independence checking
  3. ctx.fork() isolation
  4. AsyncPar with store-independence enforcement
  5. Par with forked contexts
"""

import pytest
from lambdagent.store_analysis import (
    writes, reads, check_store_independence, StoreConflictError,
)
from lambdagent.core import Context
from lambdagent.primitives import Lam, Compose, If, Loop, Pair, Tool
from lambdagent.extensions import Par, Route, Memory, Guard
from lambdagent.multiagent import AsyncPar


# ============================================================
# 1. writes() Analysis
# ============================================================

class TestWritesAnalysis:

    def test_lam_no_writes(self):
        """Lam doesn't write to store"""
        lam = Lam("test", "prompt")
        assert writes(lam) == frozenset()

    def test_tool_no_writes(self):
        """Tool doesn't write to store"""
        tool = Tool("t", lambda x: x)
        assert writes(tool) == frozenset()

    def test_memory_writes_keys(self):
        """Memory writes all keys in its store"""
        mem = Memory(Tool("t", lambda x: x), store={"key1": "v1", "key2": "v2"})
        w = writes(mem)
        assert "key1" in w
        assert "key2" in w

    def test_compose_unions_writes(self):
        """Compose writes = union of all stage writes"""
        m1 = Memory(Tool("t1", lambda x: x), store={"a": 1})
        m2 = Memory(Tool("t2", lambda x: x), store={"b": 2})
        comp = Compose(m1, m2)
        w = writes(comp)
        assert "a" in w
        assert "b" in w

    def test_guard_propagates_writes(self):
        """Guard propagates inner agent writes"""
        inner = Memory(Tool("t", lambda x: x), store={"key": "v"})
        guard = Guard(inner, lambda x: True)
        assert "key" in writes(guard)

    def test_pair_unions_writes(self):
        """Pair writes = first writes ∪ second writes"""
        m1 = Memory(Tool("t1", lambda x: x), store={"a": 1})
        m2 = Memory(Tool("t2", lambda x: x), store={"b": 2})
        pair = Pair(m1, m2)
        w = writes(pair)
        assert "a" in w
        assert "b" in w

    def test_explicit_writes_annotation(self):
        """Term with _writes attribute"""
        tool = Tool("t", lambda x: x)
        tool._writes = {"custom_key"}
        assert "custom_key" in writes(tool)


# ============================================================
# 2. Store Independence Checking
# ============================================================

class TestStoreIndependence:

    def test_independent_agents_pass(self):
        """Agents with no writes → passes"""
        agents = [
            Lam("a", "prompt1"),
            Lam("b", "prompt2"),
            Tool("c", lambda x: x),
        ]
        check_store_independence(agents)  # Should not raise

    def test_disjoint_writes_pass(self):
        """Agents writing to different keys → passes"""
        agents = [
            Memory(Tool("t1", lambda x: x), store={"a": 1}),
            Memory(Tool("t2", lambda x: x), store={"b": 2}),
        ]
        check_store_independence(agents)  # Should not raise

    def test_overlapping_writes_fail(self):
        """Agents writing to same key → fails"""
        agents = [
            Memory(Tool("t1", lambda x: x), store={"shared": 1}),
            Memory(Tool("t2", lambda x: x), store={"shared": 2}),
        ]
        with pytest.raises(StoreConflictError) as exc_info:
            check_store_independence(agents)
        assert "shared" in str(exc_info.value)

    def test_error_message_quality(self):
        """Error message includes agent names and conflicting keys"""
        m1 = Memory(Tool("agent_A", lambda x: x), store={"key": 1})
        m2 = Memory(Tool("agent_B", lambda x: x), store={"key": 2})
        with pytest.raises(StoreConflictError) as exc_info:
            check_store_independence([m1, m2])
        err = str(exc_info.value)
        assert "Pair confluence" in err
        assert "Prop. 30" in err


# ============================================================
# 3. ctx.fork() Isolation
# ============================================================

class TestContextFork:

    def test_fork_independent_trace(self):
        """Forked context has independent trace"""
        parent = Context()
        parent.log("parent_op", "id1", "in", "out", 10.0)
        child = parent.fork()
        child.log("child_op", "id2", "in", "out", 5.0)
        assert len(parent.trace) == 1
        assert len(child.trace) == 1

    def test_fork_independent_memory(self):
        """Forked context has independent memory"""
        parent = Context(memory={"key": "original"})
        child = parent.fork()
        child.memory["key"] = "modified"
        assert parent.memory["key"] == "original"
        assert child.memory["key"] == "modified"

    def test_fork_shares_bindings(self):
        """Forked context copies bindings (read-only sharing)"""
        parent = Context(bindings={"x": 42})
        child = parent.fork()
        assert child.bindings["x"] == 42

    def test_merge_trace(self):
        """merge_trace() combines child trace into parent"""
        parent = Context()
        child = parent.fork()
        child.log("child_op", "id1", "in", "out", 5.0)
        parent.merge_trace(child)
        assert len(parent.trace) == 1
        assert parent.trace[0].term_name == "child_op"


# ============================================================
# 4. AsyncPar with Store Independence
# ============================================================

class TestAsyncParStoreIndependence:

    def test_asyncpar_passes_with_pure_agents(self):
        """AsyncPar with pure agents (no writes) passes store check"""
        agents = [
            Tool("t1", lambda x: f"result1: {x}"),
            Tool("t2", lambda x: f"result2: {x}"),
        ]
        ap = AsyncPar(*agents)
        result = ap.apply("test", Context())
        assert len(result) == 2

    def test_asyncpar_fails_with_conflicting_writes(self):
        """AsyncPar detects store conflicts"""
        m1 = Memory(Tool("t1", lambda x: x), store={"shared": 1})
        m2 = Memory(Tool("t2", lambda x: x), store={"shared": 2})
        ap = AsyncPar(m1, m2)
        with pytest.raises(StoreConflictError):
            ap.apply("test", Context())

    def test_asyncpar_skip_check(self):
        """AsyncPar with check_store_independence=False skips check"""
        m1 = Memory(Tool("t1", lambda x: x), store={"shared": 1})
        m2 = Memory(Tool("t2", lambda x: x), store={"shared": 2})
        ap = AsyncPar(m1, m2, check_store_independence=False)
        # Should not raise StoreConflictError (may still have race conditions)
        result = ap.apply("test", Context())
        assert len(result) == 2

    def test_asyncpar_forked_contexts(self):
        """AsyncPar uses forked contexts — traces are merged after"""
        agents = [
            Tool("t1", lambda x: f"r1:{x}"),
            Tool("t2", lambda x: f"r2:{x}"),
        ]
        ap = AsyncPar(*agents)
        ctx = Context()
        result = ap.apply("input", ctx)
        # Each agent produces a trace entry + AsyncPar itself
        assert len(ctx.trace) >= 2  # at least agent traces + asyncpar log


# ============================================================
# 5. Par with Forked Contexts
# ============================================================

class TestParForkedContexts:

    def test_par_forked_contexts(self):
        """Par uses forked contexts for thread safety"""
        agents = [
            Tool("t1", lambda x: f"r1:{x}"),
            Tool("t2", lambda x: f"r2:{x}"),
        ]
        par = Par(*agents)
        ctx = Context()
        result = par.apply("input", ctx)
        assert len(result) == 2
        # Traces from forked contexts should be merged back
        assert len(ctx.trace) >= 2
