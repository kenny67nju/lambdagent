"""
Tests for lambdagent.multiagent — Channel, Send, Receive, GroupChat,
Handoff, AsyncPar, SharedMemory.

All tests use Tool(name, lambda) for deterministic, fast execution.
No real LLM calls.
"""

from __future__ import annotations

import threading
import time

import pytest

from lambdagent.core import Context, LambdagentError
from lambdagent.primitives import Tool
from lambdagent.multiagent import (
    Channel,
    ChannelClosed,
    Send,
    Receive,
    GroupChat,
    Handoff,
    HandoffError,
    AsyncPar,
    SharedMemory,
)
from lambdagent.handlers import TestHandler, set_current_handler


# ── helpers ─────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_handler():
    set_current_handler(None)
    yield
    set_current_handler(None)


def _tool(name: str, fn):
    return Tool(name, fn)


# ════════════════════════════════════════════════════════════
# Channel tests
# ════════════════════════════════════════════════════════════

class TestChannel:

    def test_send_receive_basic(self):
        ch = Channel("test")
        ch.send("hello")
        assert ch.receive(timeout=1.0) == "hello"

    def test_send_receive_ordering(self):
        ch = Channel("fifo")
        ch.send("first")
        ch.send("second")
        ch.send("third")
        assert ch.receive(timeout=1.0) == "first"
        assert ch.receive(timeout=1.0) == "second"
        assert ch.receive(timeout=1.0) == "third"

    def test_channel_capacity_blocking(self):
        """A channel with capacity=1 blocks on second send until consumed."""
        ch = Channel("bounded", capacity=1)
        ch.send("msg1")

        # Second send should time out since buffer is full
        with pytest.raises(ChannelClosed, match="timeout"):
            ch.send("msg2", timeout=0.1)

        # After consuming, send should work
        assert ch.receive(timeout=1.0) == "msg1"
        ch.send("msg2", timeout=1.0)
        assert ch.receive(timeout=1.0) == "msg2"

    def test_close_prevents_send(self):
        ch = Channel("closable")
        ch.close()
        assert ch.is_closed
        with pytest.raises(ChannelClosed):
            ch.send("after close")

    def test_close_allows_drain(self):
        """Messages already in channel can still be received after close."""
        ch = Channel("drain")
        ch.send("before_close")
        ch.close()
        assert ch.receive(timeout=1.0) == "before_close"

    def test_history_records_messages(self):
        ch = Channel("history")
        ch.send("a")
        ch.receive(timeout=1.0)
        assert len(ch.history) == 2
        assert ch.history[0][0] == "send"
        assert ch.history[0][1] == "a"
        assert ch.history[1][0] == "recv"
        assert ch.history[1][1] == "a"

    def test_pending_count(self):
        ch = Channel("pending")
        ch.send("x")
        ch.send("y")
        assert ch.pending == 2
        ch.receive(timeout=1.0)
        assert ch.pending == 1


# ════════════════════════════════════════════════════════════
# Send / Receive tests
# ════════════════════════════════════════════════════════════

class TestSendReceive:

    def test_send_puts_result_on_channel(self):
        ch = Channel("pipe")
        agent = _tool("upper", lambda x: str(x).upper())
        send = Send(agent, ch)

        result = send("hello")
        assert result == "HELLO"  # Send also returns the result
        assert ch.receive(timeout=1.0) == "HELLO"

    def test_receive_gets_message(self):
        ch = Channel("pipe")
        ch.send("payload")

        recv = Receive(ch, timeout=1.0)
        result = recv("ignored_input")
        assert result == "payload"

    def test_receive_with_handler(self):
        ch = Channel("pipe")
        ch.send("raw")

        handler_agent = _tool("process", lambda x: f"processed({x})")
        recv = Receive(ch, handler=handler_agent, timeout=1.0)
        result = recv("ignored")
        assert result == "processed(raw)"

    def test_send_receive_roundtrip(self):
        """Full send-receive pipeline with threading."""
        ch = Channel("roundtrip")
        sender_agent = _tool("double", lambda x: int(x) * 2)
        send_term = Send(sender_agent, ch)

        results = {}

        def sender():
            results["sent"] = send_term("5")

        def receiver():
            recv = Receive(ch, timeout=5.0)
            results["received"] = recv("_")

        t1 = threading.Thread(target=sender)
        t2 = threading.Thread(target=receiver)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert results["sent"] == 10
        assert results["received"] == 10


# ════════════════════════════════════════════════════════════
# GroupChat tests
# ════════════════════════════════════════════════════════════

class TestGroupChat:

    def test_round_robin_two_agents(self):
        agent_a = _tool("alice", lambda x: "alice says hi")
        agent_b = _tool("bob", lambda x: "bob says hi")
        # Terminate after 2 rounds
        chat = GroupChat(
            agents=[agent_a, agent_b],
            max_rounds=2,
            scheduler="round_robin",
            termination=lambda state, r: r >= 2,
        )
        result = chat("start")
        assert isinstance(result, str)

    def test_groupchat_terminates_on_keyword(self):
        """GroupChat stops when the state contains a termination keyword."""
        round_count = {"n": 0}

        def agent_fn(x):
            round_count["n"] += 1
            if round_count["n"] >= 2:
                return "I think we reached CONSENSUS"
            return "still discussing"

        agent_a = _tool("agent_a", agent_fn)
        agent_b = _tool("agent_b", lambda x: "ok")
        chat = GroupChat(
            agents=[agent_a, agent_b],
            max_rounds=20,
            scheduler="round_robin",
            # Use default termination which checks for keywords like CONSENSUS
        )
        result = chat("topic")
        # Should stop well before 20 rounds
        assert "CONSENSUS" in result or round_count["n"] <= 5

    def test_groupchat_max_rounds_limit(self):
        call_count = {"n": 0}

        def counting_agent(x):
            call_count["n"] += 1
            return f"round {call_count['n']}"

        agent = _tool("counter", counting_agent)
        chat = GroupChat(
            agents=[agent],
            max_rounds=3,
            scheduler="round_robin",
            termination=lambda state, r: False,  # never terminate
        )
        chat("go")
        assert call_count["n"] == 3


# ════════════════════════════════════════════════════════════
# Handoff tests
# ════════════════════════════════════════════════════════════

class TestHandoff:

    def test_handoff_dispatches_to_correct_agent(self):
        math_agent = _tool("math", lambda x: f"math({x})")
        code_agent = _tool("code", lambda x: f"code({x})")
        selector = lambda x: "math"

        handoff = Handoff(
            selector=selector,
            registry={"math": math_agent, "code": code_agent},
        )
        assert handoff("2+2") == "math(2+2)"

    def test_handoff_with_fallback(self):
        fallback = _tool("fallback", lambda x: f"fallback({x})")
        handoff = Handoff(
            selector=lambda x: "nonexistent",
            registry={"math": _tool("math", lambda x: "m")},
            fallback=fallback,
        )
        assert handoff("hello") == "fallback(hello)"

    def test_handoff_raises_without_fallback(self):
        handoff = Handoff(
            selector=lambda x: "nonexistent",
            registry={"math": _tool("math", lambda x: "m")},
        )
        with pytest.raises(HandoffError, match="not found"):
            handoff("hello")

    def test_register_and_unregister(self):
        handoff = Handoff(
            selector=lambda x: "new_agent",
            registry={},
        )
        new_agent = _tool("new_agent", lambda x: f"new({x})")

        # Before register: should raise
        with pytest.raises(HandoffError):
            handoff("test")

        # Register and use
        handoff.register("new_agent", new_agent)
        assert handoff("test") == "new(test)"

        # Unregister
        handoff.unregister("new_agent")
        with pytest.raises(HandoffError):
            handoff("test")

    def test_handoff_with_term_selector(self):
        selector_agent = _tool("selector", lambda x: "code")
        code_agent = _tool("code", lambda x: f"code({x})")
        handoff = Handoff(
            selector=selector_agent,
            registry={"code": code_agent},
        )
        assert handoff("write hello world") == "code(write hello world)"


# ════════════════════════════════════════════════════════════
# AsyncPar tests
# ════════════════════════════════════════════════════════════

class TestAsyncPar:

    def test_async_par_all_results_present(self):
        a = _tool("double", lambda x: int(x) * 2)
        b = _tool("triple", lambda x: int(x) * 3)
        c = _tool("negate", lambda x: -int(x))

        par = AsyncPar(a, b, c, check_store_independence=False)
        results = par("5")
        assert results == (10, 15, -5)

    def test_async_par_preserves_order(self):
        """Results must match agent order regardless of completion time."""
        import time as _time

        def slow(x):
            _time.sleep(0.05)
            return "slow"

        def fast(x):
            return "fast"

        par = AsyncPar(
            _tool("slow", slow),
            _tool("fast", fast),
            check_store_independence=False,
        )
        results = par("x")
        assert results == ("slow", "fast")

    def test_async_par_merges_traces(self):
        a = _tool("a", lambda x: "ra")
        b = _tool("b", lambda x: "rb")
        ctx = Context()
        AsyncPar(a, b, check_store_independence=False).apply("x", ctx)
        # Each agent logs once, both traces merged into parent
        assert len(ctx.trace) >= 2
        names = {e.term_name for e in ctx.trace}
        assert "a" in names
        assert "b" in names

    def test_async_par_with_timeout(self):
        """If an agent exceeds timeout, AsyncPar should raise."""
        import time as _time

        def hang(x):
            _time.sleep(10)
            return "done"

        par = AsyncPar(
            _tool("hang", hang),
            timeout=0.2,
            check_store_independence=False,
        )
        with pytest.raises(Exception):
            par("x")

    def test_async_par_single_agent(self):
        a = _tool("only", lambda x: f"result({x})")
        par = AsyncPar(a, check_store_independence=False)
        assert par("in") == ("result(in)",)


# ════════════════════════════════════════════════════════════
# SharedMemory tests
# ════════════════════════════════════════════════════════════

class TestSharedMemory:

    def test_read_write(self):
        sm = SharedMemory()
        sm.write("key", "value")
        assert sm.read("key") == "value"

    def test_read_default(self):
        sm = SharedMemory()
        assert sm.read("missing", "default") == "default"
        assert sm.read("missing") is None

    def test_read_all(self):
        sm = SharedMemory(store={"a": 1, "b": 2})
        all_data = sm.read_all()
        assert all_data == {"a": 1, "b": 2}
        # read_all returns a copy
        all_data["c"] = 3
        assert sm.read("c") is None

    def test_append_only_mode_allows_new_keys(self):
        sm = SharedMemory(append_only=True)
        sm.write("key", "value")
        assert sm.read("key") == "value"

    def test_append_only_mode_allows_same_type_update(self):
        sm = SharedMemory(append_only=True)
        sm.write("counter", 1)
        sm.write("counter", 2)  # same type (int)
        assert sm.read("counter") == 2

    def test_append_only_mode_rejects_type_change(self):
        sm = SharedMemory(append_only=True)
        sm.write("key", "string_value")
        with pytest.raises(TypeError, match="type violation"):
            sm.write("key", 42)  # str -> int is a type violation

    def test_mutable_mode_allows_type_change(self):
        sm = SharedMemory(append_only=False)
        sm.write("key", "string")
        sm.write("key", 42)  # no error in mutable mode
        assert sm.read("key") == 42

    def test_wrap_agent(self):
        """SharedMemory.wrap injects memory into agent input."""
        captured = {}

        def capture(x):
            captured["input"] = x
            return x

        sm = SharedMemory(store={"context": "important"})
        agent = _tool("echo", capture)
        wrapped = sm.wrap(agent)
        wrapped("query")

        assert "important" in captured["input"]
        assert "query" in captured["input"]

    def test_thread_safety(self):
        """Concurrent writes should not corrupt the store."""
        sm = SharedMemory()
        errors = []

        def writer(key, value, n):
            try:
                for i in range(n):
                    sm.write(f"{key}_{i}", value)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=("a", "val_a", 50)),
            threading.Thread(target=writer, args=("b", "val_b", 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        all_data = sm.read_all()
        assert len(all_data) == 100
