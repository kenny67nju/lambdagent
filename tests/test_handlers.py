"""
Tests for Paper III §6 Algebraic Effect Handlers.

Tests cover:
  1. TestHandler — Mock LLM + Mock tools (deterministic)
  2. TraceHandler — real execution with logging
  3. with_handler context manager
  4. Handler type preservation (switching handler preserves behavior)
  5. Lam + Tool integration with handlers
  6. Composed agents with handlers
"""

import pytest
from lambdagent.handlers import (
    EffectHandler, ProductionHandler, TestHandler, TraceHandler,
    get_current_handler, set_current_handler, with_handler,
)
from lambdagent.primitives import Lam, Compose, Tool, Pair
from lambdagent.extensions import Guard, Memory
from lambdagent.core import Context


# ============================================================
# 1. TestHandler — Mock LLM + Mock Tools
# ============================================================

class TestTestHandler:

    def test_default_llm_mock(self):
        """Default mock returns generic response"""
        handler = TestHandler()
        result = handler.handle_llm("prompt", "input", "model")
        assert result == "Mock LLM response"

    def test_pattern_matching_llm(self):
        """Mock with pattern matching"""
        handler = TestHandler()
        handler.mock_llm("summarize", "This is a concise summary.")
        result = handler.handle_llm("Summarize this text", "long text here", "claude")
        assert result == "This is a concise summary."

    def test_pattern_case_insensitive(self):
        """Pattern matching is case-insensitive"""
        handler = TestHandler()
        handler.mock_llm("TRANSLATE", "翻译结果")
        result = handler.handle_llm("translate this", "hello", "model")
        assert result == "翻译结果"

    def test_tool_mock(self):
        """Mock tool returns preset result"""
        handler = TestHandler()
        handler.mock_tool("search", {"results": ["a", "b"]})
        result = handler.handle_tool("search", lambda x: None, "query")
        assert result == {"results": ["a", "b"]}

    def test_default_tool_mock(self):
        """Default tool mock returns generic result"""
        handler = TestHandler()
        result = handler.handle_tool("unknown_tool", lambda x: x, "input")
        assert result == "Mock tool result"

    def test_call_log(self):
        """All calls are logged"""
        handler = TestHandler()
        handler.handle_llm("prompt", "input", "model")
        handler.handle_tool("tool", lambda x: x, "val")
        assert len(handler.call_log) == 2
        assert len(handler.llm_calls) == 1
        assert len(handler.tool_calls) == 1

    def test_reset(self):
        """reset() clears all mocks and logs"""
        handler = TestHandler()
        handler.mock_llm("test", "response")
        handler.handle_llm("test", "input", "model")
        handler.reset()
        assert len(handler.call_log) == 0
        assert handler.handle_llm("test", "input", "model") == "Mock LLM response"

    def test_state_operations(self):
        """State read/write use internal dict"""
        handler = TestHandler()
        handler.handle_state_write({}, "key", "value")
        result = handler.handle_state_read({}, "key")
        assert result == "value"


# ============================================================
# 2. TraceHandler
# ============================================================

class TestTraceHandler:

    def test_tool_tracing(self):
        """TraceHandler records tool calls"""
        handler = TraceHandler()
        result = handler.handle_tool("double", lambda x: int(x) * 2, "5")
        assert result == 10
        assert len(handler.trace) == 1
        assert handler.trace[0]["tool_name"] == "double"

    def test_state_tracing(self):
        """TraceHandler records state operations"""
        handler = TraceHandler()
        store = {"key": "old"}
        handler.handle_state_write(store, "key", "new")
        handler.handle_state_read(store, "key")
        assert len(handler.trace) == 2
        assert handler.trace[0]["type"] == "state_write"
        assert handler.trace[1]["type"] == "state_read"

    def test_cost_tracking(self):
        """TraceHandler accumulates cost"""
        handler = TraceHandler()
        handler.handle_cost(100, 500.0, "claude")
        handler.handle_cost(200, 300.0, "claude")
        assert handler.total_tokens == 300
        assert handler.total_latency_ms == 800.0

    def test_summary(self):
        """summary() returns structured info"""
        handler = TraceHandler()
        handler.handle_tool("t", lambda x: x, "val")
        handler.handle_cost(100, 50.0, "model")
        summary = handler.summary()
        assert summary["total_events"] == 2
        assert summary["tool_calls"] == 1
        assert summary["total_tokens"] == 100


# ============================================================
# 3. with_handler Context Manager
# ============================================================

class TestWithHandler:

    def test_handler_activation(self):
        """with_handler sets current handler"""
        handler = TestHandler()
        assert get_current_handler() is None
        with with_handler(handler) as h:
            assert get_current_handler() is handler
            assert h is handler
        assert get_current_handler() is None

    def test_handler_nesting(self):
        """Nested handlers restore correctly"""
        h1 = TestHandler()
        h2 = TestHandler()
        with with_handler(h1):
            assert get_current_handler() is h1
            with with_handler(h2):
                assert get_current_handler() is h2
            assert get_current_handler() is h1
        assert get_current_handler() is None

    def test_handler_exception_safety(self):
        """Handler restored even on exception"""
        handler = TestHandler()
        try:
            with with_handler(handler):
                raise ValueError("test error")
        except ValueError:
            pass
        assert get_current_handler() is None


# ============================================================
# 4. Lam Integration with Handler
# ============================================================

class TestLamWithHandler:

    def test_lam_uses_test_handler(self):
        """Lam delegates to TestHandler.handle_llm()"""
        handler = TestHandler()
        handler.mock_llm("summarize", "Mock summary result")
        lam = Lam("summarizer", "Summarize the following:")

        with with_handler(handler):
            result = lam("Please summarize this long text about AI")
        assert result == "Mock summary result"

    def test_lam_records_in_handler(self):
        """Lam calls are logged in handler"""
        handler = TestHandler()
        lam = Lam("test", "System prompt")

        with with_handler(handler):
            lam("user input")
        assert len(handler.llm_calls) == 1
        assert "System prompt" in handler.llm_calls[0]["prompt"]


# ============================================================
# 5. Tool Integration with Handler
# ============================================================

class TestToolWithHandler:

    def test_tool_uses_test_handler(self):
        """Tool delegates to TestHandler.handle_tool()"""
        handler = TestHandler()
        handler.mock_tool("search", {"results": ["found it"]})
        tool = Tool("search", lambda x: {"results": ["real result"]})

        with with_handler(handler):
            result = tool("query")
        assert result == {"results": ["found it"]}

    def test_tool_no_handler_uses_real_fn(self):
        """Without handler, Tool uses real function"""
        tool = Tool("double", lambda x: int(x) * 2)
        result = tool("5")
        assert result == 10

    def test_tool_trace_handler_uses_real_fn(self):
        """TraceHandler still executes real function"""
        handler = TraceHandler()
        tool = Tool("double", lambda x: int(x) * 2)

        with with_handler(handler):
            result = tool("5")
        assert result == 10
        assert len(handler.trace) == 1


# ============================================================
# 6. Composed Agents with Handlers
# ============================================================

class TestComposedAgentsWithHandler:

    def test_compose_with_test_handler(self):
        """Compose(Lam, Tool) works with TestHandler"""
        handler = TestHandler()
        handler.mock_llm_default("42")
        handler.mock_tool("double", 84)

        lam = Lam("to_num", "Convert to number")
        tool = Tool("double", lambda x: int(x) * 2)
        pipeline = lam >> tool

        with with_handler(handler):
            result = pipeline("what is 21 * 2?")
        assert result == 84
        assert len(handler.llm_calls) == 1
        assert len(handler.tool_calls) == 1

    def test_guard_with_test_handler(self):
        """Guard retries work with mock responses"""
        handler = TestHandler()
        # First call returns "invalid", second returns "valid"
        call_count = [0]
        original_handle = handler.handle_llm

        def counting_handle(prompt, input_text, model, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 1:
                return "invalid"
            return "valid_output"

        handler.handle_llm = counting_handle
        lam = Lam("agent", "Do something")
        guard = Guard(lam, lambda x: "valid" in x, retry=2)

        with with_handler(handler):
            result = guard("test input")
        assert "valid" in result

    def test_handler_preserves_types(self):
        """Handler switching preserves type annotations (Paper III Theorem)"""
        lam = Lam("typed", "Process input")
        from lambdagent.types import T_STR, T_JSON
        lam.input_type = T_STR
        lam.output_type = T_JSON({"type": "object"})

        handler = TestHandler()
        handler.mock_llm_default('{"result": "ok"}')

        with with_handler(handler):
            result = lam("test")
            # Type annotations are preserved regardless of handler
            assert lam.input_type == T_STR
            assert lam.output_type == T_JSON({"type": "object"})
