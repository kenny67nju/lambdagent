"""
Integration tests for Phase 3: P2 Competitiveness Improvements

Tests:
  - T18: OpenTelemetry integration (AgentTracer)
  - T19: Token budget management
  - T20: Tool concurrency safety
  - T21: Rate limiter
  - T22: Resilient MCP client
"""

from __future__ import annotations

import asyncio
import time
import pytest


# ════════════════════════════════════════════════════════════
# T18: Observability (AgentTracer)
# ════════════════════════════════════════════════════════════


class TestObservability:
    def test_tracer_span(self):
        from lambdagent.observability import AgentTracer

        tracer = AgentTracer("test")
        with tracer.span("test_op", key="value") as span:
            time.sleep(0.01)
        assert len(tracer.spans) == 1
        assert tracer.spans[0].name == "test_op"
        assert tracer.spans[0].attributes["key"] == "value"
        assert tracer.spans[0].duration_ms >= 10

    def test_tracer_error_span(self):
        from lambdagent.observability import AgentTracer

        tracer = AgentTracer("test")
        with pytest.raises(ValueError):
            with tracer.span("failing_op") as span:
                raise ValueError("boom")
        assert len(tracer.spans) == 1
        assert "error" in tracer.spans[0].status

    def test_record_reduction(self):
        from lambdagent.observability import AgentTracer

        tracer = AgentTracer("test")
        tracer.record_reduction(
            "think", "Lam", "input", "output", 150.0, model="claude-3", tokens=500
        )
        assert len(tracer.spans) == 1
        assert tracer.spans[0].attributes["tokens.used"] == 500

    def test_export_json(self):
        from lambdagent.observability import AgentTracer

        tracer = AgentTracer("test")
        with tracer.span("op1"):
            pass
        exported = tracer.export_json()
        assert len(exported) == 1
        assert "name" in exported[0]
        assert "duration_ms" in exported[0]

    def test_summary(self):
        from lambdagent.observability import AgentTracer

        tracer = AgentTracer("test")
        with tracer.span("op1"):
            pass
        s = tracer.summary()
        assert "1 spans" in s


# ════════════════════════════════════════════════════════════
# T19: Token Budget
# ════════════════════════════════════════════════════════════


class TestTokenBudget:
    def test_basic_tracking(self):
        from lambdagent.token_budget import TokenBudget

        budget = TokenBudget(max_tokens=1000)
        assert budget.remaining == 1000
        budget.record(100, 50, model="claude-3")
        assert budget.used == 150
        assert budget.remaining == 850

    def test_can_afford(self):
        from lambdagent.token_budget import TokenBudget

        budget = TokenBudget(max_tokens=100)
        assert budget.can_afford(50)
        assert budget.can_afford(99)
        assert not budget.can_afford(100)

    def test_budget_exhausted_error(self):
        from lambdagent.token_budget import TokenBudget, BudgetExhaustedError

        budget = TokenBudget(max_tokens=100)
        budget.record(60, 50)
        with pytest.raises(BudgetExhaustedError):
            budget.check()

    def test_by_model_tracking(self):
        from lambdagent.token_budget import TokenBudget

        budget = TokenBudget(max_tokens=10000)
        budget.record(100, 50, model="claude-3")
        budget.record(200, 100, model="gpt-4")
        budget.record(50, 25, model="claude-3")
        assert budget._by_model["claude-3"] == 225
        assert budget._by_model["gpt-4"] == 300

    def test_estimate_cost(self):
        from lambdagent.token_budget import TokenBudget

        budget = TokenBudget()
        assert budget.estimate_cost("hello world") == 2  # 11 chars / 4


# ════════════════════════════════════════════════════════════
# T20: Tool concurrency safety
# ════════════════════════════════════════════════════════════


class TestConcurrentTools:
    def test_concurrent_tool_creation(self):
        from lambdagent.concurrent_tools import ConcurrentTool

        safe = ConcurrentTool("search", lambda x: x, concurrent_safe=True)
        unsafe = ConcurrentTool("write", lambda x: x, concurrent_safe=False)
        assert safe.concurrent_safe is True
        assert unsafe.concurrent_safe is False

    def test_partition_by_safety(self):
        from lambdagent.concurrent_tools import ConcurrentTool, partition_by_safety

        tools = [
            ConcurrentTool("read1", lambda x: x, concurrent_safe=True),
            ConcurrentTool("write1", lambda x: x, concurrent_safe=False),
            ConcurrentTool("read2", lambda x: x, concurrent_safe=True),
        ]
        safe, unsafe = partition_by_safety(tools)
        assert len(safe) == 2
        assert len(unsafe) == 1

    def test_helper_constructors(self):
        from lambdagent.concurrent_tools import read_tool, write_tool

        r = read_tool("search", lambda x: x)
        w = write_tool("save", lambda x: x)
        assert r.concurrent_safe is True
        assert w.concurrent_safe is False


# ════════════════════════════════════════════════════════════
# T21: Rate Limiter
# ════════════════════════════════════════════════════════════


class TestRateLimiter:
    def test_basic_acquire(self):
        from lambdagent.rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=600)  # 10/sec
        wait = limiter.acquire()
        assert wait == 0.0

    def test_rate_limiting(self):
        from lambdagent.rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=60)  # 1/sec
        # Drain all tokens
        for _ in range(60):
            limiter.acquire()
        # Next should have to wait
        t0 = time.time()
        limiter.acquire()
        elapsed = time.time() - t0
        assert elapsed >= 0.5  # Should wait ~1 second

    def test_async_acquire(self):
        from lambdagent.rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=600)

        async def _run():
            wait = await limiter.aacquire()
            return wait

        loop = asyncio.get_event_loop()
        wait = loop.run_until_complete(_run())
        assert wait == 0.0

    def test_summary(self):
        from lambdagent.rate_limiter import RateLimiter

        limiter = RateLimiter(60)
        limiter.acquire()
        s = limiter.summary()
        assert "1 acquired" in s


# ════════════════════════════════════════════════════════════
# T22: Resilient MCP Client
# ════════════════════════════════════════════════════════════


class TestResilientMCP:
    def test_client_creation(self):
        from lambdagent.resilient_mcp import ResilientMCPClient

        client = ResilientMCPClient("http://localhost:8080", name="test")
        assert client.name == "test"
        assert client.full_url == "http://localhost:8080"

    def test_tool_cache(self):
        from lambdagent.resilient_mcp import ResilientMCPClient

        client = ResilientMCPClient("http://localhost:8080")
        # Manually set cache
        from lambdagent.resilient_mcp import MCPToolSchema

        client._tool_cache = [MCPToolSchema(name="test_tool", description="A test")]
        client._cache_time = time.time()
        tools = client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"

    def test_call_tool_with_offline_server(self):
        from lambdagent.resilient_mcp import ResilientMCPClient
        from lambdagent.retry import RetryPolicy

        client = ResilientMCPClient(
            "http://localhost:99999",
            retry_policy=RetryPolicy(max_attempts=1, base_delay=0.01),
            name="offline",
        )
        result = client.call_tool("test", {"input": "hello"})
        assert "MCP_ERROR" in result

    def test_summary(self):
        from lambdagent.resilient_mcp import ResilientMCPClient

        client = ResilientMCPClient("http://localhost:8080", name="test")
        s = client.summary()
        assert "test" in s
        assert "0 calls" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
