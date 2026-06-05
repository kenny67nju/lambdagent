"""
Integration tests for Phase 1: P0 Engineering Improvements

Tests:
  - T01: Async Term (aapply)
  - T02: LLM streaming interface
  - T03/T04: Async Executor + ReActEngine
  - T05: Retry + CircuitBreaker
  - T06: Configurable timeouts
  - T07: CancellationToken
  - T08: Workspace isolation
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import time
import pytest
from unittest.mock import MagicMock, patch


# Helper for running async tests without pytest-asyncio auto mode
def run_async(coro):
    """Run async test in a new event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ════════════════════════════════════════════════════════════
# T05: Retry + CircuitBreaker
# ════════════════════════════════════════════════════════════


class TestRetryPolicy:
    def test_sync_retry_success(self):
        from lambdagent.retry import RetryPolicy, with_retry_sync

        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "ok"

        policy = RetryPolicy(max_attempts=3, base_delay=0.01, jitter=False)
        result = with_retry_sync(flaky, policy)
        assert result == "ok"
        assert call_count == 3

    def test_sync_retry_exhausted(self):
        from lambdagent.retry import RetryPolicy, with_retry_sync

        def always_fail():
            raise ConnectionError("permanent")

        policy = RetryPolicy(max_attempts=2, base_delay=0.01, jitter=False)
        with pytest.raises(ConnectionError):
            with_retry_sync(always_fail, policy)

    def test_async_retry_success(self):
        from lambdagent.retry import RetryPolicy, with_retry

        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("slow")
            return "done"

        async def _run():
            policy = RetryPolicy(max_attempts=3, base_delay=0.01, jitter=False)
            return await with_retry(flaky, policy)

        result = run_async(_run())
        assert result == "done"

    def test_async_retry_with_timeout(self):
        from lambdagent.retry import RetryPolicy, with_retry

        async def slow():
            await asyncio.sleep(10)
            return "never"

        async def _run():
            policy = RetryPolicy(max_attempts=1, base_delay=0.01)
            await with_retry(slow, policy, timeout=0.1)

        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            run_async(_run())


class TestCircuitBreaker:
    def test_circuit_opens_after_failures(self):
        from lambdagent.retry import CircuitBreaker, CircuitOpenError

        cb = CircuitBreaker(failure_threshold=2, reset_timeout=60)

        def fail():
            raise RuntimeError("boom")

        for _ in range(2):
            try:
                cb.call_sync(fail)
            except RuntimeError:
                pass

        with pytest.raises(CircuitOpenError):
            cb.call_sync(lambda: "ok")

    def test_circuit_resets(self):
        from lambdagent.retry import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        def fail():
            raise RuntimeError("boom")

        for _ in range(2):
            try:
                cb.call_sync(fail)
            except RuntimeError:
                pass

        # Wait for reset
        time.sleep(0.15)
        result = cb.call_sync(lambda: "recovered")
        assert result == "recovered"


# ════════════════════════════════════════════════════════════
# T07: CancellationToken
# ════════════════════════════════════════════════════════════


class TestCancellationToken:
    def test_basic_cancel(self):
        from lambdagent.cancellation import CancellationToken, CancelledError

        token = CancellationToken()
        assert not token.is_cancelled
        token.cancel("test")
        assert token.is_cancelled
        with pytest.raises(CancelledError):
            token.check()

    def test_parent_child_propagation(self):
        from lambdagent.cancellation import CancellationToken

        parent = CancellationToken()
        child1 = parent.child()
        child2 = parent.child()
        grandchild = child1.child()

        parent.cancel("cascade")
        assert child1.is_cancelled
        assert child2.is_cancelled
        assert grandchild.is_cancelled

    def test_child_cancel_doesnt_affect_parent(self):
        from lambdagent.cancellation import CancellationToken

        parent = CancellationToken()
        child = parent.child()
        child.cancel("local")
        assert not parent.is_cancelled
        assert child.is_cancelled

    def test_null_token_never_cancels(self):
        from lambdagent.cancellation import NullCancellationToken

        token = NullCancellationToken()
        token.cancel("should not work")
        assert not token.is_cancelled
        token.check()  # should not raise

    def test_callback_on_cancel(self):
        from lambdagent.cancellation import CancellationToken

        called = []
        token = CancellationToken()
        token.on_cancel(lambda: called.append("called"))
        token.cancel("test_reason")
        assert called == ["called"]


# ════════════════════════════════════════════════════════════
# T08: Workspace Isolation
# ════════════════════════════════════════════════════════════


class TestWorkspaceIsolation:
    def test_none_isolation(self):
        from lambdagent.isolation import WorkspaceManager, IsolationLevel

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = WorkspaceManager(tmpdir)
            ws = mgr.create("agent-1", IsolationLevel.NONE)
            assert ws.workspace_path == os.path.abspath(tmpdir)
            assert ws.isolation_level == IsolationLevel.NONE
            mgr.cleanup("agent-1")

    def test_directory_isolation(self):
        from lambdagent.isolation import WorkspaceManager, IsolationLevel

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file in base dir
            with open(os.path.join(tmpdir, "test.txt"), "w") as f:
                f.write("hello")

            mgr = WorkspaceManager(tmpdir)
            ws = mgr.create("agent-2", IsolationLevel.DIRECTORY)
            assert ws.workspace_path != tmpdir
            assert os.path.exists(ws.workspace_path)
            # File should be copied
            assert os.path.exists(os.path.join(ws.workspace_path, "test.txt"))
            mgr.cleanup("agent-2", force=True)

    def test_worktree_isolation(self):
        """Test git worktree isolation (requires git repo)."""
        from lambdagent.isolation import WorkspaceManager, IsolationLevel

        with tempfile.TemporaryDirectory() as tmpdir:
            # Init a git repo
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=tmpdir,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"], cwd=tmpdir, capture_output=True
            )
            with open(os.path.join(tmpdir, "file.txt"), "w") as f:
                f.write("content")
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True
            )

            mgr = WorkspaceManager(tmpdir)
            ws = mgr.create("coder", IsolationLevel.WORKTREE)
            assert ws.isolation_level == IsolationLevel.WORKTREE
            assert ws.branch_name is not None
            assert os.path.exists(os.path.join(ws.workspace_path, "file.txt"))

            # Write in workspace shouldn't affect base
            with open(os.path.join(ws.workspace_path, "new.txt"), "w") as f:
                f.write("isolated")
            assert not os.path.exists(os.path.join(tmpdir, "new.txt"))
            assert ws.has_changes()

            mgr.cleanup("coder", force=True)

    def test_slug_validation(self):
        from lambdagent.isolation import WorkspaceManager

        mgr = WorkspaceManager(".")
        with pytest.raises(Exception):  # IsolationError or ValueError
            mgr._validate_slug("../escape")
        with pytest.raises(Exception):
            mgr._validate_slug("a" * 65)


# ════════════════════════════════════════════════════════════
# T01: Async Term (aapply)
# ════════════════════════════════════════════════════════════


class TestAsyncTerm:
    def test_tool_aapply(self):
        import lambdagent.async_core  # trigger patching
        from lambdagent.primitives import Tool
        from lambdagent.core import Context

        async def _run():
            tool = Tool("double", lambda x: int(x) * 2)
            return await tool.aapply("5", Context())

        result = run_async(_run())
        assert result == 10

    def test_compose_aapply(self):
        import lambdagent.async_core
        from lambdagent.primitives import Compose, Tool
        from lambdagent.core import Context

        async def _run():
            add1 = Tool("add1", lambda x: int(x) + 1)
            mul2 = Tool("mul2", lambda x: int(x) * 2)
            pipeline = Compose(add1, mul2)
            return await pipeline.aapply("3", Context())

        result = run_async(_run())
        assert result == 8  # (3+1)*2

    def test_par_aapply_parallel(self):
        import lambdagent.async_core
        from lambdagent.extensions import Par
        from lambdagent.primitives import Tool
        from lambdagent.core import Context

        async def _run():
            t0 = time.time()
            slow1 = Tool("slow1", lambda x: (time.sleep(0.1), "a")[1])
            slow2 = Tool("slow2", lambda x: (time.sleep(0.1), "b")[1])
            par = Par(slow1, slow2)
            result = await par.aapply("x", Context())
            elapsed = time.time() - t0
            return result, elapsed

        result, elapsed = run_async(_run())
        assert result == ("a", "b")
        # Should run in parallel: ~0.1s, not ~0.2s
        assert elapsed < 0.25  # generous margin for CI

    def test_loop_aapply_with_cancel(self):
        import lambdagent.async_core
        from lambdagent.primitives import Loop, Tool
        from lambdagent.cancellation import CancellationToken, CancelledError
        from lambdagent.core import Context

        counter = [0]

        def inc(x):
            counter[0] += 1
            time.sleep(0.005)  # Small delay to let cancel_later run
            return str(counter[0])

        async def _run():
            body = Tool("inc", inc)
            loop = Loop(body, condition=lambda r, s: int(r) >= 100, max_steps=100)
            cancel = CancellationToken()

            async def cancel_later():
                await asyncio.sleep(0.05)
                cancel.cancel("timeout")

            asyncio.ensure_future(cancel_later())
            await loop.aapply("0", Context(), cancel)

        with pytest.raises(CancelledError):
            run_async(_run())
        assert counter[0] < 100


# ════════════════════════════════════════════════════════════
# T06: Configurable Timeouts
# ════════════════════════════════════════════════════════════


class TestConfigurableTimeouts:
    def test_timeout_config_defaults(self):
        from lambdagent.agentruntime.config import TimeoutConfig

        tc = TimeoutConfig()
        assert tc.llm_call == 120
        assert tc.tool_call == 30
        assert tc.mcp_call == 30
        assert tc.shell == 30

    def test_timeout_call_in_compiler(self):
        from lambdagent.fromconfig.compiler import _timeout_call
        from lambdagent.primitives import Tool

        fast = Tool("fast", lambda x: "ok")
        assert _timeout_call(fast, "input", 5) == "ok"

    def test_timeout_call_timeout(self):
        from lambdagent.fromconfig.compiler import _timeout_call
        from lambdagent.primitives import Tool

        slow = Tool("slow", lambda x: (time.sleep(5), "never")[1])
        with pytest.raises(TimeoutError):
            _timeout_call(slow, "input", 0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
