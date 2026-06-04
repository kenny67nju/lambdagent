"""
Integration tests for Phase 2: P1 Reliability Improvements

Tests:
  - T13: Par true parallel
  - T14: Context compaction
  - T15: Tool input validation
  - T16: GroupChat anti-state-explosion
  - T17: Execution checkpoint
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import pytest

from lambdagent.core import Context
from lambdagent.primitives import Tool, Lam, Compose, Loop


# ════════════════════════════════════════════════════════════
# T13: Par true parallel
# ════════════════════════════════════════════════════════════

class TestParTrueParallel:
    def test_par_runs_parallel(self):
        from lambdagent.extensions import Par
        t0 = time.time()
        slow1 = Tool("s1", lambda x: (time.sleep(0.1), "a")[1])
        slow2 = Tool("s2", lambda x: (time.sleep(0.1), "b")[1])
        par = Par(slow1, slow2)
        result = par.apply("x", Context())
        elapsed = time.time() - t0
        assert result == ("a", "b")
        # Parallel should run ~0.1s, sequential would be ~0.2s. Threshold of
        # 0.19s gives generous headroom for CI scheduling jitter while still
        # comfortably below the 0.2s sequential baseline.
        assert elapsed < 0.19, f"expected parallel ~0.1s, got {elapsed:.3f}s (sequential would be ~0.2s)"

    def test_par_single_agent(self):
        from lambdagent.extensions import Par
        tool = Tool("only", lambda x: "result")
        par = Par(tool)
        result = par.apply("x", Context())
        assert result == ("result",)

    def test_par_preserves_order(self):
        from lambdagent.extensions import Par
        # Agent 1 is slow, Agent 2 is fast — results should maintain order
        slow = Tool("slow", lambda x: (time.sleep(0.05), "slow")[1])
        fast = Tool("fast", lambda x: "fast")
        par = Par(slow, fast)
        result = par.apply("x", Context())
        assert result == ("slow", "fast")


# ════════════════════════════════════════════════════════════
# T14: Context compaction
# ════════════════════════════════════════════════════════════

class TestContextCompaction:
    def test_should_compact(self):
        from lambdagent.context_manager import ContextManager
        cm = ContextManager(max_tokens=100, compact_threshold=0.8)
        short = "hello"
        assert not cm.should_compact(short)
        long = "a" * 400  # 100 tokens at 4 chars/token
        assert cm.should_compact(long)

    def test_compact_preserves_recent(self):
        from lambdagent.context_manager import ContextManager
        cm = ContextManager(max_tokens=10000, keep_recent=2)

        state = "User: what's 2+2?\n\n"
        state += "[Step 1]\nThought: need to calculate\nAction: calc\nObservation: 4\n\n"
        state += "[Step 2]\nThought: got result\nAction: verify\nObservation: correct\n\n"
        state += "[Step 3]\nThought: verified\nAction: terminate\nObservation: done\n\n"
        state += "[Step 4]\nThought: final\nAction: report\nObservation: 2+2=4"

        compacted = cm.compact(state)
        # Recent 2 steps should be intact
        assert "[Step 4]" in compacted
        assert "[Step 3]" in compacted
        # Old steps should be summarized
        assert "[Previous Steps Summary]" in compacted

    def test_compact_no_op_for_short(self):
        from lambdagent.context_manager import ContextManager
        cm = ContextManager(keep_recent=5)
        state = "User: hi\n\n[Step 1]\nThought: hello"
        assert cm.compact(state) == state


# ════════════════════════════════════════════════════════════
# T15: Tool input validation
# ════════════════════════════════════════════════════════════

class TestToolValidation:
    def test_validated_tool_with_schema(self):
        from lambdagent.validated_tool import ValidatedTool, ShellToolInput

        def mock_shell(input_dict):
            return f"executed: {input_dict['command']}"

        tool = ValidatedTool("shell", mock_shell, schema=ShellToolInput)
        result = tool.apply(json.dumps({"command": "ls -la"}), Context())
        assert "executed: ls -la" in result

    def test_validated_tool_validation_error(self):
        from lambdagent.validated_tool import ValidatedTool, ShellToolInput

        tool = ValidatedTool("shell", lambda x: x, schema=ShellToolInput)
        result = tool.apply(json.dumps({"command": ""}), Context())
        assert "VALIDATION_ERROR" in result

    def test_validated_tool_no_schema(self):
        from lambdagent.validated_tool import ValidatedTool

        tool = ValidatedTool("passthrough", lambda x: f"got: {x}")
        result = tool.apply("hello", Context())
        assert result == "got: hello"


# ════════════════════════════════════════════════════════════
# T16: GroupChat anti-state-explosion
# ════════════════════════════════════════════════════════════

class TestGroupChatAntiExplosion:
    def test_build_speaker_input_short(self):
        from lambdagent.multiagent import GroupChat
        a1 = Tool("alice", lambda x: "hi")
        a2 = Tool("bob", lambda x: "hello")
        chat = GroupChat([a1, a2], max_rounds=5)

        conversation = [
            {"speaker": "alice", "content": "hello", "round": 0},
            {"speaker": "bob", "content": "hi there", "round": 0},
        ]
        result = chat._build_speaker_input("alice", conversation, "task", 1)
        # Short conversation: should include everything
        assert "hello" in result
        assert "hi there" in result

    def test_build_speaker_input_long(self):
        from lambdagent.multiagent import GroupChat
        a1 = Tool("alice", lambda x: "hi")
        a2 = Tool("bob", lambda x: "hello")
        chat = GroupChat([a1, a2], max_rounds=20)

        # Create a long conversation
        conversation = []
        for i in range(20):
            conversation.append({
                "speaker": "alice" if i % 2 == 0 else "bob",
                "content": f"message {i}",
                "round": i // 2,
            })

        result = chat._build_speaker_input("alice", conversation, "task", 10)
        # Should have topic, own messages, and recent
        assert "[Original Task]" in result or "task" in result
        # Should NOT have ALL messages (that would be state explosion)
        assert len(result) < sum(len(m["content"]) for m in conversation) * 2


# ════════════════════════════════════════════════════════════
# T17: Execution checkpoint
# ════════════════════════════════════════════════════════════

class TestExecutionCheckpoint:
    def test_checkpoint_save_load(self):
        from lambdagent.execution_checkpoint import ExecutionCheckpoint, StackFrame

        cp = ExecutionCheckpoint(
            last_input="test input",
            description="test checkpoint",
        )
        cp.push_frame("Loop", "react_loop", 5, result="step 5 result")
        cp.update_step(5, "intermediate")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            cp.save(path)
            loaded = ExecutionCheckpoint.load(path)
            assert loaded.last_input == "test input"
            assert loaded.resume_step == 5
            assert loaded.term_type == "Loop"
            assert len(loaded.execution_stack) == 1
        finally:
            os.unlink(path)

    def test_checkpoint_manager(self):
        from lambdagent.execution_checkpoint import ExecutionCheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExecutionCheckpointManager(
                directory=tmpdir, save_every_n_steps=2, max_checkpoints=3
            )
            cp = mgr.begin("Loop", "test_loop", "hello")

            for i in range(6):
                saved = mgr.step(i, f"result_{i}")

            mgr.finish()

            # Should have saved checkpoints
            latest = mgr.latest()
            assert latest is not None
            assert latest.resume_step >= 4

    def test_stack_frame_serialization(self):
        from lambdagent.execution_checkpoint import StackFrame
        frame = StackFrame("GroupChat", "discussion", 3, {"round": 3, "speakers": 5})
        d = frame.to_dict()
        restored = StackFrame.from_dict(d)
        assert restored.term_type == "GroupChat"
        assert restored.step_index == 3
        assert restored.local_state["round"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
