"""
Tests for A17: Three-layer Hook system

Layer 1: Decorators (pre_hook, post_hook, guard_hook)
Layer 2: HookTerm (first-class Lambda term)
Layer 3: HookRegistry (machine-level observer)
"""

from __future__ import annotations

import asyncio
import time
import pytest

from lambdagent.core import Context
from lambdagent.primitives import Tool, Lam, Compose
from lambdagent.hooks import (
    HookRegistry,
    HookTerm,
    pre_hook,
    post_hook,
    guard_hook,
    compile_hooks_from_config,
    compile_shell_hook,
)


def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ════════════════════════════════════════════════════════════
# Layer 3: HookRegistry
# ════════════════════════════════════════════════════════════


class TestHookRegistry:
    def test_register_and_fire(self):
        reg = HookRegistry()
        calls = []
        reg.register(
            "pre_tool", lambda **kw: calls.append(("pre_tool", kw.get("input")))
        )
        reg.fire("pre_tool", input="hello", term=None)
        assert len(calls) == 1
        assert calls[0] == ("pre_tool", "hello")

    def test_multiple_hooks(self):
        reg = HookRegistry()
        log = []
        reg.register("on_step", lambda **kw: log.append("a"))
        reg.register("on_step", lambda **kw: log.append("b"))
        reg.fire("on_step")
        assert log == ["a", "b"]

    def test_hook_error_doesnt_break(self, capsys):
        """Hook errors are caught and logged, not raised."""
        reg = HookRegistry()
        reg.register("pre_llm", lambda **kw: 1 / 0)  # will raise
        reg.register("pre_llm", lambda **kw: None)  # should still run
        reg.fire("pre_llm", term=None, input="x", ctx=None)
        captured = capsys.readouterr()
        assert "HookError" in captured.err

    def test_post_llm_modify_output(self):
        """post_llm hooks can modify output via dict pattern."""
        reg = HookRegistry()

        def uppercase_hook(**kwargs):
            if "output" in kwargs and isinstance(kwargs["output"], dict):
                kwargs["output"]["value"] = str(kwargs["output"]["value"]).upper()

        reg.register("post_llm", uppercase_hook)
        output = {"value": "hello world"}
        reg.fire("post_llm", term=None, input="x", output=output, usage=None, ctx=None)
        assert output["value"] == "HELLO WORLD"

    def test_async_fire(self):
        reg = HookRegistry()
        log = []

        async def async_hook(**kwargs):
            log.append("async")

        reg.register("pre_llm", async_hook)
        reg.register("pre_llm", lambda **kw: log.append("sync"))
        run_async(reg.afire("pre_llm", term=None, input="x", ctx=None))
        assert log == ["async", "sync"]

    def test_unregister(self):
        reg = HookRegistry()
        fn = lambda **kw: None
        reg.register("on_step", fn)
        assert len(reg.on_step) == 1
        reg.unregister("on_step", fn)
        assert len(reg.on_step) == 0

    def test_clear(self):
        reg = HookRegistry()
        reg.register("pre_llm", lambda **kw: None)
        reg.register("post_tool", lambda **kw: None)
        reg.clear()
        assert len(reg.pre_llm) == 0
        assert len(reg.post_tool) == 0

    def test_invalid_event(self):
        reg = HookRegistry()
        with pytest.raises(ValueError, match="Unknown"):
            reg.register("nonexistent_event", lambda: None)

    def test_summary(self):
        reg = HookRegistry()
        reg.register("pre_llm", lambda **kw: None)
        reg.register("pre_llm", lambda **kw: None)
        reg.register("on_step", lambda **kw: None)
        s = reg.summary()
        assert "pre_llm: 2" in s
        assert "on_step: 1" in s


# ════════════════════════════════════════════════════════════
# Layer 2: HookTerm
# ════════════════════════════════════════════════════════════


class TestHookTerm:
    def test_pre_hook(self):
        """Pre hook runs before agent, doesn't modify input."""
        log = []
        agent = Tool("double", lambda x: int(x) * 2)
        hooked = HookTerm(agent, pre=lambda x: log.append(f"pre:{x}"))
        result = hooked.apply("5", Context())
        assert result == 10
        assert log == ["pre:5"]

    def test_post_hook(self):
        """Post hook can modify output."""
        agent = Tool("greet", lambda x: f"hello {x}")
        hooked = HookTerm(agent, post=lambda x: x.upper())
        result = hooked.apply("world", Context())
        assert result == "HELLO WORLD"

    def test_error_hook(self):
        """on_error hook provides fallback."""
        agent = Tool("fail", lambda x: 1 / 0)
        hooked = HookTerm(agent, on_error=lambda e, x: f"fallback for {x}")
        result = hooked.apply("test", Context())
        assert result == "fallback for test"

    def test_error_reraise(self):
        """Without on_error, exceptions propagate."""
        agent = Tool("fail", lambda x: 1 / 0)
        hooked = HookTerm(agent)
        with pytest.raises(ZeroDivisionError):
            hooked.apply("test", Context())

    def test_combined_pre_post(self):
        log = []
        agent = Tool("id", lambda x: x)
        hooked = HookTerm(
            agent,
            pre=lambda x: log.append("pre"),
            post=lambda x: (log.append("post"), x + "!")[1],
        )
        result = hooked.apply("hello", Context())
        assert result == "hello!"
        assert log == ["pre", "post"]

    def test_async_aapply(self):
        """HookTerm.aapply works async."""
        import lambdagent.async_core  # patch aapply

        agent = Tool("double", lambda x: int(x) * 2)
        hooked = HookTerm(agent, post=lambda x: x + 1)
        result = run_async(hooked.aapply("5", Context()))
        assert result == 11

    def test_trace_logged(self):
        agent = Tool("id", lambda x: x)
        hooked = HookTerm(agent, name="MyHook")
        ctx = Context()
        hooked.apply("input", ctx)
        # Both Tool and HookTerm log to trace
        names = [e.term_name for e in ctx.trace]
        assert "id" in names
        assert "MyHook" in names


# ════════════════════════════════════════════════════════════
# Layer 1: Decorators
# ════════════════════════════════════════════════════════════


class TestDecorators:
    def test_pre_hook_decorator(self):
        log = []
        base = Tool("echo", lambda x: x)
        agent = pre_hook(lambda x: log.append(x))(base)
        result = agent.apply("hello", Context())
        assert result == "hello"
        assert "hello" in log

    def test_post_hook_decorator(self):
        base = Tool("echo", lambda x: x)
        agent = post_hook(lambda x: x.upper())(base)
        result = agent.apply("hello", Context())
        assert result == "HELLO"

    def test_guard_hook_decorator(self):
        from lambdagent.core import ValidationError

        base = Tool("short", lambda x: "hi")
        agent = guard_hook(lambda x: len(str(x)) > 3, retry=1)(base)
        with pytest.raises(ValidationError):
            agent.apply("input", Context())

    def test_stacked_decorators(self):
        """Multiple decorators compose correctly."""
        log = []
        base = Tool("echo", lambda x: x)
        agent = pre_hook(lambda x: log.append("pre"))(
            post_hook(lambda x: x + "!")(base)
        )
        result = agent.apply("hi", Context())
        assert result == "hi!"
        assert "pre" in log


# ════════════════════════════════════════════════════════════
# YAML Hook compilation
# ════════════════════════════════════════════════════════════


class TestYAMLHooks:
    def test_compile_hooks_from_config(self):
        cfg = {
            "pre_tool": [
                {"command": "echo hook_fired"},
            ],
        }
        registry = compile_hooks_from_config(cfg)
        assert len(registry.pre_tool) == 1

    def test_compile_empty_config(self):
        registry = compile_hooks_from_config({})
        assert len(registry.pre_llm) == 0

    def test_shell_hook_execution(self):
        """Shell hook runs without error."""
        fn = compile_shell_hook("echo test", "pre_tool")
        # Should not raise
        mock_term = type("MockTerm", (), {"_name": "test"})()
        fn(term=mock_term, input="hello")


# ════════════════════════════════════════════════════════════
# Integration: Three layers coexist
# ════════════════════════════════════════════════════════════


class TestThreeLayerIntegration:
    def test_all_layers_fire(self):
        """Layer 1 + Layer 2 + Layer 3 all fire in one execution."""
        log = []

        # Layer 3: global observer
        registry = HookRegistry()
        registry.register("pre_tool", lambda **kw: log.append("L3:pre_tool"))

        # Layer 2: HookTerm wrapper
        inner = Tool("compute", lambda x: int(x) * 2)
        hooked = HookTerm(
            inner,
            pre=lambda x: log.append("L2:pre"),
            post=lambda x: (log.append("L2:post"), x)[1],
        )

        # Layer 1: inline compose
        from lambdagent.primitives import Compose

        pipeline = Compose(
            Tool("L1:log", lambda x: (log.append("L1:pre"), x)[1]),
            hooked,
        )

        ctx = Context()
        result = pipeline.apply("5", ctx)
        assert result == 10

        # Layer 1 and Layer 2 should have fired
        assert "L1:pre" in log
        assert "L2:pre" in log
        assert "L2:post" in log

        # Layer 3 fires when tools are called through AsyncExecutor (not direct apply)
        # So we manually fire to verify it works
        registry.fire("pre_tool", term=inner, input="5", ctx=ctx)
        assert "L3:pre_tool" in log


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
