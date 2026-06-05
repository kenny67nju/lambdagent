"""
lambdagent.handlers — Paper III §6 代数效果处理器

实现论文 III 的效果处理器系统:
  - EffectHandler: 效果处理器抽象接口
  - ProductionHandler: 生产环境 — 真实 LLM + 真实工具
  - TestHandler: 测试环境 — Mock LLM + Mock 工具（确定性）
  - TraceHandler: 调试环境 — 真实调用 + 完整日志记录

核心概念:
    同一 Agent，不同执行语义:
        handler.handle_llm(prompt, input, model, ...) → response
        handler.handle_tool(tool_fn, input) → result
        handler.handle_state_read(key) → value
        handler.handle_state_write(key, value) → None

    切换处理器 = 切换执行环境，不修改 Agent 代码。

    Paper III 定理 (Handler Type Preservation):
        如果 agent: A →^ε B，则对任何合法的 handler h，
        h(agent): A →^ε' B（类型保持，效果可能变化）。
"""

from __future__ import annotations

import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================
# EffectHandler: 效果处理器抽象接口
# ============================================================


class EffectHandler(ABC):
    """
    代数效果处理器 — Paper III §6。

    每个 handler 定义了如何处理 4 种效果:
        LLM    → handle_llm()
        IO     → handle_tool()
        State  → handle_state_read() / handle_state_write()
        Cost   → handle_cost() (可选)
    """

    @abstractmethod
    def handle_llm(
        self,
        prompt: str,
        input_text: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """处理 LLM 效果 — 自回归解码"""
        ...

    @abstractmethod
    def handle_tool(
        self,
        tool_name: str,
        tool_fn: Callable,
        input_val: Any,
    ) -> Any:
        """处理 IO 效果 — 工具调用"""
        ...

    def handle_state_read(
        self, store: Dict[str, Any], key: str, default: Any = None
    ) -> Any:
        """处理 State 读取效果"""
        return store.get(key, default)

    def handle_state_write(self, store: Dict[str, Any], key: str, value: Any) -> None:
        """处理 State 写入效果"""
        store[key] = value

    def handle_cost(self, tokens: int, latency_ms: float, model: str) -> None:
        """处理 Cost 效果（可选钩子）"""
        pass


# ============================================================
# ProductionHandler: 生产环境
# ============================================================


class PassthroughHandler(EffectHandler):
    """Handlers that delegate to the original Term.apply() for LLM/Tool calls.

    DESIGN-01: Explicit marker class. The CEK machine and Lam.apply() check
    isinstance(handler, PassthroughHandler) to know whether to delegate
    or use the handler's handle_llm/handle_tool methods.
    """

    pass


class ProductionHandler(PassthroughHandler):
    """
    生产效果处理器 — 真实执行。

    | 效果 | 行为 |
    |------|------|
    | llm(m) | 真实 LLM API 调用 |
    | io | 真实工具执行 |
    | state(s) | 真实 dict/Redis/DB |
    | cost | 记录到 trace |
    """

    def handle_llm(
        self,
        prompt: str,
        input_text: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """真实 LLM 调用 — 委托给 Lam._call_llm"""
        # 生产处理器直接返回 None，让 Lam 自己调用
        # 这是一个 passthrough — Lam 检查 handler 返回 None 时自行调用
        raise NotImplementedError(
            "ProductionHandler.handle_llm should not be called directly. "
            "Lam uses its own _call_llm when handler is ProductionHandler."
        )

    def handle_tool(self, tool_name: str, tool_fn: Callable, input_val: Any) -> Any:
        """真实工具执行"""
        return tool_fn(input_val)


# ============================================================
# TestHandler: 测试环境（确定性 Mock）
# ============================================================


class TestHandler(EffectHandler):
    """
    测试效果处理器 — 确定性 Mock。

    | 效果 | 行为 |
    |------|------|
    | llm(m) | 返回预设 Mock 响应 |
    | io | 返回预设 Mock 工具结果 |
    | state(s) | 内存字典 |
    | cost | 无操作 |

    使用方式:
        handler = TestHandler()
        handler.mock_llm("summarize", "This is a summary.")
        handler.mock_tool("search", {"results": ["a", "b"]})
    """

    def __init__(self):
        self._llm_mocks: Dict[str, str] = {}  # prompt_pattern → response
        self._llm_default: str = "Mock LLM response"
        self._tool_mocks: Dict[str, Any] = {}  # tool_name → response
        self._tool_default: Any = "Mock tool result"
        self._state: Dict[str, Any] = {}  # 内存状态存储
        self._call_log: List[Dict[str, Any]] = []  # 调用记录

    def mock_llm(self, pattern: str, response: str):
        """设置 LLM Mock: 当 prompt 或 input 包含 pattern 时返回 response"""
        self._llm_mocks[pattern.lower()] = response

    def mock_llm_default(self, response: str):
        """设置默认 LLM Mock 响应"""
        self._llm_default = response

    def mock_tool(self, tool_name: str, response: Any):
        """设置工具 Mock: tool_name → response"""
        self._tool_mocks[tool_name] = response

    def mock_tool_default(self, response: Any):
        """设置默认工具 Mock 响应"""
        self._tool_default = response

    def handle_llm(
        self,
        prompt: str,
        input_text: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """返回 Mock LLM 响应（确定性，无网络调用）"""
        self._call_log.append(
            {
                "type": "llm",
                "prompt": prompt[:200],
                "input": input_text[:200],
                "model": model,
            }
        )

        # 尝试匹配 pattern
        combined = f"{prompt} {input_text}".lower()
        for pattern, response in self._llm_mocks.items():
            if pattern in combined:
                return response

        return self._llm_default

    def handle_tool(self, tool_name: str, tool_fn: Callable, input_val: Any) -> Any:
        """返回 Mock 工具结果（确定性，无真实执行）"""
        self._call_log.append(
            {
                "type": "tool",
                "tool_name": tool_name,
                "input": str(input_val)[:200],
            }
        )

        if tool_name in self._tool_mocks:
            return self._tool_mocks[tool_name]
        return self._tool_default

    def handle_state_read(
        self, store: Dict[str, Any], key: str, default: Any = None
    ) -> Any:
        """从内存字典读取"""
        return self._state.get(key, default)

    def handle_state_write(self, store: Dict[str, Any], key: str, value: Any) -> None:
        """写入内存字典"""
        self._state[key] = value

    @property
    def call_log(self) -> List[Dict[str, Any]]:
        """所有 Mock 调用记录"""
        return self._call_log

    @property
    def llm_calls(self) -> List[Dict[str, Any]]:
        """仅 LLM 调用记录"""
        return [c for c in self._call_log if c["type"] == "llm"]

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        """仅工具调用记录"""
        return [c for c in self._call_log if c["type"] == "tool"]

    def reset(self):
        """重置所有 Mock 和记录"""
        self._llm_mocks.clear()
        self._tool_mocks.clear()
        self._state.clear()
        self._call_log.clear()
        self._llm_default = "Mock LLM response"
        self._tool_default = "Mock tool result"


# ============================================================
# TraceHandler: 调试/审计环境
# ============================================================


class TraceHandler(PassthroughHandler):
    """
    调试效果处理器 — 真实调用 + 完整日志。

    | 效果 | 行为 |
    |------|------|
    | llm(m) | 真实调用 + 记录输入/输出/延迟/token |
    | io | 真实执行 + 记录 I/O |
    | state(s) | 真实存储 + 审计轨迹 |
    | cost | 累积并记录 |

    适用于调试和审计 — 生产功能不变，但增加完整可观测性。
    """

    def __init__(self):
        self._trace: List[Dict[str, Any]] = []
        self._total_tokens = 0
        self._total_cost_usd = 0.0
        self._total_latency_ms = 0.0

    def handle_llm(
        self,
        prompt: str,
        input_text: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """真实调用 + 完整记录"""
        # TraceHandler 也是 passthrough — 让 Lam 自行调用
        raise NotImplementedError(
            "TraceHandler.handle_llm: Lam should call _call_llm directly "
            "and report via handle_cost."
        )

    def handle_tool(self, tool_name: str, tool_fn: Callable, input_val: Any) -> Any:
        """真实工具执行 + I/O 记录"""
        t0 = time.time()
        result = tool_fn(input_val)
        elapsed = (time.time() - t0) * 1000

        self._trace.append(
            {
                "type": "tool",
                "tool_name": tool_name,
                "input": str(input_val)[:500],
                "output": str(result)[:500],
                "latency_ms": elapsed,
            }
        )

        return result

    def handle_state_read(
        self, store: Dict[str, Any], key: str, default: Any = None
    ) -> Any:
        """真实读取 + 审计"""
        value = store.get(key, default)
        self._trace.append(
            {
                "type": "state_read",
                "key": key,
                "value": str(value)[:200],
            }
        )
        return value

    def handle_state_write(self, store: Dict[str, Any], key: str, value: Any) -> None:
        """真实写入 + 审计"""
        old_value = store.get(key)
        store[key] = value
        self._trace.append(
            {
                "type": "state_write",
                "key": key,
                "old_value": str(old_value)[:200] if old_value is not None else None,
                "new_value": str(value)[:200],
            }
        )

    def handle_cost(self, tokens: int, latency_ms: float, model: str) -> None:
        """累积成本"""
        self._total_tokens += tokens
        self._total_latency_ms += latency_ms
        self._trace.append(
            {
                "type": "cost",
                "tokens": tokens,
                "latency_ms": latency_ms,
                "model": model,
            }
        )

    @property
    def trace(self) -> List[Dict[str, Any]]:
        return self._trace

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_latency_ms(self) -> float:
        return self._total_latency_ms

    def summary(self) -> Dict[str, Any]:
        """生成调试摘要"""
        return {
            "total_events": len(self._trace),
            "llm_calls": sum(1 for t in self._trace if t["type"] == "cost"),
            "tool_calls": sum(1 for t in self._trace if t["type"] == "tool"),
            "state_reads": sum(1 for t in self._trace if t["type"] == "state_read"),
            "state_writes": sum(1 for t in self._trace if t["type"] == "state_write"),
            "total_tokens": self._total_tokens,
            "total_latency_ms": self._total_latency_ms,
        }


# ============================================================
# HandlerContext: 当前活跃的 Handler（线程本地）
# ============================================================

import threading

_handler_local = threading.local()


def get_current_handler() -> Optional[EffectHandler]:
    """获取当前线程的效果处理器"""
    return getattr(_handler_local, "handler", None)


def set_current_handler(handler: Optional[EffectHandler]) -> None:
    """设置当前线程的效果处理器"""
    _handler_local.handler = handler


class with_handler:
    """
    效果处理器上下文管理器。

    使用方式:
        with with_handler(TestHandler()) as h:
            agent("input")  # 使用 TestHandler
        # 退出后恢复之前的 handler

    或者用于测试:
        handler = TestHandler()
        handler.mock_llm("summarize", "Mock summary")
        with with_handler(handler):
            result = my_agent("Summarize this")
            assert result == "Mock summary"
    """

    def __init__(self, handler: EffectHandler):
        self.handler = handler
        self._prev_handler: Optional[EffectHandler] = None

    def __enter__(self) -> EffectHandler:
        self._prev_handler = get_current_handler()
        set_current_handler(self.handler)
        return self.handler

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_current_handler(self._prev_handler)
        return False
