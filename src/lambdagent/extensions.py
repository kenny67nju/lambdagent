"""
lambdagent.extensions — Agent 专用扩展

Par     并行执行     同时运行多个 Agent
Route   多路分发     分类器 Agent 选路（广义 Church 布尔）
Memory  有状态 Agent 持久记忆（环境扩展）
Guard   输出验证     依赖类型 / 类型约束
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Optional

from .core import Term, Context, RouteError, ValidationError


# ============================================================
# Par: 并行执行
# ============================================================


class Par(Term):
    """
    并行求值: PAR(f, g) = λx. (f(x), g(x))

    对同一输入并行运行多个 Agent，返回元组。
    语法糖: f | g | h
    """

    def __init__(self, *agents: Term):
        name = " | ".join(a._name for a in agents)
        super().__init__(name)
        self.agents = list(agents)

    def apply(self, input: Any, ctx: Context | None = None) -> tuple:
        ctx = ctx or Context()
        if len(self.agents) <= 1:
            return tuple(a.apply(input, ctx) for a in self.agents)
        # True parallel via thread pool — each branch gets forked context (Paper II Prop. 30)
        results = [None] * len(self.agents)
        forked_ctxs = [ctx.fork() for _ in self.agents]
        with ThreadPoolExecutor(max_workers=len(self.agents)) as pool:
            futures = {
                pool.submit(a.apply, input, forked_ctxs[i]): i
                for i, a in enumerate(self.agents)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        # Merge traces back
        for fork_ctx in forked_ctxs:
            ctx.merge_trace(fork_ctx)
        return tuple(results)

    def __or__(self, other: Term) -> Par:
        """展平: (f | g) | h = Par(f, g, h)"""
        if isinstance(other, Par):
            return Par(*self.agents, *other.agents)
        return Par(*self.agents, other)


# ============================================================
# Route: 多路分发（广义 Church 布尔）
# ============================================================


class Route(Term):
    """
    路由/分发: 分类器选择下游 Agent。

    Lambda: 广义 Church 布尔 —— 不是 TRUE/FALSE 选两个，
    而是 classifier 在 N 个分支中选一个。

        Route(classifier, {"code": coder, "math": mathbot})
        ≡ λx. CASE (classifier x) [(l₁, a₁), (l₂, a₂), ...]
    """

    def __init__(
        self,
        classifier: Term,
        routes: Dict[str, Term],
        default: Term | None = None,
    ):
        super().__init__(f"Route({classifier._name})")
        self.classifier = classifier
        self.routes = routes
        self.default = default

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        ctx = ctx or Context()
        label = str(self.classifier.apply(input, ctx)).strip().lower()
        # 尝试精确匹配，再尝试包含匹配
        agent = self.routes.get(label)
        if agent is None:
            for key, val in self.routes.items():
                if key.lower() in label:
                    agent = val
                    break
        if agent is None:
            agent = self.default
        if agent is None:
            raise RouteError(
                f"No route for '{label}'. Available: {list(self.routes.keys())}"
            )
        return agent.apply(input, ctx)


# ============================================================
# Memory: 有状态 Agent（环境扩展）
# ============================================================


class Memory(Term):
    """
    有状态 Agent: 扩展环境 Γ 加入持久记忆。

    Lambda: Memory(agent) = λx. agent(x) [Γ ∪ store]

    记忆通过 prompt 注入 Agent。跨调用持久。
    """

    def __init__(self, agent: Term, store: Dict[str, Any] | None = None):
        super().__init__(f"Memory({agent._name})")
        self.agent = agent
        self.store = store or {}

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        ctx = ctx or Context()
        # 将记忆注入输入
        if self.store:
            memory_str = "\n".join(f"- {k}: {v}" for k, v in self.store.items())
            augmented = f"[Memory]\n{memory_str}\n\n[Input]\n{input}"
        else:
            augmented = str(input)
        return self.agent.apply(augmented, ctx)

    def remember(self, key: str, value: Any):
        """写入记忆"""
        self.store[key] = value

    def forget(self, key: str):
        """删除记忆"""
        self.store.pop(key, None)


# ============================================================
# Guard: 输出验证（依赖类型）
# ============================================================


class Guard(Term):
    """
    输出验证: 约束 Agent 输出满足谓词。

    Lambda: Guard(agent, P) = λx. let r = agent(x) in
                                    if P(r) then r else retry...

    对应依赖类型 —— 输出必须满足 validator 约束。
    失败时重试或执行 fallback。
    """

    def __init__(
        self,
        agent: Term,
        validator: Callable[[Any], bool] | Term,
        retry: int = 0,
        on_fail: Callable[[Any], Any] | None = None,
    ):
        super().__init__(f"Guard({agent._name})")
        self.agent = agent
        self.validator = validator
        self.retry = retry
        self.on_fail = on_fail

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        ctx = ctx or Context()
        last_result = None
        for attempt in range(1 + self.retry):
            result = self.agent.apply(input, ctx)
            last_result = result
            if isinstance(self.validator, Term):
                valid = self.validator.apply(result, ctx)
                # FIX-02: 内联 truthy 检查，避免循环导入 If 类
                if isinstance(valid, str):
                    valid = valid.strip().upper() in ("TRUE", "YES", "1")
                else:
                    valid = bool(valid)
            else:
                valid = self.validator(result)
            if valid:
                return result
        if self.on_fail:
            return self.on_fail(last_result)
        raise ValidationError(
            f"Guard({self.agent._name}) failed after {1 + self.retry} attempts. "
            f"Last output: {last_result}"
        )


# 避免循环导入
from .primitives import If
