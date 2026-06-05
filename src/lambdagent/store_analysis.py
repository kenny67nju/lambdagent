"""
lambdagent.store_analysis — Paper II Proposition 30 存储独立性分析

实现论文 II 命题 30 的前提条件检查:
    Pair confluence: writes(f) ∩ writes(g) = ∅ → schedule-independent

核心功能:
  - writes(term): 静态分析 Term 可能写入的存储键集合
  - reads(term): 静态分析 Term 可能读取的存储键集合
  - check_store_independence(terms): 检查多个 Term 的写入集合不相交
  - StoreConflictError: 存储冲突错误
"""

from __future__ import annotations

from typing import Any, FrozenSet, List, Set

from .core import Term, LambdagentError


# ============================================================
# 异常
# ============================================================


class StoreConflictError(LambdagentError):
    """
    存储冲突错误 — Paper II Proposition 30 违反。

    当 writes(f) ∩ writes(g) ≠ ∅ 时，Pair(f, g) 的结果
    依赖于调度策略，不满足合流性。
    """

    def __init__(self, agent_a: str, agent_b: str, conflicting_keys: FrozenSet[str]):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.conflicting_keys = conflicting_keys
        keys_str = ", ".join(sorted(conflicting_keys))
        super().__init__(
            f"Pair confluence violation (Paper II Prop. 30): "
            f"'{agent_a}' and '{agent_b}' both write to: {{{keys_str}}}. "
            f"Parallel execution is schedule-dependent. "
            f"Fix: ensure writes({agent_a}) ∩ writes({agent_b}) = ∅, "
            f"or use sequential Compose instead of Pair/Par."
        )


# ============================================================
# 写入集合分析: writes(term)
# ============================================================


def writes(term: Term) -> FrozenSet[str]:
    """
    静态分析 Term 可能写入的存储键集合。

    Paper II Proposition 30:
        writes(f) ∩ writes(g) = ∅ is required for schedule-independence

    规则:
        writes(Lam)      = ∅ (LLM 调用不写入存储)
        writes(Tool)     = ∅ (工具调用不直接写入存储，除非显式标注)
        writes(Compose)  = ∪ writes(stage_i)
        writes(Pair)     = writes(first) ∪ writes(second)
        writes(Par)      = ∪ writes(agent_i)
        writes(If)       = writes(then_) ∪ writes(else_)
        writes(Loop)     = writes(body)
        writes(Memory)   = {all keys in Memory.store} ∪ writes(inner)
        writes(Guard)    = writes(inner)
        writes(Route)    = ∪ writes(route_i)
        writes(SharedMemoryAgent) = {all possible keys}
    """
    from .primitives import Lam, Compose, If, Loop, Pair, Fst, Snd, Tool
    from .extensions import Par, Route, Memory, Guard

    result: Set[str] = set()

    if isinstance(term, (Lam, Tool, Fst, Snd)):
        # 纯函数或外部调用 — 不写入 Lambda 存储（除非有显式标注）
        if hasattr(term, "_writes"):
            return frozenset(term._writes)
        return frozenset()

    elif isinstance(term, Compose):
        for stage in term.stages:
            result.update(writes(stage))

    elif isinstance(term, Pair):
        result.update(writes(term.first))
        result.update(writes(term.second))

    elif isinstance(term, Par):
        for agent in term.agents:
            result.update(writes(agent))

    elif isinstance(term, If):
        result.update(writes(term.then_))
        result.update(writes(term.else_))
        if isinstance(term.cond, Term):
            result.update(writes(term.cond))

    elif isinstance(term, Loop):
        result.update(writes(term.body))

    elif isinstance(term, Memory):
        # Memory 可以写入其 store 的所有键
        if term.store:
            result.update(term.store.keys())
        result.update(writes(term.agent))

    elif isinstance(term, Guard):
        result.update(writes(term.agent))

    elif isinstance(term, Route):
        result.update(writes(term.classifier))
        for route_agent in term.routes.values():
            result.update(writes(route_agent))

    else:
        # 多智能体扩展: 尝试分析
        try:
            from .multiagent import AsyncPar, _SharedMemoryAgent, GroupChat, Handoff

            if isinstance(term, AsyncPar):
                for agent in term.agents:
                    result.update(writes(agent))
            elif isinstance(term, _SharedMemoryAgent):
                # SharedMemory agent 写入所有可能的键
                all_keys = set(term.shared._store.keys())
                result.update(all_keys)
                result.add("__shared_memory__")  # sentinel
                result.update(writes(term.agent))
            elif isinstance(term, GroupChat):
                for agent in term.agent_list:
                    result.update(writes(agent))
            elif isinstance(term, Handoff):
                for agent in term.registry.values():
                    result.update(writes(agent))
        except ImportError:
            pass

    # 检查是否有显式 _writes 标注
    if hasattr(term, "_writes"):
        result.update(term._writes)

    return frozenset(result)


def reads(term: Term) -> FrozenSet[str]:
    """
    静态分析 Term 可能读取的存储键集合。

    类似 writes() 但分析读取操作。
    """
    from .primitives import Lam, Compose, If, Loop, Pair, Fst, Snd, Tool
    from .extensions import Par, Route, Memory, Guard

    result: Set[str] = set()

    if isinstance(term, (Lam, Tool, Fst, Snd)):
        return frozenset()

    elif isinstance(term, Compose):
        for stage in term.stages:
            result.update(reads(stage))

    elif isinstance(term, Pair):
        result.update(reads(term.first))
        result.update(reads(term.second))

    elif isinstance(term, Par):
        for agent in term.agents:
            result.update(reads(agent))

    elif isinstance(term, Memory):
        if term.store:
            result.update(term.store.keys())
        result.update(reads(term.agent))

    elif isinstance(term, Guard):
        result.update(reads(term.agent))

    # 检查显式 _reads 标注
    if hasattr(term, "_reads"):
        result.update(term._reads)

    return frozenset(result)


# ============================================================
# 存储独立性检查 (Paper II Proposition 30)
# ============================================================


def check_store_independence(terms: List[Term]) -> None:
    """
    检查多个 Term 的写入集合两两不相交。

    Paper II Proposition 30:
        ∀ i < j: writes(terms[i]) ∩ writes(terms[j]) = ∅

    Raises:
        StoreConflictError: 存在写入冲突
    """
    write_sets = [(t, writes(t)) for t in terms]

    for i in range(len(write_sets)):
        for j in range(i + 1, len(write_sets)):
            term_a, writes_a = write_sets[i]
            term_b, writes_b = write_sets[j]
            conflict = writes_a & writes_b
            if conflict:
                raise StoreConflictError(
                    term_a._name,
                    term_b._name,
                    conflict,
                )
