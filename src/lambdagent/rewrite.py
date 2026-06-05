"""
lambdagent.rewrite — Paper II 代数定律作为 AST 重写规则

实现论文 II 定理 36–41 的 6 条代数定律:

  1. 组合结合律 (Thm 36):  (f >> g) >> h ≡ f >> (g >> h)
     → Compose 已经自动展平（不需要额外重写）

  2. 左单位元 (Thm 37):    Id >> f ≡ f
     → 消除 Identity agent

  3. 右单位元 (Thm 38):    f >> Id ≡ f
     → 消除尾部 Identity agent

  4. 循环展开 (Thm 39):    Loop(b, c, n) ≡ If(c, Id, b >> Loop(b, c, n-1))
     → Loop 优化（当 n=1 时简化为 If）

  5. 路由分配 (Thm 40):    Route(c, {li: fi}) >> g ≡ Route(c, {li: fi >> g})
     → 将后处理推入 Route 分支

  6. 对对称性 (Thm 41):    Pair(f, g) ≡ swap ∘ Pair(g, f)
     → 信息性（不改变语义，用于验证）

反模式 (Prop 42):
  Guard(a, P, k) >> g ≢ Guard(a >> g, P', k)
  → Guard 不满足对组合的分配律 — 发出警告

使用方式:
    from lambdagent.rewrite import optimize, RewriteLog
    optimized_agent, log = optimize(agent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from .core import Term


# ============================================================
# 重写日志
# ============================================================

@dataclass
class RewriteEntry:
    """一条重写的记录"""
    law: str            # 定律名 (e.g. "Thm37: Left Unit")
    description: str    # 描述
    before: str         # 重写前的 repr
    after: str          # 重写后的 repr


@dataclass
class RewriteLog:
    """重写过程的完整日志"""
    entries: List[RewriteEntry] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"RewriteLog({self.count} rewrites, {len(self.warnings)} warnings)"


# ============================================================
# Identity Term (用于检测单位元)
# ============================================================

class _IdentityTerm(Term):
    """Identity agent: Id(x) = x"""
    def __init__(self):
        super().__init__("Id")

    def apply(self, input: Any, ctx=None) -> Any:
        return input


def _is_identity(term: Term) -> bool:
    """检查是否为 Identity agent"""
    if isinstance(term, _IdentityTerm):
        return True
    # 检查 Tool(lambda x: x) 模式
    from .primitives import Tool
    if isinstance(term, Tool):
        if term._name in ("Id", "id", "identity", "passthrough", "noop"):
            return True
    return False


# ============================================================
# 重写规则
# ============================================================

def _rewrite_left_unit(term: Term, log: RewriteLog) -> Term:
    """
    定理 37: 左单位元 — Id >> f ≡ f

    消除组合链中的前导 Identity。
    """
    from .primitives import Compose
    if not isinstance(term, Compose):
        return term
    new_stages = [s for s in term.stages if not _is_identity(s)]
    if len(new_stages) < len(term.stages):
        removed = len(term.stages) - len(new_stages)
        before = repr(term)
        if not new_stages:
            result = _IdentityTerm()
        elif len(new_stages) == 1:
            result = new_stages[0]
        else:
            result = Compose(*new_stages)
        log.entries.append(RewriteEntry(
            law="Thm37/38: Unit Laws",
            description=f"Eliminated {removed} identity agent(s) from pipeline",
            before=before,
            after=repr(result),
        ))
        return result
    return term


def _rewrite_route_distribution(term: Term, log: RewriteLog) -> Term:
    """
    定理 40: 路由分配 — Route(c, {li: fi}) >> g ≡ Route(c, {li: fi >> g})

    当 Compose 的最后一个 stage 前面是 Route 时，
    将后续 stage 推入每个路由分支。
    """
    from .primitives import Compose
    from .extensions import Route

    if not isinstance(term, Compose) or len(term.stages) < 2:
        return term

    # 查找 Route 后跟其他 stages 的模式
    for i, stage in enumerate(term.stages):
        if isinstance(stage, Route) and i < len(term.stages) - 1:
            route = stage
            remaining = term.stages[i + 1:]
            suffix = remaining[0] if len(remaining) == 1 else Compose(*remaining)

            # 将 suffix 推入每个路由分支
            new_routes = {}
            for label, agent in route.routes.items():
                new_routes[label] = Compose(agent, suffix) if not isinstance(agent, Compose) else Compose(*agent.stages, suffix)

            new_default = None
            if route.default:
                new_default = Compose(route.default, suffix)

            new_route = Route(route.classifier, new_routes, new_default)

            # 重建前缀
            prefix = term.stages[:i]
            before = repr(term)
            if prefix:
                result = Compose(*prefix, new_route)
            else:
                result = new_route
            log.entries.append(RewriteEntry(
                law="Thm40: Route Distribution",
                description=f"Pushed post-processing into {len(new_routes)} route branches",
                before=before,
                after=repr(result),
            ))
            return result

    return term


def _rewrite_loop_simplify(term: Term, log: RewriteLog) -> Term:
    """
    定理 39: 循环展开 — Loop(b, c, 1) ≡ If(c, Id, b)

    当 max_steps=1 时，Loop 退化为 If。
    """
    from .primitives import Loop, If

    if isinstance(term, Loop) and term.max_steps == 1:
        before = repr(term)
        result = If(
            cond=lambda x: term.condition(x, 0),
            then_=_IdentityTerm(),
            else_=term.body,
        )
        log.entries.append(RewriteEntry(
            law="Thm39: Loop Unfolding",
            description="Loop(body, cond, 1) simplified to If(cond, Id, body)",
            before=before,
            after=repr(result),
        ))
        return result
    return term


def _check_guard_distribution(term: Term, log: RewriteLog) -> None:
    """
    命题 42 (反模式): Guard(a, P, k) >> g ≢ Guard(a >> g, P', k)

    检查是否有开发者错误地将 Guard 和 Compose 互换。
    这不是重写，而是一个警告。
    """
    from .primitives import Compose
    from .extensions import Guard

    if isinstance(term, Compose):
        for i, stage in enumerate(term.stages):
            if isinstance(stage, Guard) and i < len(term.stages) - 1:
                # Guard 后面有其他 stage — 这是合法的
                # 但如果 Guard 内部的 agent 又是一个 Compose，
                # 可能意味着开发者把后处理错误地放在了 Guard 内部
                if isinstance(stage.agent, Compose):
                    log.warnings.append(
                        f"⚠ Proposition 42 (Paper II): "
                        f"Guard at step {i} wraps a Compose ({stage.agent._name}). "
                        f"Guard does NOT distribute over composition. "
                        f"Verify: Guard(a >> g, P) ≠ Guard(a, P) >> g"
                    )


# ============================================================
# 递归重写
# ============================================================

def _rewrite_recursive(term: Term, log: RewriteLog) -> Term:
    """递归地对所有子项应用重写规则"""
    from .primitives import Compose, If, Loop, Pair
    from .extensions import Par, Route, Memory, Guard

    # 先对子项重写
    if isinstance(term, Compose):
        new_stages = [_rewrite_recursive(s, log) for s in term.stages]
        term = Compose(*new_stages) if len(new_stages) > 1 else new_stages[0]
    elif isinstance(term, Pair):
        first = _rewrite_recursive(term.first, log)
        second = _rewrite_recursive(term.second, log)
        term = Pair(first, second)
    elif isinstance(term, If):
        then_ = _rewrite_recursive(term.then_, log)
        else_ = _rewrite_recursive(term.else_, log)
        term = If(term.cond, then_, else_)
    elif isinstance(term, Loop):
        body = _rewrite_recursive(term.body, log)
        term = Loop(body, term.condition, term.max_steps)
    elif isinstance(term, Guard):
        agent = _rewrite_recursive(term.agent, log)
        term = Guard(agent, term.validator, term.retry, term.on_fail)
    elif isinstance(term, Memory):
        agent = _rewrite_recursive(term.agent, log)
        term = Memory(agent, term.store)
    elif isinstance(term, Route):
        new_routes = {k: _rewrite_recursive(v, log) for k, v in term.routes.items()}
        term = Route(term.classifier, new_routes, term.default)

    # 应用本层重写
    term = _rewrite_left_unit(term, log)
    term = _rewrite_loop_simplify(term, log)
    term = _rewrite_route_distribution(term, log)

    # 检查反模式
    _check_guard_distribution(term, log)

    return term


# ============================================================
# Public API
# ============================================================

def optimize(term: Term) -> Tuple[Term, RewriteLog]:
    """
    应用所有代数定律重写规则优化 Agent AST。

    Paper II Theorems 36-41:
        - Identity 消除（左/右单位元）
        - 路由分配优化
        - Loop(1) 简化
        - Guard 分配反模式警告（Prop 42）

    Returns:
        (optimized_term, log): 优化后的 term 和重写日志
    """
    log = RewriteLog()
    result = _rewrite_recursive(term, log)
    return result, log
