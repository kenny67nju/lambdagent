"""
lambdagent.effects — Paper III 效果代数

实现论文 III 的效果系统 (§4):
  - Effect: 效果类型 (pure, llm(m), io, state(s))
  - 效果组合: 串行 (·), 并行 (∥), 迭代 (εⁿ)
  - 效果子类型格: pure ≤ ε for all ε (Definition 9)
  - 效果标注: 每个 Term 有计算效果 (Definition 6-7)

核心方程:
    ε1 · ε2 = 串行效果组合 (Compose)
    ε1 ∥ ε2 = 并行效果组合 (Pair/Par)
    εⁿ      = 迭代效果 (Loop)

效果格偏序:
    pure ≤ llm(m) ≤ llm(m) · io ≤ ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, FrozenSet, List, Optional, Set, Tuple


# ============================================================
# 效果种类 (Paper III Definition 6)
# ============================================================


class EffectKind(Enum):
    """基本效果种类"""

    PURE = "pure"  # 纯函数 — 无副作用
    LLM = "llm"  # LLM 调用 — 自回归解码
    IO = "io"  # I/O — 工具调用、外部 API
    STATE = "state"  # 状态 — 读写 Memory/SharedMemory


# ============================================================
# Effect: 效果类型 (Paper III Definition 7)
# ============================================================


@dataclass(frozen=True)
class Effect:
    """
    效果类型。

    Paper III Definition 7:
        ε ::= pure | llm(m) | io | state(s) | ε1 · ε2 | ε1 ∥ ε2 | εⁿ

    每个 Effect 是基本效果的组合。
    """

    kind: EffectKind
    # llm(m): 模型名称 (when kind == LLM)
    model: Optional[str] = None
    # state(s): 状态键集合 (when kind == STATE)
    state_keys: Optional[FrozenSet[str]] = None

    def __repr__(self) -> str:
        if self.kind == EffectKind.PURE:
            return "pure"
        elif self.kind == EffectKind.LLM:
            if self.model:
                return f"llm({self.model})"
            return "llm"
        elif self.kind == EffectKind.IO:
            return "io"
        elif self.kind == EffectKind.STATE:
            if self.state_keys:
                keys = ", ".join(sorted(self.state_keys))
                return f"state({keys})"
            return "state"
        return f"Effect({self.kind})"


# ============================================================
# 效果常量
# ============================================================

PURE = Effect(EffectKind.PURE)
IO = Effect(EffectKind.IO)


def LLM(model: str | None = None) -> Effect:
    """构造 llm(m) 效果"""
    return Effect(EffectKind.LLM, model=model)


def STATE(*keys: str) -> Effect:
    """构造 state(s) 效果"""
    return Effect(EffectKind.STATE, state_keys=frozenset(keys) if keys else None)


# ============================================================
# 组合效果 (Paper III Definition 7 continued)
# ============================================================


@dataclass(frozen=True)
class ComposedEffect:
    """
    组合效果 — 多个基本效果的组合。

    串行 (·): Compose(f, g) 的效果 = ε_f · ε_g
    并行 (∥): Pair(f, g) 的效果 = ε_f ∥ ε_g
    迭代 (εⁿ): Loop(body, n) 的效果 = ε_body^n
    """

    effects: Tuple[Effect, ...]
    mode: str = "serial"  # "serial" (·), "parallel" (∥), "iterate" (εⁿ)
    iterations: Optional[int] = None  # only for iterate

    @property
    def is_pure(self) -> bool:
        """是否所有效果都是 pure"""
        return all(e.kind == EffectKind.PURE for e in self.effects)

    @property
    def has_llm(self) -> bool:
        """是否包含 LLM 调用"""
        return any(e.kind == EffectKind.LLM for e in self.effects)

    @property
    def has_io(self) -> bool:
        """是否包含 I/O"""
        return any(e.kind == EffectKind.IO for e in self.effects)

    @property
    def has_state(self) -> bool:
        """是否包含状态操作"""
        return any(e.kind == EffectKind.STATE for e in self.effects)

    @property
    def all_state_keys(self) -> FrozenSet[str]:
        """收集所有状态键（用于 store-independence 检查）"""
        keys: Set[str] = set()
        for e in self.effects:
            if e.kind == EffectKind.STATE and e.state_keys:
                keys.update(e.state_keys)
        return frozenset(keys)

    @property
    def models_used(self) -> FrozenSet[str]:
        """收集所有使用的 LLM 模型"""
        models: Set[str] = set()
        for e in self.effects:
            if e.kind == EffectKind.LLM and e.model:
                models.add(e.model)
        return frozenset(models)

    def __repr__(self) -> str:
        if not self.effects:
            return "pure"
        if len(self.effects) == 1:
            base = repr(self.effects[0])
            if self.mode == "iterate" and self.iterations:
                return f"{base}^{self.iterations}"
            return base
        sep = " · " if self.mode == "serial" else " ∥ "
        parts = sep.join(repr(e) for e in self.effects)
        if self.mode == "iterate" and self.iterations:
            return f"({parts})^{self.iterations}"
        return parts


# ============================================================
# 效果组合运算 (Paper III Definitions 6-7)
# ============================================================


def serial(*effects: Effect | ComposedEffect) -> ComposedEffect:
    """
    串行效果组合: ε1 · ε2

    对应 Compose(f, g) — 先执行 f 的效果，再执行 g 的效果。
    """
    flat: List[Effect] = []
    for e in effects:
        if isinstance(e, ComposedEffect):
            flat.extend(e.effects)
        else:
            flat.append(e)
    # 过滤 pure
    non_pure = [e for e in flat if e.kind != EffectKind.PURE]
    if not non_pure:
        return ComposedEffect(effects=(PURE,), mode="serial")
    return ComposedEffect(effects=tuple(non_pure), mode="serial")


def parallel(*effects: Effect | ComposedEffect) -> ComposedEffect:
    """
    并行效果组合: ε1 ∥ ε2

    对应 Pair(f, g) — f 和 g 的效果同时发生。
    """
    flat: List[Effect] = []
    for e in effects:
        if isinstance(e, ComposedEffect):
            flat.extend(e.effects)
        else:
            flat.append(e)
    non_pure = [e for e in flat if e.kind != EffectKind.PURE]
    if not non_pure:
        return ComposedEffect(effects=(PURE,), mode="parallel")
    return ComposedEffect(effects=tuple(non_pure), mode="parallel")


def iterate(effect: Effect | ComposedEffect, n: int) -> ComposedEffect:
    """
    迭代效果: εⁿ

    对应 Loop(body, n) — body 的效果重复 n 次。
    """
    if isinstance(effect, ComposedEffect):
        return ComposedEffect(effects=effect.effects, mode="iterate", iterations=n)
    return ComposedEffect(effects=(effect,), mode="iterate", iterations=n)


# ============================================================
# 效果子类型格 (Paper III Definition 9)
# ============================================================

# 偏序: pure ≤ ε for all ε
_EFFECT_ORDER = {
    EffectKind.PURE: 0,
    EffectKind.STATE: 1,
    EffectKind.IO: 2,
    EffectKind.LLM: 3,
}


def effect_leq(e1: Effect, e2: Effect) -> bool:
    """
    效果子类型: e1 ≤ e2

    Paper III Definition 9:
        pure ≤ ε for all ε (Proposition 10: monotonicity)

    偏序: pure ≤ state ≤ io ≤ llm
    """
    if e1.kind == EffectKind.PURE:
        return True
    if e1.kind == e2.kind:
        return True
    return _EFFECT_ORDER.get(e1.kind, 0) <= _EFFECT_ORDER.get(e2.kind, 0)


def max_effect(*effects: Effect) -> Effect:
    """取效果格中的最大元素 (join / supremum)"""
    if not effects:
        return PURE
    result = effects[0]
    for e in effects[1:]:
        if _EFFECT_ORDER.get(e.kind, 0) > _EFFECT_ORDER.get(result.kind, 0):
            result = e
    return result


# ============================================================
# 效果推断辅助
# ============================================================


def parse_effect_annotation(annotation: Any) -> Effect | ComposedEffect:
    """
    从 YAML 配置中的效果标注解析为 Effect。

    支持的格式:
        "pure"              → PURE
        "llm(qwen3-max)"   → LLM("qwen3-max")
        "io"                → IO
        "state(memory)"     → STATE("memory")
        "llm(claude) · io"  → serial(LLM("claude"), IO)
    """
    if annotation is None:
        return ComposedEffect(effects=(PURE,))

    if isinstance(annotation, str):
        annotation = annotation.strip()

        # 组合效果: "ε1 · ε2"
        if " · " in annotation:
            parts = [
                parse_effect_annotation(p.strip()) for p in annotation.split(" · ")
            ]
            return serial(*parts)

        # 组合效果: "ε1 ∥ ε2"
        if " ∥ " in annotation or " || " in annotation:
            sep = " ∥ " if " ∥ " in annotation else " || "
            parts = [parse_effect_annotation(p.strip()) for p in annotation.split(sep)]
            return parallel(*parts)

        if annotation == "pure":
            return PURE
        if annotation == "io":
            return IO
        if annotation.startswith("llm(") and annotation.endswith(")"):
            model = annotation[4:-1]
            return LLM(model)
        if annotation.startswith("llm"):
            return LLM()
        if annotation.startswith("state(") and annotation.endswith(")"):
            keys = annotation[6:-1].split(",")
            return STATE(*[k.strip() for k in keys])
        if annotation == "state":
            return STATE()

    return ComposedEffect(effects=(PURE,))


# ============================================================
# Term 效果推断 (Paper III §4)
# ============================================================


def infer_effect_for_term(term: Any) -> Effect | ComposedEffect:
    """
    为 Term 推断效果。

    Paper III §4 效果推断规则:
        Lam     → llm(model)
        Tool    → io
        Memory  → state · inner_effect
        Guard   → inner_effect^(1+retry)
        Compose → ε1 · ε2 · ... · εn
        Pair    → ε1 ∥ ε2
        Par     → ε1 ∥ ε2 ∥ ... ∥ εn
        If      → ε_cond · (ε_then | ε_else)
        Loop    → ε_body^n
        Route   → ε_classifier · max(ε_routes)
    """
    # 延迟导入避免循环
    from .primitives import Lam, Compose, If, Loop, Pair, Tool
    from .extensions import Par, Route, Memory, Guard

    if isinstance(term, Lam):
        return LLM(term.model)

    elif isinstance(term, Tool):
        return IO

    elif isinstance(term, Compose):
        stage_effects = [infer_effect_for_term(s) for s in term.stages]
        return serial(*stage_effects)

    elif isinstance(term, Pair):
        return parallel(
            infer_effect_for_term(term.first),
            infer_effect_for_term(term.second),
        )

    elif isinstance(term, Par):
        agent_effects = [infer_effect_for_term(a) for a in term.agents]
        return parallel(*agent_effects)

    elif isinstance(term, If):
        cond_eff = (
            infer_effect_for_term(term.cond)
            if isinstance(term.cond, type) and hasattr(term.cond, "apply")
            else PURE
        )
        then_eff = infer_effect_for_term(term.then_)
        else_eff = infer_effect_for_term(term.else_)
        # If 的效果 = ε_cond · max(ε_then, ε_else)
        branch_max = max_effect(
            then_eff
            if isinstance(then_eff, Effect)
            else then_eff.effects[0]
            if then_eff.effects
            else PURE,
            else_eff
            if isinstance(else_eff, Effect)
            else else_eff.effects[0]
            if else_eff.effects
            else PURE,
        )
        return serial(
            cond_eff if isinstance(cond_eff, Effect) else cond_eff, branch_max
        )

    elif isinstance(term, Loop):
        body_eff = infer_effect_for_term(term.body)
        return iterate(body_eff, term.max_steps)

    elif isinstance(term, Memory):
        inner_eff = infer_effect_for_term(term.agent)
        return serial(STATE(), inner_eff)

    elif isinstance(term, Guard):
        inner_eff = infer_effect_for_term(term.agent)
        return iterate(inner_eff, 1 + term.retry)

    elif isinstance(term, Route):
        cls_eff = infer_effect_for_term(term.classifier)
        route_effects = [infer_effect_for_term(r) for r in term.routes.values()]
        if route_effects:
            route_max = route_effects[0]
            for re in route_effects[1:]:
                if isinstance(re, Effect):
                    route_max = re
                elif isinstance(route_max, ComposedEffect) and isinstance(
                    re, ComposedEffect
                ):
                    if len(re.effects) > len(route_max.effects):
                        route_max = re
            return serial(cls_eff, route_max)
        return cls_eff

    # DESIGN-07: Multi-agent constructs effect inference
    try:
        from .multiagent import GroupChat, AsyncPar, Handoff, Send, Receive
    except ImportError:
        GroupChat = AsyncPar = Handoff = Send = Receive = None

    if GroupChat and isinstance(term, GroupChat):
        # GroupChat = Y_n(agents) -> iterated parallel effects
        agents = getattr(term, "agent_list", None) or list(term.agents.values())
        agent_effects = [infer_effect_for_term(a) for a in agents]
        combined = agent_effects[0] if agent_effects else PURE
        for e in agent_effects[1:]:
            combined = parallel(combined, e)
        max_rounds = getattr(term, "max_rounds", 10)
        return iterate(combined, max_rounds)

    if AsyncPar and isinstance(term, AsyncPar):
        # AsyncPar = Pair(f1, ..., fn) -> parallel effects
        agent_effects = [infer_effect_for_term(a) for a in term.agents]
        combined = agent_effects[0] if agent_effects else PURE
        for e in agent_effects[1:]:
            combined = parallel(combined, e)
        return combined

    if Handoff and isinstance(term, Handoff):
        # Handoff = Route with dynamic registry -> union of all agent effects
        from .core import Term as _Term

        registry_effects = [
            infer_effect_for_term(a)
            for a in term.registry.values()
            if isinstance(a, _Term)
        ]
        combined = PURE
        for e in registry_effects:
            combined = serial(combined, e)  # worst case: any agent could be called
        return combined

    if Send and isinstance(term, Send):
        inner = infer_effect_for_term(term.agent) if hasattr(term, "agent") else PURE
        return serial(inner, Effect(EffectKind.STATE))  # writes to channel

    if Receive and isinstance(term, Receive):
        handler_effect = (
            infer_effect_for_term(term.handler)
            if hasattr(term, "handler") and term.handler
            else PURE
        )
        return serial(Effect(EffectKind.STATE), handler_effect)  # reads from channel

    return PURE
