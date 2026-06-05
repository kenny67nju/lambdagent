"""
lambdagent.cost_grade — Paper III §4.3 分级类型用于成本预测

实现论文 III 的分级类型系统 (Definitions 11-12):
  - CostGrade: 静态成本上界 (p, t, l, m)
      p = 成功概率
      t = token 数上界
      l = 延迟上界 (秒)
      m = 成本上界 (USD)
  - 分级组合规则:
      串行  g1 · g2 = (p1*p2, t1+t2, l1+l2, m1+m2)
      并行  g1 ∥ g2 = (p1*p2, t1+t2, max(l1,l2), m1+m2)
      迭代  g^n     = (p^n, n*t, n*l, n*m)
      Guard g(k)    = (1-(1-p)^k, k*t, k*l, k*m)

核心方程:
    estimate_cost(agent) → CostGrade

依赖:
    types.py (AgentType), effects.py (Effect)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .core import Term


# ============================================================
# CostGrade (Paper III Definition 11)
# ============================================================


@dataclass(frozen=True)
class CostGrade:
    """
    分级类型: 静态成本上界。

    Paper III Definition 11 (修正版):
        g = (p, p_fatal, t, l, m) where:
        p       ∈ [0, 1]  — 端到端成功概率 (修正: 只算致命失败)
        p_fatal ∈ [0, 1]  — 单步致命失败率 (API 错误/解析异常/工具异常)
        t       ∈ ℕ       — token 数上界
        l       ∈ ℝ⁺      — 延迟上界 (秒)
        m       ∈ ℝ⁺      — 成本上界 (USD)

    概率模型修正:
        旧模型: p_total = p_step^n  (每步成功率连乘 → 步数越多越低 → 过度悲观)
        新模型: p_total = (1 - p_fatal)^n  (只对致命失败连乘)

        区分两种"失败":
          致命失败: API 500/解析错误/工具异常 → 管道无法继续 → 降低 p
          非致命失败: LLM 想错方向/搜索没找到 → Agent 自我纠正 → 不降低 p

        这更符合 ReAct 的实际行为:
          10 步循环不是"10 次独立实验都要成功"
          而是"10 步里至少走对一条路，且不出致命错误"
    """

    probability: float = 1.0  # p: 端到端成功概率
    tokens: int = 0  # t: token 数上界
    latency: float = 0.0  # l: 延迟上界 (秒)
    money: float = 0.0  # m: 成本上界 (USD)
    p_fatal: float = 0.0  # 单步致命失败率 (用于迭代计算)

    def __repr__(self) -> str:
        return (
            f"CostGrade(p={self.probability:.2%}, "
            f"t={self.tokens}, "
            f"l={self.latency:.1f}s, "
            f"m=${self.money:.4f})"
        )

    @property
    def is_free(self) -> bool:
        """是否零成本（纯计算）"""
        return self.tokens == 0 and self.money == 0.0


# ============================================================
# 分级组合规则 (Paper III Definition 12)
# ============================================================


def grade_serial(g1: CostGrade, g2: CostGrade) -> CostGrade:
    """
    串行组合: g1 · g2

    Paper III Definition 12 (修正版):
        p = p1 × p2       概率: 两个都不出致命错误
        t = t1 + t2       Token: 累加
        l = l1 + l2       延迟: 累加
        m = m1 + m2       成本: 累加
        p_fatal = 合并致命率
    """
    # 合并致命率: 串行中任一步致命失败都会导致整体失败
    # P(至少一步致命) = 1 - P(两步都不致命)
    p_fatal_combined = 1.0 - (1.0 - g1.p_fatal) * (1.0 - g2.p_fatal)
    return CostGrade(
        probability=g1.probability * g2.probability,
        tokens=g1.tokens + g2.tokens,
        latency=g1.latency + g2.latency,
        money=g1.money + g2.money,
        p_fatal=p_fatal_combined,
    )


def grade_parallel(g1: CostGrade, g2: CostGrade) -> CostGrade:
    """
    并行组合: g1 ∥ g2

    Paper III Definition 12:
        p = p1 × p2       概率: 两个都不出致命错误
        t = t1 + t2       Token: 累加 (并行不省 token)
        l = max(l1, l2)   延迟: 取最慢 (并行省时间!)
        m = m1 + m2       成本: 累加 (并行不省钱)
    """
    p_fatal_combined = 1.0 - (1.0 - g1.p_fatal) * (1.0 - g2.p_fatal)
    return CostGrade(
        probability=g1.probability * g2.probability,
        tokens=g1.tokens + g2.tokens,
        latency=max(g1.latency, g2.latency),
        money=g1.money + g2.money,
        p_fatal=p_fatal_combined,
    )


def grade_iterate(g: CostGrade, n: int) -> CostGrade:
    """
    迭代: g^n (修正版)

    旧模型 (过度悲观):
        p = p_step^n  → 每步成功率连乘 → 步数越多概率越低
        问题: 把"LLM 想错方向"也算成失败

    修正模型 (区分致命/非致命):
        p = (1 - p_fatal_per_step)^n  → 只算致命失败的连乘
        t = n × t                     → 成本上界不变
        l = n × l
        m = n × m

    致命失败: API 错误, 解析异常, 工具崩溃 → 管道无法继续
    非致命失败: 想错方向, 搜索没结果 → Agent 会自我纠正, 不算失败

    这更符合 ReAct 循环的实际行为:
        10 步不是"10 次独立实验都要成功"
        而是"10 步探索中不出致命错误"
    """
    # 致命失败率: 从 body 的 p_fatal 获取，如果没有则从 probability 推导
    p_fatal_per_step = g.p_fatal if g.p_fatal > 0 else (1.0 - g.probability) * 0.1

    # n 步都不出致命错误的概率
    p_no_fatal = (1.0 - p_fatal_per_step) ** n

    return CostGrade(
        probability=p_no_fatal,
        tokens=n * g.tokens,
        latency=n * g.latency,
        money=n * g.money,
        p_fatal=1.0 - p_no_fatal,  # 整体致命失败率
    )


def grade_guard(g: CostGrade, retries: int) -> CostGrade:
    """
    Guard 重试: (1+k) 次尝试

    Paper III (修正版):
        p = 1 - (1-p_useful)^k  (至少一次通过验证的概率)
        t = k × t               (成本按最坏 k 次算)
        l = k × l
        m = k × m

    Guard 的语义: 重试是为了通过验证器 P，不是为了修复致命错误。
    所以 Guard 的概率模型用的是"至少一次通过验证"，和致命失败无关。
    """
    k = 1 + retries
    # Guard 的概率: 至少一次通过验证
    # p_useful = g.probability (单次通过验证的概率)
    p_useful = g.probability
    p_at_least_once = 1.0 - (1.0 - p_useful) ** k

    return CostGrade(
        probability=p_at_least_once,
        tokens=k * g.tokens,
        latency=k * g.latency,
        money=k * g.money,
        p_fatal=g.p_fatal,  # 致命率不变 (Guard 不修复致命错误)
    )


# ============================================================
# 模型成本配置
# ============================================================

# 每个模型的默认成本参数
_MODEL_COSTS: Dict[str, Dict[str, float]] = {
    "claude-sonnet-4-20250514": {
        "tokens_per_call": 800,
        "latency": 2.0,
        "price_per_1k": 0.003,
    },
    "claude-opus-4-20250514": {
        "tokens_per_call": 1200,
        "latency": 5.0,
        "price_per_1k": 0.015,
    },
    "claude-haiku-4-5-20251001": {
        "tokens_per_call": 500,
        "latency": 0.5,
        "price_per_1k": 0.00025,
    },
    "gpt-4": {"tokens_per_call": 1000, "latency": 3.0, "price_per_1k": 0.03},
    "gpt-4o": {"tokens_per_call": 800, "latency": 1.5, "price_per_1k": 0.0025},
    "qwen3-max": {"tokens_per_call": 600, "latency": 1.0, "price_per_1k": 0.001},
}

_DEFAULT_MODEL_COST = {"tokens_per_call": 800, "latency": 2.0, "price_per_1k": 0.003}


def _get_model_cost(model: str) -> Dict[str, float]:
    """获取模型的成本参数"""
    for key, cost in _MODEL_COSTS.items():
        if key in model.lower():
            return cost
    return _DEFAULT_MODEL_COST


# ============================================================
# 成本估算: estimate_cost(term) → CostGrade
# ============================================================

# ============================================================
# 概率参数 (修正版)
# ============================================================

# 致命失败率: API 错误 / 解析异常 / 工具崩溃 — 导致管道无法继续
_DEFAULT_LLM_FATAL_RATE = 0.01  # LLM 每次调用 1% 致命失败 (API 500/超时)
_DEFAULT_TOOL_FATAL_RATE = 0.02  # Tool 每次调用 2% 致命失败 (文件不存在/异常)

# 单步成功概率: 用于 Guard 的"通过验证"概率
_DEFAULT_LLM_SUCCESS_PROB = 0.95  # LLM 单次生成有用结果的概率
_DEFAULT_TOOL_SUCCESS_PROB = 0.98  # Tool 单次返回正确结果的概率


def estimate_cost(
    term: Term, model_costs: Dict[str, Dict[str, float]] | None = None
) -> CostGrade:
    """
    静态估算 Agent 的最坏情况成本。

    Paper III §4.3 (修正版): 编译时为每个 agent 流水线计算:
      - 成本上界 (t, l, m): 最坏情况全部跑满
      - 成功概率 (p): 只对致命失败做串行连乘, 非致命探索不降低概率

    Args:
        term: Agent term
        model_costs: 自定义模型成本参数 (可选)

    Returns:
        CostGrade: 成本上界 + 成功概率
    """
    from .primitives import Lam, Compose, If, Loop, Pair, Fst, Snd, Tool
    from .extensions import Par, Route, Memory, Guard

    if model_costs:
        _MODEL_COSTS.update(model_costs)

    if isinstance(term, Lam):
        mc = _get_model_cost(term.model)
        tokens = mc["tokens_per_call"]
        return CostGrade(
            probability=1.0 - _DEFAULT_LLM_FATAL_RATE,  # 99% 不出致命错误
            tokens=tokens,
            latency=mc["latency"],
            money=tokens / 1000.0 * mc["price_per_1k"],
            p_fatal=_DEFAULT_LLM_FATAL_RATE,  # 1% 致命失败率
        )

    elif isinstance(term, Tool):
        return CostGrade(
            probability=1.0 - _DEFAULT_TOOL_FATAL_RATE,  # 98% 不出致命错误
            tokens=0,
            latency=0.1,
            money=0.0,
            p_fatal=_DEFAULT_TOOL_FATAL_RATE,  # 2% 致命失败率
        )

    elif isinstance(term, (Fst, Snd)):
        return CostGrade()  # 零成本

    elif isinstance(term, Compose):
        grades = [estimate_cost(s) for s in term.stages]
        result = grades[0]
        for g in grades[1:]:
            result = grade_serial(result, g)
        return result

    elif isinstance(term, Pair):
        g1 = estimate_cost(term.first)
        g2 = estimate_cost(term.second)
        return grade_parallel(g1, g2)

    elif isinstance(term, Par):
        grades = [estimate_cost(a) for a in term.agents]
        result = grades[0]
        for g in grades[1:]:
            result = grade_parallel(result, g)
        return result

    elif isinstance(term, If):
        g_then = estimate_cost(term.then_)
        g_else = estimate_cost(term.else_)
        # 最坏情况: 取成本更高的分支
        cond_cost = CostGrade()
        if isinstance(term.cond, Term):
            cond_cost = estimate_cost(term.cond)
        worst = g_then if g_then.money >= g_else.money else g_else
        return grade_serial(cond_cost, worst)

    elif isinstance(term, Loop):
        body_cost = estimate_cost(term.body)
        return grade_iterate(body_cost, term.max_steps)

    elif isinstance(term, Guard):
        inner_cost = estimate_cost(term.agent)
        return grade_guard(inner_cost, term.retry)

    elif isinstance(term, Memory):
        inner_cost = estimate_cost(term.agent)
        # Memory 本身几乎零成本
        return grade_serial(CostGrade(latency=0.001), inner_cost)

    elif isinstance(term, Route):
        cls_cost = estimate_cost(term.classifier)
        # 最坏情况: 选成本最高的路由
        route_costs = [estimate_cost(r) for r in term.routes.values()]
        if route_costs:
            worst_route = max(route_costs, key=lambda g: g.money)
            return grade_serial(cls_cost, worst_route)
        return cls_cost

    else:
        # 多智能体扩展等 — 尝试分析
        try:
            from .multiagent import AsyncPar

            if isinstance(term, AsyncPar):
                grades = [estimate_cost(a) for a in term.agents]
                result = grades[0]
                for g in grades[1:]:
                    result = grade_parallel(result, g)
                return result
        except ImportError:
            pass

    return CostGrade()  # 未知 term → 零成本（保守下界）


def validate_cost(
    predicted: CostGrade,
    actual_tokens: int,
    actual_money: float,
    deviation_threshold: float = 2.0,
) -> Dict[str, Any]:
    """
    DESIGN-08: Cross-validate cost prediction against actual execution.

    Returns dict with:
        - valid: bool (True if actual within threshold of predicted)
        - deviation_tokens: float (actual/predicted ratio)
        - deviation_money: float (actual/predicted ratio)
        - alert: str or None (warning message if deviation exceeds threshold)
    """
    result: Dict[str, Any] = {
        "valid": True,
        "deviation_tokens": 0.0,
        "deviation_money": 0.0,
        "alert": None,
    }

    if predicted.tokens > 0:
        result["deviation_tokens"] = actual_tokens / predicted.tokens
        if result["deviation_tokens"] > deviation_threshold:
            result["valid"] = False
            result["alert"] = (
                f"COST_ANOMALY: Actual tokens ({actual_tokens}) are "
                f"{result['deviation_tokens']:.1f}x the predicted upper bound "
                f"({predicted.tokens}). Threshold: {deviation_threshold}x."
            )

    if predicted.money > 0:
        result["deviation_money"] = actual_money / predicted.money
        if result["deviation_money"] > deviation_threshold:
            result["valid"] = False
            if result["alert"]:
                result["alert"] += (
                    f" Money: ${actual_money:.4f} vs predicted ${predicted.money:.4f}."
                )
            else:
                result["alert"] = (
                    f"COST_ANOMALY: Actual cost (${actual_money:.4f}) is "
                    f"{result['deviation_money']:.1f}x the predicted upper bound "
                    f"(${predicted.money:.4f})."
                )

    return result


def format_cost_estimate(grade: CostGrade) -> str:
    """格式化成本估算为人类可读字符串"""
    lines = [
        f"┌─ Cost Estimate (Paper III §4.3) ─────────────┐",
        f"│ Success probability: {grade.probability:.1%}",
        f"│ Max tokens:         {grade.tokens:,}",
        f"│ Max latency:        {grade.latency:.1f}s",
        f"│ Max cost:           ${grade.money:.4f}",
        f"└──────────────────────────────────────────────┘",
    ]
    return "\n".join(lines)
