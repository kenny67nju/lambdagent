"""
lambdagent.patterns — Layer 2: Reusable Collaboration Patterns

可复用的多智能体协作模式，位于 Skill（原子能力）和 Orchestration（业务编排）之间。

每个 Pattern 是一个高阶函数：接受 Term（Skill），返回 Term（组合后的协作流程）。
Pattern 自身不包含业务逻辑——它定义的是"协作结构"。

Lambda 语义:
    Pattern = (Term → ... → Term) — 高阶函数
    review_pattern(producer, reviewer) = Loop(producer >> Guard(reviewer))
    fan_out_merge(agents, merger)      = Par(agents) >> merger
    escalation(agents)                 = If(ok?, done, next_agent)

基于 Paper II 代数定律:
    - fan_out_merge 利用 Pair 对称律 (Thm 41)
    - pipeline 利用组合结合律 (Thm 36)
    - review_pattern 利用循环展开律 (Thm 39)
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union
from .core import Term, Context
from .primitives import Lam, Compose, If, Loop, Pair, Tool
from .extensions import Par, Route, Guard, Memory


# ════════════════════════════════════════════════════════════
# Pattern Registry
# ════════════════════════════════════════════════════════════

_PATTERN_REGISTRY: Dict[str, Callable] = {}


def pattern(name: str):
    """Decorator to register a pattern in the global registry."""

    def decorator(fn):
        _PATTERN_REGISTRY[name] = fn
        fn.pattern_name = name
        return fn

    return decorator


def get_pattern(name: str) -> Callable:
    """Look up a registered pattern by name."""
    if name not in _PATTERN_REGISTRY:
        raise KeyError(
            f"Unknown pattern: '{name}'. Available: {list(_PATTERN_REGISTRY.keys())}"
        )
    return _PATTERN_REGISTRY[name]


def list_patterns() -> List[Dict[str, str]]:
    """List all registered patterns with descriptions."""
    return [
        {"name": name, "doc": fn.__doc__ or ""}
        for name, fn in _PATTERN_REGISTRY.items()
    ]


# ════════════════════════════════════════════════════════════
# Pattern 1: Review (Producer → Guard → Loop)
# ════════════════════════════════════════════════════════════


@pattern("review")
def review_pattern(
    producer: Term,
    reviewer: Term,
    max_rounds: int = 3,
    approval_keyword: str = "APPROVED",
) -> Term:
    """
    生产者-审查者模式: producer 产出 → reviewer 审查 → 通过或重做。

    Lambda: Y_n(λself.λx. let r = producer(x) in
                          IF reviewer(r) = APPROVED THEN r ELSE self(feedback(r)))

    用途: 代码审查、论文修改、翻译校对

    Args:
        producer: 生产者 Agent（生成内容）
        reviewer: 审查者 Agent（评价质量，输出含 approval_keyword 表示通过）
        max_rounds: 最大审查轮数
        approval_keyword: 审查通过标志

    Example:
        pipeline = review_pattern(
            Lam("writer", "写一篇技术博客"),
            Lam("editor", "审查文章质量，通过则回复 APPROVED"),
            max_rounds=3,
        )
    """
    validated = Guard(
        producer,
        validator=lambda r: approval_keyword in str(r),
        retry=0,  # Guard 内部不重试，由外层 Loop 控制
    )

    def _not_approved(result, step):
        return approval_keyword in str(result)

    return Loop(
        body=Compose(validated, reviewer),
        condition=_not_approved,
        max_steps=max_rounds,
    )


# ════════════════════════════════════════════════════════════
# Pattern 2: Fan-Out Merge (Parallel → Synthesize)
# ════════════════════════════════════════════════════════════


@pattern("fan_out_merge")
def fan_out_merge(
    agents: List[Term],
    merger: Term,
) -> Term:
    """
    扇出并行执行，然后合并结果。

    Lambda: λx. merger(Par(a₁(x), a₂(x), ..., aₙ(x)))
    Paper II Thm 41 (Pair 对称律): 并行顺序不影响结果

    用途: 多角度分析、多源搜索、专家委员会

    Args:
        agents: 并行执行的 Agent 列表
        merger: 合并 Agent（接收所有结果，综合输出）

    Example:
        analysis = fan_out_merge(
            agents=[researcher, competitor_analyst, audience_profiler],
            merger=Lam("synthesizer", "综合所有分析结果"),
        )
    """
    parallel = Par(*agents)

    # 将 tuple 结果转为 merger 可读的文本
    def _format_results(results):
        if isinstance(results, (tuple, list)):
            parts = []
            for i, r in enumerate(results):
                parts.append(f"[Result {i + 1}]\n{r}")
            return "\n\n".join(parts)
        return str(results)

    formatter = Tool("format_parallel_results", _format_results)
    return Compose(parallel, Compose(formatter, merger))


# ════════════════════════════════════════════════════════════
# Pattern 3: Pipeline (Sequential Chain)
# ════════════════════════════════════════════════════════════


@pattern("pipeline")
def pipeline_pattern(*stages: Term) -> Term:
    """
    顺序管道: f >> g >> h

    Lambda: λx. h(g(f(x)))
    Paper II Thm 36 (结合律): (f>>g)>>h ≡ f>>(g>>h)

    用途: 翻译→润色→校对, 提取→分析→报告

    Example:
        pipe = pipeline_pattern(translator, polisher, reviewer)
    """
    if len(stages) == 0:
        return Tool("identity", lambda x: x)
    if len(stages) == 1:
        return stages[0]
    result = stages[0]
    for s in stages[1:]:
        result = Compose(result, s)
    return result


# ════════════════════════════════════════════════════════════
# Pattern 4: Escalation Chain
# ════════════════════════════════════════════════════════════


@pattern("escalation")
def escalation_pattern(
    agents: List[Term],
    escalation_keyword: str = "ESCALATE",
) -> Term:
    """
    逐级升级: 前一个搞不定就交给下一个。

    Lambda: λx. IF ok(a₁(x)) THEN a₁(x)
                ELSE IF ok(a₂(x)) THEN a₂(x)
                ELSE ... ELSE aₙ(x)

    用途: L1→L2→L3 客服、简单→复杂模型、快速→精确

    Args:
        agents: 按能力递增排列的 Agent 列表
        escalation_keyword: 输出包含此关键词表示需要升级

    Example:
        support = escalation_pattern(
            [Lam("L1", "基础客服"), Lam("L2", "高级客服"), Lam("L3", "专家")],
            escalation_keyword="ESCALATE",
        )
    """
    if not agents:
        raise ValueError("escalation_pattern requires at least 1 agent")
    if len(agents) == 1:
        return agents[0]

    # 从最后一个开始构建嵌套 If
    result = agents[-1]  # 最后一级不检查，直接执行
    for agent in reversed(agents[:-1]):
        result = If(
            cond=lambda x, a=agent: escalation_keyword not in str(a.apply(x)),
            then_=agent,
            else_=result,
        )
    return result


# ════════════════════════════════════════════════════════════
# Pattern 5: Map-Reduce
# ════════════════════════════════════════════════════════════


@pattern("map_reduce")
def map_reduce_pattern(
    splitter: Term,
    mapper: Term,
    reducer: Term,
) -> Term:
    """
    分而治之: 拆分 → 并行处理每块 → 合并。

    Lambda: λx. reducer(Par(mapper(chunk) for chunk in splitter(x)))

    用途: 大文档分析、多文件处理、批量数据转换

    Args:
        splitter: 将输入拆分为多个块
        mapper: 对每个块独立处理
        reducer: 将所有处理结果合并

    Example:
        doc_analyzer = map_reduce_pattern(
            splitter=Tool("split", lambda doc: doc.split("\\n\\n")),
            mapper=Lam("analyze", "分析这段文本的关键论点"),
            reducer=Lam("merge", "综合所有段落的分析"),
        )
    """

    def _map_and_reduce(input_val):
        # Split
        chunks = splitter.apply(input_val)
        if isinstance(chunks, str):
            chunks = chunks.split("\n---\n")  # default split
        if not isinstance(chunks, (list, tuple)):
            chunks = [chunks]

        # Map (parallel via threads)
        from concurrent.futures import ThreadPoolExecutor
        from .core import Context

        results = []
        with ThreadPoolExecutor(max_workers=min(len(chunks), 8)) as pool:
            futures = [pool.submit(mapper.apply, chunk, Context()) for chunk in chunks]
            results = [f.result() for f in futures]

        # Reduce
        combined = "\n\n".join(f"[Block {i + 1}]\n{r}" for i, r in enumerate(results))
        return reducer.apply(combined)

    return Tool("map_reduce", _map_and_reduce)


# ════════════════════════════════════════════════════════════
# Pattern 6: Debate (对抗辩论)
# ════════════════════════════════════════════════════════════


@pattern("debate")
def debate_pattern(
    proponent: Term,
    opponent: Term,
    judge: Term,
    max_rounds: int = 3,
) -> Term:
    """
    辩论模式: 正方 vs 反方，评委裁决。

    Lambda: Y_n(λself. λstate.
        let pro  = proponent(state) in
        let con  = opponent(pro) in
        let verdict = judge(pro + con) in
        IF "FINAL" in verdict THEN verdict ELSE self(state + verdict))

    用途: 决策分析、风险评估、方案对比

    Example:
        decision = debate_pattern(
            proponent=Lam("pro", "论证为什么应该采用方案 A"),
            opponent=Lam("con", "反驳方案 A 的问题"),
            judge=Lam("judge", "综合正反方观点，给出裁决。最终裁决请以 FINAL 开头"),
        )
    """

    def _one_round(state):
        ctx = Context()
        pro_arg = proponent.apply(state, ctx)
        con_arg = opponent.apply(str(pro_arg), ctx)
        verdict = judge.apply(f"正方: {pro_arg}\n\n反方: {con_arg}", ctx)
        return str(verdict)

    round_tool = Tool("debate_round", _one_round)
    return Loop(
        body=round_tool,
        condition=lambda result, step: "FINAL" in str(result),
        max_steps=max_rounds,
    )
