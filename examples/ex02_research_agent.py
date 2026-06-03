"""
示例 2: 多步研究 Agent — 完整的 Lambda 演算程序

Lambda 演算解读:
    research = decompose >> map_research >> synthesize >> Loop(critique >> refine)

    其中:
    - decompose:    λ 抽象，将问题分解为子问题
    - map_research: 高阶函数，对每个子问题应用 search + analyze
    - synthesize:   λ 抽象，综合多个发现
    - Loop(critique >> refine): Y 组合子，迭代优化直到不动点

用法:
    export ANTHROPIC_API_KEY=sk-...
    python examples/ex02_research_agent.py
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lambdagent import Lam, Compose, Tool, Loop, Par, If, Context


def main():
    ctx = Context()

    # ════════════════════════════════════════════
    # Stage 1: Decompose（分解问题 → 子问题列表）
    # Lambda: λ question. [sub_q1, sub_q2, sub_q3]
    # ════════════════════════════════════════════

    decompose = Lam(
        "decompose",
        prompt=(
            "You are a research planner. Given a research question, "
            "break it into exactly 3 focused sub-questions. "
            "Output as a JSON array of strings. Example: "
            '[\"What is X?\", \"How does X affect Y?\", \"What are alternatives to X?\"]'
        ),
        output_parser=lambda x: json.loads(x) if x.strip().startswith("[") else [x],
    )

    # ════════════════════════════════════════════
    # Stage 2: Research（对每个子问题进行研究）
    # Lambda: MAP research_one sub_questions
    # ════════════════════════════════════════════

    research_one = Lam(
        "research_one",
        prompt=(
            "You are a research analyst. Given a sub-question, provide a concise, "
            "factual answer in 2-3 sentences based on your knowledge. "
            "Focus on accuracy and cite specific facts when possible."
        ),
    )

    def map_research(sub_questions):
        """高阶函数: MAP research_one over list"""
        results = []
        for i, q in enumerate(sub_questions):
            print(f"    研究子问题 {i+1}/{len(sub_questions)}: {q[:50]}...")
            answer = research_one(q, ctx)
            results.append(f"Q: {q}\nA: {answer}")
        return "\n\n".join(results)

    map_research_tool = Tool("map_research", map_research)

    # ════════════════════════════════════════════
    # Stage 3: Synthesize（综合发现）
    # Lambda: λ findings. summary
    # ════════════════════════════════════════════

    synthesize = Lam(
        "synthesize",
        prompt=(
            "You are a research synthesizer. Given research findings from multiple "
            "sub-questions, write a coherent 2-paragraph research summary. "
            "Organize by themes, not by question. Be precise and balanced."
        ),
    )

    # ════════════════════════════════════════════
    # Stage 4: Critique + Refine Loop（Y 组合子）
    # Lambda: Y(λself. λdraft. if good(critique(draft)) then draft else self(refine(draft)))
    # ════════════════════════════════════════════

    critique = Lam(
        "critique",
        prompt=(
            "You are a critical reviewer. Review this research summary for:\n"
            "1) Factual accuracy\n"
            "2) Missing perspectives\n"
            "3) Clarity\n\n"
            "If excellent, respond with exactly: APPROVED\n"
            "Otherwise, provide 2-3 specific improvement suggestions."
        ),
    )

    refine = Lam(
        "refine",
        prompt=(
            "You are an editor. Given a research summary followed by reviewer feedback, "
            "produce an improved version. Address all feedback points. "
            "Output ONLY the improved summary."
        ),
    )

    def critique_and_refine(draft):
        """Y 组合子的一步展开: critique → (if approved return, else refine)"""
        review = critique(draft, ctx)
        if "APPROVED" in review.upper():
            return draft  # 不动点！
        return refine(f"Summary:\n{draft}\n\nFeedback:\n{review}", ctx)

    refine_loop = Loop(
        Tool("critique_refine", critique_and_refine),
        condition=lambda result, step: step >= 2,  # 最多 3 轮
        max_steps=3,
    )

    # ════════════════════════════════════════════
    # 组装完整管道: >> 就是函数组合
    # ════════════════════════════════════════════

    research_agent = decompose >> map_research_tool >> synthesize >> refine_loop

    # ════════════════════════════════════════════
    # 执行
    # ════════════════════════════════════════════

    question = "What are the implications of quantum computing for modern cryptography?"

    print("=" * 60)
    print("lambdagent: 多步研究 Agent")
    print("=" * 60)
    print(f"\n研究问题: {question}\n")
    print("执行管道: decompose >> map_research >> synthesize >> Loop(critique/refine)")
    print("-" * 60)

    result = research_agent(question, ctx)

    print("\n" + "=" * 60)
    print("最终报告:")
    print("=" * 60)
    print(result)

    print("\n" + "=" * 60)
    print(f"β-规约追踪 ({len(ctx.trace)} 步):")
    print("=" * 60)
    ctx.print_trace()


if __name__ == "__main__":
    main()
