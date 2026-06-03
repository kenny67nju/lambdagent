"""
lambdagent 进阶演示 — 4 个真实场景

Demo 1: 自纠正翻译器 (Loop + Guard + Compose)
    翻译 → 回译 → 比较 → 不一致则重翻 → 直到收敛

Demo 2: 多视角分析 (Par + Route + Compose)
    同一问题 → 乐观分析 | 悲观分析 | 中立分析 → 综合判断

Demo 3: 递归文档生成器 (Loop + Pair + If)
    大纲 → 逐段扩写 → 质量检查 → 不合格则重写 → 拼装成文

Demo 4: 自学函数 (Dataset + Loop + Guard)
    给几个例子 → 学出函数 → 测试 → 失败则补充例子 → 再学

用法:
    export ANTHROPIC_API_KEY=sk-...
    python examples/demo_advanced.py [1|2|3|4|all]
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lambdagent import (
    Lam, Tool, Compose, If, Loop, Pair, Fst, Snd,
    Route, Guard, Context, Dataset,
)
from lambdagent.extensions import Memory, Par


# ════════════════════════════════════════════════════════════
# Demo 1: 自纠正翻译器
#
# Lambda 结构:
#   translate = Y(λself. λ(src, ref).
#       let zh    = en2zh(src) in
#       let back  = zh2en(zh) in
#       let score = compare(src, back) in
#       IF (score > 0.8)
#           THEN zh                    ← base case: 翻译质量达标
#           ELSE self(src, zh)         ← 递归: 带着上次翻译重试
#   )
#
# 展示: Loop (Y) + Guard + Compose + Pair + Tool
# ════════════════════════════════════════════════════════════

def demo_self_correcting_translator():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Demo 1: 自纠正翻译器 (Round-trip Translation)      ║")
    print("║  Loop(translate >> back-translate >> compare)        ║")
    print("╚══════════════════════════════════════════════════════╝")

    ctx = Context()

    # 英→中
    en2zh = Lam(
        "en→zh",
        "Translate the English text to Chinese. "
        "If a previous translation attempt is provided, improve upon it. "
        "Output ONLY the Chinese translation, nothing else.",
        max_tokens=256,
    )

    # 中→英 (回译)
    zh2en = Lam(
        "zh→en",
        "Translate the Chinese text back to English. "
        "Output ONLY the English translation, nothing else.",
        max_tokens=256,
    )

    # 比较原文和回译
    judge = Lam(
        "judge",
        "Compare the original English text with the back-translated version. "
        "Rate semantic preservation on a scale 1-10. "
        "Output ONLY the number.",
        max_tokens=8,
    )

    source = (
        "The paradox of tolerance states that if a society is tolerant "
        "without limit, its ability to be tolerant is eventually seized "
        "or destroyed by the intolerant."
    )

    print(f"\n  Original: {source}\n")

    # 状态: "ORIGINAL: ...\nPREVIOUS_ATTEMPT: ...(如果有)"
    state = f"ORIGINAL: {source}"
    best_zh = None
    best_score = 0

    for step in range(4):  # Y₄: 最多 4 次展开
        print(f"  ── β-reduction round {step} ──")

        # Phase 1: 翻译 (β-规约)
        zh = en2zh(state, ctx)
        print(f"  [en→zh] {zh[:80]}...")

        # Phase 2: 回译 (β-规约)
        back = zh2en(zh, ctx)
        print(f"  [zh→en] {back[:80]}...")

        # Phase 3: 评分 (β-规约)
        score_raw = judge(f"Original: {source}\nBack-translated: {back}", ctx)
        try:
            score = int(score_raw.strip())
        except:
            score = 5
        print(f"  [score] {score}/10")

        if score > best_score:
            best_score = score
            best_zh = zh

        # Phase 4: base case 检测
        if score >= 8:
            print(f"  ✓ Score ≥ 8, terminate (base case reached)")
            break
        else:
            print(f"  → Score < 8, refining... (Y combinator continues)")
            state = f"ORIGINAL: {source}\nPREVIOUS_ATTEMPT: {zh}\nSCORE: {score}/10. Please improve."

    print(f"\n  Final translation (score={best_score}/10):")
    print(f"  {best_zh}")
    print(f"\n  β-reductions this demo: {len(ctx.trace)}")
    return ctx


# ════════════════════════════════════════════════════════════
# Demo 2: 多视角分析
#
# Lambda 结构:
#   analyze = Par(optimist, pessimist, realist)
#             >> synthesize
#
# 即: λx. synthesize(TRIPLE (optimist x) (pessimist x) (realist x))
#
# 展示: Par (并行) + Compose + Lam
# ════════════════════════════════════════════════════════════

def demo_multi_perspective():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  Demo 2: 多视角分析 (Par + Synthesize)              ║")
    print("║  Par(optimist, pessimist, realist) >> synthesize     ║")
    print("╚══════════════════════════════════════════════════════╝")

    ctx = Context()

    optimist = Lam(
        "optimist",
        "You are an extreme optimist. Analyze the topic focusing ONLY on opportunities "
        "and positive outcomes. 2-3 sentences. Start with 'BULL CASE:'",
        max_tokens=128,
    )

    pessimist = Lam(
        "pessimist",
        "You are a harsh critic. Analyze the topic focusing ONLY on risks "
        "and worst-case scenarios. 2-3 sentences. Start with 'BEAR CASE:'",
        max_tokens=128,
    )

    realist = Lam(
        "realist",
        "You are a balanced realist. Give the most likely outcome with evidence. "
        "2-3 sentences. Start with 'BASE CASE:'",
        max_tokens=128,
    )

    synthesize = Lam(
        "synthesize",
        "You receive three perspectives (bull, bear, base case). "
        "Synthesize them into a single balanced verdict in 2-3 sentences. "
        "Start with 'VERDICT:'",
        max_tokens=128,
    )

    # Par = Church TRIPLE: λx. (f x, g x, h x)
    multi_view = Par(optimist, pessimist, realist)

    # 格式化三元组
    format_views = Tool("format", lambda views: "\n\n".join(str(v) for v in views))

    # 完整 pipeline: Par >> format >> synthesize
    pipeline = multi_view >> format_views >> synthesize

    topic = "Should a startup adopt Rust instead of Go for their backend in 2026?"
    print(f"\n  Topic: {topic}\n")

    t0 = time.time()
    result = pipeline(topic, ctx)
    elapsed = time.time() - t0

    # 打印各视角
    print("  ── Three perspectives (parallel β-reduction) ──")
    for entry in ctx.trace:
        if entry.term_name in ("optimist", "pessimist", "realist"):
            print(f"  [{entry.term_name}] {str(entry.output)[:100]}...")
    print()
    print(f"  ── Synthesis ──")
    print(f"  {result}")
    print(f"\n  {len(ctx.trace)} β-reductions, {elapsed:.1f}s")
    return ctx


# ════════════════════════════════════════════════════════════
# Demo 3: 递归文档生成器
#
# Lambda 结构:
#   generate = outline >> MAP(expand) >> Loop(review_and_fix)
#
# 其中:
#   outline  = λtopic. [section1, section2, section3]
#   expand   = λsection. paragraph
#   review   = λdraft. (score, feedback)
#   fix      = λ(draft, feedback). improved_draft
#   Loop     = Y(λself.λdraft. let (s,fb) = review(draft) in
#                               IF s≥8 THEN draft ELSE self(fix(draft, fb)))
#
# 展示: Compose + Tool (MAP) + Loop (Y) + Pair + If
# ════════════════════════════════════════════════════════════

def demo_recursive_document():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  Demo 3: 递归文档生成器                               ║")
    print("║  outline >> expand_all >> Loop(review >> fix)         ║")
    print("╚══════════════════════════════════════════════════════╝")

    ctx = Context()

    # Step 1: 生成大纲
    outliner = Lam(
        "outliner",
        "Create a 3-section outline for a short article about the given topic. "
        "Output EXACTLY 3 lines, one section title per line. No numbering, no bullets.",
        max_tokens=128,
    )

    # Step 2: 逐段扩写 (MAP = Tool 包装的循环)
    expander = Lam(
        "expander",
        "Write a concise 2-3 sentence paragraph for the given section title. "
        "Output ONLY the paragraph text.",
        max_tokens=128,
    )

    def expand_all(outline_text):
        """MAP(expand) over sections — 高阶函数应用"""
        sections = [s.strip() for s in outline_text.strip().split("\n") if s.strip()]
        paragraphs = []
        for section in sections[:3]:
            para = expander(section, ctx)
            paragraphs.append(f"## {section}\n{para}")
        return "\n\n".join(paragraphs)

    map_expand = Tool("MAP(expand)", expand_all)

    # Step 3: 审稿 (返回 score + feedback)
    reviewer = Lam(
        "reviewer",
        "Review this draft article. "
        "Line 1: Score 1-10. "
        "Line 2: One specific improvement suggestion. "
        "Output ONLY these two lines.",
        max_tokens=64,
    )

    # Step 4: 修改 (根据反馈改进)
    fixer = Lam(
        "fixer",
        "Improve this draft based on the reviewer feedback. "
        "Keep the same structure (3 sections with ## headers). "
        "Make the specific improvement requested.",
        max_tokens=512,
    )

    topic = "Why sleep is the most underrated productivity tool"
    print(f"\n  Topic: {topic}\n")

    # ── Phase 1: outline (β-规约) ──
    print("  ── Phase 1: Outline ──")
    outline = outliner(topic, ctx)
    print(f"  {outline}\n")

    # ── Phase 2: MAP(expand) (N 次 β-规约) ──
    print("  ── Phase 2: Expand sections ──")
    draft = map_expand(outline, ctx)
    print(f"  (Expanded {outline.count(chr(10))+1} sections)\n")

    # ── Phase 3: Loop(review >> fix) — Y 组合子 ──
    print("  ── Phase 3: Review-Fix loop (Y combinator) ──")

    for iteration in range(3):  # Y₃
        # Review
        review = reviewer(draft, ctx)
        lines = review.strip().split("\n")
        try:
            score = int(lines[0].strip().split()[0])
        except:
            score = 6
        feedback = lines[1] if len(lines) > 1 else "No specific feedback"

        print(f"  [Round {iteration}] Score: {score}/10 | Feedback: {feedback[:60]}...")

        if score >= 8:
            print(f"  ✓ Score ≥ 8, Loop terminates (base case)")
            break

        # Fix
        fix_input = f"DRAFT:\n{draft}\n\nFEEDBACK (score {score}/10): {feedback}"
        draft = fixer(fix_input, ctx)
        print(f"  → Fixed. Continuing Y expansion...")

    print(f"\n  ── Final Document ──")
    # 只打印前 300 字符
    print(f"  {draft[:300]}...")
    print(f"\n  {len(ctx.trace)} β-reductions")
    return ctx


# ════════════════════════════════════════════════════════════
# Demo 4: 自学函数
#
# Lambda 结构:
#   learn = λexamples.
#       let f = Dataset(examples).to_lam() in
#       let tests = generate_tests(f) in
#       let (pass, fail) = run_tests(f, tests) in
#       IF (all_pass)
#           THEN f                              ← base case
#           ELSE learn(examples ∪ corrections)  ← 递归: 补充数据再学
#
# 展示: Dataset + Loop + Guard + Tool + If
# 核心洞察: LDS 可以从数据中"学出"函数，然后自我验证
# ════════════════════════════════════════════════════════════

def demo_self_learning():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  Demo 4: 自学函数 (Dataset → Learn → Test → Refine) ║")
    print("║  Loop(learn >> test >> IF pass THEN done ELSE refine)║")
    print("╚══════════════════════════════════════════════════════╝")

    ctx = Context()

    # 目标函数: f(x) = x² + 1 (LLM 不知道这个函数)
    # 我们只给几个例子，看 LDS 能不能学会

    true_fn = lambda x: x * x + 1

    # 初始训练集 (故意给很少)
    examples = [
        (0, 1),    # 0² + 1 = 1
        (1, 2),    # 1² + 1 = 2
        (3, 10),   # 3² + 1 = 10
    ]

    # 测试集 (LDS 从未见过)
    test_cases = [2, 4, 5, 7, 10]

    print(f"\n  Target: f(x) = x² + 1 (secret)")
    print(f"  Initial examples: {examples}")
    print(f"  Test cases: {test_cases}\n")

    for iteration in range(4):  # Y₄
        print(f"  ── Learning round {iteration} ({len(examples)} examples) ──")

        # 学习: 从 examples 构建 LDS
        learner = Dataset(
            examples=[(str(x), str(y)) for x, y in examples],
            description="Learn the pattern from examples. Given input number x, compute f(x). Output ONLY the number.",
        ).to_lam(f"f_v{iteration}", max_tokens=16)

        # 测试
        passed = 0
        failed_cases = []
        for x in test_cases:
            expected = true_fn(x)
            actual_raw = learner(str(x), ctx)
            try:
                actual = int(actual_raw.strip())
            except:
                actual = -1

            ok = actual == expected
            status = "✓" if ok else "✗"
            print(f"    {status} f({x}) = {actual} (expected {expected})")

            if ok:
                passed += 1
            else:
                failed_cases.append((x, expected, actual))

        rate = passed / len(test_cases) * 100
        print(f"    Pass rate: {passed}/{len(test_cases)} ({rate:.0f}%)")

        # base case: 全部通过
        if passed == len(test_cases):
            print(f"  ✓ All tests pass! Loop terminates.")
            break

        # 递归: 补充失败的用例到训练集
        if failed_cases:
            new_examples = [(x, expected) for x, expected, _ in failed_cases[:2]]
            examples.extend(new_examples)
            print(f"    → Adding corrections: {new_examples}")
            print(f"    → Continuing Y expansion with {len(examples)} examples...")

    print(f"\n  {len(ctx.trace)} β-reductions")
    return ctx


# ════════════════════════════════════════════════════════════

def main():
    demos = {
        "1": ("自纠正翻译器", demo_self_correcting_translator),
        "2": ("多视角分析", demo_multi_perspective),
        "3": ("递归文档生成器", demo_recursive_document),
        "4": ("自学函数", demo_self_learning),
    }

    # 解析参数
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"

    if choice == "all":
        total_ctx = Context()
        for key, (name, fn) in demos.items():
            ctx = fn()
            total_ctx.trace.extend(ctx.trace)
            print()
        print(f"═══ Total: {len(total_ctx.trace)} β-reductions across 4 demos ═══")
    elif choice in demos:
        name, fn = demos[choice]
        fn()
    else:
        print("Usage: python demo_advanced.py [1|2|3|4|all]")
        print()
        for k, (name, _) in demos.items():
            print(f"  {k}: {name}")


if __name__ == "__main__":
    main()
