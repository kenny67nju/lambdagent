"""
test_cek_claude.py — CEK Machine tests using Claude Code Max Plan (no API key needed)

Uses ClaudeLam which calls `claude -p` CLI instead of direct API calls.
Requires: Claude Code installed (`npm install -g @anthropic-ai/claude-code`)

Usage:
    python -m lambdagent.examples.test_cek_claude
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lambdagent.cek_machine import AgentCEKMachine, CostVector
from lambdagent.primitives import Tool, Compose, If, Loop, Pair
from lambdagent.extensions import Guard
from agentexample.agent67.core.claude_lam import ClaudeLam


def test_1_simple_lam():
    """T1: Single ClaudeLam call via CEK Machine."""
    print("=" * 60)
    print("T1: Simple ClaudeLam call")
    print("=" * 60)

    agent = ClaudeLam("翻译", "你是翻译官。将输入翻译为英文，只输出翻译结果。")
    machine = AgentCEKMachine()
    result = machine.run(agent, "你好世界")

    print(f"  Input:  '你好世界'")
    print(f"  Output: '{result}'")
    print(f"  Steps:  {machine.step_count}")
    print(f"  Cost:   {machine.state.cost}")
    machine.print_trace()
    print()
    return result


def test_2_compose_with_lam():
    """T2: ClaudeLam >> Tool pipeline via CEK."""
    print("=" * 60)
    print("T2: ClaudeLam >> Tool (compose)")
    print("=" * 60)

    translate = ClaudeLam("翻译", "将输入翻译为英文，只输出翻译结果，不要解释。")
    count_words = Tool("计数", lambda text: f"{len(str(text).split())} words")
    pipeline = Compose(translate, count_words)

    machine = AgentCEKMachine()
    result = machine.run(pipeline, "今天天气很好")

    print(f"  Input:  '今天天气很好'")
    print(f"  Output: '{result}'")
    print(f"  Steps:  {machine.step_count}")
    machine.print_trace()
    print()
    return result


def test_3_lam_compose_lam():
    """T3: ClaudeLam >> ClaudeLam (two LLM calls)."""
    print("=" * 60)
    print("T3: ClaudeLam >> ClaudeLam (two LLM calls)")
    print("=" * 60)

    translate = ClaudeLam("翻译", "将输入翻译为英文，只输出翻译结果。")
    summarize = ClaudeLam("摘要", "用一句话概括输入内容，只输出概括结果。")
    pipeline = Compose(translate, summarize)

    machine = AgentCEKMachine()
    result = machine.run(pipeline, "Lambda演算由Alonzo Church于1930年代提出，是计算理论的基础。")

    print(f"  Input:  'Lambda演算由Alonzo Church...'")
    print(f"  Output: '{result}'")
    print(f"  Steps:  {machine.step_count}")
    print(f"  Cost:   {machine.state.cost}")
    machine.print_trace()
    print()
    return result


def test_4_pair_with_lam():
    """T4: Pair(ClaudeLam, ClaudeLam) — parallel two LLM calls."""
    print("=" * 60)
    print("T4: Pair(ClaudeLam, ClaudeLam)")
    print("=" * 60)

    to_english = ClaudeLam("英翻", "翻译为英文，只输出结果。")
    to_french = ClaudeLam("法翻", "翻译为法文，只输出结果。")
    parallel = Pair(to_english, to_french)

    machine = AgentCEKMachine()
    result = machine.run(parallel, "你好")

    print(f"  Input:  '你好'")
    print(f"  Output: {result}")
    print(f"  Steps:  {machine.step_count}")
    machine.print_trace()
    print()
    return result


def test_5_loop_with_lam():
    """T5: Loop with ClaudeLam body — iterative refinement."""
    print("=" * 60)
    print("T5: Loop(ClaudeLam) — iterative refinement")
    print("=" * 60)

    refine = ClaudeLam(
        "优化",
        "你是写作助手。改进输入文本使其更简洁有力，只输出改进后的文本。"
    )

    # Stop after 2 iterations or if text is short enough
    loop = Loop(
        refine,
        condition=lambda text, step: step >= 2,  # max 2 refinements
        max_steps=3,
    )

    machine = AgentCEKMachine()
    result = machine.run(loop, "这个东西非常非常好，我觉得它真的特别特别棒，简直太厉害了。")

    print(f"  Input:  '这个东西非常非常好...'")
    print(f"  Output: '{result}'")
    print(f"  Steps:  {machine.step_count}")
    print(f"  Cost:   {machine.state.cost}")
    machine.print_trace()
    print()
    return result


def test_6_guard_with_lam():
    """T6: Guard(ClaudeLam) — validated output."""
    print("=" * 60)
    print("T6: Guard(ClaudeLam) — validated output")
    print("=" * 60)

    translator = ClaudeLam(
        "翻译",
        "将输入翻译为英文。只输出英文翻译，不要包含任何中文字符。"
    )

    # Validate: output should not contain Chinese characters
    def no_chinese(text):
        return not any('\u4e00' <= c <= '\u9fff' for c in str(text))

    guarded = Guard(translator, validator=no_chinese, retry=2)

    machine = AgentCEKMachine()
    result = machine.run(guarded, "机器学习是人工智能的子领域")

    print(f"  Input:  '机器学习是人工智能的子领域'")
    print(f"  Output: '{result}'")
    print(f"  Valid:  {no_chinese(str(result))}")
    print(f"  Steps:  {machine.step_count}")
    machine.print_trace()
    print()
    return result


def test_7_route_with_lam():
    """T7: Route with ClaudeLam — LLM-based dispatch."""
    print("=" * 60)
    print("T7: Route with ClaudeLam — LLM dispatch")
    print("=" * 60)

    # Classifier: determine the task type
    classifier = Tool("分类", lambda text: "翻译" if "翻译" in text else "摘要")

    translate_agent = ClaudeLam("翻译", "将输入翻译为英文，只输出结果。")
    summarize_agent = ClaudeLam("摘要", "用一句话概括输入，只输出结果。")

    from lambdagent.extensions import Route
    router = Route(
        classifier,
        routes={"翻译": translate_agent, "摘要": summarize_agent}
    )

    machine = AgentCEKMachine()
    result = machine.run(router, "请翻译：今天是个好日子")

    print(f"  Input:  '请翻译：今天是个好日子'")
    print(f"  Output: '{result}'")
    print(f"  Steps:  {machine.step_count}")
    machine.print_trace()
    print()
    return result


if __name__ == "__main__":
    print("\n🔧 CEK Machine + Claude Code Max Plan Tests\n")
    print("All tests use `claude -p` CLI — no API key required.\n")

    results = {}
    tests = [
        ("T1: Simple Lam", test_1_simple_lam),
        ("T2: Lam >> Tool", test_2_compose_with_lam),
        ("T3: Lam >> Lam", test_3_lam_compose_lam),
        ("T4: Pair(Lam, Lam)", test_4_pair_with_lam),
        ("T5: Loop(Lam)", test_5_loop_with_lam),
        ("T6: Guard(Lam)", test_6_guard_with_lam),
        ("T7: Route + Lam", test_7_route_with_lam),
    ]

    for name, test_fn in tests:
        try:
            r = test_fn()
            results[name] = ("PASS", r)
        except Exception as e:
            results[name] = ("FAIL", str(e))
            print(f"  ❌ {name}: {e}\n")

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, (status, _) in results.items():
        mark = "✅" if status == "PASS" else "❌"
        print(f"  {mark} {name}: {status}")
    passed = sum(1 for s, _ in results.values() if s == "PASS")
    print(f"\n  {passed}/{len(results)} passed")
