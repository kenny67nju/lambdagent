"""
实验 6: 失败边界测试

目的：诚实地找到 LDS 失败的场景，增强论文可信度。
系统性地增大输入规模/递归深度/数据集噪声，测量正确率衰减曲线。

用法：
    export ANTHROPIC_API_KEY=sk-...
    python examples/ex05_failure_boundary.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lambdagent import Dataset, Tool, Loop, Context


def normalize(s):
    return s.strip().strip("'\"").lower()


def main():
    ctx = Context()

    # ════════════════════════════════════════════
    # A) SUCC 大数边界：从 100 到 1,000,000
    # ════════════════════════════════════════════
    print("\n[A] SUCC 大数边界测试")
    print("    数据集: {(0,1),(1,2),(2,3),(5,6),(9,10),(99,100)}")

    succ = Dataset(
        examples=[
            ("0", "1"),
            ("1", "2"),
            ("2", "3"),
            ("5", "6"),
            ("9", "10"),
            ("99", "100"),
        ],
        description="Given a number n, output n+1.",
    ).to_lam("SUCC")

    succ_tests = [100, 255, 999, 1000, 9999, 10000, 99999, 100000, 999999]
    for n in succ_tests:
        actual = succ(str(n), ctx)
        expected = str(n + 1)
        ok = normalize(actual) == normalize(expected)
        status = "✓" if ok else "✗"
        print(f"  {status} SUCC({n}) = {actual}  (expected {expected})")

    # ════════════════════════════════════════════
    # B) 阶乘递归深度：从 2! 到 20!
    # ════════════════════════════════════════════
    print("\n[B] FACTORIAL 递归深度测试")
    print("    数据集: 4 个 CoT 示例 (0!, 1!, 3!, 5!)")

    factorial = Dataset(
        examples=[
            ("0", "0! = 1\nResult: 1"),
            ("1", "1! = 1 × 0! = 1 × 1 = 1\nResult: 1"),
            (
                "3",
                "3! = 3 × 2!\n2! = 2 × 1!\n1! = 1 × 0!\n0! = 1\nSo: 1 × 1 × 2 × 3 = 6\nResult: 6",
            ),
            (
                "5",
                "5! = 5 × 4!\n4! = 4 × 3!\n3! = 3 × 2!\n2! = 2 × 1!\n1! = 1 × 0!\n0! = 1\nSo: 1 × 1 × 2 × 3 × 4 × 5 = 120\nResult: 120",
            ),
        ],
        description="Compute n! step by step. Show recursive expansion. End with 'Result: <number>'.",
    ).to_lam("FACTORIAL", max_tokens=2048)

    fact_tests = [2, 4, 6, 7, 8, 10, 12, 15, 20]
    for n in fact_tests:
        actual_raw = factorial(str(n), ctx)
        # 提取 Result 行
        actual = actual_raw
        for line in reversed(actual_raw.split("\n")):
            if "Result:" in line:
                actual = line.split("Result:")[-1].strip().replace(",", "")
                break
        expected = str(math.factorial(n))
        ok = actual == expected
        status = "✓" if ok else "✗"
        print(f"  {status} {n}! = {actual}  (expected {expected})")

    # ════════════════════════════════════════════
    # C) Church 数 double^n(1) = 2^n：指数增长边界
    # ════════════════════════════════════════════
    print("\n[C] Church 数 double^n(1) = 2^n 边界测试")

    church_double = Dataset(
        examples=[
            ("n=0, f='double', x='1'", "1"),
            ("n=1, f='double', x='1'", "2"),
            ("n=2, f='double', x='1'", "4"),
            ("n=3, f='double', x='1'", "8"),
            ("n=4, f='double', x='1'", "16"),
        ],
        description="Apply function f to value x exactly n times. double multiplies by 2.",
    ).to_lam("CHURCH_DOUBLE")

    for n in [5, 6, 7, 8, 10, 12, 15, 20]:
        actual = church_double(f"n={n}, f='double', x='1'", ctx)
        expected = str(2**n)
        ok = normalize(actual) == normalize(expected)
        status = "✓" if ok else "✗"
        print(f"  {status} double^{n}(1) = {actual}  (expected {expected})")

    # ════════════════════════════════════════════
    # D) 噪声数据集：混入 1 个错误示例
    # ════════════════════════════════════════════
    print("\n[D] 噪声数据集测试 (1/6 错误)")
    print("    正确: (0,1),(1,2),(2,3),(5,6),(9,10)")
    print("    错误: (7,77)  ← 故意的错误")

    noisy_succ = Dataset(
        examples=[
            ("0", "1"),
            ("1", "2"),
            ("2", "3"),
            ("5", "6"),
            ("7", "77"),  # ← 错误！
            ("9", "10"),
        ],
        description="Given a number n, output n+1.",
    ).to_lam("NOISY_SUCC")

    noisy_tests = [(3, "4"), (4, "5"), (6, "7"), (7, "8"), (8, "9"), (10, "11")]
    for n, expected in noisy_tests:
        actual = noisy_succ(str(n), ctx)
        ok = normalize(actual) == normalize(expected)
        status = "✓" if ok else "✗"
        note = " ← 数据集中的错误点附近" if n in [6, 7, 8] else ""
        print(f"  {status} NOISY_SUCC({n}) = {actual}  (expected {expected}){note}")

    # ════════════════════════════════════════════
    # E) 最小数据集：逐步减少 SUCC 的示例数
    # ════════════════════════════════════════════
    print("\n[E] 最小数据集测试：SUCC 需要几个示例？")

    for num_examples in [6, 4, 3, 2, 1]:
        all_examples = [
            ("0", "1"),
            ("1", "2"),
            ("2", "3"),
            ("5", "6"),
            ("9", "10"),
            ("99", "100"),
        ]
        examples = all_examples[:num_examples]
        mini_succ = Dataset(
            examples=examples,
            description="Given a number n, output n+1.",
        ).to_lam(f"SUCC_{num_examples}")

        test_inputs = [3, 7, 15, 42]
        correct = 0
        for n in test_inputs:
            actual = mini_succ(str(n), ctx)
            if normalize(actual) == str(n + 1):
                correct += 1
        rate = correct / len(test_inputs) * 100
        print(
            f"  {num_examples} 个示例 → {correct}/{len(test_inputs)} 通过 ({rate:.0f}%)"
        )

    # ════════════════════════════════════════════
    # 汇总
    # ════════════════════════════════════════════
    print(f"\n{'=' * 50}")
    print("实验 6 完成。结果用于分析 LDS 的失败边界。")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
