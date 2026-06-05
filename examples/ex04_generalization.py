"""
实验 5: 自定义函数泛化测试

目的：消除"LLM 只是查表/用预训练知识"的质疑。
设计 LLM 不可能在预训练中见过的函数，验证 LDS 确实从 few-shot 中学到了新函数。

三类测试：
A) 自定义编码函数：encode(n) = n*3+7（LLM 见过乘法和加法，但没见过这个特定组合）
B) 自定义符号逻辑：用非标准标签（ACCEPT/DENY 代替 TRUE/FALSE）
C) 组合未见过的函数：custom_f ∘ custom_g

用法：
    export ANTHROPIC_API_KEY=sk-...
    python examples/ex04_generalization.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lambdagent import Dataset, Tool, Compose, Context


def main():
    ctx = Context()
    passed = 0
    failed = 0

    def check(name, expected, actual):
        nonlocal passed, failed
        ok = str(actual).strip() == str(expected).strip()
        status = "✓" if ok else "✗"
        print(f"  {status} {name}: expected={expected!r}, actual={actual!r}")
        if ok:
            passed += 1
        else:
            failed += 1

    # ════════════════════════════════════════════
    # A) 自定义编码函数：f(n) = n*3 + 7
    # LLM 见过乘法和加法，但这个特定组合不是标准函数
    # ════════════════════════════════════════════
    print("\n[A] 自定义编码函数: f(n) = n*3 + 7")

    encode_fn = Dataset(
        examples=[
            ("0", "7"),  # 0*3+7=7
            ("1", "10"),  # 1*3+7=10
            ("2", "13"),  # 2*3+7=13
            ("5", "22"),  # 5*3+7=22
            ("10", "37"),  # 10*3+7=37
        ],
        description="Apply the function to the input number. Output only the result number.",
    ).to_lam("custom_encode")

    test_cases_a = [
        (3, 16),
        (4, 19),
        (7, 28),
        (8, 31),
        (15, 52),
        (20, 67),
        (33, 106),
        (100, 307),
    ]
    for n, expected in test_cases_a:
        check(f"f({n})=n*3+7", str(expected), encode_fn(str(n), ctx))

    # ════════════════════════════════════════════
    # B) 自定义逆向函数：g(n) = (n - 7) / 3
    # 这是 encode 的逆函数，更难学（需要除法+减法）
    # ════════════════════════════════════════════
    print("\n[B] 自定义逆向函数: g(n) = (n-7)/3")

    decode_fn = Dataset(
        examples=[
            ("7", "0"),  # (7-7)/3=0
            ("10", "1"),  # (10-7)/3=1
            ("13", "2"),  # (13-7)/3=2
            ("22", "5"),  # (22-7)/3=5
            ("37", "10"),  # (37-7)/3=10
        ],
        description="Apply the inverse function to the input number. Output only the result number.",
    ).to_lam("custom_decode")

    test_cases_b = [
        (16, 3),
        (19, 4),
        (28, 7),
        (31, 8),
        (52, 15),
        (67, 20),
        (106, 33),
        (307, 100),
    ]
    for n, expected in test_cases_b:
        check(f"g({n})=(n-7)/3", str(expected), decode_fn(str(n), ctx))

    # ════════════════════════════════════════════
    # C) 非标准符号逻辑：用 ACCEPT/DENY 代替 TRUE/FALSE
    # 验证 LDS 学到的是"选择策略"而不是"TRUE 这个字符串的含义"
    # ════════════════════════════════════════════
    print("\n[C] 非标准符号逻辑: ACCEPT/DENY (替代 TRUE/FALSE)")

    custom_and = Dataset(
        examples=[
            ("ACCEPT, ACCEPT", "ACCEPT"),
            ("ACCEPT, DENY", "DENY"),
            ("DENY, ACCEPT", "DENY"),
            ("DENY, DENY", "DENY"),
        ],
        description="Logical AND using ACCEPT/DENY labels. Output only ACCEPT or DENY.",
    ).to_lam("custom_AND")

    custom_or = Dataset(
        examples=[
            ("ACCEPT, ACCEPT", "ACCEPT"),
            ("ACCEPT, DENY", "ACCEPT"),
            ("DENY, ACCEPT", "ACCEPT"),
            ("DENY, DENY", "DENY"),
        ],
        description="Logical OR using ACCEPT/DENY labels. Output only ACCEPT or DENY.",
    ).to_lam("custom_OR")

    custom_not = Dataset(
        examples=[
            ("ACCEPT", "DENY"),
            ("DENY", "ACCEPT"),
        ],
        description="Logical NOT using ACCEPT/DENY labels. Output only ACCEPT or DENY.",
    ).to_lam("custom_NOT")

    # 测试：AND/OR/NOT 全真值表
    check("AND(A,A)", "ACCEPT", custom_and("ACCEPT, ACCEPT", ctx))
    check("AND(A,D)", "DENY", custom_and("ACCEPT, DENY", ctx))
    check("AND(D,A)", "DENY", custom_and("DENY, ACCEPT", ctx))
    check("AND(D,D)", "DENY", custom_and("DENY, DENY", ctx))
    check("OR(A,A)", "ACCEPT", custom_or("ACCEPT, ACCEPT", ctx))
    check("OR(A,D)", "ACCEPT", custom_or("ACCEPT, DENY", ctx))
    check("OR(D,A)", "ACCEPT", custom_or("DENY, ACCEPT", ctx))
    check("OR(D,D)", "DENY", custom_or("DENY, DENY", ctx))
    check("NOT(A)", "DENY", custom_not("ACCEPT", ctx))
    check("NOT(D)", "ACCEPT", custom_not("DENY", ctx))

    # ════════════════════════════════════════════
    # D) 组合验证：decode(encode(n)) = n（往返一致性）
    # 证明自定义函数可以组合，且组合后保持正确
    # ════════════════════════════════════════════
    print("\n[D] 组合验证: decode(encode(n)) = n (往返一致性)")

    roundtrip = encode_fn >> decode_fn  # g(f(n)) = ((n*3+7)-7)/3 = n

    for n in [0, 3, 7, 12, 25, 50]:
        check(f"decode(encode({n}))={n}", str(n), roundtrip(str(n), ctx))

    # ════════════════════════════════════════════
    # E) 完全自创的映射（无数学规律）
    # 这是最强的测试：纯任意映射，LLM 无法从预训练知识推导
    # ════════════════════════════════════════════
    print("\n[E] 任意映射 (无数学规律，纯 pattern matching)")

    arbitrary = Dataset(
        examples=[
            ("apple", "7"),
            ("banana", "3"),
            ("cherry", "9"),
            ("date", "1"),
            ("elderberry", "5"),
            ("fig", "8"),
            ("grape", "2"),
        ],
        description="Map the fruit name to its assigned number. Output only the number.",
    ).to_lam("arbitrary_map")

    # 测试已有映射的准确回忆
    check("apple→7", "7", arbitrary("apple", ctx))
    check("cherry→9", "9", arbitrary("cherry", ctx))
    check("grape→2", "2", arbitrary("grape", ctx))
    check("fig→8", "8", arbitrary("fig", ctx))

    # ════════════════════════════════════════════
    # 汇总
    # ════════════════════════════════════════════
    total = passed + failed
    print(f"\n{'=' * 50}")
    print(f"实验 5 总计: {passed}/{total} 通过")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
