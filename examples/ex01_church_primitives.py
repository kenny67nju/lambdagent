"""
示例 1: 用 lambdagent DSL 重写 Church 原语实验

验证 DSL 能正确表达所有 Lambda 演算基本构件。
每个 Church 原语用 Dataset.to_lam() 构造。

用法:
    export ANTHROPIC_API_KEY=sk-...
    python examples/ex01_church_primitives.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lambdagent import Dataset, Lam, Compose, If, Loop, Pair, Fst, Snd, Tool, Context


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

    # ────────────────────────────────────────
    # 1. SUCC: 后继函数 ≡ λn.λf.λx. f(n f x)
    # ────────────────────────────────────────
    print("\n[1] SUCC (后继函数)")
    succ = Dataset(
        examples=[("0","1"),("1","2"),("2","3"),("5","6"),("9","10"),("99","100")],
        description="Given a number n, output n+1.",
    ).to_lam("SUCC")

    for n in [3, 7, 42, 255]:
        check(f"SUCC({n})", str(n+1), succ(str(n), ctx))

    # ────────────────────────────────────────
    # 2. TRUE/FALSE: Church 布尔值
    # ────────────────────────────────────────
    print("\n[2] TRUE / FALSE (Church 布尔值)")
    true_ = Dataset(
        examples=[("A='cat', B='dog'","cat"),("A='1', B='0'","1"),("A='yes', B='no'","yes")],
        description="Given two options A and B, always select A (the first one).",
    ).to_lam("TRUE")

    false_ = Dataset(
        examples=[("A='cat', B='dog'","dog"),("A='1', B='0'","0"),("A='yes', B='no'","no")],
        description="Given two options A and B, always select B (the second one).",
    ).to_lam("FALSE")

    check("TRUE(sun,moon)", "sun", true_("A='sun', B='moon'", ctx))
    check("FALSE(sun,moon)", "moon", false_("A='sun', B='moon'", ctx))
    check("TRUE(alpha,omega)", "alpha", true_("A='alpha', B='omega'", ctx))
    check("FALSE(alpha,omega)", "omega", false_("A='alpha', B='omega'", ctx))

    # ────────────────────────────────────────
    # 3. AND / OR / NOT: 逻辑运算
    # ────────────────────────────────────────
    print("\n[3] AND / OR / NOT (逻辑运算)")
    and_ = Dataset(
        examples=[("TRUE, TRUE","TRUE"),("TRUE, FALSE","FALSE"),
                  ("FALSE, TRUE","FALSE"),("FALSE, FALSE","FALSE")],
        description="Logical AND.",
    ).to_lam("AND")

    or_ = Dataset(
        examples=[("TRUE, TRUE","TRUE"),("TRUE, FALSE","TRUE"),
                  ("FALSE, TRUE","TRUE"),("FALSE, FALSE","FALSE")],
        description="Logical OR.",
    ).to_lam("OR")

    not_ = Dataset(
        examples=[("TRUE","FALSE"),("FALSE","TRUE")],
        description="Logical NOT.",
    ).to_lam("NOT")

    check("AND(T,T)", "TRUE", and_("TRUE, TRUE", ctx))
    check("AND(T,F)", "FALSE", and_("TRUE, FALSE", ctx))
    check("OR(F,T)", "TRUE", or_("FALSE, TRUE", ctx))
    check("OR(F,F)", "FALSE", or_("FALSE, FALSE", ctx))
    check("NOT(T)", "FALSE", not_("TRUE", ctx))
    check("NOT(F)", "TRUE", not_("FALSE", ctx))

    # ────────────────────────────────────────
    # 4. IF-THEN-ELSE: 条件分支
    # ────────────────────────────────────────
    print("\n[4] IF-THEN-ELSE (条件分支)")
    if_lds = Dataset(
        examples=[
            ("condition=TRUE, then='yes', else='no'", "yes"),
            ("condition=FALSE, then='yes', else='no'", "no"),
            ("condition=TRUE, then='42', else='0'", "42"),
            ("condition=FALSE, then='42', else='0'", "0"),
        ],
        description="If condition is TRUE, output THEN value. If FALSE, output ELSE value.",
    ).to_lam("IF")

    check("IF(T,accept,reject)", "accept",
          if_lds("condition=TRUE, then='accept', else='reject'", ctx))
    check("IF(F,accept,reject)", "reject",
          if_lds("condition=FALSE, then='accept', else='reject'", ctx))

    # ────────────────────────────────────────
    # 5. 函数组合: DOUBLE ∘ SUCC ≡ λx. g(f(x))
    # ────────────────────────────────────────
    print("\n[5] DOUBLE ∘ SUCC (函数组合 via >>)")
    double = Dataset(
        examples=[("0","0"),("1","2"),("2","4"),("3","6"),("5","10"),("10","20")],
        description="Given a number n, output 2*n.",
    ).to_lam("DOUBLE")

    # 用 >> 组合！这是 DSL 的核心语法
    pipeline = succ >> double  # λx. DOUBLE(SUCC(x))

    for n in [0, 3, 5, 10]:
        check(f"(DOUBLE∘SUCC)({n})", str(2*(n+1)), pipeline(str(n), ctx))

    # ────────────────────────────────────────
    # 6. PAIR / FST / SND: 有序对
    # ────────────────────────────────────────
    print("\n[6] Pair / Fst / Snd (有序对)")
    # 用 Pair 同时运行 succ 和 double
    pair_agent = Pair(succ, double)
    result = pair_agent("5", ctx)
    check("Pair(SUCC,DOUBLE)(5)", ("6", "10"), result)

    # Fst / Snd 解构
    check("Fst(6,10)", "6", Fst()(result, ctx))
    check("Snd(6,10)", "10", Snd()(result, ctx))

    # 组合: Pair >> Fst
    first_of_pair = pair_agent >> Fst()
    check("(Pair>>Fst)(5)", "6", first_of_pair("5", ctx))

    # ────────────────────────────────────────
    # 7. Tool: 外部函数 → Lambda 项
    # ────────────────────────────────────────
    print("\n[7] Tool (外部函数提升)")
    square = Tool("square", lambda x: str(int(x) ** 2))
    check("square(7)", "49", square("7", ctx))

    # Tool 也能用 >> 组合
    succ_then_square = succ >> square  # λx. square(succ(x)) = (x+1)²
    check("(SUCC>>square)(4)", "25", succ_then_square("4", ctx))

    # ────────────────────────────────────────
    # 8. Church 数: f^n(x)
    # ────────────────────────────────────────
    print("\n[8] Church 数 (f^n(x) via Loop)")
    # Church 数 3 = 将 SUCC 作用于 0 共 3 次
    # 用 Loop 实现: 迭代 body n 次
    def apply_n_times(f: Tool, n: int):
        """Church 数 c_n: λf.λx. f^n(x)"""
        return Loop(f, condition=lambda r, step: step >= n - 1, max_steps=n)

    add1 = Tool("add1", lambda x: str(int(x) + 1))
    church_3 = apply_n_times(add1, 3)
    check("c_3(add1)(0) = add1^3(0)", "3", church_3("0", ctx))

    church_5 = apply_n_times(add1, 5)
    check("c_5(add1)(0) = add1^5(0)", "5", church_5("0", ctx))

    dbl = Tool("dbl", lambda x: str(int(x) * 2))
    church_4_dbl = apply_n_times(dbl, 4)
    check("c_4(dbl)(1) = dbl^4(1) = 16", "16", church_4_dbl("1", ctx))

    # ────────────────────────────────────────
    # 汇总
    # ────────────────────────────────────────
    total = passed + failed
    print(f"\n{'='*50}")
    print(f"总计: {passed}/{total} 通过")
    print(f"{'='*50}")

    # 打印 β-规约追踪
    print(f"\nβ-规约追踪 ({len(ctx.trace)} 步):")
    ctx.print_trace()


if __name__ == "__main__":
    main()
