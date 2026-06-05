"""
lambdagent.dataset — Dataset 类：数据集即程序

Dataset 是 LDS 理论中的 D —— 通过示例定义函数行为。
构造 Dataset 等价于编写 Lambda 项。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Dataset:
    """
    数据集 D: LDS 计算模型中的 "程序"。

    examples 中的每个 (input, output) 对定义了函数的一个行为点。
    LLM 从这些示例中学习出完整的函数。

        Dataset([("0","1"), ("1","2"), ("2","3")])
        ≡ 定义了后继函数 SUCC 的 Lambda 项
    """

    examples: List[tuple[str, str]]
    description: str = ""
    system_prompt: str = ""

    def to_prompt(self) -> str:
        """将数据集编码为 prompt 文本"""
        lines = []
        if self.system_prompt:
            lines.append(self.system_prompt)
            lines.append("")
        if self.description:
            lines.append(self.description)
            lines.append("")
        lines.append("Examples:")
        for inp, out in self.examples:
            lines.append(f"  Input: {inp}")
            lines.append(f"  Output: {out}")
            lines.append("")
        lines.append(
            "Now apply the same pattern to the new input. "
            "Output ONLY the result, nothing else."
        )
        return "\n".join(lines)

    def to_lam(self, name: str, model: str = "claude-sonnet-4-20250514", **kwargs):
        """将 Dataset 转为 Lam（λ 抽象）"""
        from .primitives import Lam

        return Lam(name=name, prompt=self.to_prompt(), model=model, **kwargs)
