"""
lambdagent.core — Lambda 演算 Agent DSL 的基础抽象

每个 Agent 是一个 Lambda 项（Term）。
调用 Agent 就是函数应用（β-规约）。
Context 是求值环境（Γ）。
TraceEntry 记录每步 β-规约。
"""

from __future__ import annotations

import copy
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .lam_types import LamType, AgentType
    from .effects import Effect, ComposedEffect


# ============================================================
# 异常
# ============================================================

class LambdagentError(Exception):
    pass

class UnboundVariable(LambdagentError):
    pass

class RouteError(LambdagentError):
    pass

class ValidationError(LambdagentError):
    pass


# ============================================================
# TraceEntry: 一步 β-规约的记录
# ============================================================

@dataclass
class TraceEntry:
    """记录一次 β-规约（Agent 调用）"""
    term_name: str
    term_id: str
    input: Any
    output: Any
    duration_ms: float
    model: str = ""
    tokens_used: int = 0


# ============================================================
# Context: 求值环境 Γ
# ============================================================

@dataclass
class Context:
    """
    求值环境（Lambda 演算中的 Γ）。

    携带变量绑定、β-规约追踪、持久记忆。
    支持词法作用域（parent 链）。
    """
    bindings: Dict[str, Any] = field(default_factory=dict)
    trace: List[TraceEntry] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)
    parent: Optional[Context] = None
    # Run workspace — 每次执行的工作目录
    workspace_path: Optional[str] = None
    run_id: Optional[str] = None

    def extend(self, **bindings) -> Context:
        """创建子环境（词法作用域）"""
        return Context(
            bindings={**self.bindings, **bindings},
            trace=self.trace,
            memory=self.memory,
            parent=self,
            workspace_path=self.workspace_path,
            run_id=self.run_id,
        )

    def lookup(self, name: str) -> Any:
        """变量查找，沿作用域链向上"""
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.lookup(name)
        raise UnboundVariable(f"Unbound variable: {name}")

    def log(self, term_name: str, term_id: str, inp: Any, out: Any,
            duration_ms: float, model: str = "", tokens: int = 0):
        """记录一步 β-规约"""
        self.trace.append(TraceEntry(
            term_name=term_name, term_id=term_id,
            input=inp, output=out,
            duration_ms=duration_ms, model=model, tokens_used=tokens,
        ))

    def fork(self) -> Context:
        """
        创建独立的上下文深拷贝（Paper II Proposition 30）。

        用于并行分支: 每个分支有独立的 trace 和 memory，
        防止 writes(f) ∩ writes(g) ≠ ∅ 造成的竞态条件。

        FIX-01: 改用 deepcopy 防止嵌套对象共享引用。
        浅拷贝 dict(self.bindings) 只复制顶层 key，嵌套的 list/dict
        仍共享引用，并行修改会互相干扰。
        """
        return Context(
            bindings=copy.deepcopy(self.bindings),  # 深拷贝，隔离嵌套对象
            trace=[],                                # 独立 trace
            memory=copy.deepcopy(self.memory),       # 深拷贝，隔离嵌套对象
            parent=self.parent,                      # parent 只读，不需要拷贝
            workspace_path=self.workspace_path,      # 共享 workspace（同一次 run）
            run_id=self.run_id,                      # 共享 run_id
        )

    def merge_trace(self, other: Context):
        """合并子上下文的 trace 到当前上下文"""
        self.trace.extend(other.trace)

    def print_trace(self):
        """打印完整的 β-规约链"""
        for i, e in enumerate(self.trace):
            inp_s = str(e.input)[:60]
            out_s = str(e.output)[:60]
            print(f"  β[{i}] {e.term_name} ({e.duration_ms:.0f}ms): {inp_s} → {out_s}")


# ============================================================
# Term: 所有 Lambda 项的基类
# ============================================================

class Term(ABC):
    """
    Lambda 项基类。

    Lambda 演算:  M ::= x | λx.M | M N
    DSL:          Term 是所有 Agent 构造的基类

    每个 Term 可被调用（函数应用 = β-规约）。
    支持 >> 组合（函数组合）和 | 并行。
    """

    def __init__(self, name: str = ""):
        self._name = name or self.__class__.__name__
        self._trace_id = uuid.uuid4().hex[:8]
        # Paper III: 类型标注 (默认 Any → Any)
        self._input_type: LamType | None = None
        self._output_type: LamType | None = None
        # Paper III: 效果标注 (默认 None = 需要推断)
        self._effect: Effect | ComposedEffect | None = None

    # ── Paper III: 类型标注属性 ──

    @property
    def input_type(self) -> LamType:
        """Agent 的输入类型 (Paper III Definition 3)"""
        if self._input_type is not None:
            return self._input_type
        from .lam_types import T_ANY
        return T_ANY

    @input_type.setter
    def input_type(self, t: LamType):
        self._input_type = t

    @property
    def output_type(self) -> LamType:
        """Agent 的输出类型 (Paper III Definition 3)"""
        if self._output_type is not None:
            return self._output_type
        from .lam_types import T_ANY
        return T_ANY

    @output_type.setter
    def output_type(self, t: LamType):
        self._output_type = t

    @property
    def effect(self) -> Effect | ComposedEffect:
        """Agent 的效果标注 (Paper III Definition 6-7)"""
        if self._effect is not None:
            return self._effect
        from .effects import infer_effect_for_term
        return infer_effect_for_term(self)

    @effect.setter
    def effect(self, e: Effect | ComposedEffect):
        self._effect = e

    @property
    def agent_type(self) -> AgentType:
        """完整的 Agent 函数类型 τ1 →^ε τ2"""
        from .lam_types import AgentType
        eff = self.effect
        eff_str = repr(eff)
        return AgentType(self.input_type, self.output_type, effect=eff_str)

    @abstractmethod
    def apply(self, input: Any, ctx: Context) -> Any:
        """同步 β-规约: (self input) → result"""
        ...

    def __call__(self, input: Any, ctx: Context | None = None) -> Any:
        """语法糖: agent(x) = App(agent, x)"""
        return self.apply(input, ctx or Context())

    def __rshift__(self, other: Term) -> Term:
        """
        语法糖: f >> g = Compose(f, g) = λx. g(f(x))

        Paper III T-Compose: 检查 output(f) <: input(g)
        """
        from .primitives import Compose
        if isinstance(self, type) and issubclass(self, Term):
            raise TypeError("Use instances, not classes")

        # Paper III T-Compose: 类型检查 (仅当两端都有非 Any 类型标注时)
        from .lam_types import T_ANY, is_subtype
        f_out = self.output_type if hasattr(self, 'stages') else self.output_type
        g_in = other.input_type
        if f_out != T_ANY and g_in != T_ANY:
            if not is_subtype(f_out, g_in):
                from .lam_types import AgentTypeError
                raise AgentTypeError(
                    f"Type mismatch: {self._name} >> {other._name}",
                    source_type=f_out,
                    target_type=g_in,
                    position=0,
                )

        if hasattr(self, 'stages'):  # flatten nested Compose
            return Compose(*self.stages, other)
        return Compose(self, other)

    def __or__(self, other: Term) -> Term:
        """语法糖: f | g = Par(f, g)"""
        from .extensions import Par
        return Par(self, other)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._name!r})"
