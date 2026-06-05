"""fromconfig.errors — 编译阶段异常"""

from __future__ import annotations
from lambdagent.core import LambdagentError


class CompileError(LambdagentError):
    """编译阶段错误（YAML → Term）"""

    pass


class SchemaError(CompileError):
    """YAML schema 不合法"""

    def __init__(self, field: str, expected: str, actual=None, msg: str = ""):
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(
            msg or f"Schema error at '{field}': expected {expected}, got {actual}"
        )


class SemanticError(CompileError):
    """语义不合法（schema 合法但逻辑矛盾）"""

    def __init__(self, rule: str, msg: str = ""):
        self.rule = rule
        super().__init__(msg or f"Semantic error: rule {rule}")
