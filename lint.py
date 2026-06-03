"""
lambdagent.lint — v1 兼容层，已迁移到 lambdagent.fromconfig.lint (v2)

请使用:
    from lambdagent.fromconfig import lint_config, format_lint
"""
import warnings
warnings.warn(
    "lambdagent.lint is deprecated (v1). "
    "Use lambdagent.fromconfig.lint (v2) instead: "
    "from lambdagent.fromconfig import lint_config, format_lint",
    DeprecationWarning,
    stacklevel=2,
)

# 重导出 v2，保持旧导入路径可用
from .fromconfig import lint_config, format_lint  # noqa: F401
from .fromconfig.lint import LintResult as LintIssue  # noqa: F401 — 旧名兼容
