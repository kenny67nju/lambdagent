"""
lambdagent.from_config — v1 兼容层，已迁移到 lambdagent.fromconfig (v2)

请使用:
    from lambdagent.fromconfig import from_config, build_agent, describe_config
"""
import warnings
warnings.warn(
    "lambdagent.from_config is deprecated (v1). "
    "Use lambdagent.fromconfig (v2) instead: "
    "from lambdagent.fromconfig import from_config, lint_config, to_lambda_expr",
    DeprecationWarning,
    stacklevel=2,
)

# 重导出 v2，保持旧导入路径可用
from .fromconfig import from_config, build_agent, describe_config  # noqa: F401
