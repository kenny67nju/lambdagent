"""
lambdagent.fromconfig — YAML Agent Configuration -> Lambda Term Compiler (v2)

Enhanced compiler supporting all 5 agent types:
  simple, react, chain, router, parallel

Plus: schema validation, lint, Lambda expression export.
"""

from .compiler import from_config, build_agent
from .lambda_expr import describe_config, to_lambda_expr
from .lint import lint_config, format_lint, LintResult
from .errors import CompileError, SchemaError, SemanticError
from .schema import validate_schema

__all__ = [
    "from_config",
    "build_agent",
    "describe_config",
    "to_lambda_expr",
    "lint_config",
    "format_lint",
    "LintResult",
    "CompileError",
    "SchemaError",
    "SemanticError",
    "validate_schema",
]
